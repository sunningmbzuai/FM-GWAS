"""gene_pca_cache.py + gene_scan.py against the vendored FM-GWAS script.

The PCA is fitted once per sex sample (All, F, M). When a cell's cohort is
exactly that sample (every participant measured), this is the model the script
fits inside the cell, so the p-values must agree with a plain
`insample_hpp_AG.py run` in every sex.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import assoc_settings
import gene_pca_cache as C
import gene_scan as G

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "fmgwas" / "insample_hpp_AG.py"
GENE = "ENSG00000000001.5"
HALF = {"cage": 5, "dnase": 4, "atac": 3, "rna_seq": 6}
N = 600
N_HAPLOTYPES = 6


def features(rng: np.random.Generator, ids: np.ndarray) -> dict:
    """Raw track fields of an array-genotyped gene: a few distinct haplotypes."""
    pairs = rng.integers(0, N_HAPLOTYPES, size=(len(ids), 2))
    data = {"subject_id": ids.astype(str)}
    for track, half in HALF.items():
        haps = rng.normal(size=(N_HAPLOTYPES, half))
        body = np.hstack([haps[pairs[:, 0]], haps[pairs[:, 1]]])
        data[f"concat_{track}_mean"] = body
        data[f"concat_{track}_tss500_mean"] = (body * 0.5 + 1).astype(np.float32)
    return data, pairs


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scan")
    rng = np.random.default_rng(3)
    ids = np.arange(2_000_001, 2_000_001 + N, dtype=np.int64)
    data, pairs = features(rng, ids)
    feature_root = tmp / "features"
    (feature_root / "ENSG00000000001.9").mkdir(parents=True)  # other version on disk
    np.savez(feature_root / "ENSG00000000001.9" / "ENSG00000000001.9.npz", **data)

    sex = rng.integers(0, 2, N)
    age = rng.normal(55, 8, N)
    effect = (pairs == 2).sum(axis=1)
    quant = 0.4 * effect + 0.02 * age + rng.normal(size=N)
    binary = (rng.random(N) < 1 / (1 + np.exp(-(effect - 0.5)))).astype(float)
    gpcs = rng.normal(size=(N, 2)) + effect[:, None] * 0.3   # ancestry tracks the gene
    quant = quant + 0.5 * gpcs[:, 0]
    people = pd.DataFrame({"participant_id": ids, "sex": sex, "age": age})

    scan = tmp / "traits"
    scan.mkdir()
    people.to_csv(scan / "participants.tsv", sep="\t", index=False)
    np.save(scan / "trait_values.npy", np.column_stack([quant, binary, gpcs]))
    pd.DataFrame({"trait_id": ["q", "b", "x_genetic_pcs:pc1", "x_genetic_pcs:pc2"],
                  "column": [0, 1, 2, 3],
                  "kind": ["quantitative", "binary", "quantitative", "quantitative"]}).to_csv(
        scan / "trait_catalog.tsv", sep="\t", index=False)
    genes = tmp / "genes.tsv"
    genes.write_text(f"gene_id\n{GENE}\n")
    for name, label in (("q", quant), ("b", binary)):
        people.assign(label=label, pc1=gpcs[:, 0], pc2=gpcs[:, 1]).to_csv(
            tmp / f"{name}_All.tsv", sep="\t", index=False)

    cache, out, out_gpc = tmp / "cache", tmp / "out", tmp / "out_gpc"
    C.main(["--genes", str(genes), "--participants", str(scan / "participants.tsv"),
            "--feature-root", str(feature_root), "--cache", str(cache)])
    floor = G.MIN_CELL_N
    G.MIN_CELL_N = 100   # N=600 leaves ~300 per sex, under the production floor
    try:
        G.main(["--scan", str(scan), "--cache", str(cache), "--genes", str(genes),
                "--out", str(out)])
        G.main(["--scan", str(scan), "--cache", str(cache), "--genes", str(genes),
                "--out", str(out_gpc), "--genetic-pcs", "2"])
    finally:
        G.MIN_CELL_N = floor
    return dict(tmp=tmp, feature_root=feature_root, genes=genes, cache=cache, out=out,
                out_gpc=out_gpc, data=data)


@pytest.mark.integration
def test_cache_rebuilds_the_scripts_pca_scores_exactly(world):
    fm = C.fmgwas()
    by_sex, meta = C.load_cache(world["cache"], GENE)
    assert meta["feature_dir"] == "ENSG00000000001.9"
    assert sorted(by_sex) == ["All", "F", "M"]
    all_ids, matrices = fm.load_gene(
        world["feature_root"] / "ENSG00000000001.9" / "ENSG00000000001.9.npz",
        ["concat_cage_dnase", "add_tss_rnaseq"])
    sex = pd.read_csv(world["tmp"] / "traits" / "participants.tsv", sep="\t").sex.to_numpy()
    for name, rows in (("All", sex >= 0), ("F", sex == 0), ("M", sex == 1)):
        ids, scores = by_sex[name]
        assert np.array_equal(ids, all_ids[rows])
        assert len(scores) == 60
        for embed, matrix in matrices.items():
            expected, info = fm.fit_pca(matrix[rows], fm.parse_pca("variance:0.95"))
            assert np.array_equal(scores[embed], expected)
            assert meta["sexes"][name]["components"][embed] == info["n_components"]


def script_p(world, trait: str, test: str, sex: str = "All",
             covariates: str = "age,sex") -> pd.Series:
    out = world["tmp"] / f"script_{trait}_{sex}_{covariates.replace(',', '_')}"
    cmd = [sys.executable, str(SCRIPT), "run", trait, sex, "--gene_tsv", str(world["genes"]),
           "--feature_root", str(world["tmp"] / "script_features"),
           "--cohort_file", str(world["tmp"] / f"{trait}_All.tsv"),
           "--covariates", covariates, "--gene_pca", assoc_settings.GENE_PCA,
           "--tests", test, "--embed_type", ",".join(assoc_settings.ALL_VIEWS),
           "--out_dir", str(out), "--no_progress"]
    root = world["tmp"] / "script_features" / GENE
    if not root.exists():
        root.mkdir(parents=True)
        np.savez(root / f"{GENE}.npz", **world["data"])
    subprocess.run(cmd, check=True, capture_output=True)
    frame = pd.read_csv(next((out / "per_gene").glob("*.tsv")), sep="\t")
    return frame.set_index("embed_type")["p_F" if test == "F" else "p_LRT"]


def scan_rows(world, out: str = "out") -> pd.DataFrame:
    return pd.concat(pd.read_csv(p, sep="\t")
                     for p in (world[out] / "per_gene" / GENE).glob("part*.tsv.gz"))


def scan_p(world, trait: str, sex: str = "All", out: str = "out") -> pd.Series:
    frame = scan_rows(world, out)
    cell = frame[(frame.trait_id == trait) & (frame.sex == sex)]
    return cell.set_index("embed_type")["p"]


@pytest.mark.integration
@pytest.mark.parametrize("sex", ["All", "F", "M"])
@pytest.mark.parametrize("trait,test", [("q", "F"), ("b", "LRT")])
def test_each_sex_matches_the_vendored_script(world, trait, test, sex):
    expected, got = script_p(world, trait, test, sex), scan_p(world, trait, sex)
    assert len(got) == 60 and got.notna().all()
    np.testing.assert_allclose(got.loc[expected.index], expected, rtol=1e-8)


@pytest.mark.integration
def test_rerun_resumes_without_recomputing(world):
    part = world["out"] / "per_gene" / GENE / "part0.tsv.gz"
    before = part.stat().st_mtime_ns
    assert G.scan_gene(C.fmgwas(), G.Scan(world["tmp"] / "traits"), world["cache"],
                       GENE, world["out"], 50) == 0
    assert part.stat().st_mtime_ns == before


@pytest.mark.integration
@pytest.mark.parametrize("sex", ["All", "F"])
@pytest.mark.parametrize("trait,test", [("q", "F"), ("b", "LRT")])
def test_genetic_pcs_match_the_script_with_pc_covariates(world, trait, test, sex):
    expected = script_p(world, trait, test, sex, covariates="age,sex,pc1,pc2")
    got = scan_p(world, trait, sex, out="out_gpc")
    assert len(got) == 60 and got.notna().all()
    np.testing.assert_allclose(got.loc[expected.index], expected, rtol=1e-8)
    # and the PCs change the answer: q depends on pc1, which tracks the gene
    if trait == "q":
        assert not np.allclose(got, scan_p(world, trait, sex))


@pytest.mark.integration
@pytest.mark.parametrize("out", ["out", "out_gpc"])
def test_genetic_pcs_are_never_tested_as_traits(world, out):
    assert set(scan_rows(world, out).trait_id) == {"q", "b"}
