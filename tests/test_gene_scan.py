"""gene_pca_cache.py + gene_scan.py against the vendored FM-GWAS script.

When a cell's cohort is exactly the PCA fit sample (the full cohort, every
participant measured), fitting the PCA once and reusing it is the same model
the script fits inside the cell, so the p-values must agree with a plain
`insample_hpp_AG.py run`.
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
    people = pd.DataFrame({"participant_id": ids, "sex": sex, "age": age})

    scan = tmp / "traits"
    scan.mkdir()
    people.to_csv(scan / "participants.tsv", sep="\t", index=False)
    np.save(scan / "trait_values.npy", np.column_stack([quant, binary]))
    pd.DataFrame({"trait_id": ["q", "b"], "column": [0, 1],
                  "kind": ["quantitative", "binary"]}).to_csv(
        scan / "trait_catalog.tsv", sep="\t", index=False)
    genes = tmp / "genes.tsv"
    genes.write_text(f"gene_id\n{GENE}\n")
    for name, label in (("q", quant), ("b", binary)):
        people.assign(label=label).to_csv(tmp / f"{name}_All.tsv", sep="\t", index=False)

    cache, out = tmp / "cache", tmp / "out"
    C.main(["--genes", str(genes), "--participants", str(scan / "participants.tsv"),
            "--feature-root", str(feature_root), "--cache", str(cache)])
    G.main(["--scan", str(scan), "--cache", str(cache), "--genes", str(genes),
            "--out", str(out)])
    return dict(tmp=tmp, feature_root=feature_root, genes=genes, cache=cache, out=out,
                data=data)


@pytest.mark.integration
def test_cache_rebuilds_the_scripts_pca_scores_exactly(world):
    fm = C.fmgwas()
    ids, scores, meta = C.load_cache(world["cache"], GENE)
    assert meta["feature_dir"] == "ENSG00000000001.9"
    assert len(scores) == 60
    _, matrices = fm.load_gene(
        world["feature_root"] / "ENSG00000000001.9" / "ENSG00000000001.9.npz",
        ["concat_cage_dnase", "add_tss_rnaseq"])
    for embed, matrix in matrices.items():
        expected, info = fm.fit_pca(matrix, fm.parse_pca("variance:0.95"))
        assert np.array_equal(scores[embed], expected)
        assert meta["components"][embed] == info["n_components"]


def script_p(world, trait: str, test: str, sex: str = "All") -> pd.Series:
    out = world["tmp"] / f"script_{trait}_{sex}"
    cmd = [sys.executable, str(SCRIPT), "run", trait, sex, "--gene_tsv", str(world["genes"]),
           "--feature_root", str(world["tmp"] / "script_features"),
           "--cohort_file", str(world["tmp"] / f"{trait}_All.tsv"),
           "--covariates", "age,sex", "--gene_pca", assoc_settings.GENE_PCA,
           "--tests", test, "--embed_type", ",".join(assoc_settings.ALL_VIEWS),
           "--out_dir", str(out), "--no_progress"]
    root = world["tmp"] / "script_features" / GENE
    if not root.exists():
        root.mkdir(parents=True)
        np.savez(root / f"{GENE}.npz", **world["data"])
    subprocess.run(cmd, check=True, capture_output=True)
    frame = pd.read_csv(next((out / "per_gene").glob("*.tsv")), sep="\t")
    return frame.set_index("embed_type")["p_F" if test == "F" else "p_LRT"]


def scan_p(world, trait: str, sex: str = "All") -> pd.Series:
    frame = pd.concat(pd.read_csv(p, sep="\t")
                      for p in (world["out"] / "per_gene" / GENE).glob("part*.tsv.gz"))
    cell = frame[(frame.trait_id == trait) & (frame.sex == sex)]
    return cell.set_index("embed_type")["p"]


@pytest.mark.integration
@pytest.mark.parametrize("trait,test", [("q", "F"), ("b", "LRT")])
def test_full_cohort_matches_the_vendored_script(world, trait, test):
    expected, got = script_p(world, trait, test), scan_p(world, trait)
    assert len(got) == 60
    np.testing.assert_allclose(got.loc[expected.index], expected, rtol=1e-8)


@pytest.mark.integration
def test_every_sex_cell_is_tested(world):
    for trait in ("q", "b"):
        for sex in ("All", "F", "M"):
            assert scan_p(world, trait, sex).notna().all()


@pytest.mark.integration
def test_rerun_resumes_without_recomputing(world):
    part = world["out"] / "per_gene" / GENE / "part0.tsv.gz"
    before = part.stat().st_mtime_ns
    assert G.scan_gene(C.fmgwas(), G.Scan(world["tmp"] / "traits"), world["cache"],
                       GENE, world["out"], 50) == 0
    assert part.stat().st_mtime_ns == before
