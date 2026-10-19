"""Our layers in front of Ning's script change nothing it computes.

fmgwas/insample_hpp_AG.py is vendored unmodified (pinned by hash below). What
this pipeline adds sits in front of it: sequence dedup + fan-out
(dedup_gene_sequences.py, expand_embeddings.py), the torch-free tables
(export_features.py) and the per-trait settings (assoc_settings.py). The end to
end test builds one gene's embeddings twice -- the plain way, one model call
per participant, and our way, deduplicated, expanded and exported -- runs the
script on both exactly as run_assoc.sh does, and requires identical result
files, for the LRT (disease) and the F test (quantitative).
"""
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import assoc_pending  # noqa: E402
import assoc_settings  # noqa: E402
import dedup_gene_sequences as D  # noqa: E402
import expand_embeddings as E  # noqa: E402
import export_features as X  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "fmgwas" / "insample_hpp_AG.py"
UPSTREAM_SHA256 = "7a47698ae697beaf37409b35eb7d5f9b54fd42d962bf1db48983f1c0fc562e19"

GENE = "ENSG00000000001.5"
# Per-haplotype widths of the fake model's tracks; real ones are 546/305/167/667.
HALF = {"cage": 5, "dnase": 4, "atac": 3, "rna_seq": 6}
N_PEOPLE = 400
N_SEQUENCES = 7


def load_script():
    spec = importlib.util.spec_from_file_location("insample_hpp_AG", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_vendored_script_is_upstream_byte_for_byte():
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == UPSTREAM_SHA256


@pytest.mark.unit
def test_embedding_names_match_the_scripts_own():
    script = load_script()
    data = {f"concat_{t}_{w}": np.zeros((3, 2 * h))
            for t, h in HALF.items() for w in ("mean", "tss500_mean")}
    supported = script.available_embeddings(data, 3)
    assert assoc_settings.ALL_VIEWS == supported
    assert assoc_settings.CONCAT_TSS == [n for n in supported if n.startswith("concat_tss_")]
    assert len(assoc_settings.CONCAT_TSS) == 15
    assert script.TRACKS == assoc_settings.TRACKS


@pytest.mark.unit
def test_settings_parse_as_the_script_expects():
    script = load_script()
    for trait, test in (("Hypertension", "LRT"), ("Height", "F")):
        args = script.build_parser().parse_args(
            ["run", trait, "All", "--gene_tsv", "g", "--feature_root", "f",
             "--cohort_file", "c", "--out_dir", "o"] + assoc_settings.cli_args(trait))
        assert args.tests == test
        assert args.gene_pca == ["variance:0.95"]
        assert args.covariates == "age,sex"
        assert args.embed_type.split(",") == assoc_settings.embed_types(trait)


@pytest.mark.unit
def test_result_path_matches_the_script():
    script = load_script()
    assert Path(assoc_pending.result_path("out", GENE)) == script.gene_result_path("out", GENE)


@pytest.mark.unit
def test_export_reads_like_the_script_reads_a_pt():
    script = load_script()
    tensor = torch.randn(4, 6).to(torch.bfloat16)
    assert np.array_equal(X.as_numpy(tensor), script.numpy_value(tensor))
    wide = torch.randn(4, 6, dtype=torch.float64)
    assert X.as_numpy(wide).dtype == script.numpy_value(wide).dtype == np.float64


# ----------------------------- end to end -------------------------------------
def sequences():
    rng = np.random.default_rng(7)
    return ["".join(rng.choice(list("ACGT"), 40)) for _ in range(N_SEQUENCES)]


def fake_track(sequence: str, track: str) -> torch.Tensor:
    """Per-sequence features: a deterministic function of the sequence alone."""
    seed = int.from_bytes(hashlib.sha256((track + sequence).encode()).digest()[:4], "big")
    return torch.tensor(np.random.default_rng(seed).normal(size=HALF[track]),
                        dtype=torch.float32)


def embed_rows(tsv_path) -> dict:
    """What the GPU script saves for a gene TSV: [f(mut1), f(mut2)] per row,
    TSS tracks in bfloat16 as the real model stores them."""
    with open(tsv_path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = {E.SUBJECT_KEY: [r["participant_id"] for r in rows]}
    for track in HALF:
        stacked = torch.stack([torch.cat([fake_track(r["mutation_seq_1"], track),
                                          fake_track(r["mutation_seq_2"], track)])
                               for r in rows])
        out[f"concat_{track}_mean"] = stacked
        out[f"concat_{track}_tss500_mean"] = (stacked * 0.5 + 1).to(torch.bfloat16)
    return out


def save_pt(root: Path, data: dict) -> None:
    (root / GENE).mkdir(parents=True)
    torch.save(data, root / GENE / f"{GENE}.pt")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("agreement")
    rng = np.random.default_rng(11)
    seqs = sequences()
    pairs = rng.integers(0, N_SEQUENCES, size=(N_PEOPLE, 2))
    ids = [str(1_000_001 + i) for i in range(N_PEOPLE)]

    gene_tsv = tmp / f"{GENE}.tsv"
    with open(gene_tsv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=D.COLUMNS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for pid, (a, b) in zip(ids, pairs):
            writer.writerow({"participant_id": pid, "ref_seq": seqs[0],
                             "mutation_seq_1": seqs[a], "mutation_seq_2": seqs[b]})

    # The plain way: one model row per participant.
    direct = embed_rows(gene_tsv)
    save_pt(tmp / "pt", direct)

    # Our way: dedup to distinct sequences, embed those, fan back out, export.
    D.prepare_gene_seqs(str(gene_tsv), str(tmp / "unique"))
    index = json.loads((tmp / "unique" / f"{GENE}.index.json").read_text())
    expanded = E.expand_seqs(embed_rows(tmp / "unique" / f"{GENE}.tsv"), index)
    save_pt(tmp / "expanded", expanded)
    X.export_gene(GENE, str(tmp / "expanded"), str(tmp / "tables"))

    # Cohort: a carrier effect for the disease label, a dosage effect for the
    # quantitative one; one participant without features, one missing age.
    effect = (pairs == 2).sum(axis=1) - 0.7 * (pairs == 4).sum(axis=1)
    sex = rng.integers(0, 2, N_PEOPLE)
    age = rng.normal(55, 8, N_PEOPLE)
    disease = (rng.random(N_PEOPLE) < 1 / (1 + np.exp(-(effect - 0.3)))).astype(int)
    height = 170 + 6 * sex + 2 * effect + rng.normal(0, 3, N_PEOPLE)
    for trait, label in (("Hypertension", disease), ("Height", height)):
        frame = pd.DataFrame({"participant_id": ids + ["1999999"],
                              "sex": list(sex) + [0],
                              "age": list(age) + [50.0],
                              "label": list(label) + [label[0]]})
        frame.loc[3, "age"] = np.nan
        frame.to_csv(tmp / f"{trait}_All.tsv", sep="\t", index=False)
    pd.DataFrame({"gene_id": [GENE]}).to_csv(tmp / "genes.tsv", sep="\t", index=False)
    return dict(tmp=tmp, direct=direct, expanded=expanded)


def run_script(inputs, trait: str, feature_root: str) -> pd.DataFrame:
    """The script exactly as run_assoc.sh invokes it."""
    tmp = inputs["tmp"]
    out_dir = tmp / f"out_{trait}_{feature_root}"
    subprocess.run(
        [sys.executable, str(SCRIPT), "run", trait, "All", "--rank", "0",
         "--world_size", "1", "--gene_tsv", str(tmp / "genes.tsv"),
         "--feature_root", str(tmp / feature_root),
         "--cohort_file", str(tmp / f"{trait}_All.tsv"),
         *assoc_settings.cli_args(trait), "--out_dir", str(out_dir), "--no_progress"],
        check=True, capture_output=True, text=True)
    return pd.read_csv(assoc_pending.result_path(str(out_dir), GENE), sep="\t")


@pytest.mark.integration
def test_dedup_fan_out_reproduces_the_plain_embedding(inputs):
    direct, expanded = inputs["direct"], inputs["expanded"]
    assert expanded[E.SUBJECT_KEY] == direct[E.SUBJECT_KEY]
    for key in X.RAW_FIELDS:
        assert expanded[key].dtype == direct[key].dtype
        assert torch.equal(expanded[key], direct[key]), key


@pytest.mark.integration
@pytest.mark.parametrize("trait", ["Hypertension", "Height"])
def test_tables_from_deduplicated_embeddings_give_identical_results(inputs, trait):
    plain = run_script(inputs, trait, "pt")
    ours = run_script(inputs, trait, "tables")
    pd.testing.assert_frame_equal(plain, ours, check_exact=True)

    assert list(ours["embed_type"]) == assoc_settings.embed_types(trait)
    assert (ours["n_samples"] == N_PEOPLE - 1).all()   # missing age dropped
    p = ours[assoc_settings.p_column(trait)]
    assert p.notna().all()
    if trait == "Hypertension":
        assert (ours["lrt_status"] == "ok").all()
        assert "p_F" not in ours.columns          # --tests LRT only
        assert p.min() < 1e-3                      # the carrier effect is found
    else:
        assert "p_LRT" not in ours.columns
        assert p.min() < 1e-10
