"""Unit tests for classifying which target genes UKB can actually test.

A gene is untestable when it has no sequence at all (build_gene_sequences
skipped it) or when every participant carries the same haplotype (one distinct
sequence in the dedup index): its features are then constant across people.
Two distinct sequences is the smallest testable gene, not an untestable one.
"""
import json

import pandas as pd
import pytest

import untestable_genes as U


def write_targets(path, rows):
    pd.DataFrame(rows, columns=["gene_id", "trait", "sex"]).to_csv(
        path, sep="\t", index=False)
    return str(path)


def add_sequence(seq_dir, gene_id):
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / f"{gene_id}.tsv").write_text("participant_id\n", encoding="utf-8")


def add_index(unique_dir, gene_id, n_distinct):
    unique_dir.mkdir(parents=True, exist_ok=True)
    (unique_dir / f"{gene_id}.index.json").write_text(
        json.dumps({"mode": "seqs", "n_distinct": n_distinct}), encoding="utf-8")


@pytest.fixture
def layout(tmp_path):
    targets = write_targets(tmp_path / "targets.tsv", [
        ("ENSG1.1", "Height", "All"),   # no sequence
        ("ENSG2.1", "Height", "All"),   # one haplotype
        ("ENSG3.1", "Height", "All"),   # two haplotypes: testable
        ("ENSG4.1", "Height", "All"),   # sequence, not deduplicated yet
    ])
    seq_dir, unique_dir = tmp_path / "seqs", tmp_path / "unique"
    for gene in ("ENSG2.1", "ENSG3.1", "ENSG4.1"):
        add_sequence(seq_dir, gene)
    add_index(unique_dir, "ENSG2.1", 1)
    add_index(unique_dir, "ENSG3.1", 2)
    return targets, str(seq_dir), str(unique_dir)


@pytest.mark.unit
def test_each_gene_gets_its_status(layout):
    frame = U.classify(*layout)
    assert dict(zip(frame["gene_id"], frame["status"])) == {
        "ENSG1.1": U.NO_SEQUENCE,
        "ENSG2.1": U.SINGLE_HAPLOTYPE,
        "ENSG3.1": U.TESTABLE,
        "ENSG4.1": U.NOT_DEDUPLICATED,
    }


@pytest.mark.unit
def test_distinct_sequence_count_is_reported(layout):
    frame = U.classify(*layout).set_index("gene_id")
    assert frame.loc["ENSG3.1", "n_distinct_seqs"] == 2
    assert pd.isna(frame.loc["ENSG1.1", "n_distinct_seqs"])


@pytest.mark.unit
def test_version_mismatch_still_finds_the_sequence(tmp_path):
    targets = write_targets(tmp_path / "t.tsv", [("ENSG5.2", "Height", "All")])
    add_sequence(tmp_path / "seqs", "ENSG5.7")
    add_index(tmp_path / "unique", "ENSG5.7", 3)
    frame = U.classify(targets, str(tmp_path / "seqs"),
                       str(tmp_path / "unique"))
    assert list(frame["status"]) == [U.TESTABLE]


@pytest.mark.unit
def test_main_writes_table_and_exits_zero(layout, tmp_path, capsys):
    out = tmp_path / "status.tsv"
    targets, seq_dir, unique_dir = layout
    assert U.main(["--targets", targets, "--gene_seq_dir", seq_dir,
                   "--unique_dir", unique_dir, "--out", str(out)]) == 0
    written = pd.read_csv(out, sep="\t")
    assert len(written) == 4
    printed = capsys.readouterr().out
    assert "ENSG1.1" in printed and "ENSG2.1" in printed
