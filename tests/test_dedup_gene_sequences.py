"""Unit tests for row deduplication and the fan-out back to participants.

The property everything rests on: embedding the distinct rows and gathering
them by the recorded index must reproduce, exactly, what embedding every row
would have produced. The gather is tested against a stand-in "embedding" that
is a deterministic function of the whole row, which is the only assumption the
real pipeline makes about get_track_mean().
"""
import json

import pytest
import torch

import dedup_gene_sequences as D
import expand_embeddings as E

REF = "ACGT" * 4
HAP_A = "ACGT" * 3 + "ACGA"
# p1 and p3 are the same row; p2 and p4 are the same row. 4 rows, 2 distinct.
ROWS = [
    ("p1", REF, HAP_A, REF),
    ("p2", REF, REF, REF),
    ("p3", REF, HAP_A, REF),
    ("p4", REF, REF, REF),
]


@pytest.fixture
def gene_tsv(tmp_path):
    path = tmp_path / "ENSG1.tsv"
    lines = ["\t".join(D.COLUMNS)] + ["\t".join(r) for r in ROWS]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def fake_embedding(row) -> list[float]:
    """Stand-in for the model: any deterministic function of the whole row."""
    return [float(len(row[1])), float(sum(map(ord, row[2]))), float(hash(row[3]) % 97)]


@pytest.mark.unit
def test_stats_count_distinct_rows(gene_tsv):
    assert D.gene_stats(gene_tsv) == {
        "gene_id": "ENSG1", "participants": 4, "distinct_rows": 2, "speedup": 2.0}


@pytest.mark.unit
def test_prepare_writes_distinct_rows_and_index(tmp_path, gene_tsv):
    out = tmp_path / "unique"
    stats = D.prepare_gene(gene_tsv, str(out))

    written = (out / "ENSG1.tsv").read_text().strip().split("\n")
    index = json.loads((out / "ENSG1.index.json").read_text())

    assert stats["distinct_rows"] == 2
    assert len(written) == 3           # header + 2 distinct rows
    assert index["participant_id"] == ["p1", "p2", "p3", "p4"]
    assert index["row_index"] == [0, 1, 0, 1]


@pytest.mark.unit
def test_expand_reproduces_the_undeduplicated_embedding(tmp_path, gene_tsv):
    out = tmp_path / "unique"
    D.prepare_gene(gene_tsv, str(out))
    index = json.loads((out / "ENSG1.index.json").read_text())

    # What the model would produce run on the DISTINCT rows...
    distinct = [ROWS[0], ROWS[1]]
    deduped = {
        "subject_id": [f"uniq{i}" for i in range(len(distinct))],
        "concat_cage_mean": torch.tensor([fake_embedding(r) for r in distinct]),
    }
    # ...and what it would produce run on every row.
    full = torch.tensor([fake_embedding(r) for r in ROWS])

    expanded = E.expand(deduped, index)
    assert expanded["subject_id"] == ["p1", "p2", "p3", "p4"]
    assert torch.equal(expanded["concat_cage_mean"], full)


@pytest.mark.unit
def test_expand_maps_by_subject_name_not_position(tmp_path, gene_tsv):
    out = tmp_path / "unique"
    D.prepare_gene(gene_tsv, str(out))
    index = json.loads((out / "ENSG1.index.json").read_text())

    # Same content, rows in the reverse order the dataset happened to emit.
    deduped = {
        "subject_id": ["uniq1", "uniq0"],
        "concat_cage_mean": torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
    }
    expanded = E.expand(deduped, index)
    assert torch.equal(expanded["concat_cage_mean"],
                       torch.tensor([[0.0, 0.0], [1.0, 1.0],
                                     [0.0, 0.0], [1.0, 1.0]]))


@pytest.mark.unit
def test_expand_refuses_an_incomplete_embedding(tmp_path, gene_tsv):
    out = tmp_path / "unique"
    D.prepare_gene(gene_tsv, str(out))
    index = json.loads((out / "ENSG1.index.json").read_text())

    partial = {"subject_id": ["uniq0"],
               "concat_cage_mean": torch.tensor([[1.0]])}
    with pytest.raises(ValueError, match="different dedup runs"):
        E.expand(partial, index)


@pytest.mark.unit
def test_expand_rejects_a_tensor_that_is_not_per_subject(tmp_path, gene_tsv):
    out = tmp_path / "unique"
    D.prepare_gene(gene_tsv, str(out))
    index = json.loads((out / "ENSG1.index.json").read_text())

    wrong = {"subject_id": ["uniq0", "uniq1"],
             "concat_cage_mean": torch.tensor([[1.0], [2.0], [3.0]])}
    with pytest.raises(ValueError, match="not a per-subject tensor"):
        E.expand(wrong, index)


@pytest.mark.unit
def test_missing_column_is_an_error(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("participant_id\tref_seq\np1\tACGT\n")
    with pytest.raises(KeyError):
        D.gene_stats(str(path))


@pytest.mark.unit
def test_prepare_skips_a_gene_that_is_already_done(tmp_path, gene_tsv):
    out = tmp_path / "unique"
    first = D.prepare_gene(gene_tsv, str(out))
    assert first["skipped"] is False

    # Corrupt the source: a skipped gene must not read it at all.
    open(gene_tsv, "w").close()
    second = D.prepare_gene(gene_tsv, str(out))
    assert second["skipped"] is True
    assert second["distinct_rows"] == first["distinct_rows"]
    assert second["participants"] == first["participants"]


@pytest.mark.unit
def test_prepare_force_rebuilds(tmp_path, gene_tsv):
    out = tmp_path / "unique"
    D.prepare_gene(gene_tsv, str(out))
    again = D.prepare_gene(gene_tsv, str(out), force=True)
    assert again["skipped"] is False


@pytest.mark.unit
def test_prepare_redoes_a_gene_whose_index_is_missing(tmp_path, gene_tsv):
    out = tmp_path / "unique"
    D.prepare_gene(gene_tsv, str(out))
    (out / "ENSG1.index.json").unlink()
    # A TSV without its index cannot be fanned back out, so it is unfinished.
    assert D.prepare_gene(gene_tsv, str(out))["skipped"] is False


@pytest.mark.unit
def test_variance_check_passes_on_varying_embeddings():
    embedding = {"subject_id": ["uniq0", "uniq1"],
                 "concat_cage_mean": torch.tensor([[0.0, 1.0], [2.0, 3.0]])}
    assert E.check_variance("ENSG1", embedding, n_distinct=2)["status"] == "ok"


@pytest.mark.unit
def test_variance_check_flags_constant_embeddings():
    embedding = {"subject_id": ["uniq0", "uniq1"],
                 "concat_cage_mean": torch.tensor([[1.0, 2.0], [1.0, 2.0]])}
    result = E.check_variance("ENSG1", embedding, n_distinct=2)
    assert result["status"] == "flat"
    assert result["flat_tracks"] == ["concat_cage_mean"]


@pytest.mark.unit
def test_variance_check_flags_one_dead_track_among_live_ones():
    embedding = {
        "subject_id": ["uniq0", "uniq1"],
        "concat_cage_mean": torch.tensor([[0.0], [1.0]]),
        "concat_atac_mean": torch.tensor([[5.0], [5.0]]),
    }
    result = E.check_variance("ENSG1", embedding, n_distinct=2)
    assert result["status"] == "partially_flat"
    assert result["flat_tracks"] == ["concat_atac_mean"]


@pytest.mark.unit
def test_variance_check_accepts_a_single_distinct_sequence():
    # A gene with no variants gives every participant the same sequence, so a
    # constant embedding is the correct answer, not a failure.
    embedding = {"subject_id": ["uniq0"],
                 "concat_cage_mean": torch.tensor([[1.0, 2.0]])}
    assert E.check_variance("ENSG1", embedding, n_distinct=1)["status"] == "single"


@pytest.mark.unit
def test_check_only_reports_without_writing(tmp_path, gene_tsv, monkeypatch):
    unique = tmp_path / "unique"
    D.prepare_gene(gene_tsv, str(unique))

    embedded = tmp_path / "embedded" / "ENSG1"
    embedded.mkdir(parents=True)
    torch.save({"subject_id": ["uniq0", "uniq1"],
                "concat_cage_mean": torch.tensor([[0.0], [1.0]])},
               embedded / "ENSG1.pt")

    out_root = tmp_path / "out"
    result = E.expand_gene("ENSG1", str(embedded.parent), str(unique),
                           str(out_root), check_only=True)

    assert result["variance"] == "ok"
    assert result["path"] is None
    assert not out_root.exists()
