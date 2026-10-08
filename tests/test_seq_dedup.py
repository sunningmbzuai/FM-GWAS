"""Sequence-level dedup must reproduce the row-level result bit for bit.

The claim being tested is the one the whole optimisation rests on: because
get_track_mean() pools each haplotype separately and concatenates, a saved row
is [f(hap1), f(hap2)] for a per-sequence f. So embedding each distinct
sequence once and reassembling gives exactly what embedding each distinct
(hap1, hap2) row would have given.

The model is stood in for by a deterministic fake f -- the point is the
bookkeeping (slots, packing, halves, haplotype order), not the network. A fake
that returned the same vector for every sequence would pass any of this
vacuously, so f here separates its inputs.
"""
import csv
import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")

import dedup_gene_sequences as D
import expand_embeddings as E

csv.field_size_limit(sys.maxsize)

HALF = 4          # per-haplotype track width of the fake model
SEQUENCES = {"A": "AAAA", "B": "CCCC", "C": "GGGG", "D": "TTTT", "E": "ACGT"}


def fake_f(sequence: str) -> torch.Tensor:
    """A per-sequence feature vector, distinct for distinct sequences."""
    base = float(sum(sequence.encode()))
    return torch.tensor([base + i for i in range(HALF)], dtype=torch.float32)


def embed_rows(tsv_path: str) -> dict:
    """What the GPU script would save for a TSV: [f(mut1), f(mut2)] per row."""
    with open(tsv_path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return {
        E.SUBJECT_KEY: [r["participant_id"] for r in rows],
        "concat_cage_mean": torch.stack(
            [torch.cat([fake_f(r["mutation_seq_1"]), fake_f(r["mutation_seq_2"])])
             for r in rows]),
    }


def write_gene_tsv(path, haplotype_pairs):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=D.COLUMNS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for i, (hap1, hap2) in enumerate(haplotype_pairs):
            writer.writerow({"participant_id": f"p{i}",
                             "ref_seq": SEQUENCES["A"],
                             "mutation_seq_1": SEQUENCES[hap1],
                             "mutation_seq_2": SEQUENCES[hap2]})
    return str(path)


def expanded(tmp_path, haplotype_pairs, mode):
    """Run one dedup mode end to end and return the per-participant tensor."""
    gene_tsv = write_gene_tsv(tmp_path / f"{mode}_GENE.tsv", haplotype_pairs)
    out_dir = tmp_path / mode
    prepare = D.prepare_gene_seqs if mode == "seqs" else D.prepare_gene
    stats = prepare(gene_tsv, str(out_dir))
    with open(out_dir / f"{mode}_GENE.index.json") as fh:
        index = json.load(fh)
    embedding = embed_rows(str(out_dir / f"{mode}_GENE.tsv"))
    fan_out = E.expand_seqs if mode == "seqs" else E.expand
    return fan_out(embedding, index), stats, index


# Five participants over three distinct haplotypes, with both orders of one
# pair present: (B, C) and (C, B) are different rows but the same sequences.
PAIRS = [("A", "B"), ("B", "C"), ("A", "B"), ("C", "B"), ("A", "A")]


@pytest.mark.unit
def test_seq_dedup_matches_row_dedup_exactly(tmp_path):
    by_seq, _, _ = expanded(tmp_path, PAIRS, "seqs")
    by_row, _, _ = expanded(tmp_path, PAIRS, "rows")
    assert by_seq[E.SUBJECT_KEY] == by_row[E.SUBJECT_KEY]
    assert torch.equal(by_seq["concat_cage_mean"], by_row["concat_cage_mean"])


@pytest.mark.unit
def test_expanded_rows_are_the_participants_own_haplotypes(tmp_path):
    by_seq, _, _ = expanded(tmp_path, PAIRS, "seqs")
    for i, (hap1, hap2) in enumerate(PAIRS):
        expected = torch.cat([fake_f(SEQUENCES[hap1]), fake_f(SEQUENCES[hap2])])
        assert torch.equal(by_seq["concat_cage_mean"][i], expected)


@pytest.mark.unit
def test_haplotype_order_is_not_symmetric(tmp_path):
    """(B, C) and (C, B) must not collapse -- the two halves differ."""
    by_seq, _, _ = expanded(tmp_path, PAIRS, "seqs")
    bc = by_seq["concat_cage_mean"][1]
    cb = by_seq["concat_cage_mean"][3]
    assert not torch.equal(bc, cb)
    assert torch.equal(bc[:HALF], cb[HALF:])
    assert torch.equal(bc[HALF:], cb[:HALF])


@pytest.mark.unit
def test_packing_costs_one_pass_per_distinct_sequence(tmp_path):
    _, stats, index = expanded(tmp_path, PAIRS, "seqs")
    assert index["n_distinct"] == 3            # A, B, C
    assert index["n_packed_rows"] == 2         # (A,B), (C,C)
    assert stats["distinct_seqs"] == 3
    _, row_stats, _ = expanded(tmp_path, PAIRS, "rows")
    # 4 distinct rows x 2 haplotypes = 8 passes, against 4 packed halves.
    assert row_stats["distinct_rows"] == 4


@pytest.mark.unit
def test_odd_sequence_count_drops_the_repeated_tail(tmp_path):
    by_seq, _, index = expanded(tmp_path, [("A", "B"), ("C", "C")], "seqs")
    assert index["n_distinct"] == 3 and index["n_packed_rows"] == 2
    assert by_seq["concat_cage_mean"].shape == (2, 2 * HALF)


@pytest.mark.unit
def test_single_sequence_gene_round_trips(tmp_path):
    """A gene with no variants: every participant has the same haplotypes."""
    by_seq, _, index = expanded(tmp_path, [("A", "A"), ("A", "A")], "seqs")
    assert index["n_distinct"] == 1 and index["n_packed_rows"] == 1
    expected = torch.cat([fake_f(SEQUENCES["A"]), fake_f(SEQUENCES["A"])])
    assert torch.equal(by_seq["concat_cage_mean"][0], expected)


@pytest.mark.unit
def test_shuffled_subject_order_is_handled(tmp_path):
    """The embedding script may reorder or shard subjects."""
    gene_tsv = write_gene_tsv(tmp_path / "S_GENE.tsv", PAIRS)
    out_dir = tmp_path / "s"
    D.prepare_gene_seqs(gene_tsv, str(out_dir))
    with open(out_dir / "S_GENE.index.json") as fh:
        index = json.load(fh)
    embedding = embed_rows(str(out_dir / "S_GENE.tsv"))

    order = list(reversed(range(len(embedding[E.SUBJECT_KEY]))))
    shuffled = {
        E.SUBJECT_KEY: [embedding[E.SUBJECT_KEY][i] for i in order],
        "concat_cage_mean": embedding["concat_cage_mean"][order],
    }
    assert torch.equal(E.expand_seqs(shuffled, index)["concat_cage_mean"],
                       E.expand_seqs(embedding, index)["concat_cage_mean"])


@pytest.mark.unit
def test_short_embedding_is_an_error(tmp_path):
    """A half-finished embedding is caught by the row count."""
    gene_tsv = write_gene_tsv(tmp_path / "M_GENE.tsv", PAIRS)
    out_dir = tmp_path / "m"
    D.prepare_gene_seqs(gene_tsv, str(out_dir))
    with open(out_dir / "M_GENE.index.json") as fh:
        index = json.load(fh)
    embedding = embed_rows(str(out_dir / "M_GENE.tsv"))
    truncated = {E.SUBJECT_KEY: embedding[E.SUBJECT_KEY][:1],
                 "concat_cage_mean": embedding["concat_cage_mean"][:1]}
    with pytest.raises(ValueError, match="different dedup runs"):
        E.expand_seqs(truncated, index)


@pytest.mark.unit
def test_renamed_packed_row_is_an_error(tmp_path):
    """Right count, wrong names: the gather would silently take the wrong row."""
    gene_tsv = write_gene_tsv(tmp_path / "N_GENE.tsv", PAIRS)
    out_dir = tmp_path / "n"
    D.prepare_gene_seqs(gene_tsv, str(out_dir))
    with open(out_dir / "N_GENE.index.json") as fh:
        index = json.load(fh)
    embedding = embed_rows(str(out_dir / "N_GENE.tsv"))
    renamed = dict(embedding)
    renamed[E.SUBJECT_KEY] = ["uniq0", "uniq9"]
    with pytest.raises(ValueError, match="never embedded"):
        E.expand_seqs(renamed, index)


@pytest.mark.unit
def test_two_sequence_gene_is_not_reported_flat(tmp_path):
    """Two sequences pack into ONE row; variance lives between the halves."""
    gene_tsv = write_gene_tsv(tmp_path / "V_GENE.tsv", [("A", "B")])
    out_dir = tmp_path / "v"
    D.prepare_gene_seqs(gene_tsv, str(out_dir))
    with open(out_dir / "V_GENE.index.json") as fh:
        index = json.load(fh)
    embedding = embed_rows(str(out_dir / "V_GENE.tsv"))
    measured = {E.SUBJECT_KEY: [], **E.per_sequence(embedding, index)}
    assert E.check_variance("V_GENE", measured, index["n_distinct"])["status"] == "ok"


@pytest.mark.unit
def test_row_mode_embedding_is_refused_by_a_seqs_index(tmp_path):
    """A .pt left by the old dedup must not be read against a new index."""
    gene_tsv = write_gene_tsv(tmp_path / "X_GENE.tsv", PAIRS)
    D.prepare_gene_seqs(gene_tsv, str(tmp_path / "seq"))
    D.prepare_gene(gene_tsv, str(tmp_path / "row"))
    with open(tmp_path / "seq" / "X_GENE.index.json") as fh:
        seq_index = json.load(fh)
    stale = embed_rows(str(tmp_path / "row" / "X_GENE.tsv"))
    with pytest.raises(ValueError, match="different dedup runs"):
        E.expand_seqs(stale, seq_index)


@pytest.mark.unit
def test_packed_embedding_is_refused_by_a_rows_index(tmp_path):
    gene_tsv = write_gene_tsv(tmp_path / "Y_GENE.tsv", PAIRS)
    D.prepare_gene_seqs(gene_tsv, str(tmp_path / "seq"))
    D.prepare_gene(gene_tsv, str(tmp_path / "row"))
    with open(tmp_path / "row" / "Y_GENE.index.json") as fh:
        row_index = json.load(fh)
    packed = embed_rows(str(tmp_path / "seq" / "Y_GENE.tsv"))
    with pytest.raises(ValueError, match="different dedup runs"):
        E.expand(packed, row_index)


@pytest.mark.unit
def test_truncated_index_is_rebuilt_not_fatal(tmp_path):
    """An interrupted run leaves half-written JSON; the next run redoes it."""
    gene_tsv = write_gene_tsv(tmp_path / "T_GENE.tsv", PAIRS)
    out_dir = tmp_path / "t"
    D.prepare_gene_seqs(gene_tsv, str(out_dir))
    index_path = out_dir / "T_GENE.index.json"

    whole = index_path.read_text()
    index_path.write_text(whole[:len(whole) // 2])

    result = D.prepare_gene_seqs(gene_tsv, str(out_dir))
    assert result["skipped"] is False
    assert json.loads(index_path.read_text())["n_distinct"] == 3


@pytest.mark.unit
def test_truncated_row_index_is_rebuilt_not_fatal(tmp_path):
    gene_tsv = write_gene_tsv(tmp_path / "U_GENE.tsv", PAIRS)
    out_dir = tmp_path / "u"
    D.prepare_gene(gene_tsv, str(out_dir))
    index_path = out_dir / "U_GENE.index.json"
    index_path.write_text("{\"gene_id\": \"U_GENE\", \"participant")

    result = D.prepare_gene(gene_tsv, str(out_dir))
    assert result["skipped"] is False
    assert json.loads(index_path.read_text())["n_distinct"] == 4


@pytest.mark.unit
def test_a_rows_index_does_not_satisfy_a_seqs_run(tmp_path):
    """Switching mode must rebuild, not silently reuse the other unit."""
    gene_tsv = write_gene_tsv(tmp_path / "W_GENE.tsv", PAIRS)
    out_dir = tmp_path / "w"
    D.prepare_gene(gene_tsv, str(out_dir))
    result = D.prepare_gene_seqs(gene_tsv, str(out_dir))
    assert result["skipped"] is False
    with open(out_dir / "W_GENE.index.json") as fh:
        assert json.load(fh)["mode"] == "seqs"


def embedded_gene(tmp_path):
    """One deduplicated, 'embedded' gene laid out the way expand_gene reads it."""
    gene_tsv = write_gene_tsv(tmp_path / "GENE.tsv", PAIRS)
    index_dir, unique_root = tmp_path / "unique_seqs", tmp_path / "unique_emb"
    D.prepare_gene_seqs(gene_tsv, str(index_dir))
    (unique_root / "GENE").mkdir(parents=True)
    torch.save(embed_rows(str(index_dir / "GENE.tsv")),
               unique_root / "GENE" / "GENE.pt")
    return str(unique_root), str(index_dir), str(tmp_path / "expanded")


@pytest.mark.unit
def test_up_to_date_expansion_is_skipped(tmp_path):
    """Every run expands every embedded gene otherwise -- 100+ genes, each a
    43,706-row write to the FUSE mount, redone by every rank."""
    unique_root, index_dir, out_root = embedded_gene(tmp_path)
    first = E.expand_gene("GENE", unique_root, index_dir, out_root)
    stamp = os.stat(first["path"]).st_mtime_ns
    second = E.expand_gene("GENE", unique_root, index_dir, out_root)
    assert second["variance"] == E.UP_TO_DATE
    assert os.stat(first["path"]).st_mtime_ns == stamp


@pytest.mark.unit
def test_newer_embedding_is_re_expanded(tmp_path):
    unique_root, index_dir, out_root = embedded_gene(tmp_path)
    first = E.expand_gene("GENE", unique_root, index_dir, out_root)
    src = os.path.join(unique_root, "GENE", "GENE.pt")
    later = os.stat(first["path"]).st_mtime + 10
    os.utime(src, (later, later))
    again = E.expand_gene("GENE", unique_root, index_dir, out_root)
    assert again["variance"] != E.UP_TO_DATE


@pytest.mark.unit
def test_force_re_expands(tmp_path):
    unique_root, index_dir, out_root = embedded_gene(tmp_path)
    E.expand_gene("GENE", unique_root, index_dir, out_root)
    again = E.expand_gene("GENE", unique_root, index_dir, out_root, force=True)
    assert again["variance"] != E.UP_TO_DATE
