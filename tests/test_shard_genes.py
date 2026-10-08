"""Unit tests for splitting an assoc cell's gene file into parallel shards.

A cell used to run as one process, gene after gene, at ~45 min per gene on
43,680 participants. Shards are disjoint subsets of the genes still lacking a
result, so N processes cover the cell with no gene tested twice and the
finished ones never re-run.
"""
import pandas as pd
import pytest

import shard_genes as S

COLUMNS = ["gene_id", "num_feature", "embed_type"]


def write_gene_file(path, gene_ids):
    pd.DataFrame({"gene_id": gene_ids, "num_feature": 5,
                  "embed_type": "x"})[COLUMNS].to_csv(path, sep="\t",
                                                      index=False)
    return str(path)


def finish(save_path, gene_id):
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / f"{gene_id}.tsv").write_text("done\n", encoding="utf-8")


@pytest.mark.unit
def test_pending_skips_genes_with_a_result(tmp_path):
    genes = write_gene_file(tmp_path / "g.tsv", ["G1.1", "G2.1", "G3.1"])
    finish(tmp_path / "out", "G2.1")
    assert S.pending_genes(genes, str(tmp_path / "out")) == ["G1.1", "G3.1"]


@pytest.mark.unit
def test_shards_are_disjoint_and_cover_every_pending_gene(tmp_path):
    ids = [f"G{i}.1" for i in range(7)]
    genes = write_gene_file(tmp_path / "g.tsv", ids)
    paths = S.write_shards(genes, str(tmp_path / "out"), 3,
                           str(tmp_path / "shards"))
    assert len(paths) == 3
    seen = [g for p in paths for g in pd.read_csv(p, sep="\t")["gene_id"]]
    assert sorted(seen) == sorted(ids)
    sizes = sorted(len(pd.read_csv(p, sep="\t")) for p in paths)
    assert sizes == [2, 2, 3]


@pytest.mark.unit
def test_shards_keep_every_column(tmp_path):
    genes = write_gene_file(tmp_path / "g.tsv", ["G1.1", "G2.1"])
    paths = S.write_shards(genes, str(tmp_path / "out"), 2,
                           str(tmp_path / "shards"))
    assert list(pd.read_csv(paths[0], sep="\t").columns) == COLUMNS


@pytest.mark.unit
def test_no_more_shards_than_pending_genes(tmp_path):
    genes = write_gene_file(tmp_path / "g.tsv", ["G1.1", "G2.1", "G3.1"])
    finish(tmp_path / "out", "G1.1")
    paths = S.write_shards(genes, str(tmp_path / "out"), 8,
                           str(tmp_path / "shards"))
    assert len(paths) == 2


@pytest.mark.unit
def test_nothing_pending_writes_no_shards(tmp_path):
    genes = write_gene_file(tmp_path / "g.tsv", ["G1.1"])
    finish(tmp_path / "out", "G1.1")
    assert S.write_shards(genes, str(tmp_path / "out"), 4,
                          str(tmp_path / "shards")) == []


@pytest.mark.unit
def test_stale_shards_are_removed(tmp_path):
    genes = write_gene_file(tmp_path / "g.tsv", ["G1.1", "G2.1", "G3.1"])
    shards = tmp_path / "shards"
    S.write_shards(genes, str(tmp_path / "out"), 3, str(shards))
    S.write_shards(genes, str(tmp_path / "out"), 1, str(shards))
    assert len(list(shards.glob("*.tsv"))) == 1


@pytest.mark.unit
def test_main_count_and_split(tmp_path, capsys):
    genes = write_gene_file(tmp_path / "g.tsv", ["G1.1", "G2.1"])
    out = str(tmp_path / "out")
    assert S.main(["count", genes, out]) == 0
    assert capsys.readouterr().out.strip() == "2"
    assert S.main(["split", genes, out, "2", str(tmp_path / "shards")]) == 0
    assert len(capsys.readouterr().out.split()) == 2
