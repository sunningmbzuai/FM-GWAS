"""Unit tests for an assoc cell's pending genes and stale-failure cleanup.

fmgwas/insample_hpp_AG.py skips a gene once <out_dir>/per_gene/<gene>.tsv
exists, including the input_failed file it writes for a gene with no features
yet. Those files must be cleared once the features land, and nothing else may
be touched.
"""
import pandas as pd
import pytest

import assoc_pending as P

MISSING = ("ENSG2.1: expected exactly one .pt/.npz feature file, found []")


def write_gene_file(path, gene_ids):
    pd.DataFrame({"gene_id": gene_ids, "extra": 1}).to_csv(path, sep="\t", index=False)
    return str(path)


def write_result(out_dir, gene_id, status="ok", error=""):
    (out_dir / "per_gene").mkdir(parents=True, exist_ok=True)
    path = P.result_path(str(out_dir), gene_id)
    pd.DataFrame({"gene_id": [gene_id, gene_id], "embed_type": ["a", "b"],
                  "status": status, "error": error}).to_csv(path, sep="\t", index=False)
    return path


def add_features(root, gene_id, suffix="npz"):
    (root / gene_id).mkdir(parents=True, exist_ok=True)
    (root / gene_id / f"{gene_id}.{suffix}").write_bytes(b"x")


@pytest.mark.unit
def test_pending_skips_genes_with_a_result(tmp_path):
    genes = write_gene_file(tmp_path / "g.tsv", ["G1.1", "G2.1", "G3.1"])
    write_result(tmp_path / "out", "G2.1")
    assert P.pending_genes(genes, str(tmp_path / "out")) == ["G1.1", "G3.1"]


@pytest.mark.unit
def test_duplicate_gene_ids_count_once(tmp_path):
    genes = write_gene_file(tmp_path / "g.tsv", ["G1.1", "G1.1", "G2.1"])
    assert P.pending_genes(genes, str(tmp_path / "out")) == ["G1.1", "G2.1"]


@pytest.mark.unit
def test_missing_feature_failure_is_cleared_once_features_exist(tmp_path):
    genes = write_gene_file(tmp_path / "g.tsv", ["ENSG2.1"])
    out, feats = tmp_path / "out", tmp_path / "feats"
    write_result(out, "ENSG2.1", "input_failed", MISSING)

    assert P.clear_stale(genes, str(out), str(feats)) == []     # still no features
    add_features(feats, "ENSG2.1")
    assert P.clear_stale(genes, str(out), str(feats)) == ["ENSG2.1"]
    assert P.pending_genes(genes, str(out)) == ["ENSG2.1"]


@pytest.mark.unit
def test_other_failures_and_real_results_are_kept(tmp_path):
    genes = write_gene_file(tmp_path / "g.tsv", ["G1.1", "G2.1"])
    out, feats = tmp_path / "out", tmp_path / "feats"
    write_result(out, "G1.1", "input_failed", "Fewer than 3 participants after intersection")
    write_result(out, "G2.1", "ok")
    for gene in ("G1.1", "G2.1"):
        add_features(feats, gene)
    assert P.clear_stale(genes, str(out), str(feats)) == []
    assert P.pending_genes(genes, str(out)) == []


@pytest.mark.unit
def test_two_feature_files_do_not_count_as_features(tmp_path):
    """The script refuses a gene with both a .pt and an .npz."""
    feats = tmp_path / "feats"
    add_features(feats, "G1.1", "npz")
    assert P.has_features("G1.1", str(feats))
    add_features(feats, "G1.1", "pt")
    assert not P.has_features("G1.1", str(feats))


@pytest.mark.unit
def test_main_count(tmp_path, capsys):
    genes = write_gene_file(tmp_path / "g.tsv", ["G1.1", "G2.1"])
    assert P.main(["count", genes, str(tmp_path / "out")]) == 0
    assert capsys.readouterr().out.strip() == "2"
