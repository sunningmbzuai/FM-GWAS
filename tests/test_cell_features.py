"""Unit tests for the assoc cell completeness check.

A cell that ran before every one of its genes had features must not be marked
done, or the next run skips it and the late genes are never tested. Matching
is unversioned: gene files carry ENSG...N, feature directories may not.
"""
import pytest

import cell_features as C


def write_gene_file(path, gene_ids):
    path.write_text("gene_id\tchrom\n" + "".join(f"{g}\t1\n" for g in gene_ids),
                    encoding="utf-8")
    return str(path)


def add_table(root, gene_id, ext="npz"):
    gene_dir = root / gene_id
    gene_dir.mkdir(parents=True)
    (gene_dir / f"{gene_id}.{ext}").write_bytes(b"")


@pytest.mark.unit
def test_every_gene_with_a_table_is_complete(tmp_path):
    root = tmp_path / "tables"
    add_table(root, "ENSG1")
    add_table(root, "ENSG2")
    genes = write_gene_file(tmp_path / "g.tsv", ["ENSG1.3", "ENSG2.1"])
    assert C.genes_without_features(genes, str(root)) == []


@pytest.mark.unit
def test_genes_without_a_table_are_listed_in_file_order(tmp_path):
    root = tmp_path / "tables"
    add_table(root, "ENSG2")
    genes = write_gene_file(tmp_path / "g.tsv", ["ENSG3.1", "ENSG2.1", "ENSG1.1"])
    assert C.genes_without_features(genes, str(root)) == ["ENSG3.1", "ENSG1.1"]


@pytest.mark.unit
def test_versioned_feature_directory_matches(tmp_path):
    root = tmp_path / "embeddings"
    add_table(root, "ENSG1.7", ext="pt")
    genes = write_gene_file(tmp_path / "g.tsv", ["ENSG1.3"])
    assert C.genes_without_features(genes, str(root)) == []


@pytest.mark.unit
def test_empty_gene_directory_is_not_a_feature(tmp_path):
    root = tmp_path / "tables"
    (root / "ENSG1").mkdir(parents=True)
    genes = write_gene_file(tmp_path / "g.tsv", ["ENSG1.1"])
    assert C.genes_without_features(genes, str(root)) == ["ENSG1.1"]


@pytest.mark.unit
def test_missing_feature_root_means_nothing_has_features(tmp_path):
    genes = write_gene_file(tmp_path / "g.tsv", ["ENSG1.1"])
    assert C.genes_without_features(genes, str(tmp_path / "nope")) == ["ENSG1.1"]


@pytest.mark.unit
def test_gene_file_without_gene_id_column_is_an_error(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("chrom\n1\n", encoding="utf-8")
    with pytest.raises(KeyError):
        C.genes_without_features(str(path), str(tmp_path))


@pytest.mark.unit
def test_main_exit_code_reports_completeness(tmp_path, capsys):
    root = tmp_path / "tables"
    add_table(root, "ENSG1")
    genes = write_gene_file(tmp_path / "g.tsv", ["ENSG1.1", "ENSG2.1"])
    assert C.main([genes, str(root)]) == 1
    assert "ENSG2.1" in capsys.readouterr().out
    complete = write_gene_file(tmp_path / "ok.tsv", ["ENSG1.1"])
    assert C.main([complete, str(root)]) == 0
