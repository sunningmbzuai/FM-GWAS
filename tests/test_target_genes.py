"""Unit tests for restricting the run to the key associations.

All fixtures are temp files, so nothing here touches cluster paths. The
distinctions pinned down: version suffixes must not break matching, a cell's
gene file must keep only that cell's targets, and the "All" cell must see both
sexes' hits rather than nothing.
"""
import pandas as pd
import pytest

import target_genes as T

TARGET_ROWS = [
    ("ENSG00000174327.6", "Cholesterol", "M"),
    ("ENSG00000105131.3", "Cholesterol", "F"),
    ("ENSG00000197208.5", "Height", "F"),
]


@pytest.fixture
def targets_tsv(tmp_path):
    path = tmp_path / "top_novel_genes.tsv"
    pd.DataFrame(TARGET_ROWS, columns=["gene_id", "trait", "sex"]).to_csv(
        path, sep="\t", index=False)
    return str(path)


@pytest.mark.unit
def test_target_gene_ids_are_unversioned(targets_tsv):
    assert T.target_gene_ids(targets_tsv) == {
        "ENSG00000174327", "ENSG00000105131", "ENSG00000197208"}


@pytest.mark.unit
def test_cell_selection_is_trait_and_sex_specific(targets_tsv):
    assert T.target_gene_ids_for("Cholesterol", "M", targets_tsv) == {"ENSG00000174327"}
    assert T.target_gene_ids_for("Cholesterol", "F", targets_tsv) == {"ENSG00000105131"}
    assert T.target_gene_ids_for("Height", "M", targets_tsv) == set()


@pytest.mark.unit
def test_all_cell_unions_both_sexes(targets_tsv):
    assert T.target_gene_ids_for("Cholesterol", "All", targets_tsv) == {
        "ENSG00000174327", "ENSG00000105131"}


@pytest.mark.unit
def test_filter_gene_file_keeps_rows_and_columns(tmp_path, targets_tsv):
    hpp = tmp_path / "Cholesterol_extreme_M.tsv"
    pd.DataFrame({
        # A different version suffix on the same gene must still match.
        "gene_id": ["ENSG00000174327.9", "ENSG00000105131.3", "ENSG00000999999.1"],
        "p_F_analytic": [1e-11, 1e-10, 1e-3],
        "extra": ["a", "b", "c"],
    }).to_csv(hpp, sep="\t", index=False)

    out = tmp_path / "filtered" / "Cholesterol_extreme_M.tsv"
    kept = T.filter_gene_file(str(hpp), str(out), "Cholesterol", "M", targets_tsv)

    frame = pd.read_csv(out, sep="\t")
    assert kept == 1
    assert list(frame["gene_id"]) == ["ENSG00000174327.9"]
    assert list(frame.columns) == ["gene_id", "p_F_analytic", "extra"]


@pytest.mark.unit
def test_missing_column_is_an_error(tmp_path):
    path = tmp_path / "bad.tsv"
    pd.DataFrame({"gene_id": ["ENSG1"], "trait": ["Height"]}).to_csv(
        path, sep="\t", index=False)
    with pytest.raises(KeyError):
        T.load_targets(str(path))


@pytest.mark.unit
def test_all_cell_recovers_sex_specific_rows(tmp_path, targets_tsv):
    """A hit significant in one sex only must still reach the pooled cell.

    HPP's own Cholesterol_extreme.tsv holds the genes it called significant
    when pooled, so a men-only hit is absent from it. Intersecting with that
    file drops the gene and the pooled UKBB test never runs.
    """
    pooled = tmp_path / "Cholesterol_extreme.tsv"
    pd.DataFrame({
        "gene_id": ["ENSG00000999999.1"],
        "p_F_analytic": [1e-3],
        "extra": ["c"],
    }).to_csv(pooled, sep="\t", index=False)

    # The men-only and women-only hits live in the sex-specific files.
    pd.DataFrame({
        "gene_id": ["ENSG00000174327.6"],
        "p_F_analytic": [1e-11],
        "extra": ["a"],
    }).to_csv(tmp_path / "Cholesterol_extreme_M.tsv", sep="\t", index=False)
    pd.DataFrame({
        "gene_id": ["ENSG00000105131.3"],
        "p_F_analytic": [1e-10],
        "extra": ["b"],
    }).to_csv(tmp_path / "Cholesterol_extreme_F.tsv", sep="\t", index=False)

    out = tmp_path / "filtered" / "Cholesterol_extreme.tsv"
    kept = T.filter_gene_file(str(pooled), str(out), "Cholesterol", "All",
                              targets_tsv)

    frame = pd.read_csv(out, sep="\t")
    assert kept == 2
    assert set(frame["gene_id"]) == {"ENSG00000174327.6", "ENSG00000105131.3"}
    assert list(frame.columns) == ["gene_id", "p_F_analytic", "extra"]


@pytest.mark.unit
def test_target_absent_from_every_hpp_file_still_gets_a_row(tmp_path,
                                                            targets_tsv):
    """A UKB-only target needs just its gene_id; FM-GWAS reads nothing else.

    Ning's full-cohort hits are mostly absent from the July HPP gene files.
    Dropping them would mean they are never tested in UKB, so the row is made
    from the target list itself, with the HPP-only columns left empty.
    """
    pooled = tmp_path / "Height_extreme.tsv"
    pd.DataFrame({"gene_id": ["ENSG00000999999.1"], "p_F_analytic": [1e-3]}
                 ).to_csv(pooled, sep="\t", index=False)
    out = tmp_path / "filtered" / "Height_extreme.tsv"
    kept = T.filter_gene_file(str(pooled), str(out), "Height", "All",
                              targets_tsv)

    frame = pd.read_csv(out, sep="\t")
    assert kept == 1
    assert list(frame["gene_id"]) == ["ENSG00000197208.5"]
    assert list(frame.columns) == ["gene_id", "p_F_analytic"]
    assert frame["p_F_analytic"].isna().all()


@pytest.mark.unit
def test_target_rows_respect_the_cell_sex(tmp_path, targets_tsv):
    hpp = tmp_path / "Height_extreme_M.tsv"
    pd.DataFrame({"gene_id": ["ENSG00000999999.1"]}).to_csv(
        hpp, sep="\t", index=False)
    out = tmp_path / "filtered" / "Height_extreme_M.tsv"
    assert T.filter_gene_file(str(hpp), str(out), "Height", "M",
                              targets_tsv) == 0


@pytest.mark.unit
def test_extend_with_targets_keeps_union_ids_and_adds_the_rest(targets_tsv):
    union = ["ENSG00000174327.9", "ENSG00000999999.1"]
    kept, added = T.extend_with_targets(union, targets_tsv)
    # Union version wins where both exist; absent targets use the list's id.
    assert kept == ["ENSG00000174327.9", "ENSG00000105131.3",
                    "ENSG00000197208.5"]
    assert added == ["ENSG00000105131.3", "ENSG00000197208.5"]


@pytest.mark.unit
def test_candidate_genes_gain_missing_targets_once(tmp_path, targets_tsv):
    hpp = tmp_path / "candidate_genes.tsv"
    pd.DataFrame({"gene_id": ["ENSG00000174327.9", "ENSG00000000001.1"]}
                 ).to_csv(hpp, sep="\t", index=False)
    out = tmp_path / "out" / "candidate_genes.tsv"
    n_added = T.write_candidate_genes(str(hpp), str(out), targets_tsv)

    frame = pd.read_csv(out, sep="\t")
    assert n_added == 2
    assert list(frame["gene_id"]) == ["ENSG00000174327.9", "ENSG00000000001.1",
                                      "ENSG00000105131.3", "ENSG00000197208.5"]
