import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import summarize_replication as S  # noqa: E402


def write_assoc(root, trait, sex, gene_id, p_by_type, column="p_F"):
    cell = root / f"{trait}_{sex}" / "per_gene"
    cell.mkdir(parents=True, exist_ok=True)
    status = "lrt_status" if column == "p_LRT" else "status"
    pd.DataFrame({
        "gene_id": gene_id,
        "embed_type": list(p_by_type),
        "gene_pca_components": 3,
        "status": "ok",
        status: "ok",
        column: list(p_by_type.values()),
    }).to_csv(cell / f"{gene_id}.tsv", sep="\t", index=False)


def targets():
    return pd.DataFrame({
        "gene_id": ["ENSG01.3", "ENSG02.1", "ENSG03.2"],
        "p_F_analytic_min": [1e-10, 1e-7, 1e-6],
        "trait": ["Height", "Height", "VAT"],
        "sex": ["F", "F", "M"],
    })


def test_long_table_has_one_row_per_embedding_type(tmp_path):
    write_assoc(tmp_path, "Height", "F", "ENSG01.3",
                {"concat_tss_cage": 1e-8, "add_tss_cage": 0.2})
    long = S.per_embedding_p(targets(), str(tmp_path))
    assert len(long) == 2
    assert set(long["embed_type"]) == {"concat_tss_cage", "add_tss_cage"}


def test_gene_version_mismatch_still_joins(tmp_path):
    write_assoc(tmp_path, "Height", "F", "ENSG02.7", {"concat_cage": 0.01})
    long = S.per_embedding_p(targets(), str(tmp_path))
    assert list(long["gene_id"]) == ["ENSG02.1"]


def test_summary_marks_untested_and_replicated(tmp_path):
    write_assoc(tmp_path, "Height", "F", "ENSG01.3",
                {"concat_tss_cage": 1e-8, "add_tss_cage": 0.2})
    write_assoc(tmp_path, "Height", "F", "ENSG02.1", {"concat_cage": 0.01})
    counts = pd.DataFrame({"gene_id": ["ENSG01.3"], "n_variants": [4],
                           "n_polymorphic": [3]})
    summary = S.per_gene_summary(targets(),
                                 S.per_embedding_p(targets(), str(tmp_path)),
                                 counts)
    by_gene = summary.set_index("gene_id")
    assert by_gene.loc["ENSG01.3", "ukbb_best_embed_type"] == "concat_tss_cage"
    assert bool(by_gene.loc["ENSG01.3", "replicated_genomewide"])
    assert by_gene.loc["ENSG01.3", "n_variants"] == 4
    assert not bool(by_gene.loc["ENSG02.1", "replicated_genomewide"])
    assert by_gene.loc["ENSG03.2", "status"] == S.STATUS_NOT_TESTED


def test_rates_count_only_tested_genes_in_denominator_column(tmp_path):
    write_assoc(tmp_path, "Height", "F", "ENSG01.3", {"concat_cage": 1e-8})
    summary = S.per_gene_summary(targets(),
                                 S.per_embedding_p(targets(), str(tmp_path)),
                                 None)
    rates = S.replication_rates(summary)
    assert rates["n_targets"] == 3
    assert rates["n_tested"] == 1
    assert rates["n_replicated_genomewide"] == 1


def test_disease_cell_is_judged_on_the_lrt(tmp_path):
    rows = pd.DataFrame({"gene_id": ["ENSG05.1"], "p_F_analytic_min": [None],
                         "trait": ["Hypertension"], "sex": ["M"]})
    write_assoc(tmp_path, "Hypertension", "M", "ENSG05.1",
                {"concat_tss_cage": 1e-9, "concat_tss_dnase": 0.3}, column="p_LRT")
    long = S.per_embedding_p(rows, str(tmp_path))
    assert set(long["test"]) == {"LRT"}
    summary = S.per_gene_summary(rows, long, None).set_index("gene_id")
    assert summary.loc["ENSG05.1", "ukbb_p_min"] == 1e-9
    assert summary.loc["ENSG05.1", "ukbb_test_status"] == "ok"
    assert summary.loc["ENSG05.1", "status"] == S.STATUS_TESTED


def test_variant_table_status_column_does_not_clash(tmp_path):
    """count_gene_variants.py writes its own status column."""
    write_assoc(tmp_path, "Height", "F", "ENSG01.3", {"concat_cage": 1e-8})
    counts = pd.DataFrame({"gene_id": ["ENSG01.3"], "n_variants": [4],
                           "n_polymorphic": [3], "status": ["ok"],
                           "trait": ["Height"], "sex": ["F"]})
    summary = S.per_gene_summary(targets(),
                                 S.per_embedding_p(targets(), str(tmp_path)),
                                 counts)
    rates = S.replication_rates(summary)
    assert rates["n_tested"] == 1
    assert summary.set_index("gene_id").loc["ENSG01.3", "variant_status"] == "ok"
    assert not any(c.endswith(("_x", "_y")) for c in summary.columns)


def test_summary_carries_case_counts_and_power(tmp_path):
    rows = pd.DataFrame({"gene_id": ["ENSG05.1"], "p_F_analytic_min": [None],
                         "trait": ["Hypertension"], "sex": ["M"]})
    write_assoc(tmp_path, "Hypertension", "M", "ENSG05.1",
                {"concat_tss_cage": 0.01}, column="p_LRT")
    path = tmp_path / "Hypertension_M" / "per_gene" / "ENSG05.1.tsv"
    pd.read_csv(path, sep="\t").assign(n_samples=10_000, n_cases=500).to_csv(
        path, sep="\t", index=False)
    summary = S.per_gene_summary(rows, S.per_embedding_p(rows, str(tmp_path)),
                                 None).set_index("gene_id")
    row = summary.loc["ENSG05.1"]
    assert row["n_samples"] == 10_000
    assert row["n_cases"] == 500
    assert row["n_controls"] == 9_500
    assert abs(row["n_eff"] - 4 / (1 / 500 + 1 / 9_500)) < 1e-6
    assert 0 < row["min_detectable_r2"] < 1


def test_power_grows_with_cases():
    alpha = 0.05 / 25
    small = S.min_detectable_r2(n_eff=500, df=3, alpha=alpha)
    large = S.min_detectable_r2(n_eff=50_000, df=3, alpha=alpha)
    assert large < small


def test_power_is_nan_without_counts():
    assert pd.isna(S.min_detectable_r2(n_eff=float("nan"), df=3, alpha=0.002))
    assert pd.isna(S.min_detectable_r2(n_eff=1000, df=float("nan"), alpha=0.002))


def test_case_counts_are_nan_for_old_results(tmp_path):
    write_assoc(tmp_path, "Height", "F", "ENSG01.3", {"concat_cage": 1e-8})
    summary = S.per_gene_summary(targets(),
                                 S.per_embedding_p(targets(), str(tmp_path)),
                                 None).set_index("gene_id")
    assert pd.isna(summary.loc["ENSG01.3", "n_cases"])
    assert pd.isna(summary.loc["ENSG01.3", "min_detectable_r2"])


def write_source(path, rows):
    pd.DataFrame(rows, columns=["gene_id", "trait", "sex"]).to_csv(
        path, sep="\t", index=False)


def test_target_source_matches(tmp_path):
    src = tmp_path / "novel_genes_1019.tsv"
    write_source(src, [("ENSG05.2", "Hypertension", "M")])
    rows = pd.DataFrame({"gene_id": ["ENSG05.1"], "trait": ["Hypertension"],
                         "sex": ["M"]})
    ok, message = S.check_target_source(rows, [str(src)])
    assert ok
    assert "1 rows" in message and "novel_genes_1019.tsv" in message


def test_target_source_missing_file(tmp_path):
    ok, message = S.check_target_source(targets(), [str(tmp_path / "gone.tsv")])
    assert not ok
    assert "missing" in message and "gone.tsv" in message


def test_target_source_stale_list(tmp_path):
    src = tmp_path / "novel_genes_1019.tsv"
    write_source(src, [("ENSG01.3", "Height", "F"), ("ENSG09.1", "VAT", "F")])
    ok, message = S.check_target_source(targets(), [str(src)])
    assert not ok
    assert "1 in the export but not the target list" in message
    assert "2 in the target list but not the export" in message
