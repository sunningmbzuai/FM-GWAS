import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import summarize_replication as S  # noqa: E402


def write_assoc(root, trait, sex, gene_id, p_by_type):
    cell = root / f"{trait}_{sex}"
    cell.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "gene_id": gene_id,
        "num_feature": 3,
        "embed_type": [f"{t}_pca0.95" for t in p_by_type],
        "F_obs": 1.0,
        "p_F_analytic": list(p_by_type.values()),
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
