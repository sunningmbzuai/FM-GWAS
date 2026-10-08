"""Unit tests for the recovery-rate computation.

Uses a temp directory of fake association TSVs rather than cluster paths, so
these run anywhere. The distinction these pin down: a gene that was never
tested in UKB must not be counted the same as one that was tested and failed
to replicate.
"""
import pandas as pd
import pytest

import compute_recovery as R


def write_gene(dirpath, gene_id: str, p_values: list[float]) -> None:
    frame = pd.DataFrame({
        "gene_id": [gene_id] * len(p_values),
        "embed_type": [f"embed_{i}" for i in range(len(p_values))],
        "p_F_analytic": p_values,
    })
    frame.to_csv(dirpath / f"{gene_id}.tsv", sep="\t", index=False)


@pytest.mark.unit
def test_strip_version_makes_ids_comparable():
    assert R.strip_version("ENSG00000141510.16") == "ENSG00000141510"
    assert R.strip_version("ENSG00000141510") == "ENSG00000141510"


@pytest.mark.unit
def test_union_p_takes_the_minimum_across_embedding_types(tmp_path):
    write_gene(tmp_path, "ENSG1", [0.5, 1e-9, 0.01])

    p = R.union_p_values(str(tmp_path))

    assert p.to_dict() == {"ENSG1": 1e-9}


@pytest.mark.unit
def test_union_p_strips_versions_from_result_files(tmp_path):
    write_gene(tmp_path, "ENSG2.7", [1e-8])

    assert list(R.union_p_values(str(tmp_path)).index) == ["ENSG2"]


@pytest.mark.unit
def test_union_p_on_a_missing_directory_is_empty():
    assert R.union_p_values("/nonexistent/path/xyz").empty


@pytest.mark.unit
def test_union_p_rejects_results_without_a_p_column(tmp_path):
    pd.DataFrame({"gene_id": ["ENSG9"], "F_obs": [3.0]}).to_csv(
        tmp_path / "ENSG9.tsv", sep="\t", index=False
    )

    with pytest.raises(KeyError, match="p_F_analytic"):
        R.union_p_values(str(tmp_path))


@pytest.mark.unit
def test_significance_uses_a_strict_less_than_at_the_threshold():
    p = pd.Series({"a": R.SIGNIFICANCE_THRESHOLD, "b": R.SIGNIFICANCE_THRESHOLD / 2})

    assert R.significant_genes(p) == {"b"}


@pytest.mark.unit
def test_recovery_separates_non_replication_from_non_coverage():
    #  g1 replicates, g2 tested but not significant, g3 never tested in UKB.
    hpp = {"g1", "g2", "g3"}
    ukb = pd.Series({"g1": 1e-9, "g2": 0.3, "g4": 1e-12})

    stats = R.recovery(hpp, ukb)

    assert stats["hpp_significant"] == 3
    assert stats["recovered"] == 1
    assert stats["tested_in_ukb"] == 2
    assert stats["not_tested_in_ukb"] == 1
    assert stats["recovery_rate"] == pytest.approx(1 / 3)
    assert stats["recovery_rate_of_tested"] == pytest.approx(1 / 2)
    assert stats["ukb_significant_novel"] == 1     # g4


@pytest.mark.unit
def test_recovery_of_an_empty_hpp_list_is_not_a_zero_rate():
    """No denominator means undefined, not 0% -- 0% would read as total failure."""
    import math

    stats = R.recovery(set(), pd.Series({"g1": 1e-9}))

    assert math.isnan(stats["recovery_rate"])
