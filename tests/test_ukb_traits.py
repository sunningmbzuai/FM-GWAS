"""Unit tests for the trait label logic.

Pure-function tests only: no cluster paths, no parquet, so they run anywhere.
The definitions in ukb_fields.py / ukb_traits.py are the part of the pipeline
most likely to be silently wrong, and the cheapest to pin down.
"""
import pandas as pd
import pytest

import ukb_fields as F
import ukb_traits as T


def series(values: dict) -> pd.Series:
    return pd.Series(values, dtype="float64")


def flags(index, true_for=()) -> pd.Series:
    return pd.Series([i in set(true_for) for i in index], index=index)


@pytest.mark.unit
def test_extreme_tails_labels_only_the_tails():
    values = series({i: float(i) for i in range(100)})
    label = T.extreme_tails(values, 0.10)

    assert set(label.unique()) == {0, 1}
    assert label.loc[0] == 0 and label.loc[99] == 1
    assert 50 not in label.index          # middle dropped
    assert 20 <= len(label) <= 24         # ~2 x 10% incl. quantile ties


@pytest.mark.unit
def test_extreme_tails_rejects_a_degenerate_distribution():
    with pytest.raises(ValueError, match="degenerate"):
        T.extreme_tails(series({i: 5.0 for i in range(50)}), 0.10)


@pytest.mark.unit
def test_quantitative_continuous_keeps_values_and_drops_missing():
    values = series({1: 170.0, 2: float("nan"), 3: 180.0})
    label = T.quantitative(values, "continuous", 0.10)

    assert label.to_dict() == {1: 170.0, 3: 180.0}


@pytest.mark.unit
def test_quantitative_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="QUANT_LABEL_MODE"):
        T.quantitative(series({1: 1.0}), "deciles", 0.10)


@pytest.mark.unit
def test_obesity_is_presence_absence_with_no_dropped_middle():
    """Overweight-but-not-obese is a control, per the paper; only NaN drops."""
    bmi = series({1: 32.0, 2: 22.0, 3: 27.0, 4: float("nan")})
    label = T.obesity(bmi)

    assert label.to_dict() == {1: 1, 2: 0, 3: 0}
    assert 4 not in label.index


@pytest.mark.unit
def test_hyperlipidemia_counts_diagnosis_medication_and_measurement():
    index = [1, 2, 3, 4, 5]
    chol = pd.Series([4.0, 4.0, 4.0, 7.0, 5.6], index=index)
    label = T.hyperlipidemia(
        chol,
        icd=flags(index, [1]),
        selfrep=flags(index, [2]),
        chol_med=flags(index, [3]),
    )

    assert label.loc[[1, 2, 3]].tolist() == [1, 1, 1]  # each route makes a case
    assert label.loc[4] == 1                           # measurement alone
    assert label.loc[5] == 0                           # borderline is a control


@pytest.mark.unit
def test_hyperlipidemia_needs_a_diagnosis_or_a_measurement():
    index = [1]
    label = T.hyperlipidemia(
        pd.Series([float("nan")], index=index),
        icd=flags(index), selfrep=flags(index), chol_med=flags(index),
    )

    assert label.empty


@pytest.mark.unit
def test_diagnosis_outranks_a_normal_measurement():
    """A treated hypertensive with normal BP is a case, not a control."""
    index = [1]
    label = T.hypertension(
        sbp=pd.Series([118.0], index=index),
        dbp=pd.Series([76.0], index=index),
        icd=flags(index),
        selfrep=flags(index),
        bp_med=flags(index, [1]),
    )

    assert label.loc[1] == 1


@pytest.mark.unit
def test_hypertension_needs_both_readings_or_a_diagnosis():
    index = [1]
    label = T.hypertension(
        sbp=pd.Series([118.0], index=index),
        dbp=pd.Series([float("nan")], index=index),
        icd=flags(index),
        selfrep=flags(index),
        bp_med=flags(index),
    )

    assert label.empty


@pytest.mark.unit
def test_t2d_counts_prediabetes_and_every_diabetes_type_as_a_case():
    """Paper: "both prediabetes and diabetes are defined as 1"."""
    index = [1, 2, 3, 4, 5]
    #  1: T2D by ICD.  2: T1D -- still a case.  3: prediabetic HbA1c.
    #  4: normal HbA1c, explicit no.  5: insulin, no other evidence.
    label = T.type_2_diabetes(
        icd_t2d=flags(index, [1]),
        icd_t1d=flags(index, [2]),
        selfrep_t2d=flags(index),
        selfrep_t1d=flags(index),
        selfrep_unspec=flags(index),
        doctor_said=pd.Series([1.0, 1.0, float("nan"), 0.0, float("nan")], index=index),
        insulin=flags(index, [5]),
        hba1c=pd.Series([float("nan"), float("nan"), 42.0, 33.0, float("nan")], index=index),
    )

    assert label.to_dict() == {1: 1, 2: 1, 3: 1, 4: 0, 5: 1}


@pytest.mark.unit
def test_t2d_controls_require_positive_evidence_of_being_non_diabetic():
    """Missing both field 2443 and HbA1c is not evidence of being diabetes-free."""
    index = [1]
    label = T.type_2_diabetes(
        icd_t2d=flags(index), icd_t1d=flags(index),
        selfrep_t2d=flags(index), selfrep_t1d=flags(index),
        selfrep_unspec=flags(index),
        doctor_said=pd.Series([float("nan")], index=index),
        insulin=flags(index),
        hba1c=pd.Series([float("nan")], index=index),
    )

    assert label.empty


@pytest.mark.unit
def test_osteoporosis_counts_osteopenia_as_a_case():
    """Paper: "both osteoporosis and osteopenia are defined as 1"."""
    index = [1, 2, 3]
    tscore = pd.Series([-3.0, -0.5, -1.8], index=index)
    label = T.osteoporosis(tscore, icd=flags(index), selfrep=flags(index))

    assert label.to_dict() == {1: 1, 2: 0, 3: 1}   # -1.8 is osteopenic => case


@pytest.mark.unit
def test_case_cutoffs_are_the_inclusive_ones_the_paper_specifies():
    """Osteopenia and prediabetes cutoffs, not the strict-disease cutoffs."""
    assert F.TSCORE_OSTEOPOROSIS < F.TSCORE_LOW_BONE_MASS
    assert F.HBA1C_PREDIABETES < F.HBA1C_DIABETES


@pytest.mark.unit
def test_paper_defaults_are_the_configured_defaults():
    import config

    assert config.QUANT_LABEL_MODE == "continuous"
    assert config.GENE_UPSTREAM_BP == 0 and config.GENE_DOWNSTREAM_BP == 0
    assert config.MAX_GENE_LEN == 128 * 1024
    assert config.GENE_TYPE == "protein_coding"


@pytest.mark.unit
def test_describe_labels_summarizes_binary_as_a_case_fraction():
    from build_cohorts import describe_labels

    labels = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert describe_labels(labels) == "binary, 2 cases (20.0%)"


@pytest.mark.unit
def test_describe_labels_summarizes_continuous_as_quantiles():
    from build_cohorts import describe_labels

    summary = describe_labels(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert summary.startswith("continuous, median 3")
    assert "max 5" in summary


@pytest.mark.unit
def test_describe_labels_handles_an_empty_cohort():
    from build_cohorts import describe_labels

    assert describe_labels(pd.Series(dtype="float64")) == "no labels"
