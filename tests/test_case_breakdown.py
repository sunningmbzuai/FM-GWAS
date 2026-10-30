"""Unit tests for the case-class breakdown sent to Ning (2026-10-23 request).

Synthetic inputs only. What matters: every case lands in exactly one class, in
the listed order, so the classes sum to the case count the cohort files carry;
and the HPP comparison lines up on (trait, sex).
"""
import pandas as pd
import pytest

import case_breakdown as C


def mask(index, true_for=()) -> pd.Series:
    return pd.Series([i in set(true_for) for i in index], index=index)


@pytest.mark.unit
def test_first_matching_class_wins_and_controls_are_controls():
    idx = pd.Index([1, 2, 3, 4])
    labels = pd.Series([1, 1, 1, 0], index=idx)
    tiers = [("diagnosed", mask(idx, {1, 2})), ("measured", mask(idx, {2, 3, 4}))]
    classes = C.assign_classes(labels, tiers)
    assert classes.to_dict() == {1: "diagnosed", 2: "diagnosed",
                                 3: "measured", 4: C.CONTROL}


@pytest.mark.unit
def test_case_matching_no_class_is_reported_not_dropped():
    idx = pd.Index([1, 2])
    labels = pd.Series([1, 1], index=idx)
    classes = C.assign_classes(labels, [("diagnosed", mask(idx, {1}))])
    assert classes[2] == C.UNCLASSIFIED


@pytest.mark.unit
def test_class_rows_sum_to_the_cases():
    idx = pd.Index(range(10))
    labels = pd.Series([1] * 4 + [0] * 6, index=idx)
    classes = C.assign_classes(labels, [("a", mask(idx, {0})),
                                        ("b", mask(idx, {1, 2, 3}))])
    rows = C.class_rows("Osteoporosis", "F", classes, ["a", "b"])
    by_class = {r["class"]: r for r in rows}
    assert by_class["a"]["n"] == 1 and by_class["b"]["n"] == 3
    assert by_class[C.CONTROL]["n"] == 6
    assert by_class["b"]["pct_of_cases"] == pytest.approx(75.0)
    assert by_class["b"]["pct_of_cohort"] == pytest.approx(30.0)
    assert sum(r["n"] for r in rows if r["class"] != C.CONTROL) == 4


@pytest.mark.unit
def test_hypertension_splits_measured_cases_by_stage():
    idx = pd.Index([1, 2, 3, 4])
    no = mask(idx)
    sbp = pd.Series([170.0, 150.0, 120.0, 120.0], index=idx)
    dbp = pd.Series([80.0, 80.0, 80.0, 80.0], index=idx)
    tiers = dict(C.hypertension_tiers(sbp, dbp, icd=mask(idx, {1}),
                                      selfrep=no, bp_med=no))
    labels = pd.Series([1, 1, 0, 0], index=idx)
    classes = C.assign_classes(labels, list(tiers.items()))
    # Diagnosis outranks the measurement; an undiagnosed 150 is stage 1.
    assert classes[1] == "ICD-10 I10-I15"
    assert classes[2] == "SBP 140-159 or DBP 90-99, no diagnosis"


@pytest.mark.unit
def test_osteoporosis_separates_osteopenia_from_osteoporosis():
    idx = pd.Index([1, 2, 3])
    no = mask(idx)
    tscore = pd.Series([-3.0, -1.5, 0.0], index=idx)
    tiers = C.osteoporosis_tiers(tscore, icd=no, selfrep=no)
    classes = C.assign_classes(pd.Series([1, 1, 0], index=idx), tiers)
    assert classes[1] == "T-score <= -2.5 (osteoporosis), no diagnosis"
    assert classes[2] == "T-score -2.5 to -1.0 (osteopenia), no diagnosis"


@pytest.mark.unit
def test_case_rates_line_up_with_hpp():
    ukb = pd.DataFrame([{"trait": "Hypertension", "sex": "F",
                         "ukb_n": 100, "ukb_cases": 40}])
    hpp = pd.DataFrame([{"trait": "Hypertension", "sex": "F",
                         "n_samples": 50, "n_cases": 5}])
    row = C.case_rates(ukb, hpp).iloc[0]
    assert row["ukb_case_pct"] == pytest.approx(40.0)
    assert row["hpp_case_pct"] == pytest.approx(10.0)
    assert row["ukb_over_hpp"] == pytest.approx(4.0)
