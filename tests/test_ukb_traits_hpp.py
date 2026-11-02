"""Unit tests for the HPP-aligned disease labels (Ning, 2026-11-02)."""
import pandas as pd
import pytest

import ukb_traits_hpp as H

INDEX = [1, 2, 3, 4, 5]


def flags(true_for=()) -> pd.Series:
    return pd.Series([i in set(true_for) for i in INDEX], index=INDEX)


@pytest.mark.unit
def test_hypertension_is_diagnosis_or_selfreport_and_everyone_else_control():
    label = H.hypertension(flags([1]), flags([2]))
    assert label.to_dict() == {1: 1, 2: 1, 3: 0, 4: 0, 5: 0}


@pytest.mark.unit
def test_hyperlipidemia_ignores_nothing_but_icd_and_selfreport():
    label = H.hyperlipidemia(flags([3]), flags())
    assert label.to_dict() == {1: 0, 2: 0, 3: 1, 4: 0, 5: 0}


@pytest.mark.unit
def test_t2d_counts_field_2443_yes_only():
    doctor = pd.Series([1, 0, -1, None, 1], index=INDEX, dtype="float64")
    label = H.type_2_diabetes(flags([2]), flags([3]), doctor)
    assert label.to_dict() == {1: 1, 2: 1, 3: 1, 4: 0, 5: 1}


@pytest.mark.unit
def test_obesity_keeps_only_obese_and_normal_bmi():
    bmi = pd.Series({1: 31.0, 2: 30.0, 3: 22.0, 4: 27.0, 5: 17.0, 6: None,
                     7: 18.5, 8: 24.95})
    label = H.obesity(bmi)
    # 30.0 exactly is not > 30; 18.5 and 24.95 fall outside 18.5 < BMI < 24.9
    assert label.to_dict() == {1: 1, 3: 0}


@pytest.mark.unit
def test_osteoporosis_bmd_option():
    bmd = pd.Series([0.80, 0.95, None, 0.95, None], index=INDEX)
    label = H.osteoporosis(bmd, flags([4]), flags([5]), "bmd")
    # 1 low BMD; 2 normal BMD control; 3 no scan, undiagnosed -> dropped;
    # 4 diagnosed despite normal BMD; 5 self-reported without a scan
    assert label.to_dict() == {1: 1, 2: 0, 4: 1, 5: 1}


@pytest.mark.unit
def test_osteoporosis_diagnosis_option_ignores_bmd():
    bmd = pd.Series([0.80, 0.95, None, 0.95, None], index=INDEX)
    label = H.osteoporosis(bmd, flags([4]), flags([5]), "diagnosis")
    assert label.to_dict() == {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}


@pytest.mark.unit
def test_osteoporosis_rejects_unknown_option():
    with pytest.raises(ValueError, match="osteoporosis option"):
        H.osteoporosis(pd.Series(dtype=float), flags(), flags(), "tscore")
