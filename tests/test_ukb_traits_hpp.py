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


# ---- hpp_1103: Ning's 2026-11-03 correction, controls from the "control" class only


@pytest.mark.unit
def test_scheme_cases_icd_selfrep_and_icd_only():
    assert H.scheme_cases(flags([1]), flags([2]), "icd_selfrep").to_dict() == \
        {1: True, 2: True, 3: False, 4: False, 5: False}
    assert H.scheme_cases(flags([1]), flags([2]), "icd_only").to_dict() == \
        {1: True, 2: False, 3: False, 4: False, 5: False}


@pytest.mark.unit
def test_scheme_cases_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="case scheme"):
        H.scheme_cases(flags(), flags(), "icd_meds")


@pytest.mark.unit
def test_case_or_paper_control_drops_every_other_non_case():
    # paper label: 1 ICD case, 2 medication-only case, 3 control, 4 control,
    # 5 not assessable (absent from the paper labels)
    paper = pd.Series({1: 1, 2: 1, 3: 0, 4: 0})
    label = H.case_or_paper_control(flags([1]), paper)
    assert label.to_dict() == {1: 1, 3: 0, 4: 0}


@pytest.mark.unit
def test_case_or_paper_control_refuses_a_case_that_was_a_paper_control():
    paper = pd.Series({1: 0, 2: 0})
    with pytest.raises(ValueError, match="paper control"):
        H.case_or_paper_control(flags([1]), paper)
