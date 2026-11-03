import numpy as np
import pandas as pd
import pytest

import trait_rules as R


@pytest.mark.unit
def test_parse_dump_export_and_plain_columns():
    assert R.parse_column("4080-0.1") == R.Column("4080-0.1", "4080", 0, 1)
    assert R.parse_column("ldl_i1") == R.Column("ldl_i1", "ldl", 1, 0)
    assert R.parse_column("bmi") == R.Column("bmi", "bmi", 0, 0)


@pytest.mark.unit
def test_group_fields_drops_the_id_column():
    groups = R.group_fields(["eid", "50-0.0", "50-1.0", "ldl_i0"], "eid")
    assert sorted(groups) == ["50", "ldl"]
    assert [c.instance for c in groups["50"]] == [0, 1]


@pytest.mark.unit
def test_excluded_and_multi_select_fields():
    assert R.is_excluded("22009") and R.is_excluded("41270") and R.is_excluded("31")
    assert not R.is_excluded("50") and not R.is_excluded("ldl")
    many = [R.Column(f"20002-0.{a}", "20002", 0, a) for a in range(10)]
    assert R.too_many_arrays(many)
    assert not R.too_many_arrays(many[:2])


@pytest.mark.unit
def test_special_codes_only_cleaned_in_integer_fields():
    ints = pd.Series([1.0, 2.0, -1.0, -3.0, 5.0])
    assert R.clean_special_codes(ints).isna().sum() == 2
    floats = pd.Series([0.5, -1.0, 2.25])
    assert R.clean_special_codes(floats).isna().sum() == 0


@pytest.mark.unit
def test_best_instance_averages_arrays_of_the_best_covered_instance():
    frame = pd.DataFrame({"4080-0.0": [120.0, 130.0, np.nan],
                          "4080-0.1": [124.0, np.nan, np.nan],
                          "4080-2.0": [np.nan, np.nan, 140.0]})
    cols = [R.parse_column(c) for c in frame.columns]
    instance, merged = R.best_instance(frame, cols)
    assert instance == 0
    assert merged.tolist()[:2] == [122.0, 130.0]


def _series(values):
    return pd.Series(np.asarray(values, dtype=float))


@pytest.mark.unit
def test_classify_binary_quantitative_and_skips(monkeypatch):
    monkeypatch.setattr(R, "MIN_N", 10)
    monkeypatch.setattr(R, "MIN_MINOR_CLASS", 3)
    binary = R.classify(_series([0] * 20 + [1] * 5))
    assert binary.kind == "binary" and binary.n_minor == 5
    assert R.encode(_series([1, 2, 2]), R.classify(_series([1] * 8 + [2] * 8))).tolist() \
        == [0.0, 1.0, 1.0]
    assert R.classify(_series(np.arange(30))).kind == "quantitative"
    assert R.classify(_series([0, 1, 2, 3] * 10)).kind == "skip"          # ordinal
    assert R.classify(_series([0] * 20 + [1] * 2)).kind == "skip"         # rare class
    assert R.classify(_series([1] * 30)).reason == "constant"
    assert R.classify(_series(np.arange(5))).kind == "skip"               # small n
    assert R.classify(_series(np.arange(30)), "Categorical multiple").kind == "skip"
    assert R.classify(_series(np.arange(30)), "Date").kind == "skip"
    assert R.classify(_series(np.arange(30)), "Continuous").kind == "quantitative"


@pytest.mark.unit
def test_find_id_column():
    assert R.find_id_column(["x", "eid"]) == "eid"
    assert R.find_id_column(["participant_id", "y"]) == "participant_id"
    assert R.find_id_column(["x"]) is None
