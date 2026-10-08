"""Unit tests for the DXA column cleaning helpers.

Both failures these cover are silent: a placeholder zero becomes the leanest
participant in a continuous trait, and a single-site T-score misclassifies
people who are osteoporotic somewhere else.
"""
import pandas as pd
import pytest

import ukb_source as S


@pytest.mark.unit
def test_zero_becomes_missing_in_the_named_columns():
    frame = pd.DataFrame({"vat_volume_i2": [0.0, 1500.0], "other": [0.0, 1.0]})

    cleaned = S.mask_zeros_as_missing(frame, ["vat_volume_i2"])

    assert pd.isna(cleaned["vat_volume_i2"].iloc[0])
    assert cleaned["vat_volume_i2"].iloc[1] == 1500.0
    assert cleaned["other"].iloc[0] == 0.0      # untouched


@pytest.mark.unit
def test_masking_does_not_mutate_the_input():
    frame = pd.DataFrame({"vat_volume_i2": [0.0, 1500.0]})

    S.mask_zeros_as_missing(frame, ["vat_volume_i2"])

    assert frame["vat_volume_i2"].iloc[0] == 0.0


@pytest.mark.unit
def test_masking_ignores_columns_that_are_absent():
    frame = pd.DataFrame({"a": [0.0]})

    assert S.mask_zeros_as_missing(frame, ["not_here"]).equals(frame)


@pytest.mark.unit
def test_lowest_across_sites_takes_the_worst_site():
    frame = pd.DataFrame({
        "femur_neck_bmd_t_score_left_i2": [-0.5, -3.0],
        "l1_l4_bmd_t_score_i2": [-2.7, -0.2],
    })

    lowest = S.lowest_across_sites(frame, list(frame.columns))

    assert lowest.tolist() == [-2.7, -3.0]


@pytest.mark.unit
def test_lowest_across_sites_skips_unmeasured_sites():
    frame = pd.DataFrame({
        "femur_neck_bmd_t_score_left_i2": [float("nan")],
        "l1_l4_bmd_t_score_i2": [-1.4],
    })

    assert S.lowest_across_sites(frame, list(frame.columns)).tolist() == [-1.4]


@pytest.mark.unit
def test_lowest_across_sites_rejects_a_missing_column():
    with pytest.raises(KeyError, match="not_here"):
        S.lowest_across_sites(pd.DataFrame({"a": [1.0]}), ["a", "not_here"])
