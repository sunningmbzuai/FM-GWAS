"""Unit tests for compare_assoc_runs: tolerance, NaN handling, text columns."""
import pandas as pd
import pytest

import compare_assoc_runs as C


def frame(p, name="a"):
    return pd.DataFrame({"embed_type": [name, "b"], "p_LRT": p})


@pytest.mark.unit
def test_rounding_noise_is_within_eps():
    worst, problems = C.compare_frames(frame([0.9516126908542887, 0.1]),
                                       frame([0.9516126908542883, 0.1]), "g")
    assert problems == []
    assert worst["p_LRT"].diff < 1e-15


@pytest.mark.unit
def test_real_change_is_reported_at_its_row():
    worst, _ = C.compare_frames(frame([0.5, 0.1]), frame([0.5, 0.2]), "g")
    assert worst["p_LRT"].row == 1
    assert worst["p_LRT"].diff > C.DEFAULT_EPS


@pytest.mark.unit
def test_nan_on_both_sides_matches_on_one_side_does_not():
    nan = float("nan")
    both, _ = C.compare_frames(frame([nan, 0.1]), frame([nan, 0.1]), "g")
    one, _ = C.compare_frames(frame([nan, 0.1]), frame([0.3, 0.1]), "g")
    assert both["p_LRT"].diff == 0.0
    assert one["p_LRT"].diff == float("inf")


@pytest.mark.unit
def test_text_and_shape_mismatches_are_problems():
    _, text = C.compare_frames(frame([0.1, 0.2], "x"), frame([0.1, 0.2], "y"), "g")
    _, shape = C.compare_frames(frame([0.1, 0.2]), frame([0.1, 0.2]).iloc[:1], "g")
    assert text and text[0].startswith("TEXT")
    assert shape and shape[0].startswith("SHAPE")


@pytest.mark.unit
def test_main_end_to_end(tmp_path, capsys):
    for run, p in (("new", 0.1 + 1e-12), ("old", 0.1)):
        d = tmp_path / run / "Hypertension_F" / "per_gene"
        d.mkdir(parents=True)
        frame([p, 0.2]).to_csv(d / "ENSG1.tsv", sep="\t", index=False)
    assert C.main([str(tmp_path / "new"), str(tmp_path / "old")]) == 0
    assert "1 shared files, 0 with a difference" in capsys.readouterr().out
