"""Unit tests for the paper-vs-HPP label comparison table."""
import pandas as pd
import pytest

import compare_label_runs as C


def table(rows):
    cols = ["gene_id", "trait", "sex", *C.CARRIED]
    return pd.DataFrame(rows, columns=cols)


@pytest.mark.unit
def test_compare_joins_unversioned_and_first_old_table_wins():
    new = table([["ENSG1.2", "T2D", "F", 100, 10, 1e-3, "x", False, False]])
    old_a = table([["ENSG1.1", "T2D", "F", 200, 50, 0.5, "y", False, False]])
    old_b = table([["ENSG1.1", "T2D", "F", 300, 60, 0.9, "z", False, False]])
    export = pd.DataFrame({"gene_id": ["ENSG1.3"], "trait": ["T2D"],
                           "sex": ["F"], "Bonferroni": [True]})
    out = C.compare(new, [old_a, old_b], export)
    row = out.iloc[0]
    assert bool(row["Bonferroni"]) is True
    assert row["ukbb_p_min_hpp"] == 1e-3
    assert row["ukbb_p_min_paper"] == 0.5
    assert row["n_cases_paper"] == 50


@pytest.mark.unit
def test_compare_without_old_tables_leaves_paper_columns_empty():
    new = table([["ENSG1.2", "T2D", "F", 100, 10, 1e-3, "x", False, False]])
    export = pd.DataFrame({"gene_id": ["ENSG1.2"], "trait": ["T2D"],
                           "sex": ["F"], "Bonferroni": [False]})
    out = C.compare(new, [], export)
    assert out["ukbb_p_min_paper"].isna().all()
