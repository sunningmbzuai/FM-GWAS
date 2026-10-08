"""fix_assoc_pca95 replays FM-GWAS d17eae23a6 on the July insample_hpp_AG.py."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fix_assoc_pca95 as P  # noqa: E402

# The July lines each hunk replaces, in file order, with stand-in context
# between them so the patcher has to find each one rather than a single block.
JULY_SOURCE = "".join(old + "        # ...\n" for old, _ in P.REPLACEMENTS)


def test_every_hunk_applies_and_old_lines_are_gone():
    patched, applied = P.apply_replacements(JULY_SOURCE, P.REPLACEMENTS)
    assert applied == len(P.REPLACEMENTS)
    assert "PCA(n_components=0.99" not in patched
    assert "Z_pca" not in patched
    assert "[5] if 'add' in embed_type else [50]" not in patched
    assert "pca_component_list = [0.95]" in patched
    assert "svd_solver='auto'" in patched
    assert "LinearRegression_Ftest(X, Z, Y)" in patched


def test_second_run_is_a_no_op():
    once, _ = P.apply_replacements(JULY_SOURCE, P.REPLACEMENTS)
    twice, applied = P.apply_replacements(once, P.REPLACEMENTS)
    assert applied == 0
    assert twice == once


def test_unrecognised_source_fails_loudly():
    with pytest.raises(SystemExit, match="expected code not found"):
        P.apply_replacements("print('some other script')\n", P.REPLACEMENTS)


def test_patch_file_keeps_pristine_backup(tmp_path):
    target = tmp_path / "insample_hpp_AG.py"
    target.write_text(JULY_SOURCE)
    P.patch_file(target, P.REPLACEMENTS)
    assert (tmp_path / "insample_hpp_AG.py.orig").read_text() == JULY_SOURCE
    assert "pca_component_list = [0.95]" in target.read_text()
