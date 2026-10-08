import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

import assoc_diagnostics as D
from rank_capped_pca import RankCappedPCA, capped_components, centred_rank

REPO = Path(__file__).resolve().parents[1]


def thin_gene(n=2000, cols=300, seed=0):
    """Every participant carries one of 4 distinct embedding rows, as at a
    one-variant gene: rank 3 once centred."""
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(4, cols)).astype(np.float32)
    return rows[rng.integers(0, 4, size=n)]


def test_centred_rank_of_thin_gene():
    assert centred_rank(thin_gene()) == 3


def test_rich_matrix_keeps_full_request():
    X = np.random.default_rng(1).normal(size=(500, 80))
    assert capped_components(50, X) == 50


def test_non_integer_requests_untouched():
    X = thin_gene()
    assert capped_components(None, X) is None
    assert capped_components(0.95, X) == 0.95


def test_fit_transform_returns_rank_columns():
    out = RankCappedPCA(n_components=50, whiten=True).fit_transform(thin_gene())
    assert out.shape[1] == 3


def test_refit_on_richer_data_uses_original_request():
    pca = RankCappedPCA(n_components=50)
    pca.fit(thin_gene())
    rich = np.random.default_rng(2).normal(size=(500, 80))
    assert pca.fit_transform(rich).shape[1] == 50


def test_pipeline_f_test_no_longer_depends_on_row_order():
    X = thin_gene()
    rng = np.random.default_rng(3)
    y = rng.normal(size=len(X))
    cov = rng.normal(size=(len(X), 2))
    ps = []
    for seed in range(4):
        order = np.random.default_rng(seed).permutation(len(X))
        feats = RankCappedPCA(n_components=50, whiten=True).fit_transform(X[order])
        ps.append(D.pipeline_f_test(feats, cov[order], y[order])[1])
    # float32 PCA noise only; the bug moved p by tens of orders of magnitude
    assert np.allclose(ps, ps[0], rtol=1e-4)


def test_wrapper_patches_upstream_import(tmp_path):
    script = tmp_path / "upstream.py"
    script.write_text(
        "import sys\n"
        "import numpy as np\n"
        "from sklearn.decomposition import PCA\n"
        "rng = np.random.default_rng(0)\n"
        "X = rng.normal(size=(4, 30))[rng.integers(0, 4, size=400)]\n"
        "print(PCA(n_components=50).fit_transform(X).shape[1], sys.argv[1])\n"
    )
    out = subprocess.run(
        [sys.executable, str(REPO / "assoc_rank_capped.py"), str(script), "arg1"],
        capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["3", "arg1"]
