"""PCA that never returns more components than the data spans.

insample_hpp_AG.py asks PCA(n_components=50, whiten=True) for every gene. At a
gene with few variants the embedding matrix spans 2 or 3 dimensions, so 47 of
the 50 returned columns are an arbitrary orthonormal completion chosen by the
solver, whitened to unit variance, and the F-test charges k = 50 for them.
Those columns depend on solver internals (row order included), not on the
data, so the p-value does too.

RankCappedPCA caps n_components at the matrix rank before fitting, so every
column it returns is a function of the embeddings and LinearRegression_Ftest's
k = X.shape[1] is the rank. That is the rank-trimmed test diagnose_assoc.py
validated against permutation. Genes whose rank reaches the requested count
are fitted exactly as before.

install() swaps it in for sklearn's PCA; assoc_rank_capped.py does that before
running the upstream script, so FM-GWAS itself stays unedited.
"""
import numpy as np
from sklearn import decomposition
from sklearn.decomposition import _pca

# Same rule as assoc_diagnostics.pca_features: a direction is real when its
# singular value is above this fraction of the largest one.
RANK_RELATIVE_TOL = 1e-10


def centred_rank(matrix: np.ndarray) -> int:
    """Rank of the column-centred matrix, the dimensions PCA can recover.

    Computed on the distinct rows only: a thin gene has 43k rows but a handful
    of distinct ones, and the centred rows span the same space either way.
    """
    matrix = np.asarray(matrix)
    distinct = np.unique(matrix, axis=0).astype(np.float64)
    centred = distinct - matrix.mean(axis=0, dtype=np.float64)
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular.size == 0 or singular[0] == 0:
        return 0
    return int(np.sum(singular > singular[0] * RANK_RELATIVE_TOL))


def capped_components(requested, matrix: np.ndarray):
    """min(requested, rank) for an integer request; other requests untouched."""
    if isinstance(requested, (bool, np.bool_)) or not isinstance(
            requested, (int, np.integer)):
        return requested
    return max(1, min(int(requested), centred_rank(matrix)))


class RankCappedPCA(decomposition.PCA):
    """sklearn PCA with n_components capped at the centred matrix rank.

    Sets n_components on the instance because sklearn reads it from there; the
    original request is kept so a refit on richer data is not stuck at an
    earlier, lower cap.
    """

    def _apply_cap(self, X) -> None:
        requested = self.__dict__.setdefault("requested_components_",
                                             self.n_components)
        self.n_components = capped_components(requested, X)

    def fit(self, X, y=None):
        self._apply_cap(X)
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        self._apply_cap(X)
        return super().fit_transform(X, y)


def install() -> None:
    """Make `from sklearn.decomposition import PCA` return RankCappedPCA."""
    decomposition.PCA = RankCappedPCA
    _pca.PCA = RankCappedPCA
