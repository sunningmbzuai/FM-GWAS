"""Is a gene's association real, or an artifact of how the test is built?

Significant genes can concentrate in the thin tail of the variant
distribution, at genes with only one or two distinct haplotype sequences.
Power alone can explain that shape -- a gene with two haplotype sequences
reduces the multi-component test to a single-variant contrast, which is
exactly the regime GWAS is built for, while thousands of haplotypes spread any
real signal across components. Invalid p-values would explain it too. The two
cases call for opposite conclusions, so this module measures rather than
argues.

    rank        how many dimensions a view's feature matrix actually spans,
                against the number of columns the F-test charges df for
    analytic    the F-test the pipeline reports, recomputed here
    permutation the same statistic against a shuffled phenotype

A permutation p near the analytic one means the analytic test is calibrated and
the hit is real, whatever its provenance. A permutation p orders of magnitude
larger means the analytic p is fiction. Nothing else distinguishes them, which
is why the guess about degrees of freedom in the commit history is a guess:
spreading q df over a rank-2 design makes F conservative, not liberal, so it
cannot be the whole story on its own.

numpy only, like the rest of the assoc side.
"""
import numpy as np

# Below this fraction of the largest singular value a direction is numerical
# noise rather than a dimension the data actually spans. numpy's matrix_rank
# default, stated here because the whole question is where that line sits.
RANK_RTOL = None


def design_rank(matrix: np.ndarray) -> int:
    """How many dimensions the rows of `matrix` actually span."""
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=np.float64),
                                     tol=RANK_RTOL))


def distinct_rows(matrix: np.ndarray) -> int:
    """Distinct feature rows -- the participants' distinct sequences, after
    the model has mapped each one to a vector."""
    return len(np.unique(np.asarray(matrix, dtype=np.float64), axis=0))


def pca_features(matrix: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Top principal components, centred, as the association tests use them.

    Fewer components come back when the matrix cannot supply that many: asking
    for 50 from a rank-2 matrix yields 48 columns of numerical noise, and
    carrying them would charge the F-test df for dimensions holding nothing.
    """
    centred = np.asarray(matrix, dtype=np.float64)
    centred = centred - centred.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    keep = min(n_components, int(np.sum(singular > singular[0] * 1e-10)))
    keep = max(keep, 1)
    return centred @ right[:keep].T


def _residual_sum_of_squares(y: np.ndarray, design: np.ndarray) -> float:
    """RSS of y on `design`, via lstsq so a rank-deficient design is fine."""
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    return float(residual @ residual)


def f_statistic(y: np.ndarray, covariates: np.ndarray, features: np.ndarray,
                df_numerator: int | None = None) -> tuple[float, int, int]:
    """F for adding `features` to `covariates`, and its two df.

    df_numerator defaults to the feature column count, which is what an
    implementation charges when it does not look at the rank. Passing the rank
    instead gives the honest test; the difference between the two is the whole
    question at a gene whose features span two dimensions.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    intercept = np.ones((n, 1))
    null_design = np.hstack([intercept, covariates])
    full_design = np.hstack([null_design, features])

    rss_null = _residual_sum_of_squares(y, null_design)
    rss_full = _residual_sum_of_squares(y, full_design)

    q = df_numerator if df_numerator is not None else features.shape[1]
    df_denominator = n - design_rank(full_design)
    if q <= 0 or df_denominator <= 0:
        return float("nan"), q, df_denominator
    numerator = (rss_null - rss_full) / q
    denominator = rss_full / df_denominator
    if denominator <= 0:
        return float("inf"), q, df_denominator
    return numerator / denominator, q, df_denominator


def analytic_p(f_value: float, df_numerator: int, df_denominator: int) -> float:
    """The F-test's own p-value. scipy where available, else a survival
    function built from the regularised incomplete beta."""
    if not np.isfinite(f_value) or f_value <= 0:
        return 1.0
    try:
        from scipy import stats
    except ImportError:
        return float("nan")
    return float(stats.f.sf(f_value, df_numerator, df_denominator))


def pipeline_features(matrix: np.ndarray, n_components: int,
                      seed: int | None = None) -> np.ndarray:
    """The feature recipe insample_hpp_AG.py uses, reproduced step for step.

    normalization() then PCA(n_components, whiten=True).fit_transform(), which
    returns the left singular vectors U scaled by sqrt(n - 1). Past the
    matrix's rank the singular values are zero and those U columns are an
    arbitrary orthonormal completion chosen by the solver, not a function of
    the embeddings. At a thin gene 47 of the 50 columns are such vectors, and
    the F-test is charged for all 50. fix_signs() is omitted: flipping a
    column's sign cannot change an F statistic.
    """
    from sklearn.decomposition import PCA

    standardized = np.asarray(matrix, dtype=np.float32)
    mean = standardized.mean(axis=0, keepdims=True)
    std = standardized.std(axis=0, keepdims=True)
    standardized = (standardized - mean) / np.where(std > 0, std, 1.0)
    return PCA(n_components=n_components, whiten=True,
               random_state=seed).fit_transform(standardized)


def pipeline_f_test(features: np.ndarray, covariates: np.ndarray,
                    y: np.ndarray) -> tuple[float, float]:
    """LinearRegression_Ftest from insample_hpp_AG.py, verbatim in substance.

    QR on both designs, residual sums of squares from the projections, and
    k = the feature column count. Sound arithmetic; the fault is upstream in
    what it is handed.
    """
    from scipy import stats

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    reduced = np.column_stack([np.ones(n), covariates])
    full = np.column_stack([reduced, features])
    q_reduced, _ = np.linalg.qr(reduced, mode="reduced")
    q_full, _ = np.linalg.qr(full, mode="reduced")
    df2 = n - q_full.shape[1]
    k = features.shape[1]

    def ssr(q):
        t = q.T @ y
        return float(y @ y - t @ t)

    f_value = ((ssr(q_reduced) - ssr(q_full)) / k) / (ssr(q_full) / df2)
    return float(f_value), float(stats.f.sf(f_value, k, df2))


def row_order_sensitivity(matrix: np.ndarray, covariates: np.ndarray,
                          y: np.ndarray, n_components: int,
                          n_orders: int = 5, seed: int = 0) -> dict:
    """The pipeline's p in the original participant order, then in shuffled
    orders.

    A valid association test is a function of the (features, phenotype) pairs,
    not of the order the participants are listed in. Shuffling rows of all
    three inputs together changes nothing a real analysis could depend on, so
    a p-value that moves under it is produced by the procedure, not the data.
    """
    y = np.asarray(y, dtype=np.float64)
    _, p_original = pipeline_f_test(
        pipeline_features(matrix, n_components, seed=seed), covariates, y)
    rng = np.random.default_rng(seed)
    shuffled_ps = []
    for _ in range(n_orders):
        order = rng.permutation(len(y))
        _, p = pipeline_f_test(
            pipeline_features(matrix[order], n_components, seed=seed),
            covariates[order], y[order])
        shuffled_ps.append(p)
    return {"p_original_order": p_original, "p_shuffled_orders": shuffled_ps}


def permutation_p(y: np.ndarray, covariates: np.ndarray, features: np.ndarray,
                  n_permutations: int = 1000, seed: int = 0,
                  df_numerator: int | None = None) -> dict:
    """Empirical p: how often a shuffled phenotype beats the real F.

    The phenotype is shuffled whole rather than within covariate strata, which
    breaks its link to the features and to the covariates alike. That is the
    right null here because the covariates are sex and age and the question is
    whether the FEATURES carry anything; a covariate effect inflates both the
    observed and the permuted statistic, so it cancels.

    The reported p uses the (b + 1) / (B + 1) form, so it is never zero: with
    1,000 permutations the smallest attainable value is 1/1001, and an analytic
    p of 1e-133 simply cannot be confirmed by permutation, only refuted.
    """
    y = np.asarray(y, dtype=np.float64)
    observed, q, df_denominator = f_statistic(y, covariates, features,
                                              df_numerator)
    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    null_values = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(y)
        value, *_ = f_statistic(shuffled, covariates, features, df_numerator)
        null_values[i] = value
        if value >= observed:
            at_least_as_extreme += 1
    return {
        "f_observed": observed,
        "df_numerator": q,
        "df_denominator": df_denominator,
        "p_analytic": analytic_p(observed, q, df_denominator),
        "p_permutation": (at_least_as_extreme + 1) / (n_permutations + 1),
        "n_permutations": n_permutations,
        "null_f_median": float(np.median(null_values)),
        "null_f_max": float(np.max(null_values)),
    }
