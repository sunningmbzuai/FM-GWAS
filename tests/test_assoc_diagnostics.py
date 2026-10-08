"""Unit tests for the association diagnostics.

Synthetic data throughout, so the calibration claims are checked against a
known truth rather than against the cluster's results. What is pinned: rank
sees through duplicated rows, the permutation p is calibrated when the null
holds, it agrees with the analytic p when there is real signal, and a
two-dimensional feature matrix does not by itself manufacture significance.
"""
import numpy as np
import pytest

import assoc_diagnostics as D


def two_sequence_features(n_samples: int, n_columns: int,
                          seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A thin gene: every participant carries one of two haplotype vectors.

    Returns the feature matrix and the genotype indicator behind it, which is
    what a real effect would act through.
    """
    rng = np.random.default_rng(seed)
    genotype = rng.integers(0, 2, size=n_samples)
    haplotypes = rng.normal(size=(2, n_columns))
    return haplotypes[genotype], genotype.astype(float)


@pytest.fixture
def covariates():
    rng = np.random.default_rng(7)
    n = 400
    return np.column_stack([rng.integers(0, 2, size=n).astype(float),
                            rng.normal(50, 8, size=n)])


@pytest.mark.unit
def test_rank_sees_through_duplicated_rows():
    features, _ = two_sequence_features(n_samples=300, n_columns=64)
    assert features.shape[1] == 64
    assert D.distinct_rows(features) == 2
    # Two distinct rows span one dimension once centred, two uncentred.
    assert D.design_rank(features) <= 2


@pytest.mark.unit
def test_pca_drops_components_the_matrix_cannot_supply():
    features, _ = two_sequence_features(n_samples=300, n_columns=64)
    assert D.pca_features(features, n_components=50).shape[1] == 1


@pytest.mark.unit
def test_pca_keeps_all_components_when_the_matrix_is_rich():
    rng = np.random.default_rng(1)
    rich = rng.normal(size=(300, 64))
    assert D.pca_features(rich, n_components=50).shape[1] == 50


@pytest.mark.unit
def test_permutation_p_is_calibrated_under_the_null(covariates):
    """No effect: the permutation p must be unremarkable, not tiny."""
    rng = np.random.default_rng(3)
    n = covariates.shape[0]
    features, _ = two_sequence_features(n_samples=n, n_columns=64, seed=11)
    y = covariates @ np.array([0.5, 0.02]) + rng.normal(size=n)

    result = D.permutation_p(y, covariates, D.pca_features(features),
                             n_permutations=200, seed=5)
    assert result["p_permutation"] > 0.05


@pytest.mark.unit
def test_thin_features_do_not_manufacture_significance(covariates):
    """The pattern in the run: significance concentrated at two-sequence genes.

    If a rank-one feature matrix were enough on its own, this null case would
    come back significant. It must not.
    """
    rng = np.random.default_rng(4)
    n = covariates.shape[0]
    features, _ = two_sequence_features(n_samples=n, n_columns=64, seed=12)
    y = rng.normal(size=n)

    result = D.permutation_p(y, covariates, D.pca_features(features),
                             n_permutations=200, seed=6)
    assert result["p_analytic"] > 0.01
    assert result["p_permutation"] > 0.01


@pytest.mark.unit
def test_real_effect_is_found_by_both_tests(covariates):
    """A genuine genotype effect: analytic and permutation must agree that it
    is there, which is what makes disagreement elsewhere informative."""
    rng = np.random.default_rng(5)
    n = covariates.shape[0]
    features, genotype = two_sequence_features(n_samples=n, n_columns=64,
                                               seed=13)
    y = 0.8 * genotype + rng.normal(size=n)

    result = D.permutation_p(y, covariates, D.pca_features(features),
                             n_permutations=200, seed=7)
    assert result["p_analytic"] < 1e-6
    assert result["p_permutation"] == pytest.approx(1 / 201, rel=1e-6)


@pytest.mark.unit
def test_df_numerator_defaults_to_column_count_and_can_be_overridden():
    """The inflated-df question, made explicit: charging 50 df for a rank-1
    design lowers F rather than raising it, so it cannot explain a small p."""
    rng = np.random.default_rng(8)
    n = 300
    features, genotype = two_sequence_features(n_samples=n, n_columns=64,
                                               seed=14)
    covariates = rng.normal(size=(n, 1))
    y = 0.5 * genotype + rng.normal(size=n)
    padded = np.hstack([D.pca_features(features), rng.normal(size=(n, 49))])

    honest, q_honest, _ = D.f_statistic(y, covariates, padded, df_numerator=1)
    charged, q_charged, _ = D.f_statistic(y, covariates, padded)
    assert (q_honest, q_charged) == (1, 50)
    assert charged < honest


@pytest.mark.unit
def test_pipeline_recipe_returns_columns_past_the_rank():
    """The defect itself: 50 columns out of a matrix that spans two."""
    features, _ = two_sequence_features(n_samples=400, n_columns=64, seed=15)
    produced = D.pipeline_features(features, n_components=50, seed=0)
    assert produced.shape[1] == 50
    assert D.design_rank(features - features.mean(axis=0)) <= 2


@pytest.mark.unit
def test_pipeline_p_depends_on_row_order_when_rank_is_short():
    """Row order must not matter to a valid test. At a thin gene whose label
    tracks row position -- a sorted cohort file is enough -- the pipeline's p
    moves when only the order of participants changes."""
    rng = np.random.default_rng(9)
    n = 600
    features, _ = two_sequence_features(n_samples=n, n_columns=64, seed=16)
    covariates = rng.normal(size=(n, 1))
    # No genotype effect at all; the label merely trends with position.
    y = np.linspace(-1, 1, n) + rng.normal(scale=0.5, size=n)

    result = D.row_order_sensitivity(features, covariates, y,
                                     n_components=50, n_orders=3, seed=0)
    spread = (np.log10(max(result["p_shuffled_orders"]))
              - np.log10(result["p_original_order"]))
    assert spread > 2


@pytest.mark.unit
def test_honest_test_does_not_depend_on_row_order(covariates):
    rng = np.random.default_rng(10)
    n = covariates.shape[0]
    features, genotype = two_sequence_features(n_samples=n, n_columns=64,
                                               seed=17)
    y = 0.3 * genotype + rng.normal(size=n)
    order = rng.permutation(n)

    before, *_ = D.f_statistic(y, covariates, D.pca_features(features))
    after, *_ = D.f_statistic(y[order], covariates[order],
                              D.pca_features(features[order]))
    assert before == pytest.approx(after, rel=1e-8)
