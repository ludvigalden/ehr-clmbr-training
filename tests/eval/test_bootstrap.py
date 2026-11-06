"""Known-value tests for bootstrap confidence intervals.

Hand-checkable case: with a constant ``y_score`` and the mean as the
metric, every resample's metric value equals the constant, so the
percentile interval collapses to ``low == high == estimate``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ehrfm.eval._bootstrap import bootstrap_ci, paired_bootstrap_ci, resample_indices
from ehrfm.eval.metrics import auroc


def _mean_metric(y_true: NDArray[np.float64], y_score: NDArray[np.float64]) -> float:
    return float(np.mean(y_score))


def _sample_data(n: int = 200, seed: int = 5) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(np.float64)
    s = np.clip(rng.random(n) + 0.5 * y, 0.0, 1.5)
    return y, s


def test_same_seed_reproduces_distribution_exactly() -> None:
    y, s = _sample_data()
    first = bootstrap_ci(auroc, y, s, resamples=200, seed=123)
    second = bootstrap_ci(auroc, y, s, resamples=200, seed=123)
    assert np.array_equal(first.distribution, second.distribution)
    assert first.low == second.low
    assert first.high == second.high


def test_different_seed_changes_distribution() -> None:
    y, s = _sample_data()
    first = bootstrap_ci(auroc, y, s, resamples=200, seed=1)
    second = bootstrap_ci(auroc, y, s, resamples=200, seed=2)
    assert not np.array_equal(first.distribution, second.distribution)


def test_stratified_resamples_preserve_class_counts_exactly() -> None:
    rng = np.random.default_rng(9)
    y = np.array([0.0] * 12 + [1.0] * 8)
    indices = resample_indices(y, 50, rng, stratified=True)
    assert indices.shape == (50, 20)
    for row in indices:
        resampled_labels = y[row]
        assert int(np.count_nonzero(resampled_labels == 1)) == 8
        assert int(np.count_nonzero(resampled_labels == 0)) == 12


def test_interval_brackets_estimate_on_well_behaved_case() -> None:
    y, s = _sample_data()
    result = bootstrap_ci(auroc, y, s, resamples=500, seed=17)
    assert result.estimate == pytest.approx(auroc(y, s), abs=1e-12)
    assert result.low < result.estimate < result.high
    assert result.finite_resamples == result.resamples
    assert result.alpha == 0.05


def test_constant_scores_collapse_interval_to_estimate() -> None:
    y, _ = _sample_data()
    constant = np.full(y.size, 0.4)
    result = bootstrap_ci(_mean_metric, y, constant, resamples=300, seed=7)
    assert result.estimate == 0.4
    assert result.low == 0.4
    assert result.high == 0.4
    assert result.finite_resamples == result.resamples


def test_resamples_and_alpha_are_validated() -> None:
    y, s = _sample_data()
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_ci(auroc, y, s, resamples=0, seed=1)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(auroc, y, s, resamples=10, alpha=0.0, seed=1)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(auroc, y, s, resamples=10, alpha=1.0, seed=1)


def test_stratified_requires_both_classes() -> None:
    y = np.zeros(10)
    s = np.full(10, 0.5)
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_ci(auroc, y, s, resamples=10, seed=1)


def test_paired_bootstrap_of_a_model_with_itself_is_zero() -> None:
    y, s = _sample_data()
    result = paired_bootstrap_ci(auroc, y, s, s.copy(), resamples=300, seed=31)
    assert result.estimate == pytest.approx(0.0, abs=1e-12)
    assert result.low == pytest.approx(0.0, abs=1e-12)
    assert result.high == pytest.approx(0.0, abs=1e-12)
