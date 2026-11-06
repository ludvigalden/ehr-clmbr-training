"""Known-value tests for the vendored DeLong implementation.

Known values and their sources:

- Hanley and McNeil (1982) rating-scale example: 58 negative and 51
  positive subjects rated on a 5-point scale (positive counts 3, 2, 2,
  11, 33 and negative counts 33, 6, 6, 11, 2 per rating 1..5). The
  pairwise Mann-Whitney AUROC from the table is 2642 / 2958 =
  0.89317..., within 1e-3 of the published 0.893.
- DeLong, DeLong and Clarke-Pearson (1988) structural components
  (V10/V01 placements, S10/S01 sample variances): an independent
  O(n^2) re-implementation of the variance formula
  ``S01/m + S10/n`` (single model) and of the paired-difference
  variance ``(S01_a + S01_b - 2*C01)/m + (S10_a + S10_b - 2*C10)/n``
  on a seeded random dataset, compared to 1e-12.
- Tiny hand example: positives [0.9, 0.4], negatives [0.6, 0.1].
  Win matrix W = [[1, 1], [0, 1]], so xi01 = [1.0, 0.5] and
  xi10 = [0.5, 1.0]; the sample variances (ddof=1) are S01 = S10 =
  0.125 and the DeLong variance is S01/m + S10/n = 0.125/2 + 0.125/2
  = 0.125. AUROC = 3/4.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import norm

from ehrfm.eval._delong import (
    calc_pvalue,
    compute_ground_truth_statistics,
    delong_roc_test,
    fast_delong,
)
from ehrfm.eval.metrics import bonferroni, delong_auroc, delong_compare


def _hanley_mcneil_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the Hanley and McNeil (1982) rating-scale example arrays."""
    abnormal_counts = [3, 2, 2, 11, 33]  # label 1, ratings 1..5
    normal_counts = [33, 6, 6, 11, 2]  # label 0, ratings 1..5
    labels: list[int] = []
    scores: list[int] = []
    for rating in range(1, 6):
        labels.extend([1] * abnormal_counts[rating - 1])
        scores.extend([rating] * abnormal_counts[rating - 1])
        labels.extend([0] * normal_counts[rating - 1])
        scores.extend([rating] * normal_counts[rating - 1])
    return np.asarray(labels, dtype=np.float64), np.asarray(scores, dtype=np.float64)


def _mann_whitney_auc(y: NDArray[np.float64], s: NDArray[np.float64]) -> float:
    """O(n^2) pairwise Mann-Whitney AUROC with half credit for ties."""
    positive = s[y == 1]
    negative = s[y == 0]
    diff = positive[:, np.newaxis] - negative[np.newaxis, :]
    wins = (diff > 0).astype(np.float64) + 0.5 * (diff == 0)
    return float(wins.mean())


def _structural_delong_variance(y: NDArray[np.float64], s: NDArray[np.float64]) -> float:
    """O(n^2) structural DeLong variance (DeLong et al. 1988).

    Computes the placement statistics xi01 (per positive subject) and
    xi10 (per negative subject), their sample variances S01 and S10
    (ddof=1), and returns S01/m + S10/n.
    """
    positive = s[y == 1]
    negative = s[y == 0]
    m = positive.size
    n = negative.size
    diff = positive[:, np.newaxis] - negative[np.newaxis, :]
    wins = (diff > 0).astype(np.float64) + 0.5 * (diff == 0)
    xi01 = wins.mean(axis=1)
    xi10 = wins.mean(axis=0)
    s01 = float(np.var(xi01, ddof=1))
    s10 = float(np.var(xi10, ddof=1))
    return s01 / m + s10 / n


def _structural_delong_pair_variance(
    y: NDArray[np.float64],
    score_a: NDArray[np.float64],
    score_b: NDArray[np.float64],
) -> float:
    """O(n^2) structural variance of the paired AUROC difference.

    Computes xi01/xi10 for both models, their sample variances and
    cross-covariances (ddof=1), and returns
    ``(S01_a + S01_b - 2*C01)/m + (S10_a + S10_b - 2*C10)/n``.
    """
    m = int(np.count_nonzero(y == 1))
    n = int(np.count_nonzero(y == 0))

    stats: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for scores in (score_a, score_b):
        positive = scores[y == 1]
        negative = scores[y == 0]
        diff = positive[:, np.newaxis] - negative[np.newaxis, :]
        wins = (diff > 0).astype(np.float64) + 0.5 * (diff == 0)
        stats.append((wins.mean(axis=1), wins.mean(axis=0)))

    xi01_a, xi10_a = stats[0]
    xi01_b, xi10_b = stats[1]
    s01_a = float(np.var(xi01_a, ddof=1))
    s10_a = float(np.var(xi10_a, ddof=1))
    s01_b = float(np.var(xi01_b, ddof=1))
    s10_b = float(np.var(xi10_b, ddof=1))
    c01 = float(np.cov(np.vstack((xi01_a, xi01_b)), ddof=1)[0, 1])
    c10 = float(np.cov(np.vstack((xi10_a, xi10_b)), ddof=1)[0, 1])
    return (s01_a + s01_b - 2.0 * c01) / m + (s10_a + s10_b - 2.0 * c10) / n


def test_hanley_mcneil_rating_example() -> None:
    y, s = _hanley_mcneil_data()
    expected_auc = 2642.0 / 2958.0  # 0.89317... from the 58 x 51 table
    assert _mann_whitney_auc(y, s) == pytest.approx(expected_auc, abs=1e-12)
    result = delong_auroc(y, s)
    assert result.auroc == pytest.approx(expected_auc, abs=1e-3)
    assert abs(result.auroc - 0.893) < 1e-3


def test_structural_delong_matches_single_model_variance() -> None:
    rng = np.random.default_rng(7)
    n = 120
    y = (rng.random(n) < 0.35).astype(np.float64)
    noise = rng.normal(0.0, 1.0, n)
    s = np.clip(0.9 * y + noise, -3.0, 3.0)

    result = delong_auroc(y, s)
    structural = _structural_delong_variance(y, s)
    assert result.auroc == pytest.approx(_mann_whitney_auc(y, s), abs=1e-12)
    assert result.variance == pytest.approx(structural, abs=1e-12)
    assert result.standard_error == pytest.approx(np.sqrt(structural), abs=1e-12)


def test_structural_delong_matches_paired_comparison() -> None:
    rng = np.random.default_rng(13)
    n = 150
    y = (rng.random(n) < 0.4).astype(np.float64)
    signal_a = rng.normal(0.0, 1.0, n)
    signal_b = 0.8 * signal_a + rng.normal(0.0, 0.8, n)
    score_a = np.clip(signal_a + 0.6 * y, -3.0, 3.0)
    score_b = np.clip(signal_b + 0.6 * y, -3.0, 3.0)

    result = delong_compare(y, score_a, score_b)
    structural_variance = _structural_delong_pair_variance(y, score_a, score_b)
    expected_z = result.difference / np.sqrt(structural_variance)
    assert result.variance == pytest.approx(structural_variance, abs=1e-12)
    assert result.z == pytest.approx(expected_z, abs=1e-12)


def test_identical_predictors_give_z_zero_and_p_one() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(60) < 0.5).astype(np.float64)
    s = rng.random(60)
    result = delong_compare(y, s, s.copy())
    assert result.z == 0.0
    assert result.p_value == 1.0


def test_p_value_equals_vendored_calc_pvalue() -> None:
    rng = np.random.default_rng(21)
    n = 100
    y = (rng.random(n) < 0.4).astype(np.float64)
    score_a = np.clip(rng.random(n) + 0.5 * y, 0.0, 1.5)
    score_b = np.clip(rng.random(n) + 0.5 * y, 0.0, 1.5)

    order, label_1_count = compute_ground_truth_statistics(y)
    stacked = np.vstack((score_a, score_b))[:, order]
    aucs, sigma = fast_delong(stacked, label_1_count)
    log10_p, _ = delong_roc_test(y, score_a, score_b)
    assert log10_p == pytest.approx(calc_pvalue(aucs, sigma), abs=1e-12)

    result = delong_compare(y, score_a, score_b)
    assert result.p_value == pytest.approx(10.0 ** calc_pvalue(aucs, sigma), abs=1e-14)


def test_tiny_hand_example_variance() -> None:
    # Hand computation (see module docstring): AUROC = 3/4 and the
    # DeLong variance is S_01/m + S_10/n = 0.125.
    y = np.array([0.0, 0.0, 1.0, 1.0])
    s = np.array([0.6, 0.1, 0.9, 0.4])
    result = delong_auroc(y, s)
    alpha = 0.05
    z_critical = float(norm.ppf(1.0 - alpha / 2.0))
    se = np.sqrt(0.125)
    assert result.auroc == pytest.approx(0.75, abs=1e-12)
    assert result.variance == pytest.approx(0.125, abs=1e-12)
    assert result.standard_error == pytest.approx(se, abs=1e-12)
    assert result.low == pytest.approx(float(np.clip(0.75 - z_critical * se, 0.0, 1.0)), abs=1e-12)
    assert result.high == pytest.approx(float(np.clip(0.75 + z_critical * se, 0.0, 1.0)), abs=1e-12)


def test_delong_auroc_requires_both_classes() -> None:
    y = np.zeros(10)
    s = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="both classes"):
        delong_auroc(y, s)
    with pytest.raises(ValueError, match="both classes"):
        delong_compare(y, s, s)


def test_bonferroni_known_values() -> None:
    result = bonferroni(np.array([0.01, 0.02, 0.5]), alpha=0.05)
    assert result.adjusted.tolist() == [0.03, 0.06, 1.0]
    assert result.reject.tolist() == [True, False, False]
    assert result.alpha == 0.05
    with pytest.raises(ValueError, match="p_values must lie in"):
        bonferroni(np.array([-0.1, 0.5]))
