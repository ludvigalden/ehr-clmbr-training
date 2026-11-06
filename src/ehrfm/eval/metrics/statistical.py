"""DeLong AUROC inference and family-wise p-value correction.

Single-model: the DeLong variance of an AUROC estimator with a normal
confidence interval clipped to ``[0, 1]``. Paired: the DeLong test for
two correlated AUROCs evaluated on the same subjects — the difference
``auroc_a - auroc_b``, its variance ``c' Sigma c`` with contrast
``c = (1, -1)`` from the vendored :mod:`ehrfm.eval._delong`
computation, the z-statistic, and the two-sided normal p-value.
Identical predictors give z = 0 and p = 1.

Bonferroni correction multiplies each raw p-value by the family size
and clamps at 1.0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ehrfm.eval._delong import (
    compute_ground_truth_statistics,
    delong_roc_variance,
    fast_delong,
)
from ehrfm.eval.metrics._validation import FloatArray, LabelArray, binary_labels, scores

__all__ = [
    "BonferroniResult",
    "DeLongAUROC",
    "DeLongComparison",
    "bonferroni",
    "delong_auroc",
    "delong_compare",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class DeLongAUROC:
    """Single-model AUROC with its DeLong variance and normal interval.

    Attributes:
        auroc: AUROC point estimate from the DeLong computation.
        variance: DeLong variance of the AUROC estimator.
        standard_error: Square root of ``variance``.
        low: Lower confidence bound ``auroc - z * se``, clipped to 1.
        high: Upper confidence bound ``auroc + z * se``, clipped to 1.
        alpha: Significance level used for the interval.
    """

    auroc: float
    variance: float
    standard_error: float
    low: float
    high: float
    alpha: float


@dataclass(frozen=True, kw_only=True, slots=True)
class DeLongComparison:
    """Paired DeLong comparison of two AUROCs on a shared cohort.

    Attributes:
        auroc_a: AUROC of model A from the DeLong computation.
        auroc_b: AUROC of model B from the DeLong computation.
        difference: ``auroc_a - auroc_b`` (signed).
        variance: Variance of the difference, ``c' Sigma c`` with
            ``c = (1, -1)``.
        z: Standardized statistic ``difference / sqrt(variance)``; 0.0
            in the limiting case of identical predictors or numerically
            zero variance.
        p_value: Two-sided p-value ``2 * norm.sf(|z|)``; 1.0 in the
            limiting case above.
        low: Lower confidence bound on the difference (unclipped).
        high: Upper confidence bound on the difference (unclipped).
        alpha: Significance level used for the interval.
    """

    auroc_a: float
    auroc_b: float
    difference: float
    variance: float
    z: float
    p_value: float
    low: float
    high: float
    alpha: float


@dataclass(frozen=True, kw_only=True, slots=True)
class BonferroniResult:
    """Bonferroni-corrected p-values and rejection flags.

    Attributes:
        adjusted: ``min(p * k, 1.0)`` element-wise for a family of ``k``
            p-values.
        reject: Boolean array, True where ``adjusted < alpha``.
        alpha: Significance level used for ``reject``.
    """

    adjusted: NDArray[np.float64]
    reject: NDArray[np.bool_]
    alpha: float


def _validate_alpha(alpha: float) -> None:
    """Validate a confidence-level parameter.

    Raises:
        ValueError: If ``alpha`` lies outside ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must lie strictly inside (0, 1); got {alpha}"
        raise ValueError(msg)


def _require_both_classes(y: NDArray[np.float64]) -> None:
    """Require both label classes for DeLong inference.

    Raises:
        ValueError: If ``y`` is single-class.
    """
    if np.unique(y).size < 2:
        msg = "DeLong inference requires both classes present in y_true"
        raise ValueError(msg)


def delong_auroc(
    y_true: LabelArray,
    y_score: FloatArray,
    *,
    alpha: float = 0.05,
) -> DeLongAUROC:
    """AUROC with its DeLong variance and clipped normal interval.

    Args:
        y_true: Binary labels (0 or 1) of shape ``(n,)`` containing
            both classes.
        y_score: Predicted scores of shape ``(n,)``.
        alpha: Significance level. Default 0.05 (a 95% interval).

    Returns:
        :class:`DeLongAUROC` with the estimate, variance, standard
        error, and the normal interval clipped to ``[0, 1]``.

    Raises:
        ValueError: If the inputs fail label/score validation,
            ``y_true`` is single-class, or ``alpha`` lies outside
            ``(0, 1)``.
    """
    _validate_alpha(alpha)
    y = binary_labels(y_true)
    s = scores(y_score, y.size)
    _require_both_classes(y)

    auc_value, covariance = delong_roc_variance(y, s)
    # np.cov collapses to a scalar for the single-model input (one
    # row), so squeeze rather than index the 1x1 shape.
    variance = float(np.asarray(covariance).item())
    standard_error = float(np.sqrt(variance))
    z_critical = float(norm.ppf(1.0 - alpha / 2.0))
    low = float(np.clip(auc_value - z_critical * standard_error, 0.0, 1.0))
    high = float(np.clip(auc_value + z_critical * standard_error, 0.0, 1.0))
    return DeLongAUROC(
        auroc=auc_value,
        variance=variance,
        standard_error=standard_error,
        low=low,
        high=high,
        alpha=alpha,
    )


def delong_compare(
    y_true: LabelArray,
    score_a: FloatArray,
    score_b: FloatArray,
    *,
    alpha: float = 0.05,
) -> DeLongComparison:
    """Paired DeLong test for two AUROCs on the same subjects.

    The score arrays must be aligned subject-by-subject: the test
    exploits the within-subject correlation between the two AUROC
    estimators.

    Args:
        y_true: Binary labels (0 or 1) of shape ``(n,)`` containing
            both classes.
        score_a: Scores of model A of shape ``(n,)``.
        score_b: Scores of model B of shape ``(n,)``.
        alpha: Significance level. Default 0.05 (a 95% interval).

    Returns:
        :class:`DeLongComparison` with both AUROCs, the signed
        difference, its variance, z-statistic, two-sided p-value, and
        the unclipped confidence interval on the difference.

    Raises:
        ValueError: If the inputs fail label/score validation,
            ``y_true`` is single-class, or ``alpha`` lies outside
            ``(0, 1)``.
    """
    _validate_alpha(alpha)
    y = binary_labels(y_true)
    a = scores(score_a, y.size)
    b = scores(score_b, y.size)
    _require_both_classes(y)

    order, label_1_count = compute_ground_truth_statistics(y)
    predictions_sorted_transposed = np.vstack((a, b))[:, order]
    aucs, covariance = fast_delong(predictions_sorted_transposed, label_1_count)
    auroc_a = float(aucs[0])
    auroc_b = float(aucs[1])
    difference = auroc_a - auroc_b

    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ covariance @ contrast.T)[0, 0])

    if variance <= 0.0 or difference == 0.0:
        # Limiting case: identical predictors or numerically zero
        # variance; the statistic is zero and the p-value is 1.
        z = 0.0
        p_value = 1.0
        low = difference
        high = difference
    else:
        z = difference / np.sqrt(variance)
        p_value = float(2.0 * norm.sf(abs(z)))
        z_critical = float(norm.ppf(1.0 - alpha / 2.0))
        half_width = z_critical * float(np.sqrt(variance))
        low = difference - half_width
        high = difference + half_width

    return DeLongComparison(
        auroc_a=auroc_a,
        auroc_b=auroc_b,
        difference=difference,
        variance=variance,
        z=float(z),
        p_value=p_value,
        low=low,
        high=high,
        alpha=alpha,
    )


def bonferroni(
    p_values: Sequence[float] | FloatArray,
    *,
    alpha: float = 0.05,
) -> BonferroniResult:
    """Bonferroni correction for a family of p-values.

    Multiplies each raw p-value by the family size (the number of
    input p-values) and clamps at 1.0.

    Args:
        p_values: Raw p-values in ``[0, 1]``.
        alpha: Significance level for the rejection flags. Default
            0.05.

    Returns:
        :class:`BonferroniResult` with the adjusted p-values and
        rejection flags.

    Raises:
        ValueError: If ``p_values`` is empty or not 1-D, contains
            non-finite values or values outside ``[0, 1]``, or
            ``alpha`` lies outside ``(0, 1)``.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        msg = "p_values must be a non-empty 1-D array"
        raise ValueError(msg)
    if not np.all(np.isfinite(p)):
        msg = "p_values must be finite"
        raise ValueError(msg)
    if bool(np.any(p < 0.0)) or bool(np.any(p > 1.0)):
        msg = f"p_values must lie in [0, 1]; got min={float(p.min())}, max={float(p.max())}"
        raise ValueError(msg)
    _validate_alpha(alpha)

    adjusted: NDArray[np.float64] = np.minimum(p * float(p.size), 1.0)
    reject: NDArray[np.bool_] = adjusted < alpha
    return BonferroniResult(adjusted=adjusted, reject=reject, alpha=alpha)
