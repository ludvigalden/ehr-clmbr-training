"""Calibration metrics: reliability bins, ECE, Brier score and logistic fit.

Expected Calibration Error (Naeini et al. 2015, Guo et al. 2017)
partitions ``[0, 1]`` into equal-width bins, computes the absolute gap
between mean predicted probability and observed positive rate within
each occupied bin, and weights each gap by the bin's sample fraction.
Empty bins contribute zero weight.

The calibration fit summarizes two standard logistic diagnostics
(Cox 1958; Miller et al. 1993; Van Calster et al. 2016):

- ``slope``: the coefficient of ``logit(p)`` in an unpenalised logistic
  regression of the outcome on ``a + b * logit(p)``; ``b = 1`` with
  ``a = 0`` is perfectly calibrated.
- ``intercept``: calibration-in-the-large, the MLE of ``a`` in ``logit
  P(y = 1) = a + logit(p)`` with the slope fixed at 1 (an offset-only
  fit); ``a = 0`` means the average prediction equals the observed rate.

Both fits run Newton-Raphson (at most 100 iterations, convergence
tolerance 1e-10 on the update step) and clip probabilities to
``[eps, 1 - eps]`` before the logit transform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from ehrfm.eval.metrics._validation import FloatArray, LabelArray, binary_labels, probabilities

__all__ = [
    "CalibrationBin",
    "CalibrationFit",
    "brier_score",
    "calibration_bins",
    "calibration_fit",
    "expected_calibration_error",
]

_MAX_ITERATIONS = 100
_TOLERANCE = 1e-10


@dataclass(frozen=True, kw_only=True, slots=True)
class CalibrationBin:
    """One equal-width reliability bin over ``[0, 1]``.

    Attributes:
        lower: Lower bin edge.
        upper: Upper bin edge; the last bin is closed on the right so
            predictions of exactly ``1.0`` land inside it.
        n: Number of samples in the bin. Empty bins carry ``n = 0``.
        mean_predicted: Mean predicted probability within the bin;
            ``None`` for an empty bin.
        observed_rate: Observed positive rate within the bin; ``None``
            for an empty bin.
    """

    lower: float
    upper: float
    n: int
    mean_predicted: float | None
    observed_rate: float | None


@dataclass(frozen=True, kw_only=True, slots=True)
class CalibrationFit:
    """Logistic calibration fit for predicted probabilities.

    Attributes:
        slope: Coefficient of ``logit(p)`` in the unpenalised logistic
            regression ``y ~ a + b * logit(p)``. ``nan`` when that fit
            did not converge (for example when every prediction is
            identical and the slope is unidentifiable).
        intercept: Calibration-in-the-large, the MLE of ``a`` in
            ``logit P(y = 1) = a + logit(p)`` with the slope fixed at 1.
            ``nan`` when that fit did not converge.
        converged: True only when both fits converged within the
            iteration and tolerance budget.
    """

    slope: float
    intercept: float
    converged: bool


def _validate_bins(bins: int) -> None:
    """Validate the bin count.

    Raises:
        ValueError: If ``bins < 1``.
    """
    if bins < 1:
        msg = f"bins must be >= 1; got {bins}"
        raise ValueError(msg)


def calibration_bins(
    y_true: LabelArray,
    y_prob: FloatArray,
    *,
    bins: int = 15,
) -> list[CalibrationBin]:
    """Partition predictions into equal-width bins over ``[0, 1]``.

    Args:
        y_true: Binary labels (0 or 1) of shape ``(n,)``.
        y_prob: Predicted probabilities in ``[0, 1]`` of shape ``(n,)``.
        bins: Number of equal-width bins. Default 15.

    Returns:
        One :class:`CalibrationBin` per bin in edge order; the last bin
        includes ``1.0``. Empty bins carry ``n = 0`` and ``None`` means.

    Raises:
        ValueError: If the inputs fail label/probability validation or
            ``bins < 1``.
    """
    y = binary_labels(y_true)
    p = probabilities(y_prob, y.size)
    _validate_bins(bins)

    edges = np.linspace(0.0, 1.0, num=bins + 1)
    interior = edges[1:-1]
    indices: NDArray[np.intp] = np.clip(np.digitize(p, interior, right=False), 0, bins - 1)

    result: list[CalibrationBin] = []
    for b in range(bins):
        mask = indices == b
        n_in_bin = int(np.count_nonzero(mask))
        if n_in_bin > 0:
            result.append(
                CalibrationBin(
                    lower=float(edges[b]),
                    upper=float(edges[b + 1]),
                    n=n_in_bin,
                    mean_predicted=float(p[mask].mean()),
                    observed_rate=float(y[mask].mean()),
                )
            )
        else:
            result.append(
                CalibrationBin(
                    lower=float(edges[b]),
                    upper=float(edges[b + 1]),
                    n=0,
                    mean_predicted=None,
                    observed_rate=None,
                )
            )
    return result


def expected_calibration_error(
    y_true: LabelArray,
    y_prob: FloatArray,
    *,
    bins: int = 15,
) -> float:
    """Expected Calibration Error with equal-width binning.

    Computes ``sum_b (n_b / N) * |observed_b - mean_predicted_b|`` over
    the occupied bins ``b``; empty bins contribute zero weight.

    Args:
        y_true: Binary labels (0 or 1) of shape ``(n,)``.
        y_prob: Predicted probabilities in ``[0, 1]`` of shape ``(n,)``.
        bins: Number of equal-width bins. Default 15.

    Returns:
        ECE in ``[0, 1]``; lower is better.

    Raises:
        ValueError: If the inputs fail label/probability validation or
            ``bins < 1``.
    """
    rows = calibration_bins(y_true, y_prob, bins=bins)
    total = sum(row.n for row in rows)
    ece = 0.0
    for row in rows:
        if row.n == 0 or row.mean_predicted is None or row.observed_rate is None:
            continue
        ece += (row.n / total) * abs(row.observed_rate - row.mean_predicted)
    return ece


def brier_score(
    y_true: LabelArray,
    y_prob: FloatArray,
) -> float:
    """Brier score: the mean squared error of predicted probabilities.

    Args:
        y_true: Binary labels (0 or 1) of shape ``(n,)``.
        y_prob: Predicted probabilities in ``[0, 1]`` of shape ``(n,)``.

    Returns:
        Brier score in ``[0, 1]``; lower is better.

    Raises:
        ValueError: If the inputs fail label/probability validation.
    """
    y = binary_labels(y_true)
    p = probabilities(y_prob, y.size)
    return float(np.mean((p - y) ** 2))


def calibration_fit(
    y_true: LabelArray,
    y_prob: FloatArray,
    *,
    eps: float = 1e-12,
) -> CalibrationFit:
    """Fit the logistic calibration slope and calibration-in-the-large.

    Probabilities are clipped to ``[eps, 1 - eps]`` before the logit
    transform. The slope comes from an unpenalised two-parameter
    Newton-Raphson logistic fit of ``y`` on ``logit(p)``; the intercept
    comes from a one-parameter Newton-Raphson fit of the offset-only
    model ``logit P(y = 1) = a + logit(p)``.

    Args:
        y_true: Binary labels (0 or 1) of shape ``(n,)``.
        y_prob: Predicted probabilities in ``[0, 1]`` of shape ``(n,)``.
        eps: Clipping bound applied before the logit transform.
            Default 1e-12.

    Returns:
        :class:`CalibrationFit`. When ``y_true`` contains a single
        class, both values are ``nan`` and ``converged`` is False. When
        an individual fit fails to converge (singular Hessian or a
        diverging iterate within 100 iterations at tolerance 1e-10),
        that fit's value is ``nan`` and ``converged`` is False; the
        other fit's value is still returned when it converged.

    Raises:
        ValueError: If the inputs fail label/probability validation.
    """
    y = binary_labels(y_true)
    p = probabilities(y_prob, y.size)
    clipped = np.clip(p, eps, 1.0 - eps)
    logit_p = np.log(clipped / (1.0 - clipped))

    if np.unique(y).size < 2:
        return CalibrationFit(slope=float("nan"), intercept=float("nan"), converged=False)

    design = np.column_stack((np.ones_like(logit_p), logit_p))
    slope_result = _newton_logistic_slope(design, y)
    intercept_result = _newton_calibration_in_the_large(logit_p, y)

    slope = slope_result if slope_result is not None else float("nan")
    intercept = intercept_result if intercept_result is not None else float("nan")
    converged = slope_result is not None and intercept_result is not None
    return CalibrationFit(slope=slope, intercept=intercept, converged=converged)


def _newton_logistic_slope(
    design: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float | None:
    """Unpenalised two-parameter Newton-Raphson logistic fit.

    Args:
        design: ``(n, 2)`` design matrix ``[1, logit(p)]``.
        y: Binary outcome as ``float64``.

    Returns:
        The slope coefficient, or ``None`` on a singular Hessian, a
        non-finite iterate, or a non-converged iteration budget.
    """
    beta = np.zeros(2, dtype=np.float64)
    for _ in range(_MAX_ITERATIONS):
        mu = expit(design @ beta)
        weights = mu * (1.0 - mu)
        gradient = design.T @ (y - mu)
        hessian = design.T @ (design * weights[:, np.newaxis])
        try:
            step: NDArray[np.float64] = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if not bool(np.all(np.isfinite(beta))):
            return None
        if float(np.max(np.abs(step))) < _TOLERANCE:
            return float(beta[1])
    return None


def _newton_calibration_in_the_large(
    logit_p: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float | None:
    """Offset-only Newton-Raphson fit (slope fixed at 1).

    Maximizes the log-likelihood of ``logit P(y = 1) = a + logit(p)``
    over the scalar offset ``a``.

    Args:
        logit_p: Clipped logit-transformed probabilities of shape
            ``(n,)``.
        y: Binary outcome as ``float64``.

    Returns:
        The offset ``a``, or ``None`` on a degenerate curvature, a
        non-finite iterate, or a non-converged iteration budget.
    """
    a = 0.0
    for _ in range(_MAX_ITERATIONS):
        mu = expit(a + logit_p)
        gradient = float(np.sum(y - mu))
        curvature = float(np.sum(mu * (1.0 - mu)))
        if curvature <= 0.0:
            return None
        step = gradient / curvature
        a += step
        if not math.isfinite(a):
            return None
        if abs(step) < _TOLERANCE:
            return a
    return None
