"""Decision-curve analysis: net benefit of a prediction model.

Net benefit (Vickers and Elkin 2006, "Decision Curve Analysis: A Novel
Method for Evaluating Prediction Models", Med Decis Making 26(6)) at a
threshold probability ``t`` — the probability above which a subject is
counted as positive — is::

    NB_model(t) = TP(t)/N - FP(t)/N * (t / (1 - t))
    NB_all(t)   = prevalence - (1 - prevalence) * (t / (1 - t))
    NB_none(t)  = 0

with the ``>=`` convention for the positive decision (``y_prob >= t``).
A model is clinically useful at threshold ``t`` only when its net
benefit exceeds both reference policies at that threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ehrfm.eval.metrics._validation import FloatArray, LabelArray, binary_labels, probabilities

__all__ = [
    "DecisionCurve",
    "decision_curve",
    "net_benefit",
    "net_benefit_treat_all",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class DecisionCurve:
    """Net-benefit curves across threshold probabilities.

    Attributes:
        thresholds: Threshold probabilities of shape
            ``(n_thresholds,)``, monotonically increasing.
        net_benefit_model: Per-threshold model net benefit of shape
            ``(n_thresholds,)``.
        net_benefit_all: Per-threshold treat-all reference net benefit
            of shape ``(n_thresholds,)``.
        net_benefit_none: Treat-none reference, always 0.0 by
            definition; carried explicitly so plotting code does not
            infer the reference line from absence.
        prevalence: Positive prevalence of the evaluated cohort.
    """

    thresholds: NDArray[np.float64]
    net_benefit_model: NDArray[np.float64]
    net_benefit_all: NDArray[np.float64]
    net_benefit_none: float
    prevalence: float


def _validate_threshold(threshold: float) -> None:
    """Validate a single threshold probability.

    Raises:
        ValueError: If ``threshold`` lies outside the open interval
            (0, 1) or is non-finite.
    """
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        msg = f"threshold must lie strictly inside (0, 1); got {threshold}"
        raise ValueError(msg)


def net_benefit(
    y_true: LabelArray,
    y_prob: FloatArray,
    threshold: float,
) -> float:
    """Model net benefit at a single threshold probability.

    Computes ``TP/N - FP/N * (t / (1 - t))`` where the positive decision
    is ``y_prob >= t``.

    Args:
        y_true: Binary labels (0 or 1) of shape ``(n,)``.
        y_prob: Predicted probabilities in ``[0, 1]`` of shape ``(n,)``.
        threshold: Threshold probability strictly inside ``(0, 1)``.

    Returns:
        Net benefit at the threshold.

    Raises:
        ValueError: If the inputs fail label/probability validation or
            the threshold lies outside ``(0, 1)``.
    """
    y = binary_labels(y_true)
    p = probabilities(y_prob, y.size)
    _validate_threshold(threshold)
    n = y.size
    predicted_positive = p >= threshold
    tp = int(np.count_nonzero(predicted_positive & (y == 1)))
    fp = int(np.count_nonzero(predicted_positive & (y == 0)))
    odds = threshold / (1.0 - threshold)
    return tp / n - (fp / n) * odds


def net_benefit_treat_all(prevalence: float, threshold: float) -> float:
    """Treat-all reference net benefit at a single threshold.

    Computes ``prevalence - (1 - prevalence) * (t / (1 - t))``: the net
    benefit of acting on every subject regardless of the model.

    Args:
        prevalence: Positive prevalence in ``[0, 1]``.
        threshold: Threshold probability strictly inside ``(0, 1)``.

    Returns:
        Treat-all net benefit at the threshold.

    Raises:
        ValueError: If ``prevalence`` lies outside ``[0, 1]`` or is
            non-finite, or the threshold lies outside ``(0, 1)``.
    """
    if not np.isfinite(prevalence) or not 0.0 <= prevalence <= 1.0:
        msg = f"prevalence must lie in [0, 1]; got {prevalence}"
        raise ValueError(msg)
    _validate_threshold(threshold)
    odds = threshold / (1.0 - threshold)
    return prevalence - (1.0 - prevalence) * odds


def decision_curve(
    y_true: LabelArray,
    y_prob: FloatArray,
    *,
    thresholds: FloatArray | None = None,
    threshold_min: float = 0.01,
    threshold_max: float = 0.20,
    step: float = 0.01,
) -> DecisionCurve:
    """Decision-curve analysis across threshold probabilities.

    When ``thresholds`` is ``None``, sweeps an explicit grid from
    ``threshold_min`` to ``threshold_max`` in steps of ``step`` (both
    endpoints included when the step divides the range cleanly); grid
    values are rounded to 10 decimals to avoid floating-point drift.
    When ``thresholds`` is given, that grid is used verbatim and the
    range parameters are ignored.

    Args:
        y_true: Binary labels (0 or 1) of shape ``(n,)``.
        y_prob: Predicted probabilities in ``[0, 1]`` of shape ``(n,)``.
        thresholds: Optional explicit threshold grid, strictly inside
            ``(0, 1)``.
        threshold_min: Inclusive lower bound of the generated grid.
            Default 0.01.
        threshold_max: Inclusive upper bound of the generated grid.
            Default 0.20.
        step: Spacing of the generated grid. Default 0.01.

    Returns:
        :class:`DecisionCurve` with the model and treat-all net-benefit
        arrays and the zero treat-none reference.

    Raises:
        ValueError: If the inputs fail label/probability validation;
            if ``thresholds`` is given but is empty, non-1-D,
            non-finite, or outside ``(0, 1)``; or if the range
            parameters are outside their valid intervals.
    """
    y = binary_labels(y_true)
    p = probabilities(y_prob, y.size)
    prevalence = float(np.mean(y))

    if thresholds is None:
        if not np.isfinite(threshold_min) or not 0.0 < threshold_min < 1.0:
            msg = f"threshold_min must lie strictly inside (0, 1); got {threshold_min}"
            raise ValueError(msg)
        if not np.isfinite(threshold_max) or not 0.0 < threshold_max < 1.0:
            msg = f"threshold_max must lie strictly inside (0, 1); got {threshold_max}"
            raise ValueError(msg)
        if threshold_min > threshold_max:
            msg = (
                f"threshold_min must be <= threshold_max; got "
                f"threshold_min={threshold_min}, threshold_max={threshold_max}"
            )
            raise ValueError(msg)
        if step <= 0.0:
            msg = f"step must be positive; got {step}"
            raise ValueError(msg)
        n_steps = int(round((threshold_max - threshold_min) / step)) + 1
        grid = np.round(np.linspace(threshold_min, threshold_max, num=n_steps), decimals=10)
    else:
        provided = np.asarray(thresholds, dtype=np.float64)
        if provided.ndim != 1 or provided.size == 0:
            msg = "thresholds must be a non-empty 1-D array"
            raise ValueError(msg)
        if not np.all(np.isfinite(provided)):
            msg = "thresholds must be finite"
            raise ValueError(msg)
        if not np.all((provided > 0.0) & (provided < 1.0)):
            msg = "thresholds must lie strictly inside (0, 1)"
            raise ValueError(msg)
        grid = provided

    nb_model = np.array([net_benefit(y, p, float(t)) for t in grid], dtype=np.float64)
    nb_all = np.array([net_benefit_treat_all(prevalence, float(t)) for t in grid], dtype=np.float64)
    return DecisionCurve(
        thresholds=grid,
        net_benefit_model=nb_model,
        net_benefit_all=nb_all,
        net_benefit_none=0.0,
        prevalence=prevalence,
    )
