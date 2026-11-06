"""Bootstrap confidence intervals for scalar evaluation metrics.

Percentile-based confidence intervals for any scalar metric function,
with stratified resampling by default: positive and negative indices
are drawn separately within each resample so every replicate preserves
the original class counts exactly. Replicates on which the metric is
undefined (a ``nan`` return value) are excluded from the quantile
computation and counted in ``finite_resamples``.

``seed`` is keyword-only and required so every confidence interval in
the evaluation toolkit is reproducible by construction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ehrfm.eval.metrics._validation import FloatArray, LabelArray, binary_labels, scores

__all__ = [
    "BootstrapCI",
    "bootstrap_ci",
    "paired_bootstrap_ci",
    "resample_indices",
]

MetricFn = Callable[[NDArray[np.float64], NDArray[np.float64]], float]
"""Metric signature: ``metric(y_true, y_score) -> float``.

The metric receives the resampled label array first and the resampled
score array second, matching the argument order of the public metric
functions (``auroc(y_true, y_score)``, ``brier_score(y_true, y_prob)``,
and so on). It may return ``nan`` on degenerate replicates; such
replicates are excluded from the interval.
"""


@dataclass(frozen=True, kw_only=True, slots=True)
class BootstrapCI:
    """Percentile bootstrap confidence interval for a scalar metric.

    Attributes:
        estimate: Metric value on the full, unresampled data. Exactly
            ``metric(y_true, y_score)``.
        low: Lower percentile bound (``alpha / 2`` quantile of the
            finite bootstrap distribution). ``nan`` when no replicate is
            finite.
        high: Upper percentile bound (``1 - alpha / 2`` quantile of the
            finite bootstrap distribution). ``nan`` when no replicate is
            finite.
        alpha: Significance level; ``alpha = 0.05`` yields the 2.5th and
            97.5th percentiles.
        resamples: Number of bootstrap resamples drawn.
        finite_resamples: Number of replicates whose metric value was
            finite and therefore entered the quantile computation.
        distribution: Raw per-resample metric values of shape
            ``(resamples,)``. Degenerate replicates carry ``nan``.
    """

    estimate: float
    low: float
    high: float
    alpha: float
    resamples: int
    finite_resamples: int
    distribution: NDArray[np.float64]


def _validate_resamples(resamples: int) -> None:
    """Validate the resample count.

    Raises:
        ValueError: If ``resamples < 1``.
    """
    if resamples < 1:
        msg = f"resamples must be >= 1; got {resamples}"
        raise ValueError(msg)


def _validate_alpha(alpha: float) -> None:
    """Validate the significance level.

    Raises:
        ValueError: If ``alpha`` is outside (0, 1).
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must lie strictly inside (0, 1); got {alpha}"
        raise ValueError(msg)


def resample_indices(
    y_true: LabelArray,
    resamples: int,
    rng: np.random.Generator,
    *,
    stratified: bool = True,
) -> NDArray[np.intp]:
    """Draw bootstrap resample indices for every replicate.

    Args:
        y_true: Binary labels (0 or 1) of shape ``(n,)``.
        resamples: Number of replicates to draw. Must be >= 1.
        rng: ``np.random.Generator`` driving the draws.
        stratified: When True (default), sample positives and negatives
            separately within each replicate so every replicate keeps the
            original class counts exactly. Requires both classes to be
            present. When False, draw uniformly with replacement from all
            ``n`` indices.

    Returns:
        Index array of shape ``(resamples, n)``; row ``i`` holds the
        indices for replicate ``i``.

    Raises:
        ValueError: If ``resamples < 1``, ``y_true`` is not binary, or a
            stratified draw is requested with a single class present.
    """
    y = binary_labels(y_true)
    _validate_resamples(resamples)
    n = y.size
    if stratified:
        positive_indices: NDArray[np.intp] = np.flatnonzero(y == 1).astype(np.intp)
        negative_indices: NDArray[np.intp] = np.flatnonzero(y == 0).astype(np.intp)
        if positive_indices.size == 0 or negative_indices.size == 0:
            msg = (
                "stratified resampling requires both classes present in y_true; "
                f"got {int(positive_indices.size)} positives and "
                f"{int(negative_indices.size)} negatives"
            )
            raise ValueError(msg)
        positive_draws: NDArray[np.intp] = rng.choice(
            positive_indices, size=(resamples, int(positive_indices.size)), replace=True
        )
        negative_draws: NDArray[np.intp] = rng.choice(
            negative_indices, size=(resamples, int(negative_indices.size)), replace=True
        )
        return np.concatenate([positive_draws, negative_draws], axis=1)
    return rng.integers(low=0, high=n, size=(resamples, n), dtype=np.intp)


_CHUNK_CELLS: Final = 1 << 22
"""Upper bound on index cells drawn at once, so memory stays bounded for large ``n``."""


def _replicates(
    y: NDArray[np.float64], resamples: int, rng: np.random.Generator, *, stratified: bool
) -> Iterator[tuple[int, NDArray[np.intp]]]:
    """``(replicate, indices)`` pairs, drawn in chunks of :func:`resample_indices`."""
    chunk = max(1, _CHUNK_CELLS // max(1, y.size))
    done = 0
    while done < resamples:
        size = min(chunk, resamples - done)
        for offset, row in enumerate(resample_indices(y, size, rng, stratified=stratified)):
            yield done + offset, row
        done += size


def bootstrap_ci(
    metric: MetricFn,
    y_true: LabelArray,
    y_score: FloatArray,
    *,
    resamples: int = 10000,
    alpha: float = 0.05,
    seed: int,
    stratified: bool = True,
) -> BootstrapCI:
    """Percentile bootstrap confidence interval for a scalar metric.

    The point estimate is ``metric(y_true, y_score)`` on the full data.
    Each replicate evaluates the metric on a bootstrap resample drawn
    with :func:`resample_indices`; replicates whose metric value is
    ``nan`` are excluded from the percentile computation.

    Args:
        metric: Callable with the ``metric(y_true, y_score)`` signature;
            see :data:`MetricFn` for the argument order and ``nan``
            convention.
        y_true: Binary labels (0 or 1) of shape ``(n,)``.
        y_score: Predicted scores of shape ``(n,)``.
        resamples: Number of bootstrap resamples. Default 10000.
        alpha: Significance level. Default 0.05 (a 95% interval).
        seed: Seed for ``np.random.default_rng``. Required so results
            are reproducible.
        stratified: When True (default), preserves class counts in every
            replicate.

    Returns:
        :class:`BootstrapCI` with the estimate, percentile bounds, and
        the raw bootstrap distribution.

    Raises:
        ValueError: If ``resamples < 1``, ``alpha`` is outside (0, 1),
            the inputs fail label/score validation, or a stratified draw
            is requested with a single class present.
    """
    y = binary_labels(y_true)
    s = scores(y_score, y.size)
    _validate_resamples(resamples)
    _validate_alpha(alpha)
    rng = np.random.default_rng(seed)
    estimate = float(metric(y, s))
    distribution = np.empty(resamples, dtype=np.float64)
    for i, selected in _replicates(y, resamples, rng, stratified=stratified):
        try:
            distribution[i] = float(metric(y[selected], s[selected]))
        except ValueError:
            distribution[i] = float("nan")
    return _summarize(
        estimate=estimate, alpha=alpha, resamples=resamples, distribution=distribution
    )


def paired_bootstrap_ci(
    metric: MetricFn,
    y_true: LabelArray,
    score_a: FloatArray,
    score_b: FloatArray,
    *,
    resamples: int = 10000,
    alpha: float = 0.05,
    seed: int,
    stratified: bool = True,
) -> BootstrapCI:
    """Percentile bootstrap CI for the paired metric difference.

    Computes ``metric(a) - metric(b)`` on the full data and on every
    bootstrap resample, using the same index draw for both score arrays
    so the difference is paired subject-by-subject.

    Args:
        metric: Callable with the ``metric(y_true, y_score)`` signature.
        y_true: Binary labels (0 or 1) of shape ``(n,)``.
        score_a: Scores of model A of shape ``(n,)``.
        score_b: Scores of model B of shape ``(n,)``.
        resamples: Number of bootstrap resamples. Default 10000.
        alpha: Significance level. Default 0.05.
        seed: Seed for ``np.random.default_rng``. Required.
        stratified: When True (default), preserves class counts in every
            replicate.

    Returns:
        :class:`BootstrapCI` for the signed difference ``metric(a) -
        metric(b)``.

    Raises:
        ValueError: If ``resamples < 1``, ``alpha`` is outside (0, 1),
            the inputs fail validation, or a stratified draw is
            requested with a single class present.
    """
    y = binary_labels(y_true)
    a = scores(score_a, y.size)
    b = scores(score_b, y.size)
    _validate_resamples(resamples)
    _validate_alpha(alpha)
    rng = np.random.default_rng(seed)
    estimate = float(metric(y, a) - metric(y, b))
    distribution = np.empty(resamples, dtype=np.float64)
    for i, selected in _replicates(y, resamples, rng, stratified=stratified):
        try:
            distribution[i] = float(
                metric(y[selected], a[selected]) - metric(y[selected], b[selected])
            )
        except ValueError:
            distribution[i] = float("nan")
    return _summarize(
        estimate=estimate, alpha=alpha, resamples=resamples, distribution=distribution
    )


def _summarize(
    *,
    estimate: float,
    alpha: float,
    resamples: int,
    distribution: NDArray[np.float64],
) -> BootstrapCI:
    """Reduce a raw bootstrap distribution to a :class:`BootstrapCI`.

    NaN replicates are excluded from the quantiles; when no replicate is
    finite the bounds are ``nan``.
    """
    finite = distribution[~np.isnan(distribution)]
    finite_resamples = int(finite.size)
    if finite_resamples == 0:
        low = float("nan")
        high = float("nan")
    else:
        low = float(np.quantile(finite, alpha / 2.0))
        high = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return BootstrapCI(
        estimate=estimate,
        low=low,
        high=high,
        alpha=alpha,
        resamples=resamples,
        finite_resamples=finite_resamples,
        distribution=distribution,
    )
