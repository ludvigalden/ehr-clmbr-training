"""Stratification: per-stratum evaluation over user-supplied stratum columns.

Strata are never built in: the caller supplies one or more stratum columns
(a value per evaluated row) and every distinct value of every column becomes
a stratum. :func:`bin_numeric` turns a numeric column (an age, a count) into
interval labels over caller-chosen edges; stratum columns usually come from a
pinned covariate table (:mod:`ehrfm.eval.tables`).

Each stratum carries its own ``n``, ``n_positive`` and ``n_negative``, so a
small-cell policy that masks small counts also masks every metric of a small
stratum.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ehrfm.eval.tables import MISSING


class StratificationError(ValueError):
    """Stratum columns or the stratum table are malformed."""


@dataclass(frozen=True, slots=True)
class Stratum:
    """One stratum: a column, one of its values and the rows that have it."""

    column: str
    value: str
    mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class StratumResult:
    """Metrics of one stratum."""

    column: str
    value: str
    n: int
    n_positive: int
    n_negative: int
    metrics: dict[str, float]

    def to_record(self) -> dict[str, str | int | float | None]:
        """A flat record: identity, counts and metrics side by side (non-finite as None)."""
        record: dict[str, str | int | float | None] = {
            "column": self.column,
            "value": self.value,
            "n": self.n,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }
        for name, value in self.metrics.items():
            record[name] = value if np.isfinite(value) else None
        return record


def bin_numeric(
    values: Sequence[float | None] | NDArray[np.float64], edges: Sequence[float]
) -> NDArray[np.str_]:
    """Interval labels ``[lo, hi)`` of ``values`` over strictly increasing ``edges``.

    Values outside ``[edges[0], edges[-1])`` and missing values get
    :data:`MISSING`.
    """
    bounds = [float(edge) for edge in edges]
    if len(bounds) < 2 or any(b <= a for a, b in zip(bounds, bounds[1:], strict=False)):
        raise StratificationError("edges must be at least two strictly increasing numbers")
    array = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    labels = np.full(array.shape, MISSING, dtype=object)
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        inside = (array >= lo) & (array < hi)
        labels[inside] = f"[{lo:g}, {hi:g})"
    return labels.astype(np.str_)


def strata(columns: Mapping[str, Sequence[str] | NDArray[np.str_]], n_rows: int) -> list[Stratum]:
    """Every (column, value) stratum of ``columns``, columns in the given order, values sorted."""
    result: list[Stratum] = []
    for column, raw in columns.items():
        values = np.asarray([str(value) for value in raw], dtype=np.str_)
        if values.shape != (n_rows,):
            raise StratificationError(
                f"stratum column {column!r} has {values.shape[0]} values for {n_rows} rows"
            )
        for value in sorted(set(values.tolist())):
            result.append(Stratum(column, value, values == value))
    return result


def stratify(
    y_true: NDArray[np.float64],
    y_prob: NDArray[np.float64],
    columns: Mapping[str, Sequence[str] | NDArray[np.str_]],
    evaluate: Callable[[NDArray[np.float64], NDArray[np.float64]], dict[str, float]],
) -> list[StratumResult]:
    """Apply ``evaluate`` to the rows of every stratum of ``columns``."""
    if y_true.shape != y_prob.shape or y_true.ndim != 1:
        raise StratificationError("y_true and y_prob must be 1-D arrays of the same length")
    results: list[StratumResult] = []
    for stratum in strata(columns, int(y_true.shape[0])):
        labels = y_true[stratum.mask]
        positives = int(labels.sum())
        results.append(
            StratumResult(
                column=stratum.column,
                value=stratum.value,
                n=int(labels.shape[0]),
                n_positive=positives,
                n_negative=int(labels.shape[0]) - positives,
                metrics=evaluate(labels, y_prob[stratum.mask]),
            )
        )
    return results


__all__ = [
    "MISSING",
    "StratificationError",
    "Stratum",
    "StratumResult",
    "bin_numeric",
    "strata",
    "stratify",
]
