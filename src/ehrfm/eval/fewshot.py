"""Few-shot evaluation in the EHRSHOT protocol.

The protocol follows the EHRSHOT benchmark (Wornow et al., "EHRSHOT: An EHR
Benchmark for Few-Shot Evaluation of Foundation Models", NeurIPS Datasets and
Benchmarks, 2023): for each shot count ``k`` and each replicate, draw ``k``
positive and ``k`` negative examples from the training split to fit a
logistic-regression probe on frozen representations, draw ``k`` positive and
``k`` negative examples from the tuning split to choose its regularisation
strength from ``c_grid`` (the EHRSHOT grid by default; see
:class:`ehrfm.config.heads.EHRSHOT_C_GRID`), and evaluate the probe on the
whole held-out split. Repeating the
draw over replicates shows how much the score depends on which few examples
were seen. No EHRSHOT data ships here; the protocol runs on any boolean
representation set.

Draws are seeded per ``(seed, k, replicate)`` through
:class:`numpy.random.SeedSequence`, so they are reproducible and independent
of the order in which ``k`` values are listed. A ``k`` larger than the
available examples of either class is reported as skipped rather than drawn
with replacement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ehrfm.config.heads import EHRSHOT_C_GRID
from ehrfm.eval.baselines import fit_logistic
from ehrfm.eval.metrics import auprc, auroc

DEFAULT_SHOTS: Final = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128)
"""Shot counts of the EHRSHOT benchmark."""
DEFAULT_REPLICATES: Final = 5


class FewShotError(ValueError):
    """The few-shot settings or inputs are invalid."""


type Scorer = Callable[[NDArray[np.float64]], NDArray[np.float64]]
type ShotFitter = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[Scorer, float],
]
"""Fits a probe on ``(x_train, y_train, x_tune, y_tune)``; returns its scorer and strength."""


def logistic_fitter(c_grid: Sequence[float], seed: int) -> ShotFitter:
    """The default fitter: :func:`ehrfm.eval.baselines.fit_logistic` (scikit-learn)."""

    def fit(
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_tune: NDArray[np.float64],
        y_tune: NDArray[np.float64],
    ) -> tuple[Scorer, float]:
        fitted = fit_logistic(x_train, y_train, x_tune, y_tune, c_grid=c_grid, seed=seed)
        return fitted.model.predict, fitted.model.c

    return fit


@dataclass(frozen=True, slots=True)
class ShotDraw:
    """The rows one replicate of one shot count uses."""

    k: int
    replicate: int
    train_rows: NDArray[np.intp]
    tune_rows: NDArray[np.intp]


def _rng(seed: int, k: int, replicate: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, k, replicate]))


def _balanced(
    labels: NDArray[np.float64], k: int, rng: np.random.Generator
) -> NDArray[np.intp] | None:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if positives.shape[0] < k or negatives.shape[0] < k:
        return None
    chosen = np.concatenate(
        [rng.choice(positives, size=k, replace=False), rng.choice(negatives, size=k, replace=False)]
    )
    return np.sort(chosen).astype(np.intp)


def draw_shots(
    train_labels: NDArray[np.float64],
    tune_labels: NDArray[np.float64],
    k: int,
    replicate: int,
    *,
    seed: int,
) -> ShotDraw | None:
    """The ``k``-shot draw of one replicate, or None when a split lacks ``k`` of a class.

    When the tuning split lacks ``k`` of a class the draw keeps no tuning rows
    (the probe then uses the default strength).
    """
    if k < 1:
        raise FewShotError(f"shot counts must be positive, got {k}")
    rng = _rng(seed, k, replicate)
    train = _balanced(train_labels, k, rng)
    if train is None:
        return None
    tune = _balanced(tune_labels, k, rng)
    return ShotDraw(
        k=k,
        replicate=replicate,
        train_rows=train,
        tune_rows=tune if tune is not None else np.empty(0, dtype=np.intp),
    )


@dataclass(frozen=True, slots=True)
class ShotResult:
    """One replicate of one shot count, scored on the held-out split."""

    k: int
    replicate: int
    n_train: int
    n_tune: int
    c: float
    auroc: float
    auprc: float


@dataclass(frozen=True, slots=True)
class ShotSummary:
    """All replicates of one shot count."""

    k: int
    replicates: int
    """Replicates that could be drawn (0 when the training split lacks k of a class)."""
    auroc_mean: float
    auroc_std: float
    auprc_mean: float
    auprc_std: float


@dataclass(frozen=True, slots=True)
class FewShotResult:
    """The few-shot curve of one representation source."""

    n_held_out: int
    n_held_out_positive: int
    runs: tuple[ShotResult, ...]
    summaries: tuple[ShotSummary, ...]


def _summary(k: int, runs: Sequence[ShotResult]) -> ShotSummary:
    nan = float("nan")
    if not runs:
        return ShotSummary(k, 0, nan, nan, nan, nan)
    aurocs = np.asarray([run.auroc for run in runs], dtype=np.float64)
    auprcs = np.asarray([run.auprc for run in runs], dtype=np.float64)
    return ShotSummary(
        k=k,
        replicates=len(runs),
        auroc_mean=float(np.nanmean(aurocs)) if np.isfinite(aurocs).any() else nan,
        auroc_std=float(np.nanstd(aurocs)) if np.isfinite(aurocs).any() else nan,
        auprc_mean=float(np.nanmean(auprcs)) if np.isfinite(auprcs).any() else nan,
        auprc_std=float(np.nanstd(auprcs)) if np.isfinite(auprcs).any() else nan,
    )


def run_few_shot(
    train: tuple[NDArray[np.float64], NDArray[np.float64]],
    tune: tuple[NDArray[np.float64], NDArray[np.float64]],
    held_out: tuple[NDArray[np.float64], NDArray[np.float64]],
    *,
    shots: Sequence[int] = DEFAULT_SHOTS,
    replicates: int = DEFAULT_REPLICATES,
    seed: int,
    c_grid: Sequence[float] = EHRSHOT_C_GRID,
    fitter: ShotFitter | None = None,
) -> FewShotResult:
    """Run the protocol; each split is ``(features, labels)``.

    ``fitter`` fits each k-shot probe; the default is :func:`logistic_fitter`
    over ``c_grid``.
    """
    if replicates < 1:
        raise FewShotError("replicates must be at least 1")
    if not shots or len(set(shots)) != len(shots):
        raise FewShotError("shots must be a non-empty list of distinct counts")
    x_train, y_train = train
    x_tune, y_tune = tune
    x_test, y_test = held_out
    fit = fitter if fitter is not None else logistic_fitter(c_grid, seed)
    runs: list[ShotResult] = []
    summaries: list[ShotSummary] = []
    for k in sorted(shots):
        per_k: list[ShotResult] = []
        for replicate in range(replicates):
            draw = draw_shots(y_train, y_tune, k, replicate, seed=seed)
            if draw is None:
                break
            score, c = fit(
                x_train[draw.train_rows],
                y_train[draw.train_rows],
                x_tune[draw.tune_rows],
                y_tune[draw.tune_rows],
            )
            scores = score(x_test)
            per_k.append(
                ShotResult(
                    k=k,
                    replicate=replicate,
                    n_train=int(draw.train_rows.shape[0]),
                    n_tune=int(draw.tune_rows.shape[0]),
                    c=c,
                    auroc=auroc(y_test, scores),
                    auprc=auprc(y_test, scores),
                )
            )
        runs.extend(per_k)
        summaries.append(_summary(k, per_k))
    return FewShotResult(
        n_held_out=int(y_test.shape[0]),
        n_held_out_positive=int(y_test.sum()),
        runs=tuple(runs),
        summaries=tuple(summaries),
    )


__all__ = [
    "DEFAULT_REPLICATES",
    "DEFAULT_SHOTS",
    "FewShotError",
    "FewShotResult",
    "Scorer",
    "ShotFitter",
    "ShotDraw",
    "ShotResult",
    "ShotSummary",
    "draw_shots",
    "logistic_fitter",
    "run_few_shot",
]
