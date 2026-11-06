"""Logistic models fitted on fixed features: embedding probes and covariate baselines.

:class:`LogisticModel` is an L2-regularised logistic regression on
standardised features (scikit-learn, L-BFGS). :func:`fit_logistic` fits one
for each regularisation strength of a grid, selects the strength with the
best AUROC on a tuning set (when the tuning set has both classes; otherwise
the grid's middle value) and refits it on the training set.

:meth:`LogisticModel.to_linear_head` folds the standardisation into the
weights and returns an :class:`ehrfm.models.head.LinearHead` that gives the
same probabilities, so a fitted probe can be saved with
:func:`ehrfm.inference.head_loader.save_head` and served.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from ehrfm.config.heads import EHRSHOT_C_GRID
from ehrfm.eval.metrics import auroc
from ehrfm.models.head import HeadSpec, LinearHead


class BaselineError(ValueError):
    """A model cannot be fitted on the given features and labels."""


def _check(x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise BaselineError(f"features {x.shape} and labels {y.shape} do not align")
    if not np.all(np.isfinite(x)):
        raise BaselineError("features must be finite")
    if not np.all((y == 0) | (y == 1)):
        raise BaselineError("labels must be 0 or 1")
    if y.shape[0] == 0 or y.min() == y.max():
        raise BaselineError("fitting needs both classes in the training labels")


@dataclass(frozen=True, slots=True)
class LogisticModel:
    """A fitted logistic regression on standardised features."""

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    coef: NDArray[np.float64]
    intercept: float
    c: float

    def decision(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Log-odds of the positive class."""
        if x.ndim != 2 or x.shape[1] != self.coef.shape[0]:
            raise BaselineError(f"expected {self.coef.shape[0]} features, got {x.shape}")
        return ((x - self.mean) / self.scale) @ self.coef + self.intercept

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Probability of the positive class."""
        return np.asarray(expit(self.decision(x)), dtype=np.float64)

    def to_linear_head(self, task_name: str) -> LinearHead:
        """A boolean :class:`LinearHead` with the same probabilities."""
        weights = self.coef / self.scale
        bias = self.intercept - float(np.dot(weights, self.mean))
        spec = HeadSpec(
            kind=LinearHead.kind,
            task_name=task_name,
            label_type="boolean",
            input_dim=int(self.coef.shape[0]),
        )
        head = LinearHead(spec)
        with torch.no_grad():
            head.linear.weight.copy_(torch.from_numpy(weights[np.newaxis, :]).to(torch.float32))
            head.linear.bias.copy_(torch.tensor([bias], dtype=torch.float32))
        head.eval()
        return head


def _fit_one(
    x: NDArray[np.float64], y: NDArray[np.float64], c: float, *, seed: int, max_iter: int
) -> LogisticModel:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    model = LogisticRegression(C=c, max_iter=max_iter, random_state=seed)
    model.fit((x - mean) / scale, y.astype(np.int64))
    return LogisticModel(
        mean=mean,
        scale=scale,
        coef=np.asarray(model.coef_, dtype=np.float64).ravel(),
        intercept=float(np.asarray(model.intercept_).ravel()[0]),
        c=c,
    )


@dataclass(frozen=True, slots=True)
class FittedLogistic:
    """The selected model and the tuning AUROC of every strength tried."""

    model: LogisticModel
    tuning_auroc: dict[float, float]


def fit_logistic(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_tune: NDArray[np.float64] | None = None,
    y_tune: NDArray[np.float64] | None = None,
    *,
    c_grid: Sequence[float] = EHRSHOT_C_GRID,
    seed: int = 0,
    max_iter: int = 1000,
) -> FittedLogistic:
    """Fit on the training set, choosing the strength by tuning AUROC (see the module docstring)."""
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    _check(x_train, y_train)
    grid = tuple(float(c) for c in c_grid)
    if not grid or any(not math.isfinite(c) or c <= 0 for c in grid):
        raise BaselineError("c_grid must hold positive finite numbers")
    usable = (
        x_tune is not None
        and y_tune is not None
        and y_tune.shape[0] > 0
        and y_tune.min() != y_tune.max()
    )
    if not usable or len(grid) == 1:
        middle = sorted(grid)[len(grid) // 2]
        return FittedLogistic(_fit_one(x_train, y_train, middle, seed=seed, max_iter=max_iter), {})
    assert x_tune is not None and y_tune is not None
    scores: dict[float, float] = {}
    best: LogisticModel | None = None
    best_score = -math.inf
    for c in grid:
        candidate = _fit_one(x_train, y_train, c, seed=seed, max_iter=max_iter)
        score = auroc(np.asarray(y_tune, dtype=np.float64), candidate.predict(x_tune))
        scores[c] = score
        if score > best_score:
            best, best_score = candidate, score
    assert best is not None
    return FittedLogistic(best, scores)


SECONDS_PER_YEAR: Final = 365.25 * 24 * 3600


def age_in_years(
    times: NDArray[np.int64], birth_seconds: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Age at each prediction time (epoch seconds) given birth times (epoch seconds)."""
    return (times.astype(np.float64) - birth_seconds) / SECONDS_PER_YEAR


__all__ = [
    "SECONDS_PER_YEAR",
    "BaselineError",
    "FittedLogistic",
    "LogisticModel",
    "age_in_years",
    "fit_logistic",
]
