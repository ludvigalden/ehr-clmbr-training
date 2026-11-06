"""Fitting logistic linear heads on frozen representations.

:func:`fit_linear_probe` fits a boolean :class:`~ehrfm.models.head.LinearHead`
with full-batch L-BFGS in float64 on the CPU, from zero weights, for every
strength in the grid, keeps the strength with the best AUROC on the tuning
split, and returns the head that strength produced. The result depends only on
the inputs and the configuration: there is no random initialisation and no
sampling, so two fits of the same data give byte-identical heads.

The optimised parameters are the head that is saved and served (the feature
standardisation is folded into its weights), so the published predictions are
exactly what :meth:`~ehrfm.models.head.TaskHead.predict` returns at inference.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray

from ehrfm.config.heads import LinearProbeConfig
from ehrfm.eval.metrics import auroc
from ehrfm.eval.metrics._validation import FloatArray
from ehrfm.models.head import HeadSpec, LinearHead, TaskHead

_TOLERANCE_GRAD: Final = 1e-10
_TOLERANCE_CHANGE: Final = 1e-12
_HISTORY: Final = 20


class ProbeError(ValueError):
    """A probe cannot be fitted on the given features and labels."""


def _labels(x: FloatArray, y: FloatArray, *, need_both: bool) -> None:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ProbeError(f"features {x.shape} and labels {y.shape} do not align")
    if not np.all(np.isfinite(x)):
        raise ProbeError("features must be finite")
    if not np.all((y == 0) | (y == 1)):
        raise ProbeError("labels must be 0 or 1")
    if need_both and (y.shape[0] == 0 or y.min() == y.max()):
        raise ProbeError("fitting needs both classes in the training labels")


@dataclass(frozen=True, slots=True)
class _Fit:
    weight: NDArray[np.float64]
    bias: float
    iterations: int
    converged: bool


def _sample_weights(y: NDArray[np.float64], class_weight: str) -> NDArray[np.float64]:
    if class_weight == "none":
        return np.ones_like(y)
    n, positives = y.shape[0], float(y.sum())
    negatives = n - positives
    return np.where(y == 1, n / (2 * positives), n / (2 * negatives))


def _lbfgs(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    sample_weights: NDArray[np.float64],
    c: float,
    max_iter: int,
) -> _Fit:
    features = torch.from_numpy(x)
    targets = torch.from_numpy(y)
    weights = torch.from_numpy(sample_weights)
    n = float(x.shape[0])
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [w, b],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=_TOLERANCE_GRAD,
        tolerance_change=_TOLERANCE_CHANGE,
        history_size=_HISTORY,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = features @ w + b
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # The scikit-learn objective divided by n, for a well-scaled line search.
        loss = ((weights * losses).sum() + (w @ w) / (2.0 * c)) / n
        loss.backward()
        return loss

    optimizer.step(closure)
    iterations = int(optimizer.state[optimizer.param_groups[0]["params"][0]].get("n_iter", 0))
    return _Fit(
        weight=w.detach().numpy().copy(),
        bias=float(b.detach()),
        iterations=iterations,
        converged=iterations < max_iter,
    )


@dataclass(frozen=True, slots=True)
class ProbeFit:
    """The selected head and how it was chosen."""

    head: LinearHead
    c: float
    iterations: int
    converged: bool
    tuning_auroc: dict[float, float]
    """Tuning AUROC of every strength tried (empty when no strength was selected on tuning)."""

    def predict(self, x: FloatArray) -> NDArray[np.float64]:
        """Positive-class probabilities, as the served head computes them."""
        return predict_probabilities(self.head, x)


def predict_probabilities(head: TaskHead, x: FloatArray) -> NDArray[np.float64]:
    """``head.predict`` on float32 features, as float64 probabilities (boolean heads)."""
    if head.spec.label_type != "boolean":
        raise ProbeError(f"head {head.spec.task_name!r} is not a boolean head")
    features = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return head.predict(features).numpy().astype(np.float64)


def _head(
    task_name: str, fit: _Fit, mean: NDArray[np.float64], scale: NDArray[np.float64]
) -> LinearHead:
    weight = fit.weight / scale
    bias = fit.bias - float(np.dot(weight, mean))
    head = LinearHead(
        HeadSpec(
            kind=LinearHead.kind,
            task_name=task_name,
            label_type="boolean",
            input_dim=int(weight.shape[0]),
        )
    )
    with torch.no_grad():
        head.linear.weight.copy_(torch.from_numpy(weight[np.newaxis, :]).to(torch.float32))
        head.linear.bias.copy_(torch.tensor([bias], dtype=torch.float32))
    head.eval()
    return head


def fit_linear_probe(
    task_name: str,
    x_train: FloatArray,
    y_train: FloatArray,
    x_tune: FloatArray | None = None,
    y_tune: FloatArray | None = None,
    *,
    config: LinearProbeConfig | None = None,
) -> ProbeFit:
    """Fit a boolean linear head on ``x_train``; choose its strength on the tuning split."""
    config = config if config is not None else LinearProbeConfig()
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    _labels(x, y, need_both=True)
    if config.standardize:
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
    else:
        mean = np.zeros(x.shape[1])
        scale = np.ones(x.shape[1])
    standardized = (x - mean) / scale
    sample_weights = _sample_weights(y, config.class_weight)
    grid: Sequence[float] = config.c_grid
    tune_x = None if x_tune is None else np.asarray(x_tune, dtype=np.float64)
    tune_y = None if y_tune is None else np.asarray(y_tune, dtype=np.float64)
    if tune_x is not None and tune_y is not None:
        _labels(tune_x, tune_y, need_both=False)
    usable = (
        tune_x is not None
        and tune_y is not None
        and tune_y.size > 0
        and tune_y.min() != tune_y.max()
    )
    if not usable or len(grid) == 1:
        c = sorted(grid)[len(grid) // 2]
        fit = _lbfgs(standardized, y, sample_weights, c, config.max_iter)
        head = _head(task_name, fit, mean, scale)
        return ProbeFit(head, c, fit.iterations, fit.converged, {})
    assert tune_x is not None and tune_y is not None
    scores: dict[float, float] = {}
    best: tuple[float, _Fit, LinearHead] | None = None
    best_score = -math.inf
    for c in grid:
        fit = _lbfgs(standardized, y, sample_weights, c, config.max_iter)
        head = _head(task_name, fit, mean, scale)
        score = auroc(tune_y, predict_probabilities(head, tune_x))
        scores[c] = score
        if score > best_score:
            best, best_score = (c, fit, head), score
    assert best is not None
    c, fit, head = best
    return ProbeFit(head, c, fit.iterations, fit.converged, scores)


__all__ = ["ProbeError", "ProbeFit", "fit_linear_probe", "predict_probabilities"]
