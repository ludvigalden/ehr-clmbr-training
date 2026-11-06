"""Forgetting summaries: a base model versus a continued-pretrained model.

Sign conventions, documented explicitly because the two task-level
summaries carry opposite signs by construction:

- ``forgetting = base - continued`` for a higher-is-better metric.
  Positive means the continued model lost ground (forgetting);
  negative means it improved.
- ``backward_transfer = mean(continued - base)`` across tasks
  (Lopez-Paz and Ranzato 2017, "Gradient Episodic Memory for Continual
  Learning", NeurIPS). Negative means forgetting; positive means
  positive backward transfer.
- ``loss_increase = continued_loss - base_loss`` for a held-out loss.
  Positive means the continued model got worse.

Consequently ``backward_transfer`` over the same tasks equals the
negated mean of the per-task ``forgetting`` values.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "backward_transfer",
    "forgetting",
    "loss_increase",
]


def _finite_scalar(value: float, *, name: str) -> float:
    """Validate a finite scalar input.

    Raises:
        ValueError: If ``value`` is not finite.
    """
    if not np.isfinite(value):
        msg = f"{name} must be finite; got {value}"
        raise ValueError(msg)
    return float(value)


def forgetting(base: float, continued: float) -> float:
    """Per-task forgetting of a higher-is-better metric.

    Computes ``base - continued``. Positive means the continued model
    scored lower than the base model (forgetting); negative means it
    improved.

    Args:
        base: Metric value of the base model.
        continued: Metric value of the continued-pretrained model.

    Returns:
        Signed forgetting rate.

    Raises:
        ValueError: If either input is non-finite.
    """
    base_value = _finite_scalar(base, name="base")
    continued_value = _finite_scalar(continued, name="continued")
    return base_value - continued_value


def backward_transfer(base: Sequence[float], continued: Sequence[float]) -> float:
    """Backward transfer across tasks (Lopez-Paz and Ranzato 2017).

    Computes ``mean(continued - base)`` over the task sequence.
    Negative means forgetting on average; positive means positive
    backward transfer. Opposite in sign to the per-task
    :func:`forgetting` values.

    Args:
        base: Metric values of the base model, one per task.
        continued: Metric values of the continued-pretrained model, in
            the same task order.

    Returns:
        Mean backward transfer.

    Raises:
        ValueError: If either sequence is empty or non-1-D, the
            lengths differ, or any value is non-finite.
    """
    base_arr = np.asarray(base, dtype=np.float64)
    continued_arr = np.asarray(continued, dtype=np.float64)
    if base_arr.ndim != 1 or base_arr.size == 0:
        msg = "base must be a non-empty 1-D sequence"
        raise ValueError(msg)
    if continued_arr.ndim != 1 or continued_arr.size == 0:
        msg = "continued must be a non-empty 1-D sequence"
        raise ValueError(msg)
    if base_arr.size != continued_arr.size:
        msg = (
            "base and continued must have equal length; "
            f"got {base_arr.size} and {continued_arr.size}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(base_arr)) or not np.all(np.isfinite(continued_arr)):
        msg = "base and continued must contain only finite values"
        raise ValueError(msg)
    return float(np.mean(continued_arr - base_arr))


def loss_increase(base_loss: float, continued_loss: float) -> float:
    """Held-out loss increase after continued pretraining.

    Computes ``continued_loss - base_loss``. Positive means the
    continued model got worse on the held-out data; negative means it
    improved.

    Args:
        base_loss: Held-out loss of the base model.
        continued_loss: Held-out loss of the continued-pretrained
            model.

    Returns:
        Signed loss change.

    Raises:
        ValueError: If either input is non-finite.
    """
    base_value = _finite_scalar(base_loss, name="base_loss")
    continued_value = _finite_scalar(continued_loss, name="continued_loss")
    return continued_value - base_value
