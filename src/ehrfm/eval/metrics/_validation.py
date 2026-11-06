"""Shared input validation for the metric modules.

Every public metric function validates its arrays through these helpers
so a malformed input always fails loudly with a ``ValueError`` naming
the offending condition: labels must be 1-D, non-empty and exactly
binary in ``{0, 1}``; scores must be 1-D, finite and match the label
count; probabilities additionally lie in ``[0, 1]``. The helpers return
``float64`` arrays so downstream mean and squared-error computations
are stable regardless of the input dtype.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FloatArray",
    "LabelArray",
    "binary_labels",
    "probabilities",
    "scores",
]

LabelArray = NDArray[np.integer[Any]] | NDArray[np.floating[Any]]
"""Binary-label array; integer or floating dtype."""
FloatArray = NDArray[np.floating[Any]]
"""Score/probability array; floating dtype."""


def binary_labels(y: LabelArray) -> NDArray[np.float64]:
    """Validate binary labels and return them as a ``float64`` array.

    Args:
        y: Labels expected to contain exactly the values ``{0, 1}``.

    Returns:
        The labels as a 1-D ``float64`` array.

    Raises:
        ValueError: If ``y`` is not 1-D, is empty, contains non-finite
            values, or holds values other than ``{0, 1}``.
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        msg = f"y_true must be 1-D; got shape={arr.shape}"
        raise ValueError(msg)
    if arr.size == 0:
        msg = "y_true must be non-empty"
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = "y_true must contain only finite values"
        raise ValueError(msg)
    if not np.all(np.isin(arr, (0.0, 1.0))):
        unique = np.unique(arr)
        msg = (
            f"y_true must contain exactly the values {{0, 1}}; got unique values {unique.tolist()}"
        )
        raise ValueError(msg)
    return arr


def scores(y_score: FloatArray, n: int) -> NDArray[np.float64]:
    """Validate a score array against an expected row count.

    Args:
        y_score: Predicted scores or probabilities.
        n: Expected array length (typically ``y_true.size``).

    Returns:
        The scores as a 1-D ``float64`` array of length ``n``.

    Raises:
        ValueError: If ``y_score`` is not 1-D, its length differs from
            ``n``, or it contains non-finite values.
    """
    arr = np.asarray(y_score, dtype=np.float64)
    if arr.ndim != 1:
        msg = f"y_score must be 1-D; got shape={arr.shape}"
        raise ValueError(msg)
    if arr.size != n:
        msg = f"y_score must have the same length as y_true; got {arr.size}, expected {n}"
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = "y_score must contain only finite values"
        raise ValueError(msg)
    return arr


def probabilities(y_prob: FloatArray, n: int) -> NDArray[np.float64]:
    """Validate a probability array against an expected row count.

    Applies the :func:`scores` checks and additionally requires every
    value to lie in ``[0, 1]``.

    Args:
        y_prob: Predicted probabilities.
        n: Expected array length.

    Returns:
        The probabilities as a 1-D ``float64`` array of length ``n``.

    Raises:
        ValueError: If ``y_prob`` fails the :func:`scores` checks or any
            value lies outside ``[0, 1]``.
    """
    arr = scores(y_prob, n)
    if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
        msg = f"y_prob must lie in [0, 1]; got min={float(arr.min())}, max={float(arr.max())}"
        raise ValueError(msg)
    return arr
