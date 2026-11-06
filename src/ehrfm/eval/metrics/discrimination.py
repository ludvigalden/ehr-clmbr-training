"""Discrimination metrics: AUROC and AUPRC for a binary outcome.

Thin wrappers over the scikit-learn backends
(``roc_auc_score``, ``average_precision_score``) with loud input
validation and documented degenerate-case behavior: a single-class
label array makes AUROC undefined (``nan``), and a label array without
positives makes AUPRC undefined (``nan``). Both cases return ``nan``
instead of raising so a bootstrap resample loop can record the
degenerate replicate and continue.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ehrfm.eval.metrics._validation import FloatArray, LabelArray, binary_labels, scores

__all__ = [
    "auprc",
    "auroc",
]


def auroc(y_true: LabelArray, y_score: FloatArray) -> float:
    """Area under the receiver operating characteristic curve.

    Args:
        y_true: Binary labels (0 or 1) of shape ``(n,)``.
        y_score: Predicted scores of shape ``(n,)``.

    Returns:
        AUROC in ``[0, 1]``. Returns ``nan`` when ``y_true`` contains a
        single class, since the metric is undefined there.

    Raises:
        ValueError: If the arrays are not 1-D, their lengths differ,
            the scores are non-finite, or the labels are not exactly
            ``{0, 1}``.
    """
    y = binary_labels(y_true)
    s = scores(y_score, y.size)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auprc(y_true: LabelArray, y_score: FloatArray) -> float:
    """Area under the precision-recall curve (average precision).

    Args:
        y_true: Binary labels (0 or 1) of shape ``(n,)``.
        y_score: Predicted scores of shape ``(n,)``.

    Returns:
        AUPRC in ``[0, 1]``. Returns ``nan`` when ``y_true`` contains no
        positives, since the metric is undefined there.

    Raises:
        ValueError: If the arrays are not 1-D, their lengths differ,
            the scores are non-finite, or the labels are not exactly
            ``{0, 1}``.
    """
    y = binary_labels(y_true)
    s = scores(y_score, y.size)
    if not np.any(y == 1):
        return float("nan")
    return float(average_precision_score(y, s))
