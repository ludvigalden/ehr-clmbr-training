"""Vendored DeLong test implementation for paired AUROC comparison.

Vendored from yandexdataschool/roc_comparison (MIT License):
    https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py

Algorithm: Xu Sun and Weichao Xu, "Fast Implementation of DeLong's
Algorithm for Comparing the Areas Under Correlated Receiver Operating
Characteristic Curves", IEEE Signal Processing Letters, 21(11),
1389-1393, 2014.

The algorithm computes U-statistics for paired AUROCs and their
covariance, then derives a z-statistic for the null hypothesis that
both AUROCs are equal. The implementation is O(N log N) per model
through midrank computation rather than the O(N^2) classic DeLong
formulation.

License: MIT. Upstream copyright line: "Copyright (c) 2021 Nikita
Kazeev". The full MIT permission notice ships verbatim in
LICENSES/MIT-roc_comparison.txt at the repository root and is referenced
from NOTICE. The code below is a near-verbatim transcription with
minor adjustments for basedpyright strict mode (explicit type
annotations on local variables, removal of the unused ``pandas`` import
from the upstream file).

Public symbols (consumed by :mod:`ehrfm.eval.metrics.statistical`):

- ``delong_roc_test(ground_truth, predictions_one, predictions_two)``
  returns ``(log10_p_value, aucs)``: the two-sided ``log10(p-value)``
  of the paired test and the per-model AUROCs.
- ``delong_roc_variance(ground_truth, predictions)`` returns
  ``(auc, delong_covariance)`` for a single model.

Both functions assume ``ground_truth`` contains exactly the values
``{0, 1}`` (asserted via ``compute_ground_truth_statistics``) and that
the prediction arrays share the same indexing as the ground truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.stats
from numpy.typing import NDArray


def compute_midrank(x: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Compute midranks for a 1D array.

    Midranks resolve ties by averaging the ranks of equal-valued
    elements. The midrank vector is returned in the original index
    order; ``T2[i]`` is the midrank of ``x[i]``.

    Args:
        x: 1D numpy array of floats.

    Returns:
        Array of midranks, same shape as ``x``, in original index order.
    """
    j_indices = np.argsort(x)
    z_sorted = x[j_indices]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z_sorted[j] == z_sorted[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1)
        i = j
    t2: NDArray[np.float64] = np.empty(n, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper.
    t2[j_indices] = t + 1
    return t2


def fast_delong(
    predictions_sorted_transposed: NDArray[np.float64],
    label_1_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fast DeLong covariance computation for paired AUROCs.

    Args:
        predictions_sorted_transposed: 2D array of shape
            ``(n_classifiers, n_examples)`` sorted such that examples
            with label ``1`` come first.
        label_1_count: Number of positive examples (``m`` in the paper).

    Returns:
        Tuple of ``(aucs, delong_covariance)``: ``aucs`` is a 1D array of
        per-classifier AUROCs (length ``n_classifiers``), and
        ``delong_covariance`` is the covariance matrix between the
        classifier AUROCs (shape ``(n_classifiers, n_classifiers)``).

    Reference:
        Xu Sun and Weichao Xu, "Fast Implementation of DeLong's
        Algorithm for Comparing the Areas Under Correlated Receiver
        Operating Characteristic Curves", IEEE Signal Processing
        Letters, 21(11), 1389-1393, 2014.
    """
    # Short variables are named as they are in the paper.
    m: int = label_1_count
    n: int = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k: int = predictions_sorted_transposed.shape[0]

    tx: NDArray[np.float64] = np.empty([k, m], dtype=float)
    ty: NDArray[np.float64] = np.empty([k, n], dtype=float)
    tz: NDArray[np.float64] = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs: NDArray[np.float64] = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01: NDArray[np.float64] = (tz[:, :m] - tx[:, :]) / n
    v10: NDArray[np.float64] = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov: NDArray[np.float64] = sx / m + sy / n
    return aucs, delong_cov


def calc_pvalue(
    aucs: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> float:
    """Compute ``log10(p-value)`` from paired AUROCs and their covariance.

    Args:
        aucs: 1D array of two AUCs (one per classifier).
        sigma: Covariance matrix between the two AUC estimators.

    Returns:
        ``log10(p_value)`` for the two-sided test of ``H0: AUROC_a ==
        AUROC_b``. When the AUC difference is exactly zero (identical
        predictors), returns ``0.0`` (i.e., ``p = 1``).
    """
    contrast = np.array([[1, -1]])
    diff = float(np.diff(aucs)[0])
    variance = float(np.dot(np.dot(contrast, sigma), contrast.T)[0, 0])
    if variance <= 0.0 or diff == 0.0:
        # Limiting case: identical predictors or numerically zero
        # variance; the test statistic is zero, p_value is 1.
        return 0.0
    z = float(np.abs(diff) / np.sqrt(variance))
    log10_p: float = float(
        np.log10(2.0) + float(scipy.stats.norm.logsf(z, loc=0, scale=1)) / np.log(10.0)
    )
    return log10_p


def compute_ground_truth_statistics(
    ground_truth: NDArray[np.integer[Any]] | NDArray[np.floating[Any]],
) -> tuple[NDArray[np.intp], int]:
    """Validate and summarize ground-truth labels for DeLong inputs.

    Args:
        ground_truth: Binary labels in ``{0, 1}``.

    Returns:
        Tuple ``(order, label_1_count)`` where ``order`` is the
        permutation that places positives before negatives and
        ``label_1_count`` is the number of positive labels.

    Raises:
        AssertionError: If ``ground_truth`` does not contain exactly the
            values ``{0, 1}``.
    """
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order: NDArray[np.intp] = (-ground_truth).argsort()
    label_1_count: int = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(
    ground_truth: NDArray[np.integer[Any]] | NDArray[np.floating[Any]],
    predictions: NDArray[np.floating[Any]],
) -> tuple[float, NDArray[np.float64]]:
    """Compute single-model AUROC and its DeLong variance.

    Args:
        ground_truth: Binary labels in ``{0, 1}``.
        predictions: Predicted probabilities for the positive class.

    Returns:
        Tuple ``(auc, delong_covariance)``: scalar AUROC and 1x1
        covariance matrix containing its variance.
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delong_cov = fast_delong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return float(aucs[0]), delong_cov


def delong_roc_test(
    ground_truth: NDArray[np.integer[Any]] | NDArray[np.floating[Any]],
    predictions_one: NDArray[np.floating[Any]],
    predictions_two: NDArray[np.floating[Any]],
) -> tuple[float, NDArray[np.float64]]:
    """Compute paired DeLong test statistic for two AUROCs.

    Args:
        ground_truth: Binary labels in ``{0, 1}``.
        predictions_one: Predicted probabilities for model A.
        predictions_two: Predicted probabilities for model B.

    Returns:
        Tuple ``(log10_p_value, aucs)``: ``log10_p_value`` is the
        two-sided log-base-10 p-value for ``H0: AUROC_a == AUROC_b``;
        ``aucs`` is the length-2 array of model AUROCs in the order
        ``[a, b]``.
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delong_cov = fast_delong(predictions_sorted_transposed, label_1_count)
    log10_p = calc_pvalue(aucs, delong_cov)
    return log10_p, aucs


__all__ = [
    "calc_pvalue",
    "compute_ground_truth_statistics",
    "compute_midrank",
    "delong_roc_test",
    "delong_roc_variance",
    "fast_delong",
]
