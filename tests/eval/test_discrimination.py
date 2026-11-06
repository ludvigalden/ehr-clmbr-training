"""Known-value tests for the discrimination metrics.

Hand table (4 rows):

    y_true = [0, 0, 1, 1]
    y_score = [0.2, 0.6, 0.4, 0.8]

AUROC: the four positive-negative pairs are (0.4 vs 0.2) win, (0.4 vs
0.6) loss, (0.8 vs 0.2) win, (0.8 vs 0.6) win -> 3/4 = 0.75.

AUPRC: sorting by descending score gives 0.8 (y=1), 0.6 (y=0), 0.4
(y=1), 0.2 (y=0); precision at the two positive thresholds is 1/1 and
2/3 with recall steps 1/2 each, so average precision = 1/2 * 1 + 1/2 *
2/3 = 5/6.
"""

from __future__ import annotations

import numpy as np
import pytest

from ehrfm.eval.metrics import auprc, auroc


def test_auroc_hand_table() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    s = np.array([0.2, 0.6, 0.4, 0.8])
    assert auroc(y, s) == pytest.approx(0.75, abs=1e-12)


def test_auprc_hand_table() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    s = np.array([0.2, 0.6, 0.4, 0.8])
    assert auprc(y, s) == pytest.approx(5.0 / 6.0, abs=1e-12)


def test_auroc_single_class_is_nan() -> None:
    y = np.ones(6)
    s = np.linspace(0.0, 1.0, 6)
    assert np.isnan(auroc(y, s))


def test_auprc_without_positives_is_nan() -> None:
    y = np.zeros(6)
    s = np.linspace(0.0, 1.0, 6)
    assert np.isnan(auprc(y, s))


def test_validation_errors() -> None:
    y = np.array([0.0, 1.0])
    s = np.array([0.1])
    with pytest.raises(ValueError, match="same length"):
        auroc(y, s)
    with pytest.raises(ValueError, match="exactly the values"):
        auroc(np.array([0.0, 2.0]), np.array([0.1, 0.9]))
    with pytest.raises(ValueError, match="finite"):
        auprc(y, np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="1-D"):
        auprc(np.array([[0.0], [1.0]]), np.array([0.1, 0.9]))
