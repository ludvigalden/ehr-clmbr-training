"""Known-value tests for calibration metrics.

Hand tables (bins=2 for ECE, so bin 0 covers [0, 0.5) and bin 1 covers
[0.5, 1]):

- ECE rows ``(y_prob, y)``: (0.10, 0), (0.15, 0), (0.25, 0), (0.75, 1),
  (0.85, 1), (0.90, 1). Bin 0: n = 3, mean predicted 0.5/3 = 0.1666...,
  observed 0. Bin 1: n = 3, mean predicted 2.5/3 = 0.8333..., observed
  1. ECE = 3/6 * |0.1666... - 0| + 3/6 * |0.8333... - 1| = 1/6.
- Brier on the same rows: ((0.10)^2 + (0.15)^2 + (0.25)^2 + (0.25)^2
  + (0.15)^2 + (0.10)^2) / 6 = 0.19 / 6.
- Calibration-in-the-large with constant p = q: every fitted probability
  equals sigmoid(a + logit(q)), so the MLE sets it to the observed rate:
  a = logit(y_bar) - logit(q) exactly. With y_bar = 0.75 and q = 0.3:
  a = logit(0.75) - logit(0.3) = ln(7).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from ehrfm.eval.metrics import (
    brier_score,
    calibration_bins,
    calibration_fit,
    expected_calibration_error,
)


def _ece_rows() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y_prob = np.array([0.10, 0.15, 0.25, 0.75, 0.85, 0.90])
    y_true = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return y_true, y_prob


def test_expected_calibration_error_hand_table() -> None:
    y_true, y_prob = _ece_rows()
    ece = expected_calibration_error(y_true, y_prob, bins=2)
    assert ece == pytest.approx(1.0 / 6.0, abs=1e-12)


def test_brier_score_hand_table() -> None:
    y_true, y_prob = _ece_rows()
    assert brier_score(y_true, y_prob) == pytest.approx(0.19 / 6.0, abs=1e-12)


def test_calibration_bins_structure_and_last_bin_includes_one() -> None:
    y_true = np.array([1.0, 0.0])
    y_prob = np.array([1.0, 0.3])
    rows = calibration_bins(y_true, y_prob, bins=4)
    assert len(rows) == 4
    assert sum(row.n for row in rows) == 2
    assert rows[-1].n == 1  # p exactly 1.0 lands in the last bin
    assert rows[-1].mean_predicted == pytest.approx(1.0, abs=1e-15)
    assert rows[-1].observed_rate == pytest.approx(1.0, abs=1e-15)
    assert rows[0].upper == pytest.approx(0.25, abs=1e-15)
    empty = [row for row in rows if row.n == 0]
    assert len(empty) == 2
    for row in empty:
        assert row.mean_predicted is None
        assert row.observed_rate is None


def test_calibration_bins_first_bin_lower_edge() -> None:
    y_true = np.array([0.0])
    y_prob = np.array([0.0])
    rows = calibration_bins(y_true, y_prob, bins=5)
    assert rows[0].n == 1
    assert rows[0].lower == 0.0
    assert rows[0].upper == pytest.approx(0.2, abs=1e-15)


def test_calibration_fit_slope_matches_sklearn() -> None:
    rng = np.random.default_rng(11)
    n = 4000
    latent = rng.normal(0.0, 1.0, n)
    y = (rng.random(n) < expit(latent)).astype(np.float64)
    p_reported = expit(0.6 * latent + 0.3)
    logit_reported: NDArray[np.float64] = logit(np.clip(p_reported, 1e-9, 1.0 - 1e-9))

    # Unpenalised logistic regression: `C=np.inf` is the sanctioned
    # unpenalised configuration on sklearn >= 1.8 (`penalty=None` is
    # deprecated there); the installed stubs type `C` as a bounded
    # float, so the mismatch is silenced locally.
    reference = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10000)  # pyright: ignore[reportArgumentType]
    reference.fit(logit_reported.reshape(-1, 1), y.astype(int))

    result = calibration_fit(y, p_reported)
    assert result.converged
    reference_slope = float(np.asarray(reference.coef_).ravel()[0])
    # Only the slope is comparable: the reference fit's intercept belongs
    # to the two-parameter model, while `result.intercept` is the
    # calibration-in-the-large offset fit (a different estimand).
    assert result.slope == pytest.approx(reference_slope, abs=1e-5)


def test_calibration_in_the_large_hand_case() -> None:
    # Constant p = 0.3 with observed rate 0.75 gives
    # intercept = logit(0.75) - logit(0.3) = ln(7) exactly. With a
    # constant prediction the slope is unidentifiable, so the slope is
    # nan and converged is False while the intercept stays exact.
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])  # mean 0.75
    p = np.full(8, 0.3)
    result = calibration_fit(y, p)
    assert result.intercept == pytest.approx(np.log(7.0), abs=1e-9)
    assert np.isnan(result.slope)
    assert result.converged is False


def test_well_calibrated_sample_recovers_unit_slope() -> None:
    rng = np.random.default_rng(23)
    n = 200_000
    p = rng.uniform(0.01, 0.99, n)
    y = (rng.random(n) < p).astype(np.float64)
    result = calibration_fit(y, p)
    assert result.converged
    assert abs(result.slope - 1.0) < 0.1
    assert abs(result.intercept) < 0.1


def test_single_class_returns_nan_and_not_converged() -> None:
    y = np.ones(10)
    p = np.linspace(0.1, 0.9, 10)
    result = calibration_fit(y, p)
    assert np.isnan(result.slope)
    assert np.isnan(result.intercept)
    assert result.converged is False


def test_invalid_inputs_raise() -> None:
    y = np.array([0.0, 1.0])
    p_bad = np.array([0.1, 1.2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score(y, p_bad)
    with pytest.raises(ValueError, match="bins"):
        expected_calibration_error(y, np.array([0.1, 0.9]), bins=0)
    with pytest.raises(ValueError, match="exactly the values"):
        expected_calibration_error(np.array([0.0, 2.0]), np.array([0.1, 0.9]))
