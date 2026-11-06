"""Known-value tests for decision-curve analysis.

Hand table (10 rows, prevalence 5/10):

    y_prob = [0.05, 0.15, 0.20, 0.25, 0.30, 0.45, 0.55, 0.60, 0.70, 0.90]
    y      = [0,    1,    0,    1,    0,    0,    1,    0,    1,    1   ]

At threshold 0.10 exactly 9 of the 10 rows are predicted positive
(0.05 fails the ``>=`` test): TP = 5 (0.15, 0.25, 0.55, 0.70, 0.90),
FP = 4 (0.20, 0.30, 0.45, 0.60). NB = 5/10 - 4/10 * (0.1/0.9) =
0.5 - 4/90 = 41/90. Treat-all = 5/10 - 5/10 * (0.1/0.9) = 4/9.

At threshold 0.50 the predicted positives are 0.55, 0.60, 0.70, 0.90:
TP = 3 (0.55, 0.70, 0.90), FP = 1 (0.60). NB = 3/10 - 1/10 * (0.5/0.5)
= 0.2. Treat-all = 0.5 - 0.5 * 1 = 0.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from ehrfm.eval.metrics import decision_curve, net_benefit, net_benefit_treat_all


def _hand_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y_prob = np.array([0.05, 0.15, 0.20, 0.25, 0.30, 0.45, 0.55, 0.60, 0.70, 0.90])
    y_true = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    return y_true, y_prob


def test_net_benefit_hand_table() -> None:
    y_true, y_prob = _hand_data()
    assert net_benefit(y_true, y_prob, 0.10) == pytest.approx(41.0 / 90.0, abs=1e-12)
    assert net_benefit(y_true, y_prob, 0.50) == pytest.approx(0.2, abs=1e-12)


def test_treat_all_hand_values() -> None:
    # Prevalence 0.5: at 0.1 -> 0.5 - 0.5 * (1/9) = 4/9; at 0.5 -> 0.0.
    assert net_benefit_treat_all(0.5, 0.1) == pytest.approx(4.0 / 9.0, abs=1e-12)
    assert net_benefit_treat_all(0.5, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert net_benefit_treat_all(0.3, 0.2) == pytest.approx(0.125, abs=1e-12)


def test_decision_curve_on_hand_table_with_explicit_thresholds() -> None:
    y_true, y_prob = _hand_data()
    result = decision_curve(y_true, y_prob, thresholds=np.array([0.10, 0.50]))
    assert result.thresholds.tolist() == [0.10, 0.50]
    assert result.net_benefit_model[0] == pytest.approx(41.0 / 90.0, abs=1e-12)
    assert result.net_benefit_model[1] == pytest.approx(0.2, abs=1e-12)
    assert result.net_benefit_all[0] == pytest.approx(4.0 / 9.0, abs=1e-12)
    assert result.net_benefit_all[1] == pytest.approx(0.0, abs=1e-12)
    assert result.net_benefit_none == 0.0
    assert result.prevalence == pytest.approx(0.5, abs=1e-15)


def test_default_grid_is_rounded_to_step() -> None:
    y_true, y_prob = _hand_data()
    result = decision_curve(y_true, y_prob)
    assert result.thresholds[0] == pytest.approx(0.01, abs=1e-12)
    assert result.thresholds[-1] == pytest.approx(0.20, abs=1e-12)
    assert result.thresholds.size == 20
    assert np.all(np.abs(result.thresholds - np.round(result.thresholds, 2)) < 1e-10)


def test_agreement_with_dcurves_library() -> None:
    # Cross-check the net-benefit formula and the >= threshold
    # convention against the dcurves package (imported here to keep the
    # slow pandas/matplotlib import out of collection).
    from dcurves import dca

    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(rng.random(n) * 0.5 + 0.05, 0.0, 1.0)
    frame = pd.DataFrame({"y": y, "model": p})
    thresholds = np.round(np.linspace(0.01, 0.20, 20), 10)

    reference = dca(
        data=frame,
        outcome="y",
        modelnames=["model"],
        thresholds=thresholds,
    )

    result = decision_curve(y.astype(np.float64), p, thresholds=thresholds)
    model_rows = reference[reference["model"] == "model"]
    all_rows = reference[reference["model"] == "all"]
    none_rows = reference[reference["model"] == "none"]

    assert np.allclose(model_rows["threshold"].to_numpy(), result.thresholds, atol=1e-12)
    assert np.allclose(model_rows["net_benefit"].to_numpy(), result.net_benefit_model, atol=1e-12)
    assert np.allclose(all_rows["net_benefit"].to_numpy(), result.net_benefit_all, atol=1e-12)
    assert np.allclose(none_rows["net_benefit"].to_numpy(), 0.0, atol=1e-12)


def test_invalid_thresholds_and_ranges_raise() -> None:
    y_true, y_prob = _hand_data()
    with pytest.raises(ValueError, match="strictly inside \\(0, 1\\)"):
        net_benefit(y_true, y_prob, 0.0)
    with pytest.raises(ValueError, match="strictly inside \\(0, 1\\)"):
        net_benefit(y_true, y_prob, 1.0)
    with pytest.raises(ValueError, match="prevalence"):
        net_benefit_treat_all(1.5, 0.1)
    with pytest.raises(ValueError, match="strictly inside \\(0, 1\\)"):
        decision_curve(y_true, y_prob, thresholds=np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="threshold_max"):
        decision_curve(y_true, y_prob, threshold_min=0.05, threshold_max=1.0)
    with pytest.raises(ValueError, match="step"):
        decision_curve(y_true, y_prob, step=0.0)
