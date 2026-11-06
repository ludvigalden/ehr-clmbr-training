"""Metric implementations for the evaluation toolkit.

Discrimination, calibration, clinical utility, paired significance
tests, statistical inference, and forgetting summaries. Every public
name is re-exported here so callers import from
:mod:`ehrfm.eval.metrics` regardless of the implementing module.
"""

from __future__ import annotations

from ehrfm.eval.metrics.calibration import (
    CalibrationBin,
    CalibrationFit,
    brier_score,
    calibration_bins,
    calibration_fit,
    expected_calibration_error,
)
from ehrfm.eval.metrics.clinical_utility import (
    DecisionCurve,
    decision_curve,
    net_benefit,
    net_benefit_treat_all,
)
from ehrfm.eval.metrics.discrimination import auprc, auroc
from ehrfm.eval.metrics.forgetting import backward_transfer, forgetting, loss_increase
from ehrfm.eval.metrics.paired import (
    BinomialCI,
    McNemarResult,
    SignTestResult,
    exact_binomial_ci,
    mcnemar,
    sign_test,
)
from ehrfm.eval.metrics.statistical import (
    BonferroniResult,
    DeLongAUROC,
    DeLongComparison,
    bonferroni,
    delong_auroc,
    delong_compare,
)

__all__ = [
    "BinomialCI",
    "BonferroniResult",
    "CalibrationBin",
    "CalibrationFit",
    "DecisionCurve",
    "DeLongAUROC",
    "DeLongComparison",
    "McNemarResult",
    "SignTestResult",
    "auprc",
    "auroc",
    "backward_transfer",
    "bonferroni",
    "brier_score",
    "calibration_bins",
    "calibration_fit",
    "decision_curve",
    "delong_auroc",
    "delong_compare",
    "exact_binomial_ci",
    "expected_calibration_error",
    "forgetting",
    "loss_increase",
    "mcnemar",
    "net_benefit",
    "net_benefit_treat_all",
    "sign_test",
]
