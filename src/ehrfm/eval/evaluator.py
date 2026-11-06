"""The binary-outcome evaluator: one scored split in, one record of metrics out.

:func:`evaluate_binary` measures one model's probabilities against the labels
of one split:

- discrimination: AUROC with a DeLong confidence interval and a stratified
  percentile bootstrap interval, AUPRC with a bootstrap interval;
- calibration: expected calibration error, Brier score, calibration slope and
  calibration-in-the-large intercept, and the reliability bins;
- clinical utility: the decision curve (net benefit of the model, of treating
  everyone and of treating no one) over a threshold grid;
- optionally the same core metrics within each stratum of user-supplied
  stratum columns.

:func:`compare_binary` compares two models scored on the same rows: the
DeLong paired AUROC test, a paired bootstrap interval for the AUROC
difference, and McNemar's test on their decisions at a threshold.

Every record carries its own counts (``n``, ``n_positive``, ``n_negative``,
``n_true_positive`` ...), so the small-cell policy at the tracking boundary
masks a record's metrics whenever one of its counts is small.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ehrfm.eval._bootstrap import bootstrap_ci, paired_bootstrap_ci
from ehrfm.eval.metrics import (
    auprc,
    auroc,
    brier_score,
    calibration_bins,
    calibration_fit,
    decision_curve,
    delong_auroc,
    delong_compare,
    expected_calibration_error,
    mcnemar,
)
from ehrfm.eval.stratification import stratify

DEFAULT_RESAMPLES: Final = 10000
DEFAULT_ALPHA: Final = 0.05
DEFAULT_BINS: Final = 15
DEFAULT_DCA_MIN: Final = 0.01
DEFAULT_DCA_MAX: Final = 0.20
DEFAULT_DCA_STEP: Final = 0.01


class EvaluationSettings(BaseModel):
    """Statistical settings shared by the evaluation hooks."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    resamples: int = Field(default=DEFAULT_RESAMPLES, ge=1)
    """Bootstrap resamples per interval."""
    alpha: float = Field(default=DEFAULT_ALPHA, gt=0, lt=1)
    """Two-sided level: intervals cover ``1 - alpha``."""
    stratified_bootstrap: bool = True
    """Resample positives and negatives separately (keeps the class counts)."""
    calibration_bins: int = Field(default=DEFAULT_BINS, ge=2)
    dca_threshold_min: float = Field(default=DEFAULT_DCA_MIN, gt=0, lt=1)
    dca_threshold_max: float = Field(default=DEFAULT_DCA_MAX, gt=0, lt=1)
    dca_threshold_step: float = Field(default=DEFAULT_DCA_STEP, gt=0, lt=1)

    @model_validator(mode="after")
    def _threshold_range(self) -> Self:
        if self.dca_threshold_min > self.dca_threshold_max:
            raise ValueError("dca_threshold_min must not exceed dca_threshold_max")
        return self


def _finite(value: float) -> float | None:
    return value if np.isfinite(value) else None


def _both_classes(y_true: NDArray[np.float64]) -> bool:
    positives = int(y_true.sum())
    return 0 < positives < y_true.shape[0]


@dataclass(frozen=True, slots=True)
class BinaryEvaluation:
    """Everything :func:`evaluate_binary` measured on one split."""

    n: int
    n_positive: int
    n_negative: int
    metrics: dict[str, float]
    """Scalar metrics (non-finite where undefined)."""
    calibration: list[dict[str, JsonValue]]
    decision_curve: list[dict[str, JsonValue]]
    strata: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    bootstrap: dict[str, NDArray[np.float64]] = field(
        default_factory=dict[str, NDArray[np.float64]]
    )
    """Bootstrap distributions by metric, for the hook's output files (never logged)."""

    def record(self) -> dict[str, JsonValue]:
        """The JSON record: counts, metrics (non-finite as null), bins, curve and strata."""
        record: dict[str, JsonValue] = {
            "n": self.n,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }
        record.update({name: _finite(value) for name, value in self.metrics.items()})
        record["calibration"] = list(self.calibration)
        record["decision_curve"] = list(self.decision_curve)
        if self.strata:
            record["strata"] = list(self.strata)
        return record

    def flat_metrics(self, prefix: str) -> dict[str, float]:
        """Counts and finite scalar metrics as ``<prefix>/<name>``."""
        flat = {
            f"{prefix}/n": float(self.n),
            f"{prefix}/n_positive": float(self.n_positive),
            f"{prefix}/n_negative": float(self.n_negative),
        }
        flat.update(
            {
                f"{prefix}/{name}": value
                for name, value in self.metrics.items()
                if np.isfinite(value)
            }
        )
        return flat


def core_metrics(y_true: NDArray[np.float64], y_prob: NDArray[np.float64]) -> dict[str, float]:
    """Point estimates without intervals (used per stratum)."""
    if y_true.shape[0] == 0:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
    return {
        "auroc": auroc(y_true, y_prob),
        "auprc": auprc(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
    }


def _calibration_records(
    y_true: NDArray[np.float64], y_prob: NDArray[np.float64], bins: int
) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
    for item in calibration_bins(y_true, y_prob, bins=bins):
        in_bin = (y_prob >= item.lower) & (
            (y_prob <= item.upper) if item.upper >= 1.0 else (y_prob < item.upper)
        )
        positives = int(y_true[in_bin].sum())
        records.append(
            {
                "lower": item.lower,
                "upper": item.upper,
                "n": item.n,
                "n_positive": positives,
                "mean_predicted": item.mean_predicted,
                "observed_rate": item.observed_rate,
            }
        )
    return records


def _decision_curve_records(
    y_true: NDArray[np.float64], y_prob: NDArray[np.float64], settings: EvaluationSettings
) -> list[dict[str, JsonValue]]:
    curve = decision_curve(
        y_true,
        y_prob,
        threshold_min=settings.dca_threshold_min,
        threshold_max=settings.dca_threshold_max,
        step=settings.dca_threshold_step,
    )
    records: list[dict[str, JsonValue]] = []
    for index, threshold in enumerate(curve.thresholds.tolist()):
        treated = y_prob >= threshold
        records.append(
            {
                "threshold": threshold,
                "n_true_positive": int((treated & (y_true == 1)).sum()),
                "n_false_positive": int((treated & (y_true == 0)).sum()),
                "net_benefit_model": float(curve.net_benefit_model[index]),
                "net_benefit_all": float(curve.net_benefit_all[index]),
                "net_benefit_none": float(curve.net_benefit_none),
            }
        )
    return records


def evaluate_binary(
    y_true: NDArray[np.float64],
    y_prob: NDArray[np.float64],
    *,
    seed: int,
    settings: EvaluationSettings | None = None,
    strata: Mapping[str, Sequence[str] | NDArray[np.str_]] | None = None,
) -> BinaryEvaluation:
    """Measure probabilities ``y_prob`` against boolean labels ``y_true`` (0/1)."""
    settings = settings or EvaluationSettings()
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.ndim != 1 or y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must be 1-D arrays of the same length")
    n = int(y_true.shape[0])
    positives = int(y_true.sum())
    metrics: dict[str, float] = {
        **core_metrics(y_true, y_prob),
        "prevalence": positives / n if n else float("nan"),
        "ece": float("nan"),
        "calibration_slope": float("nan"),
        "calibration_intercept": float("nan"),
    }
    bootstrap: dict[str, NDArray[np.float64]] = {}
    calibration: list[dict[str, JsonValue]] = []
    curve: list[dict[str, JsonValue]] = []
    if n:
        metrics["ece"] = expected_calibration_error(y_true, y_prob, bins=settings.calibration_bins)
        calibration = _calibration_records(y_true, y_prob, settings.calibration_bins)
        curve = _decision_curve_records(y_true, y_prob, settings)
    if _both_classes(y_true):
        fit = calibration_fit(y_true, y_prob)
        metrics["calibration_slope"] = fit.slope
        metrics["calibration_intercept"] = fit.intercept
        if min(positives, n - positives) >= 2:
            # The DeLong variance needs two rows per class (sample variances).
            delong = delong_auroc(y_true, y_prob, alpha=settings.alpha)
            metrics["auroc_se"] = delong.standard_error
            metrics["auroc_delong_low"] = delong.low
            metrics["auroc_delong_high"] = delong.high
        for name, metric in (("auroc", auroc), ("auprc", auprc)):
            interval = bootstrap_ci(
                metric,
                y_true,
                y_prob,
                resamples=settings.resamples,
                alpha=settings.alpha,
                seed=seed,
                stratified=settings.stratified_bootstrap,
            )
            metrics[f"{name}_low"] = interval.low
            metrics[f"{name}_high"] = interval.high
            bootstrap[name] = interval.distribution
    strata_records: list[dict[str, JsonValue]] = []
    if strata:
        for result in stratify(y_true, y_prob, strata, core_metrics):
            strata_records.append(dict(result.to_record()))
    return BinaryEvaluation(
        n=n,
        n_positive=positives,
        n_negative=n - positives,
        metrics=metrics,
        calibration=calibration,
        decision_curve=curve,
        strata=strata_records,
        bootstrap=bootstrap,
    )


@dataclass(frozen=True, slots=True)
class PairedEvaluation:
    """Two models compared on the same rows."""

    n: int
    n_positive: int
    n_negative: int
    metrics: dict[str, float]
    mcnemar: dict[str, JsonValue]
    difference_distribution: NDArray[np.float64]

    def record(self) -> dict[str, JsonValue]:
        """The JSON record (non-finite metrics as null)."""
        record: dict[str, JsonValue] = {
            "n": self.n,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }
        record.update({name: _finite(value) for name, value in self.metrics.items()})
        record["mcnemar"] = dict(self.mcnemar)
        return record

    def flat_metrics(self, prefix: str) -> dict[str, float]:
        """Counts and finite scalar metrics as ``<prefix>/<name>``."""
        flat = {
            f"{prefix}/n": float(self.n),
            f"{prefix}/n_positive": float(self.n_positive),
            f"{prefix}/n_negative": float(self.n_negative),
        }
        flat.update(
            {
                f"{prefix}/{name}": value
                for name, value in self.metrics.items()
                if np.isfinite(value)
            }
        )
        return flat


def compare_binary(
    y_true: NDArray[np.float64],
    prob_a: NDArray[np.float64],
    prob_b: NDArray[np.float64],
    *,
    seed: int,
    threshold: float = 0.5,
    settings: EvaluationSettings | None = None,
) -> PairedEvaluation:
    """Compare model ``a`` with model ``b`` (differences are ``a - b``)."""
    settings = settings or EvaluationSettings()
    y_true = np.asarray(y_true, dtype=np.float64)
    if not (y_true.shape == prob_a.shape == prob_b.shape) or y_true.ndim != 1:
        raise ValueError("labels and both probability arrays must be 1-D and the same length")
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between 0 and 1")
    n = int(y_true.shape[0])
    positives = int(y_true.sum())
    nan = float("nan")
    metrics: dict[str, float] = {
        "auroc_a": nan,
        "auroc_b": nan,
        "auroc_difference": nan,
        "delong_z": nan,
        "delong_p": nan,
        "auroc_difference_low": nan,
        "auroc_difference_high": nan,
    }
    distribution = np.empty(0, dtype=np.float64)
    if min(positives, n - positives) >= 2:
        # The DeLong covariance needs two rows per class (sample covariances).
        delong = delong_compare(y_true, prob_a, prob_b, alpha=settings.alpha)
        metrics.update(
            auroc_a=delong.auroc_a,
            auroc_b=delong.auroc_b,
            auroc_difference=delong.difference,
            delong_z=delong.z,
            delong_p=delong.p_value,
            auroc_difference_delong_low=delong.low,
            auroc_difference_delong_high=delong.high,
        )
        interval = paired_bootstrap_ci(
            auroc,
            y_true,
            prob_a,
            prob_b,
            resamples=settings.resamples,
            alpha=settings.alpha,
            seed=seed,
            stratified=settings.stratified_bootstrap,
        )
        metrics["auroc_difference_low"] = interval.low
        metrics["auroc_difference_high"] = interval.high
        distribution = interval.distribution
    labels = y_true == 1
    test = mcnemar((prob_a >= threshold) == labels, (prob_b >= threshold) == labels)
    paired: dict[str, JsonValue] = {
        "threshold": threshold,
        "n_both_correct": test.n_both_correct,
        "n_a_only_correct": test.n_a_only_correct,
        "n_b_only_correct": test.n_b_only_correct,
        "n_both_wrong": test.n_both_wrong,
        "statistic": _finite(test.statistic),
        "p_value": _finite(test.p_value),
        "p_value_exact": _finite(test.p_value_exact),
    }
    return PairedEvaluation(
        n=n,
        n_positive=positives,
        n_negative=n - positives,
        metrics=metrics,
        mcnemar=paired,
        difference_distribution=distribution,
    )


__all__ = [
    "BinaryEvaluation",
    "EvaluationSettings",
    "PairedEvaluation",
    "compare_binary",
    "core_metrics",
    "evaluate_binary",
]
