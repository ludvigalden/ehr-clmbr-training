"""The evaluator and the evaluation hooks inside the experiment pipeline.

The pipeline tests run the toy experiment (``configs/experiment/toy.yaml``):
toy inputs, 18 pretraining steps, representations, the three toy heads and all
four evaluation hooks, then a rerun that must be a full cache hit. They check
the hook outputs themselves and that no count below the small-cell threshold
reaches the MLflow store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ehrfm.config.experiment import SmallCellPolicy
from ehrfm.config.loader import load_experiment_config
from ehrfm.eval.evaluator import EvaluationSettings, compare_binary, evaluate_binary
from ehrfm.eval.hooks import EVAL_HOOKS, HEAD_HOOKS
from ehrfm.eval.metrics import auroc
from ehrfm.eval.predictions import load_predictions
from ehrfm.experiment.hooks import HookError, registered_hooks
from ehrfm.experiment.pipeline import PipelineResult, run_experiment
from ehrfm.experiment.suppression import is_count_field, suppress
from ehrfm.inference.head_loader import load_head
from tests.experiment.store import small_counts, stored_counts, stored_metric_names

TOY = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "toy.yaml"
TASK = "demo_outcome_30d"
FAST = EvaluationSettings(resamples=200)


def _data(n: int, seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.float64)
    p = np.clip(0.3 + 0.35 * (y - 0.3) + rng.normal(0, 0.15, n), 0.001, 0.999)
    return y, p


# --------------------------------------------------------------------------- evaluator


def test_evaluate_binary_reports_every_metric_and_is_seeded() -> None:
    y, p = _data(300, 1)
    first = evaluate_binary(y, p, seed=4, settings=FAST, strata={"g": ["a", "b", "c"] * 100})
    second = evaluate_binary(y, p, seed=4, settings=FAST, strata={"g": ["a", "b", "c"] * 100})
    assert first.metrics == second.metrics
    assert first.n == 300 and first.n_positive + first.n_negative == 300
    m = first.metrics
    assert m["auroc"] == pytest.approx(auroc(y, p))
    assert m["auroc_low"] <= m["auroc"] <= m["auroc_high"]
    assert m["auroc_delong_low"] <= m["auroc"] <= m["auroc_delong_high"]
    for name in ("auprc_low", "auprc_high", "ece", "brier", "calibration_slope", "auroc_se"):
        assert np.isfinite(m[name]), name
    assert len(first.calibration) == FAST.calibration_bins
    assert sum(int(b["n"]) for b in _json(first.calibration)) == 300
    assert len(first.decision_curve) == 20
    assert [s["value"] for s in first.strata] == ["a", "b", "c"]
    assert first.bootstrap["auroc"].shape == (200,)
    other = evaluate_binary(y, p, seed=5, settings=FAST)
    assert other.metrics["auroc_low"] != m["auroc_low"]


def test_evaluate_binary_with_one_class_leaves_intervals_undefined() -> None:
    y = np.zeros(10)
    result = evaluate_binary(y, np.full(10, 0.2), seed=0, settings=FAST)
    assert np.isnan(result.metrics["auroc"]) and "auroc_low" not in result.metrics
    assert result.record()["auroc"] is None
    assert result.metrics["brier"] == pytest.approx(0.04)


def test_compare_binary_identical_models() -> None:
    y, p = _data(200, 2)
    same = compare_binary(y, p, p, seed=1, settings=FAST)
    assert same.metrics["auroc_difference"] == 0.0 and same.metrics["delong_p"] == 1.0
    assert same.metrics["auroc_difference_low"] == same.metrics["auroc_difference_high"] == 0.0
    assert same.mcnemar["n_a_only_correct"] == same.mcnemar["n_b_only_correct"] == 0


def test_compare_binary_detects_a_better_model() -> None:
    y, p = _data(400, 3)
    noise = np.random.default_rng(9).random(400)
    result = compare_binary(y, p, noise, seed=1, settings=FAST)
    assert result.metrics["auroc_difference"] > 0.2
    assert result.metrics["delong_p"] < 1e-6
    assert result.metrics["auroc_difference_low"] > 0


def _json(value: object) -> Any:
    """A JSON value as plain untyped data, for indexing in assertions."""
    return json.loads(json.dumps(value))


def test_records_put_counts_where_the_policy_masks_small_strata() -> None:
    y, p = _data(120, 4)
    groups = ["big"] * 116 + ["small"] * 4
    record = _json(evaluate_binary(y, p, seed=0, settings=FAST, strata={"g": groups}).record())
    masked = _json(suppress(record, SmallCellPolicy()).value)
    strata = masked["strata"]
    small = next(s for s in strata if s["value"] == "small")
    assert small["n"] is None and small["auroc"] is None and small["brier"] is None
    for row in record["calibration"]:
        assert any(is_count_field(key, SmallCellPolicy()) for key in row)


# --------------------------------------------------------------------------- pipeline


def test_hooks_are_registered_on_import() -> None:
    assert set(HEAD_HOOKS) <= set(registered_hooks("head"))
    assert set(EVAL_HOOKS) <= set(registered_hooks("eval"))


@pytest.fixture(scope="module")
def work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("eval-experiment")


@pytest.fixture(scope="module")
def toy(work: Path) -> PipelineResult:
    return run_experiment(load_experiment_config(TOY, [f"experiment.work_dir={work}"]))


def test_toy_experiment_runs_every_head_and_evaluation(toy: PipelineResult) -> None:
    statuses = {stage.name: stage.status for stage in toy.stages}
    for name in (*HEAD_HOOKS, *EVAL_HOOKS):
        kind = "head" if name in HEAD_HOOKS else "eval"
        assert statuses[f"{kind}-{name}"] == "computed", name
    assert toy.tracking_status == "logged"


def test_heads_publish_predictions_and_the_probe_is_servable(toy: PipelineResult) -> None:
    held_out = toy.representations.load(TASK, "held_out")
    for name in HEAD_HOOKS:
        predictions = load_predictions(toy.heads[name].output_dir, held_out)
        assert predictions.head == name and len(predictions) == len(held_out)
    loaded = load_head(
        toy.heads["logistic_probe"].output_dir / "head",
        encoder_hidden_size=held_out.meta.hidden_size,
        encoder_weights_sha256=toy.model.weights_sha256,
    )
    import torch

    with torch.no_grad():
        served = loaded.head.predict(torch.from_numpy(held_out.embeddings)).numpy()
    published = load_predictions(toy.heads["logistic_probe"].output_dir, held_out)
    np.testing.assert_allclose(served, published.y_prob, atol=1e-5)


def test_binary_hook_result_matches_a_direct_evaluation(toy: PipelineResult) -> None:
    result = toy.evaluations["binary"].result
    held_out = toy.representations.load(TASK, "held_out")
    predictions = load_predictions(toy.heads["logistic_probe"].output_dir, held_out)
    direct = evaluate_binary(
        predictions.y_true,
        predictions.y_prob,
        seed=toy.config.experiment.seed,
        settings=EvaluationSettings(resamples=1000),
    )
    record = _json(result.summary)["results"]["logistic_probe"]["held_out"]
    assert record["n"] == direct.n and record["n_positive"] == direct.n_positive
    assert {s["column"] for s in record["strata"]} == {"sex", "age"}
    files = {path.name for path in toy.evaluations["binary"].output_dir.iterdir()}
    assert "logistic_probe.held_out.json" in files and "age_baseline.tuning.json" in files


def test_few_shot_and_forgetting_results(toy: PipelineResult) -> None:
    curve = _json(toy.evaluations["few_shot"].result.summary)["curve"]
    assert [row["shots"] for row in curve] == [1, 2, 4, 8]
    forgetting = toy.evaluations["forgetting"].result
    assert forgetting.summary["applicable"] is True
    loss = forgetting.metrics["held_out/continued_loss"] - forgetting.metrics["held_out/base_loss"]
    assert forgetting.metrics["held_out/loss_increase"] == pytest.approx(loss)
    assert forgetting.summary["base_weights_sha256"] != toy.model.weights_sha256


def test_no_small_count_reaches_the_tracking_store(toy: PipelineResult, work: Path) -> None:
    policy = toy.config.experiment.suppression
    mlruns = work / "mlruns"
    counted = stored_counts(mlruns, policy)
    eval_counts = [c for c in counted if "eval" in c.where or "hooks" in c.where]
    assert eval_counts, "the evaluation hooks logged no counts at all"
    assert small_counts(mlruns, policy) == []
    # The toy held-out split has fewer than 5 positives, so its metrics must be masked.
    held_out = toy.representations.load(TASK, "held_out")
    assert held_out.labels is not None and held_out.labels.sum() < policy.threshold
    names = stored_metric_names(mlruns)
    assert "eval/binary/logistic_probe/held_out/auroc" not in names
    assert "eval/binary/logistic_probe/held_out/n_positive" not in names
    summary = json.loads(next(mlruns.glob("*/*/artifacts/hooks/eval/binary.json")).read_text())
    record = summary["results"]["logistic_probe"]["held_out"]
    assert record["n_positive"] is None and record["auroc"] is None


def test_positive_control_finds_small_counts_without_suppression(
    toy: PipelineResult, tmp_path: Path
) -> None:
    del toy
    permissive = SmallCellPolicy(threshold=1, suppress_zero=False)
    config = load_experiment_config(
        TOY,
        [
            f"experiment.work_dir={tmp_path}",
            "experiment.suppression.threshold=1",
            "experiment.suppression.suppress_zero=false",
            "eval.hooks=[{name: binary, options: {settings: {resamples: 50}}}]",
            "experiment.pretraining.enabled=false",
        ],
    )
    run_experiment(config)
    found = small_counts(tmp_path / "mlruns", SmallCellPolicy())
    assert any("binary" in count.where for count in found)
    assert small_counts(tmp_path / "mlruns", permissive) == []


def test_rerun_is_a_cache_hit(toy: PipelineResult, work: Path) -> None:
    again = run_experiment(load_experiment_config(TOY, [f"experiment.work_dir={work}"]))
    assert again.cache_hit and again.tracking_run_id == toy.tracking_run_id


def test_bad_options_fail_the_hook(tmp_path: Path) -> None:
    config = load_experiment_config(
        TOY,
        [
            f"experiment.work_dir={tmp_path}",
            "experiment.pretraining.enabled=false",
            "experiment.tracking.enabled=false",
            "tasks.heads=[{name: logistic_probe}]",
            "eval.hooks=[{name: binary, options: {resample: 3}}]",
        ],
    )
    with pytest.raises(HookError, match="invalid options"):
        run_experiment(config)
