"""Task heads: fitting, the registry and loader for serving, and the pipeline hook.

Fitting and the registry run on small synthetic arrays. One pipeline run on
the toy inputs (base checkpoint, no pretraining) checks that the
``linear_probe`` hook trains a head, that the evaluation hooks score it, that
it is servable, and that no small count from it reaches the tracking store.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from typer.testing import CliRunner

from ehrfm.cli import app
from ehrfm.config.heads import LinearProbeConfig
from ehrfm.config.loader import load_experiment_config
from ehrfm.eval.fewshot import Scorer, run_few_shot
from ehrfm.eval.metrics import auroc
from ehrfm.eval.predictions import load_predictions
from ehrfm.experiment.hooks import HookError, registered_hooks
from ehrfm.experiment.pipeline import PipelineResult, run_experiment
from ehrfm.heads.hooks import LINEAR_PROBE
from ehrfm.heads.probe import ProbeError, fit_linear_probe, predict_probabilities
from ehrfm.heads.store import find_trained_head, trained_heads
from ehrfm.inference.head_loader import HEAD_JSON, HEAD_WEIGHTS, HeadLoadError, save_head
from ehrfm.inference.registry import EncoderIdentity, HeadRegistry, require_head_name
from tests.experiment.store import small_counts, stored_metric_names

TOY = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "toy.yaml"
TASK = "demo_outcome_30d"
ENCODER = "a" * 64
OTHER_ENCODER = "b" * 64
SMALL = LinearProbeConfig(c_grid=(0.01, 1.0, 100.0))


def _data(n: int, d: int, seed: int) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    direction = rng.normal(size=d)
    y = (x @ direction + rng.normal(scale=1.0, size=n) > 0.5).astype(np.float64)
    return x, y


# --------------------------------------------------------------------------- fitting


def test_fit_is_deterministic_to_the_byte(tmp_path: Path) -> None:
    every, labels = _data(180, 8, 0)
    x, y, xt, yt = every[:120], labels[:120], every[120:], labels[120:]
    first = fit_linear_probe(TASK, x, y, xt, yt, config=SMALL)
    second = fit_linear_probe(TASK, x, y, xt, yt, config=SMALL)
    assert first.c == second.c and first.tuning_auroc == second.tuning_auroc
    a = save_head(first.head, tmp_path / "a", encoder_weights_sha256=ENCODER)
    b = save_head(second.head, tmp_path / "b", encoder_weights_sha256=ENCODER)
    assert a.weights_sha256 == b.weights_sha256
    assert (tmp_path / "a" / HEAD_JSON).read_bytes() == (tmp_path / "b" / HEAD_JSON).read_bytes()


def test_fit_learns_and_selects_on_tuning() -> None:
    every, labels = _data(600, 6, 2)
    x, y, xt, yt = every[:400], labels[:400], every[400:], labels[400:]
    fit = fit_linear_probe(TASK, x, y, xt, yt, config=SMALL)
    assert set(fit.tuning_auroc) == set(SMALL.c_grid)
    assert fit.tuning_auroc[fit.c] == max(fit.tuning_auroc.values())
    assert auroc(yt, fit.predict(xt)) > 0.85
    assert fit.converged and fit.head.spec.input_dim == 6
    # The saved head takes raw (unstandardised) representations.
    shifted = x * 50.0 + 3.0
    raw = fit_linear_probe(TASK, shifted, y, config=SMALL)
    assert auroc(y, raw.predict(shifted)) > 0.85


def test_fit_matches_scikit_learn_objective() -> None:
    from sklearn.linear_model import LogisticRegression

    x, y = _data(300, 5, 3)
    config = LinearProbeConfig(c_grid=(0.5,), standardize=False, class_weight="none")
    fit = fit_linear_probe(TASK, x, y, config=config)
    reference = LogisticRegression(C=0.5, tol=1e-10, max_iter=10000).fit(x.astype(np.float64), y)
    expected = reference.predict_proba(x.astype(np.float64))[:, 1]
    np.testing.assert_allclose(fit.predict(x), expected, atol=1e-5)


def test_fit_without_a_usable_tuning_split_uses_the_middle_strength() -> None:
    x, y = _data(80, 4, 4)
    fit = fit_linear_probe(TASK, x, y, x[:5], np.zeros(5), config=SMALL)
    assert fit.c == 1.0 and fit.tuning_auroc == {}


def test_fit_rejects_bad_inputs() -> None:
    x, y = _data(40, 3, 5)
    with pytest.raises(ProbeError, match="both classes"):
        fit_linear_probe(TASK, x, np.zeros(40))
    with pytest.raises(ProbeError, match="0 or 1"):
        fit_linear_probe(TASK, x, y * 2)
    with pytest.raises(ProbeError, match="align"):
        fit_linear_probe(TASK, x, y[:-1])
    with pytest.raises(ValueError, match="positive"):
        LinearProbeConfig(c_grid=(0.0,))
    with pytest.raises(ValueError):
        LinearProbeConfig.model_validate({"c_grid": [1.0], "standardise": True})


def test_few_shot_with_k_shot_linear_heads_is_seeded() -> None:
    x32, y = _data(200, 6, 6)
    x = x32.astype(np.float64)

    def fitter(
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        d: NDArray[np.float64],
    ) -> tuple[Scorer, float]:
        fitted = fit_linear_probe(TASK, a, b, c, d, config=SMALL)
        return fitted.predict, fitted.c

    splits = (x[:100], y[:100]), (x[100:150], y[100:150]), (x[150:], y[150:])
    first = run_few_shot(*splits, shots=(2, 8), replicates=2, seed=3, fitter=fitter)
    second = run_few_shot(*splits, shots=(2, 8), replicates=2, seed=3, fitter=fitter)
    assert first == second
    assert [s.k for s in first.summaries] == [2, 8]
    assert all(run.n_train == 2 * run.k for run in first.runs)


# --------------------------------------------------------------------------- registry


@pytest.fixture
def registry(tmp_path: Path) -> HeadRegistry:
    x, y = _data(100, 8, 7)
    fit = fit_linear_probe(TASK, x, y, config=SMALL)
    save_head(fit.head, tmp_path / "saved", encoder_weights_sha256=ENCODER, metadata={"c": fit.c})
    heads = HeadRegistry(tmp_path / "registry")
    heads.publish(tmp_path / "saved", "outcome")
    return heads


def test_registry_round_trip(registry: HeadRegistry, tmp_path: Path) -> None:
    assert registry.names() == ("outcome",)
    entry = registry.entry("outcome")
    loaded = registry.load(
        "outcome", EncoderIdentity(8, ENCODER), expected_record_sha256=entry.record_sha256
    )
    x, _ = _data(20, 8, 8)
    original = HeadRegistry(tmp_path).load("saved", EncoderIdentity(8, ENCODER))
    np.testing.assert_array_equal(
        predict_probabilities(loaded.head, x), predict_probabilities(original.head, x)
    )
    assert not loaded.head.training and loaded.record.metadata["c"] == 1.0
    assert set(registry.load_all(EncoderIdentity(8, ENCODER))) == {"outcome"}
    with pytest.raises(FileExistsError):
        registry.publish(tmp_path / "saved", "outcome")


def test_tampered_weights_fail_closed(registry: HeadRegistry) -> None:
    weights = registry.root / "outcome" / HEAD_WEIGHTS
    data = bytearray(weights.read_bytes())
    data[-1] ^= 1
    weights.write_bytes(bytes(data))
    with pytest.raises(HeadLoadError, match="SHA-256"):
        registry.load("outcome", EncoderIdentity(8, ENCODER))


def test_tampered_record_fails_closed(registry: HeadRegistry) -> None:
    pinned = registry.entry("outcome").record_sha256
    path = registry.root / "outcome" / HEAD_JSON
    record = json.loads(path.read_text())
    record["metadata"]["c"] = 5.0
    path.write_text(json.dumps(record))
    with pytest.raises(HeadLoadError, match="record has SHA-256"):
        registry.load("outcome", EncoderIdentity(8, ENCODER), expected_record_sha256=pinned)


def test_encoder_mismatch_fails_closed(registry: HeadRegistry) -> None:
    with pytest.raises(HeadLoadError, match="fitted on encoder weights"):
        registry.load("outcome", EncoderIdentity(8, OTHER_ENCODER))
    with pytest.raises(HeadLoadError, match="8-wide"):
        registry.load("outcome", EncoderIdentity(16, ENCODER))


def test_head_without_encoder_binding_is_not_servable(tmp_path: Path) -> None:
    x, y = _data(60, 4, 9)
    save_head(fit_linear_probe(TASK, x, y, config=SMALL).head, tmp_path / "unbound")
    with pytest.raises(HeadLoadError, match="records no encoder"):
        HeadRegistry(tmp_path).load("unbound", EncoderIdentity(4, ENCODER))


def test_registry_names_and_missing_heads(registry: HeadRegistry) -> None:
    for bad in ("../x", "A", "a/b", ""):
        with pytest.raises(ValueError):
            require_head_name(bad)
    with pytest.raises(HeadLoadError, match="no head named"):
        registry.load("absent", EncoderIdentity(8, ENCODER))
    assert HeadRegistry(registry.root / "nowhere").names() == ()


# --------------------------------------------------------------------------- pipeline


@pytest.fixture(scope="module")
def work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("heads-experiment")


@pytest.fixture(scope="module")
def result(work: Path) -> PipelineResult:
    config = load_experiment_config(
        TOY,
        [
            f"experiment.work_dir={work}",
            "experiment.pretraining.enabled=false",
            "tasks.heads=[{name: linear_probe, options: {c_grid: [0.01, 1.0, 100.0]}}]",
            "eval.hooks=[{name: binary, options: {splits: [tuning, held_out],"
            " settings: {resamples: 50}}}, {name: few_shot, options: {shots: [1, 2],"
            " replicates: 1, probe: linear_head, c_grid: [1.0]}}]",
        ],
    )
    return run_experiment(config)


def test_hook_is_registered_and_runs(result: PipelineResult) -> None:
    assert LINEAR_PROBE in registered_hooks("head")
    statuses = {stage.name: stage.status for stage in result.stages}
    assert statuses["head-linear_probe"] == "computed"
    assert statuses["eval-binary"] == statuses["eval-few_shot"] == "computed"


def test_predictions_are_evaluated_and_match_the_served_head(result: PipelineResult) -> None:
    head = result.heads[LINEAR_PROBE]
    held_out = result.representations.load(TASK, "held_out")
    published = load_predictions(head.output_dir, held_out)
    binary = result.evaluations["binary"].result.summary["results"]
    assert isinstance(binary, dict) and LINEAR_PROBE in binary
    registry = HeadRegistry(head.output_dir)
    encoder = EncoderIdentity(held_out.meta.hidden_size, result.model.weights_sha256)
    served = registry.load("head", encoder)
    np.testing.assert_allclose(
        predict_probabilities(served.head, held_out.embeddings), published.y_prob, atol=1e-6
    )
    few_shot = result.evaluations["few_shot"].result.summary
    assert few_shot["probe"] == "linear_head"


def test_head_metrics_and_counts_are_suppressed_in_the_store(
    result: PipelineResult, work: Path
) -> None:
    policy = result.config.experiment.suppression
    mlruns = work / "mlruns"
    assert small_counts(mlruns, policy) == []
    held_out = result.representations.load(TASK, "held_out")
    assert held_out.labels is not None and held_out.labels.sum() < policy.threshold
    names = stored_metric_names(mlruns)
    assert "head/linear_probe/train/n" in names
    assert "head/linear_probe/held_out/n_positive" not in names
    assert "eval/binary/linear_probe/held_out/auroc" not in names
    summary = json.loads(
        next(mlruns.glob("*/*/artifacts/hooks/head/linear_probe.json")).read_text()
    )
    for row in summary["selection"]:
        assert row["n_positive"] is None and row["auroc"] is None
    # The saved head carries settings, never counts.
    record = json.loads((result.heads[LINEAR_PROBE].output_dir / "head" / HEAD_JSON).read_text())
    assert not any(key.startswith("n") for key in record["metadata"])


def test_store_and_cli_list_and_publish(result: PipelineResult, work: Path) -> None:
    found = trained_heads(work)
    assert [head.hook for head in found] == [LINEAR_PROBE]
    assert find_trained_head(work, found[0].stage_key[:8]).path == found[0].path
    with pytest.raises(LookupError):
        find_trained_head(work, "absent")
    runner = CliRunner()
    listed = runner.invoke(app, ["heads", "ls", "--work-dir", str(work), "--json"])
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.output)[0]["encoder_weights_sha256"] == result.model.weights_sha256
    registry = work / "registry"
    published = runner.invoke(
        app,
        ["heads", "publish", LINEAR_PROBE, "--work-dir", str(work), "--registry", str(registry)],
    )
    assert published.exit_code == 0, published.output
    assert HeadRegistry(registry).names() == (TASK,)
    again = runner.invoke(
        app,
        ["heads", "publish", LINEAR_PROBE, "--work-dir", str(work), "--registry", str(registry)],
    )
    assert again.exit_code == 1 and "already registered" in again.output
    assert runner.invoke(app, ["heads", "ls"]).exit_code == 2
    shutil.rmtree(registry)


def test_non_boolean_or_unknown_options_fail(tmp_path: Path) -> None:
    config = load_experiment_config(
        TOY,
        [
            f"experiment.work_dir={tmp_path}",
            "experiment.pretraining.enabled=false",
            "experiment.tracking.enabled=false",
            "tasks.heads=[{name: linear_probe, options: {c: 1.0}}]",
            "eval.hooks=[]",
        ],
    )
    with pytest.raises(HookError, match="invalid options"):
        run_experiment(config)


def test_loaded_head_runs_on_torch_input(registry: HeadRegistry) -> None:
    loaded = registry.load("outcome", EncoderIdentity(8, ENCODER))
    out = loaded.head.predict(torch.zeros(3, 8))
    assert out.shape == (3,) and torch.all((out > 0) & (out < 1))
