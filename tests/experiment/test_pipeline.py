"""End-to-end experiment pipeline on the toy inputs (CPU), hooks, caching and the CLI."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ehrfm.cli import app
from ehrfm.config.experiment import ExperimentConfig, SmallCellPolicy
from ehrfm.config.loader import load_experiment_config
from ehrfm.experiment import pipeline
from ehrfm.experiment.cache import list_stages
from ehrfm.experiment.hooks import (
    HookContext,
    HookError,
    HookResult,
    register_hook,
    registered_hooks,
    unregister_hook,
)
from ehrfm.experiment.pipeline import PipelineResult, run_experiment
from ehrfm.experiment.representations import SUBJECTS_TASK
from ehrfm.inputs.contract import SPLITS
from ehrfm.training import runner
from tests.experiment.store import small_counts

TOY = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "toy.yaml"
TASK = "demo_outcome_30d"
CALLS: list[str] = []


def _mean_head(ctx: HookContext) -> HookResult:
    CALLS.append(f"head:{ctx.name}")
    train = ctx.representations.load(TASK, "train")
    assert train.labels is not None
    centroid = train.embeddings.mean(axis=0)
    np.save(ctx.output_dir / "centroid.npy", centroid)
    return HookResult(
        metrics={"train/n": float(len(train)), "train/n_positive": float(train.labels.sum())},
        summary={"hidden_size": train.meta.hidden_size, "scale": ctx.options.get("scale", 1)},
    )


def _distance_eval(ctx: HookContext) -> HookResult:
    CALLS.append(f"eval:{ctx.name}")
    head = ctx.heads["mean"]
    centroid = np.load(head.output_dir / "centroid.npy")
    held_out = ctx.representations.load(TASK, "held_out")
    distance = float(np.linalg.norm(held_out.embeddings - centroid, axis=1).mean())
    return HookResult(
        metrics={"held_out/n": float(len(held_out)), "held_out/mean_distance": distance},
        summary={
            "head_key": head.key[:16],
            "groups": [{"n": 3, "score": 0.5}, {"n": 50, "score": 0.1}],
        },
    )


@pytest.fixture(scope="module", autouse=True)
def hooks() -> Iterator[None]:
    register_hook("head", "mean", _mean_head)
    register_hook("eval", "distance", _distance_eval, version="2")
    yield
    unregister_hook("head", "mean")
    unregister_hook("eval", "distance")


@pytest.fixture(scope="module")
def work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("experiment")


def _config(work: Path, *overrides: str) -> ExperimentConfig:
    base = [
        f"experiment.work_dir={work}",
        "tasks.heads=[{name: mean, options: {scale: 2}}]",
        "eval.hooks=[{name: distance}]",
    ]
    return load_experiment_config(TOY, [*base, *overrides])


@pytest.fixture(scope="module")
def first(work: Path) -> PipelineResult:
    CALLS.clear()
    return run_experiment(_config(work))


def _statuses(result: PipelineResult) -> dict[str, str]:
    return {stage.name: stage.status for stage in result.stages}


def test_first_run_computes_every_stage(first: PipelineResult) -> None:
    assert _statuses(first) == {
        "inputs": "verified",
        "coverage": "computed",
        "pretraining": "computed",
        "representations": "computed",
        "head-mean": "computed",
        "eval-distance": "computed",
    }
    assert first.tracking_status == "logged" and first.tracking_run_id is not None
    assert not first.cache_hit
    assert CALLS == ["head:mean", "eval:distance"]
    assert first.model.training_fingerprint is not None


def test_representations_cover_every_split_with_labels(first: PipelineResult) -> None:
    index = first.representations
    assert index.tasks == (TASK,) and index.splits == SPLITS
    total = 0
    for split in SPLITS:
        loaded = index.load(TASK, split)
        assert loaded.embeddings.shape == (len(loaded), loaded.meta.hidden_size)
        assert np.isfinite(loaded.embeddings).all()
        assert loaded.labels is not None and set(np.unique(loaded.labels)) <= {0.0, 1.0}
        assert loaded.meta.model_weights_sha256 == first.model.weights_sha256
        total += len(loaded)
    assert total == 47  # the toy label set has 47 rows over 60 subjects


def test_hook_results_and_files_are_published(first: PipelineResult) -> None:
    head = first.heads["mean"]
    assert (head.output_dir / "centroid.npy").is_file()
    assert head.result.summary == {"hidden_size": 32, "scale": 2}
    evaluation = first.evaluations["distance"]
    assert evaluation.result.summary["head_key"] == head.key[:16]
    assert evaluation.result.metrics["held_out/mean_distance"] > 0


def test_rerun_is_a_full_cache_hit(
    first: PipelineResult, work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a cache hit must not train or extract")

    monkeypatch.setattr(runner, "run_pretraining", forbidden)
    monkeypatch.setattr(pipeline, "_write_representations", forbidden)
    CALLS.clear()
    rerun = run_experiment(_config(work))
    assert rerun.cache_hit
    assert set(_statuses(rerun).values()) == {"verified", "cached"}
    assert CALLS == []
    assert rerun.pipeline_key == first.pipeline_key
    assert rerun.tracking_status == "cached"
    assert rerun.tracking_run_id == first.tracking_run_id
    assert rerun.model == first.model


def test_tracked_store_holds_no_small_counts(first: PipelineResult, work: Path) -> None:
    mlruns = work / "mlruns"
    policy = first.config.experiment.suppression
    assert small_counts(mlruns, policy) == []
    artifacts = {path.name for path in mlruns.rglob("artifacts/**/*") if path.is_file()}
    assert {"coverage.json", "inputs.json", "config.json", "suppression.json"} <= artifacts
    assert {"mean.json", "distance.json"} <= artifacts
    log = json.loads(next(mlruns.rglob("artifacts/suppression.json")).read_text())
    sources = {entry["source"] for entry in log["suppressed"]}
    # The toy tuning and held-out splits have fewer than 5 positives; the
    # distance hook's small group is masked too.
    assert "metrics" in sources and "hooks/eval/distance.json" in sources
    distance = json.loads(next(mlruns.rglob("artifacts/hooks/eval/distance.json")).read_text())
    assert distance["groups"][0] == {"n": None, "score": None}


def test_a_changed_eval_option_reruns_only_the_eval_stage(
    first: PipelineResult, work: Path
) -> None:
    CALLS.clear()
    changed = run_experiment(_config(work, "eval.hooks=[{name: distance, options: {k: 1}}]"))
    statuses = _statuses(changed)
    assert statuses["eval-distance"] == "computed"
    assert {
        statuses[name] for name in ("coverage", "pretraining", "representations", "head-mean")
    } == {"cached"}
    assert CALLS == ["eval:distance"]
    assert changed.tracking_status == "logged"
    assert changed.pipeline_key != first.pipeline_key


def test_a_changed_suppression_policy_relogs_without_recomputing(
    first: PipelineResult, work: Path
) -> None:
    CALLS.clear()
    changed = run_experiment(_config(work, "experiment.suppression.threshold=11"))
    assert {s.status for s in changed.stages} == {"verified", "cached"}
    assert CALLS == []
    assert changed.tracking_status == "logged"
    assert changed.pipeline_key != first.pipeline_key


def test_a_changed_training_setting_reruns_from_pretraining(
    first: PipelineResult, tmp_path: Path
) -> None:
    config = _config(tmp_path, "experiment.pretraining.optimization.max_steps=4")
    result = run_experiment(config)
    assert result.model.training_fingerprint != first.model.training_fingerprint
    assert result.model.weights_sha256 != first.model.weights_sha256
    # Same inputs, same dictionary: the coverage key is shared with the first run.
    coverage = next(s for s in result.stages if s.name == "coverage")
    assert coverage.key == next(s.key for s in first.stages if s.name == "coverage")


def test_missing_pretrained_model_is_retrained(first: PipelineResult, tmp_path: Path) -> None:
    config = _config(tmp_path, "experiment.pretraining.optimization.max_steps=3")
    run_experiment(config)
    weights = run_experiment(config).model.model_dir / "model.safetensors"
    weights.write_bytes(b"truncated")
    again = run_experiment(config)
    assert _statuses(again)["pretraining"] == "computed"
    assert _statuses(again)["representations"] == "cached"  # same weights retrained exactly


def test_without_pretraining_or_tracking_the_base_model_is_used(tmp_path: Path) -> None:
    config = load_experiment_config(
        TOY,
        [
            f"experiment.work_dir={tmp_path}",
            "experiment.pretraining.enabled=false",
            "experiment.tracking.enabled=false",
            "experiment.coverage.enabled=false",
            "experiment.representations.splits=[held_out]",
            "tasks.heads=[]",
            "eval.hooks=[]",
        ],
    )
    result = run_experiment(config)
    assert _statuses(result)["pretraining"] == "skipped"
    assert _statuses(result)["coverage"] == "skipped"
    assert result.model.training_fingerprint is None
    assert result.tracking_status == "disabled" and not (tmp_path / "mlruns").exists()
    assert result.representations.splits == ("held_out",)


def test_unknown_hook_fails_before_any_stage(tmp_path: Path) -> None:
    config = load_experiment_config(
        TOY, [f"experiment.work_dir={tmp_path}", "eval.hooks=[{name: nope}]"]
    )
    with pytest.raises(HookError, match="no eval hook named 'nope'"):
        run_experiment(config)
    assert list_stages(tmp_path) == []


def test_hook_registry_rejects_duplicates() -> None:
    with pytest.raises(HookError, match="already registered"):
        register_hook("head", "mean", _mean_head)
    assert "mean" in registered_hooks("head")
    assert "distance" in registered_hooks("eval")


def test_subject_representations_without_label_sets(first: PipelineResult, tmp_path: Path) -> None:
    del first
    config = load_experiment_config(
        TOY,
        [
            f"experiment.work_dir={tmp_path}",
            "experiment.pretraining.enabled=false",
            "experiment.tracking.enabled=false",
        ],
    )
    inputs = pipeline.verify_inputs(config, tmp_path)
    unlabelled = pipeline.VerifiedInputs(
        extract=inputs.extract,
        label_sets=(),
        base_checkpoint=inputs.base_checkpoint,
        base_weights_sha256=inputs.base_weights_sha256,
        base_dictionary_sha256=inputs.base_dictionary_sha256,
        base_config_sha256=inputs.base_config_sha256,
    )
    model = pipeline.model_identity(unlabelled, None)
    _, _, index = pipeline.representation_stage(config, unlabelled, model, tmp_path)
    assert index.tasks == (SUBJECTS_TASK,)
    sizes = {split: len(index.load(SUBJECTS_TASK, split)) for split in SPLITS}
    assert sizes == {"train": 48, "tuning": 6, "held_out": 6}
    assert index.load(SUBJECTS_TASK, "train").labels is None


def test_policy_from_config_is_the_one_applied(first: PipelineResult) -> None:
    assert first.config.experiment.suppression == SmallCellPolicy()


# --------------------------------------------------------------------------- CLI


def test_cli_run_ls_show(
    first: PipelineResult, work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del first
    monkeypatch.setenv("EHRFM_WORK_DIR", str(work.parent))
    cli = CliRunner()
    # The CLI has no hooks configured, so it is a different pipeline: stages are
    # shared, tracking logs one new run, and a second call is a full cache hit.
    overrides = [f"experiment.work_dir={work}"]
    once = cli.invoke(app, ["run", str(TOY), *overrides])
    assert once.exit_code == 0, once.output
    twice = cli.invoke(app, ["run", "--config", str(TOY), *overrides])
    assert twice.exit_code == 0, twice.output
    summary = json.loads(twice.stdout)
    assert summary["cache_hit"] is True
    assert summary["tracking"]["status"] == "cached"
    run_id = summary["tracking"]["run_id"]

    listed = cli.invoke(app, ["ls", "--work-dir", str(work)])
    assert listed.exit_code == 0, listed.output
    assert run_id in listed.stdout and "representations" in listed.stdout

    shown = cli.invoke(app, ["show", run_id, "--work-dir", str(work)])
    assert shown.exit_code == 0, shown.output
    payload = json.loads(shown.stdout)
    assert payload["status"] == "FINISHED"
    assert "coverage.json" in payload["artifacts"]
    assert payload["tags"]["ehrfm.pipeline_key"] == summary["pipeline_key"]

    stage_key = summary["stages"][1]["key"]
    stage = cli.invoke(app, ["show", stage_key, "--work-dir", str(work)])
    assert stage.exit_code == 0 and json.loads(stage.stdout)["stage"] == "coverage"


def test_cli_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = CliRunner()
    assert cli.invoke(app, ["run"]).exit_code == 2
    assert cli.invoke(app, ["run", str(TOY), "not-an-override"]).exit_code == 2
    bad = cli.invoke(app, ["run", str(TOY), "+experiment.bogus=1"])
    assert bad.exit_code == 1 and "experiment.bogus" in bad.output
    monkeypatch.delenv("EHRFM_BASE_CHECKPOINT", raising=False)
    monkeypatch.setenv("EHRFM_WORK_DIR", str(tmp_path))
    default = TOY.parent / "default.yaml"
    missing = cli.invoke(app, ["run", str(default)])
    assert missing.exit_code == 1
    assert "environment variable" in missing.output and "is not set" in missing.output
    assert cli.invoke(app, ["ls"]).exit_code == 2
    unknown = cli.invoke(app, ["show", "0" * 32, "--work-dir", str(tmp_path)])
    assert unknown.exit_code == 1
