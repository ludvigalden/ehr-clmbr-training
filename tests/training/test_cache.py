"""Run fingerprint, cache decisions and atomic publication."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from ehrfm.config.training import (
    OptimizationConfig,
    RuntimeConfig,
    TrainingConfig,
    TrainingInputs,
    VocabExpansionConfig,
)
from ehrfm.toy.checkpoint import ToyCheckpoint
from ehrfm.training import cache
from ehrfm.training.cache import CheckpointMeta, RunPaths, RunRecord

EXTRACT_SHA = "1" * 64
WEIGHTS_SHA = "2" * 64
VERSIONS = {
    "femr": "0.2.4",
    "torch": "2.0",
    "transformers": "4.45",
    "ehr-clmbr-training": "0",
    "meds-reader": "0.1",
}


def _config(checkpoint: Path, out: Path, **optimization: Any) -> TrainingConfig:
    return TrainingConfig(
        inputs=TrainingInputs(extract=out / "extract", base_checkpoint=checkpoint),
        optimization=OptimizationConfig(**optimization),
        runtime=RuntimeConfig(output_dir=out / "run"),
    )


def _digest(config: TrainingConfig, **overrides: Any) -> str:
    kwargs: dict[str, Any] = {
        "extract_manifest_sha256": EXTRACT_SHA,
        "base_weights_sha256": WEIGHTS_SHA,
        "world_size": 1,
        "versions": VERSIONS,
    }
    kwargs.update(overrides)
    return cache.fingerprint_inputs(config, **kwargs).digest()


def test_fingerprint_is_stable(toy_checkpoint: ToyCheckpoint, tmp_path: Path) -> None:
    config = _config(toy_checkpoint.path, tmp_path)
    assert _digest(config) == _digest(config)
    assert len(_digest(config)) == 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"extract_manifest_sha256": "3" * 64},
        {"base_weights_sha256": "4" * 64},
        {"world_size": 2},
        {"versions": {**VERSIONS, "femr": "0.2.5"}},
        {"versions": {**VERSIONS, "torch": "2.1"}},
    ],
    ids=["extract", "base-weights", "world-size", "femr-version", "torch-version"],
)
def test_every_input_changes_the_fingerprint(
    toy_checkpoint: ToyCheckpoint, tmp_path: Path, overrides: dict[str, Any]
) -> None:
    config = _config(toy_checkpoint.path, tmp_path)
    assert _digest(config, **overrides) != _digest(config)


@pytest.mark.parametrize(
    "field",
    [
        {"seed": 1},
        {"tokens_per_batch": 123},
        {"epochs": 2},
        {"max_steps": 7},
        {"learning_rate": 2e-4},
        {"weight_decay": 0.0},
        {"warmup_steps": 3},
        {"schedule": "constant"},
        {"max_grad_norm": None},
        {"validation_fraction": 0.2},
        {"clmbr_vocab_size": 16},
        {"precision": "bf16"},
        {"vocab_expansion": VocabExpansionConfig(hook="x", n_replace=1)},
    ],
    ids=lambda field: next(iter(field)),
)
def test_every_optimization_setting_changes_the_fingerprint(
    toy_checkpoint: ToyCheckpoint, tmp_path: Path, field: dict[str, Any]
) -> None:
    base = _config(toy_checkpoint.path, tmp_path)
    changed = _config(toy_checkpoint.path, tmp_path, **field)
    assert _digest(changed) != _digest(base)


def test_base_dictionary_and_config_bytes_change_the_fingerprint(
    toy_checkpoint: ToyCheckpoint, tmp_path: Path
) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(toy_checkpoint.path, copy)
    config = _config(copy, tmp_path)
    before = _digest(config)
    (copy / "dictionary.msgpack").write_bytes((copy / "dictionary.msgpack").read_bytes() + b"\0")
    after_dictionary = _digest(config)
    (copy / "config.json").write_text((copy / "config.json").read_text() + " ", encoding="utf-8")
    after_config = _digest(config)
    assert len({before, after_dictionary, after_config}) == 3


def test_runtime_settings_do_not_change_the_fingerprint(
    toy_checkpoint: ToyCheckpoint, tmp_path: Path
) -> None:
    config = _config(toy_checkpoint.path, tmp_path)
    moved = config.model_copy(
        update={
            "runtime": RuntimeConfig(
                output_dir=tmp_path / "elsewhere",
                device="cuda",
                checkpoint_every=1,
                keep_checkpoints=5,
                tokenize_workers=4,
                deterministic=False,
            )
        }
    )
    assert _digest(moved) == _digest(config)


def _record(fingerprint: str, status: cache.RunStatus, weights: str | None = None) -> RunRecord:
    inputs = cache.FingerprintInputs(
        extract_manifest_sha256=EXTRACT_SHA,
        base_weights_sha256=WEIGHTS_SHA,
        base_dictionary_sha256="5" * 64,
        base_config_sha256="6" * 64,
        optimization=OptimizationConfig(),
        world_size=1,
        versions=VERSIONS,
    )
    return RunRecord(
        fingerprint=fingerprint,
        inputs=inputs,
        status=status,
        step=0,
        total_steps=4,
        started_at=cache.now_iso(),
        model_weights_sha256=weights,
    )


def _publish(paths: RunPaths, checkpoint: ToyCheckpoint) -> str:
    with cache.staged_directory(paths.model) as staging:
        for name in ("config.json", "dictionary.msgpack", "model.safetensors"):
            (staging / name).write_bytes((checkpoint.path / name).read_bytes())
    return cache.sha256_bytes((paths.model / "model.safetensors").read_bytes())


def test_decisions(toy_checkpoint: ToyCheckpoint, tmp_path: Path) -> None:
    paths = RunPaths(tmp_path / "run")
    assert cache.decide(paths, "a").action == "fresh"

    cache.write_record(paths, _record("a", "in_progress"))
    assert cache.decide(paths, "a").reason == "in_progress run without a checkpoint"

    meta = CheckpointMeta(fingerprint="a", step=2, epoch=0, train_losses=(1.0, 0.9))
    saved = cache.save_checkpoint(paths, meta, {"x": 1}, keep=2)
    decision = cache.decide(paths, "a")
    assert decision.action == "resume"
    assert decision.checkpoint == saved
    assert cache.decide(paths, "b").action == "fresh"

    digest = _publish(paths, toy_checkpoint)
    cache.write_record(paths, _record("a", "completed", digest))
    assert cache.decide(paths, "a").action == "cached"


def test_completed_record_without_matching_model_is_not_a_hit(
    toy_checkpoint: ToyCheckpoint, tmp_path: Path
) -> None:
    paths = RunPaths(tmp_path / "run")
    cache.write_record(paths, _record("a", "completed", "7" * 64))
    assert cache.decide(paths, "a").action == "fresh"  # no model at all
    digest = _publish(paths, toy_checkpoint)
    cache.write_record(paths, _record("a", "completed", digest))
    assert cache.decide(paths, "a").action == "cached"
    with (paths.model / "model.safetensors").open("ab") as handle:
        handle.write(b"\0")
    assert cache.decide(paths, "a").reason == "completed record but model missing or altered"


def test_unreadable_record_counts_as_none(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    paths.record.write_text("{not json", encoding="utf-8")
    assert cache.read_record(paths) is None
    paths.record.write_text('{"fingerprint": "a"}', encoding="utf-8")
    assert cache.read_record(paths) is None


def test_checkpoints_are_pruned_and_partial_ones_ignored(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    for step in (1, 2, 3):
        meta = CheckpointMeta(fingerprint="a", step=step, epoch=0, train_losses=())
        cache.save_checkpoint(paths, meta, {"step": step}, keep=2)
    assert [step for step, _ in cache.checkpoint_dirs(paths)] == [2, 3]
    partial = paths.checkpoints / "step-000000009"
    partial.mkdir()
    (partial / "state.pt").write_bytes(b"")  # no checkpoint.json: never written completely
    assert cache.latest_checkpoint(paths, "a") == paths.checkpoints / "step-000000003"
    loaded_meta, state = cache.load_checkpoint_state(paths.checkpoints / "step-000000003")
    assert loaded_meta.step == 3
    assert state == {"step": 3}


def test_atomic_write_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "file.json"
    cache.write_json_atomic(target, {"v": 1})

    def fail(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        cache.write_json_atomic(target, {"v": 2})
    assert target.read_text(encoding="utf-8").strip().endswith("}")
    assert '"v": 1' in target.read_text(encoding="utf-8")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["file.json"]


def test_staged_directory_is_all_or_nothing(tmp_path: Path) -> None:
    final = tmp_path / "model"
    with pytest.raises(RuntimeError), cache.staged_directory(final) as staging:
        (staging / "half.bin").write_bytes(b"x")
        raise RuntimeError("killed")
    assert not final.exists()
    assert list(tmp_path.iterdir()) == []


def test_run_lock_is_exclusive(tmp_path: Path) -> None:
    import fcntl

    paths = RunPaths(tmp_path)
    with cache.run_lock(paths):
        with paths.lock.open("a+") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with cache.run_lock(paths):
        pass
