"""End-to-end continued pretraining on the toy inputs (CPU).

One fresh run is shared by the module; other tests reuse, resume or
invalidate it. Assertions on the loss are deliberately loose: finite, and
lower at the end than at the start.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from ehrfm.cli import app
from ehrfm.data.batches import BatchBuilder
from ehrfm.data.splits import extract_split
from ehrfm.data.tokenization import extract_tokenizer
from ehrfm.inputs import ensure_database, open_extract
from ehrfm.models.checkpoint import WEIGHTS_FILE, load_checkpoint
from ehrfm.models.representations import ClmbrEncoder
from ehrfm.models.weights import weights_sha256
from ehrfm.toy.checkpoint import HIDDEN_SIZE
from ehrfm.toy.pretraining import toy_training_config
from ehrfm.training import runner
from ehrfm.training.cache import RunPaths, read_record
from ehrfm.training.runner import TrainingResult, run_pretraining, torchrun_command


@pytest.fixture(scope="module")
def toy_work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("toy-pretrain")


@pytest.fixture(scope="module")
def completed(toy_work: Path) -> TrainingResult:
    return run_pretraining(toy_training_config(toy_work))


def test_fresh_run_trains_and_loss_decreases(completed: TrainingResult) -> None:
    assert completed.status == "completed"
    assert completed.steps_run == completed.total_steps > 5
    losses = completed.train_losses
    assert all(math.isfinite(loss) for loss in losses)
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3
    assert completed.validation_loss is not None and math.isfinite(completed.validation_loss)


def test_rerun_is_a_cache_hit_without_training(
    completed: TrainingResult, toy_work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_before = (toy_work / "run" / "run.json").read_bytes()

    def no_training(*args: object, **kwargs: object) -> None:
        raise AssertionError("a cache hit must not load the base model")

    monkeypatch.setattr(runner, "prepare_base_model", no_training)
    rerun = run_pretraining(toy_training_config(toy_work))
    assert rerun.cache_hit
    assert rerun.steps_run == 0
    assert rerun.fingerprint == completed.fingerprint
    assert rerun.model_weights_sha256 == completed.model_weights_sha256
    assert rerun.duration_seconds < 2
    assert (toy_work / "run" / "run.json").read_bytes() == record_before


def test_output_loads_and_embeds_via_the_model_api(
    completed: TrainingResult, toy_work: Path
) -> None:
    model_dir = completed.model_dir
    assert completed.model_weights_sha256 == weights_sha256(model_dir / WEIGHTS_FILE)
    loaded = load_checkpoint(model_dir, expected_weights_sha256=completed.model_weights_sha256)
    # Written already hierarchical with the embed_bag key: nothing to patch or remap.
    assert not loaded.surgery.hierarchical_patched
    assert not loaded.surgery.embedding_remapped

    extract = open_extract(toy_work / "inputs" / "artifacts" / "meds" / "synthetic_demo")
    encoder = ClmbrEncoder.load(model_dir, device="cpu")
    builder = BatchBuilder(extract_tokenizer(model_dir, extract))
    database_path = ensure_database(extract, toy_work / "db")
    import meds_reader

    with meds_reader.SubjectDatabase(str(database_path)) as database:
        held_out = list(extract_split(extract, "held_out"))
        trained = encoder.subject_representations(database, held_out, builder)
        base = ClmbrEncoder.load(toy_work / "inputs" / "checkpoint", device="cpu")
        before = base.subject_representations(database, held_out, builder)
    assert trained.embeddings.shape == (len(held_out), HIDDEN_SIZE)
    assert torch.isfinite(trained.embeddings).all()
    assert not torch.allclose(trained.embeddings, before.embeddings)


def test_interrupted_run_resumes_to_the_same_weights(
    completed: TrainingResult, tmp_path: Path
) -> None:
    config = toy_training_config(tmp_path, checkpoint_every=1000)
    stop_at = 4
    first = run_pretraining(config, should_stop=lambda step: step == stop_at)
    assert first.status == "interrupted"
    assert first.step == stop_at
    record = read_record(RunPaths(config.runtime.output_dir))
    assert record is not None and record.status == "interrupted"
    assert not (config.runtime.output_dir / "model").exists()

    resumed = run_pretraining(config)
    assert resumed.status == "completed"
    assert resumed.resumed_from_step == stop_at
    assert resumed.steps_run == resumed.total_steps - stop_at
    assert resumed.train_losses == completed.train_losses
    assert resumed.model_weights_sha256 == completed.model_weights_sha256
    assert not (config.runtime.output_dir / "checkpoints").exists()


def test_failure_during_training_is_recorded_and_resumable(
    completed: TrainingResult, tmp_path: Path
) -> None:
    config = toy_training_config(tmp_path, checkpoint_every=3)

    def crash(step: int) -> bool:
        if step == 5:
            raise KeyboardInterrupt
        return False

    with pytest.raises(KeyboardInterrupt):
        run_pretraining(config, should_stop=crash)
    record = read_record(RunPaths(config.runtime.output_dir))
    assert record is not None
    assert record.status == "interrupted"
    assert record.step == 5
    assert not (config.runtime.output_dir / "model").exists()

    resumed = run_pretraining(config)
    assert resumed.resumed_from_step == 3
    assert resumed.model_weights_sha256 == completed.model_weights_sha256


def test_changed_input_invalidates_the_cache(completed: TrainingResult, toy_work: Path) -> None:
    config = toy_training_config(toy_work)
    changed = config.model_copy(
        update={
            "optimization": config.optimization.model_copy(update={"max_steps": 3}),
        }
    )
    result = run_pretraining(changed)
    assert result.status == "completed"
    assert result.fingerprint != completed.fingerprint
    assert result.steps_run == 3
    # The original configuration now retrains too: the directory holds the newer run.
    again = run_pretraining(config)
    assert not again.cache_hit
    assert again.model_weights_sha256 == completed.model_weights_sha256


def test_altered_published_model_is_retrained(tmp_path: Path) -> None:
    config = toy_training_config(tmp_path)
    config = config.model_copy(
        update={"optimization": config.optimization.model_copy(update={"max_steps": 2})}
    )
    first = run_pretraining(config)
    weights = first.model_dir / WEIGHTS_FILE
    weights.write_bytes(weights.read_bytes()[:-1])
    second = run_pretraining(config)
    assert not second.cache_hit
    assert second.model_weights_sha256 == first.model_weights_sha256


def test_cli_toy_run_then_cache_hit(tmp_path: Path) -> None:
    cli = CliRunner()
    args = ["pretrain", "--toy", "--work-dir", str(tmp_path)]
    first = cli.invoke(app, args)
    assert first.exit_code == 0, first.output
    second = cli.invoke(app, args)
    assert second.exit_code == 0, second.output
    summary = json.loads(second.output[second.output.index("{") :])
    assert summary["status"] == "cached"
    assert summary["steps_run"] == 0


def test_cli_config_file_and_errors(
    tmp_path: Path, completed: TrainingResult, toy_work: Path
) -> None:
    config_file = tmp_path / "train.yaml"
    inputs = toy_work / "inputs"
    config_file.write_text(
        "\n".join(
            [
                "inputs:",
                f"  extract: {inputs / 'artifacts' / 'meds' / 'synthetic_demo'}",
                f"  base_checkpoint: {inputs / 'checkpoint'}",
                "optimization:",
                "  tokens_per_batch: 96",
                "  max_steps: 2",
                "runtime:",
                "  output_dir: ${oc.env:EHRFM_TEST_RUN_DIR}",
                "  device: cpu",
            ]
        ),
        encoding="utf-8",
    )
    cli = CliRunner()
    run_dir = tmp_path / "run"
    result = cli.invoke(
        app, ["pretrain", "--config", str(config_file)], env={"EHRFM_TEST_RUN_DIR": str(run_dir)}
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "model" / WEIGHTS_FILE).is_file()

    neither = cli.invoke(app, ["pretrain"])
    assert neither.exit_code == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text("inputs: {extract: x}\nruntime: {output_dir: y, unknown: 1}\n", encoding="utf-8")
    invalid = cli.invoke(app, ["pretrain", "--config", str(bad)])
    assert invalid.exit_code == 1
    assert "ValidationError" in invalid.output
    del completed


def test_torchrun_command(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    single = torchrun_command(config, nproc_per_node=2, python="python")
    assert single[:5] == ["python", "-m", "torch.distributed.run", "--standalone", "--nnodes"]
    assert single[-4:] == ["ehrfm", "pretrain", "--config", str(config)]
    multi = torchrun_command(
        config, nproc_per_node=4, nnodes=2, rdzv_endpoint="head:29400", python="python"
    )
    assert "--rdzv-endpoint" in multi and "head:29400" in multi
    with pytest.raises(ValueError, match="rendezvous"):
        torchrun_command(config, nproc_per_node=1, nnodes=2)
    cli = CliRunner()
    printed = cli.invoke(app, ["pretrain", "--config", str(config), "--print-torchrun", "2"])
    assert printed.exit_code == 0
    assert "torch.distributed.run --standalone" in printed.output
    shutil.rmtree(tmp_path, ignore_errors=True)
