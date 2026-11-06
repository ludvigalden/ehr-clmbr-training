"""Fail-closed startup: pins, tampering, encoder mismatch, configuration."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from ehrfm.inference.head_loader import HeadLoadError
from ehrfm.inference.registry import HeadRegistry
from ehrfm.models.head import HeadSpec
from ehrfm.models.weights import weights_sha256
from ehrfm.server.config import ServerConfig
from ehrfm.server.state import load_state
from ehrfm.toy.checkpoint import ToyCheckpoint
from tests.server import helpers

WRONG_SHA = "0" * 64


@pytest.fixture(scope="session")
def good_registry(tmp_path_factory: pytest.TempPathFactory, toy_checkpoint: ToyCheckpoint) -> Path:
    """A registry with one valid head bound to the toy checkpoint's weights."""
    sha = weights_sha256(toy_checkpoint.path / "model.safetensors")
    stage = tmp_path_factory.mktemp("stage-good")
    helpers.make_head(
        HeadSpec(kind="linear", task_name="demo_task", label_type="boolean", input_dim=32),
        seed=3,
        out_dir=stage,
        encoder_sha=sha,
    )
    registry_dir = tmp_path_factory.mktemp("registry-good")
    HeadRegistry(registry_dir).publish(stage / "head", "alpha")
    return registry_dir


def _config(
    toy_checkpoint: ToyCheckpoint,
    heads_dir: Path,
    *,
    expected_weights_sha256: str | None = None,
    postprocess: str = "raw",
) -> ServerConfig:
    """A config over the toy checkpoint with one registry."""
    return ServerConfig(
        model_dir=toy_checkpoint.path,
        heads_dir=heads_dir,
        device="cpu",
        expected_weights_sha256=expected_weights_sha256,
        postprocess=postprocess,
        ontology="none",
    )


def test_config_requires_a_layout() -> None:
    with pytest.raises(ValidationError):
        ServerConfig()
    with pytest.raises(ValidationError):
        ServerConfig(toy=True, model_dir=Path("/tmp/x"), heads_dir=Path("/tmp/y"))


def test_from_env_parses_and_defaults() -> None:
    config = ServerConfig.from_env(
        {
            "EHRFM_SERVER_MODEL_DIR": "/tmp/model",
            "EHRFM_SERVER_HEADS_DIR": "/tmp/heads",
            "EHRFM_SERVER_PORT": "9000",
            "EHRFM_SERVER_DEVICE": "cpu",
            "EHRFM_SERVER_POSTPROCESS": "raw",
            "EHRFM_SERVER_TOKENS_PER_BATCH": "256",
            "EHRFM_SERVER_UNKNOWN_CODE_LIMIT": "5",
            "EHRFM_SERVER_ONTOLOGY": "none",
            "EHRFM_SERVER_TOY": "no",
        }
    )
    assert config.model_dir == Path("/tmp/model")
    assert config.heads_dir == Path("/tmp/heads")
    assert config.port == 9000
    assert config.tokens_per_batch == 256
    assert config.unknown_code_limit == 5
    assert config.ontology == "none"
    assert config.toy is False
    assert config.postprocess == "raw"
    defaults = ServerConfig.from_env(
        {
            "EHRFM_SERVER_MODEL_DIR": "/tmp/m",
            "EHRFM_SERVER_HEADS_DIR": "/tmp/h",
            "EHRFM_SERVER_EXTRACT": "/tmp/e",
        }
    )
    assert defaults.port == 8000
    assert defaults.device == "cpu"
    assert defaults.postprocess == "raw"
    assert defaults.toy is False
    assert defaults.ontology == "extract"
    with pytest.raises(ValidationError):
        ServerConfig.from_env({})
    with pytest.raises(ValidationError):
        ServerConfig.from_env(
            {"EHRFM_SERVER_MODEL_DIR": "/tmp/m", "EHRFM_SERVER_HEADS_DIR": "/tmp/h"}
        )


def test_from_env_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="EHRFM_SERVER_PORT"):
        ServerConfig.from_env({"EHRFM_SERVER_PORT": "not-a-number"})
    with pytest.raises(ValidationError):
        ServerConfig.from_env(
            {
                "EHRFM_SERVER_MODEL_DIR": "/tmp/model",
                "EHRFM_SERVER_HEADS_DIR": "/tmp/heads",
                "EHRFM_SERVER_WEIGHTS_SHA256": "ZZZ",
            }
        )


def test_weight_pin_mismatch_fails_startup(
    toy_checkpoint: ToyCheckpoint, good_registry: Path
) -> None:
    config = _config(toy_checkpoint, good_registry, expected_weights_sha256=WRONG_SHA)
    with pytest.raises(Exception, match="SHA-256"):
        load_state(config)


def test_encoder_mismatched_head_fails_startup(
    tmp_path_factory: pytest.TempPathFactory,
    toy_checkpoint: ToyCheckpoint,
) -> None:
    """A head fitted on other encoder weights refuses the encoder; startup dies."""
    stage = tmp_path_factory.mktemp("stage-mismatch")
    helpers.make_head(
        HeadSpec(kind="linear", task_name="demo_task", label_type="boolean", input_dim=32),
        seed=4,
        out_dir=stage,
        encoder_sha=WRONG_SHA,
    )
    registry_dir = tmp_path_factory.mktemp("registry-mismatch")
    HeadRegistry(registry_dir).publish(stage / "head", "alpha")
    with pytest.raises(HeadLoadError):
        load_state(_config(toy_checkpoint, registry_dir))


def test_tampered_head_fails_startup(
    tmp_path_factory: pytest.TempPathFactory,
    toy_checkpoint: ToyCheckpoint,
    good_registry: Path,
) -> None:
    """One flipped byte in a head's weights kills startup."""
    tampered = tmp_path_factory.mktemp("registry-tampered")
    for name in good_registry.iterdir():
        shutil.copytree(name, tampered / name.name)
    weights_path = tampered / "alpha" / "weights.safetensors"
    data = bytearray(weights_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    weights_path.write_bytes(bytes(data))
    with pytest.raises(HeadLoadError):
        load_state(_config(toy_checkpoint, tampered))


def test_unknown_postprocess_fails_startup(
    toy_checkpoint: ToyCheckpoint, good_registry: Path
) -> None:
    with pytest.raises(ValueError, match="unknown postprocess"):
        load_state(_config(toy_checkpoint, good_registry, postprocess="missing"))


def test_empty_registry_fails_startup(
    tmp_path_factory: pytest.TempPathFactory, toy_checkpoint: ToyCheckpoint
) -> None:
    """A registry directory with no heads must not start a ready server."""
    empty = tmp_path_factory.mktemp("registry-empty")
    with pytest.raises(HeadLoadError, match="holds no heads"):
        load_state(_config(toy_checkpoint, empty))


def test_missing_registry_fails_startup(
    tmp_path_factory: pytest.TempPathFactory, toy_checkpoint: ToyCheckpoint
) -> None:
    missing = tmp_path_factory.mktemp("registry-parent") / "nope"
    with pytest.raises(HeadLoadError, match="no head registry"):
        load_state(_config(toy_checkpoint, missing))


def test_width_mismatched_head_fails_startup(
    tmp_path_factory: pytest.TempPathFactory, toy_checkpoint: ToyCheckpoint
) -> None:
    """A head fitted on another encoder width refuses to load."""
    sha = weights_sha256(toy_checkpoint.path / "model.safetensors")
    stage = tmp_path_factory.mktemp("stage-width")
    helpers.make_head(
        HeadSpec(kind="linear", task_name="demo_task", label_type="boolean", input_dim=64),
        seed=5,
        out_dir=stage,
        encoder_sha=sha,
    )
    registry_dir = tmp_path_factory.mktemp("registry-width")
    HeadRegistry(registry_dir).publish(stage / "head", "alpha")
    with pytest.raises(HeadLoadError, match="64-wide"):
        load_state(_config(toy_checkpoint, registry_dir))


def test_head_without_encoder_binding_is_not_servable(
    tmp_path_factory: pytest.TempPathFactory, toy_checkpoint: ToyCheckpoint
) -> None:
    """A head that records no encoder is refused at startup."""
    stage = tmp_path_factory.mktemp("stage-unbound")
    helpers.make_head(
        HeadSpec(kind="linear", task_name="demo_task", label_type="boolean", input_dim=32),
        seed=6,
        out_dir=stage,
        encoder_sha=weights_sha256(toy_checkpoint.path / "model.safetensors"),
    )
    # Rewrite the record without the encoder binding.
    import json

    record_path = stage / "head" / "head.json"
    record = json.loads(record_path.read_text())
    del record["encoder_weights_sha256"]
    record_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    registry_dir = tmp_path_factory.mktemp("registry-unbound")
    HeadRegistry(registry_dir).publish(stage / "head", "alpha")
    with pytest.raises(HeadLoadError, match="records no encoder"):
        load_state(_config(toy_checkpoint, registry_dir))


def test_pinned_record_mismatch_is_refused(
    toy_checkpoint: ToyCheckpoint, good_registry: Path
) -> None:
    """Registry.load refuses a head whose record digest differs from the pin."""
    from ehrfm.inference.registry import EncoderIdentity

    sha = weights_sha256(toy_checkpoint.path / "model.safetensors")
    identity = EncoderIdentity(hidden_size=32, weights_sha256=sha)
    with pytest.raises(HeadLoadError, match="expected"):
        HeadRegistry(good_registry).load("alpha", identity, expected_record_sha256="f" * 64)


def test_toy_config_ignores_ontology_validation() -> None:
    """Toy mode derives its ontology from the toy extract; no extract_dir needed."""
    config = ServerConfig(toy=True)
    assert config.ontology == "extract"
    assert config.extract_dir is None


def test_extract_ontology_requires_extract_dir() -> None:
    with pytest.raises(ValidationError, match="extract_dir"):
        ServerConfig(model_dir=Path("/tmp/m"), heads_dir=Path("/tmp/h"))
    config = ServerConfig(
        model_dir=Path("/tmp/m"), heads_dir=Path("/tmp/h"), extract_dir=Path("/tmp/e")
    )
    assert config.ontology == "extract"
    assert (
        ServerConfig(model_dir=Path("/tmp/m"), heads_dir=Path("/tmp/h"), ontology="none").ontology
        == "none"
    )
