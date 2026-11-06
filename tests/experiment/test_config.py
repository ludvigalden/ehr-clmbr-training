"""Experiment configuration: Hydra composition into strict, frozen models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ehrfm.config.experiment import ExperimentConfig
from ehrfm.config.loader import ConfigError, load_experiment_config, validate_experiment
from ehrfm.toy.pretraining import toy_optimization

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
TOY = CONFIGS / "experiment" / "toy.yaml"
DEFAULT = CONFIGS / "experiment" / "default.yaml"


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for name in ("EHRFM_WORK_DIR", "EHRFM_EXTRACT", "EHRFM_BASE_CHECKPOINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EHRFM_WORK_DIR", str(tmp_path / "work"))
    return tmp_path


def test_toy_config_composes(env: Path) -> None:
    config = load_experiment_config(TOY)
    assert config.experiment.name == "toy"
    assert config.experiment.work_dir == env / "work" / "toy"
    assert config.data.toy is not None and config.data.extract is None
    assert config.model.base_checkpoint is None
    assert config.experiment.pretraining.enabled
    # The YAML toy optimisation is the package's toy optimisation, field for field.
    assert config.experiment.pretraining.optimization == toy_optimization()
    assert config.experiment.suppression.threshold == 5


def test_default_config_reads_paths_from_the_environment(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EHRFM_EXTRACT", str(env / "extract"))
    monkeypatch.setenv("EHRFM_BASE_CHECKPOINT", str(env / "ckpt"))
    config = load_experiment_config(DEFAULT)
    assert config.data.extract == env / "extract"
    assert config.model.base_checkpoint == env / "ckpt"
    assert config.experiment.work_dir == env / "work" / "default"
    assert not config.experiment.pretraining.enabled


def test_missing_environment_variable_fails_closed(env: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        load_experiment_config(DEFAULT)
    message = str(caught.value)
    assert "environment variable EHRFM_BASE_CHECKPOINT is not set" in message
    assert "model.base_checkpoint" in message


def test_toy_without_work_dir_variable_uses_a_relative_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EHRFM_WORK_DIR", raising=False)
    config = load_experiment_config(TOY)
    assert config.experiment.work_dir == Path("outputs/toy")


@pytest.mark.parametrize(
    ("override", "where"),
    [
        ("+experiment.bogus=1", "experiment.bogus"),
        ("+experiment.suppression.minimum=3", "experiment.suppression.minimum"),
        ("+model.extra=x", "model.extra"),
        ("+unknown_group=1", "unknown_group"),
    ],
)
def test_unknown_key_fails(env: Path, override: str, where: str) -> None:
    with pytest.raises(ConfigError, match="Extra inputs are not permitted") as caught:
        load_experiment_config(TOY, [override])
    assert where in str(caught.value)


def test_overriding_a_key_that_does_not_exist_fails_in_composition(env: Path) -> None:
    with pytest.raises(ConfigError, match="Could not override"):
        load_experiment_config(TOY, ["experiment.bogus=1"])


@pytest.mark.parametrize(
    "override",
    [
        "experiment.seed=abc",
        "experiment.suppression.threshold=0",
        "experiment.representations.splits=[train,train]",
        "experiment.name=Has Spaces",
        "model.expected_weights_sha256=ABC",
    ],
)
def test_invalid_values_fail(env: Path, override: str) -> None:
    with pytest.raises(ConfigError, match="invalid"):
        load_experiment_config(TOY, [override])


def test_overrides_apply(env: Path) -> None:
    config = load_experiment_config(
        TOY, ["experiment.seed=3", "experiment.suppression.threshold=11"]
    )
    assert config.experiment.seed == 3
    assert config.experiment.suppression.threshold == 11


def test_cross_field_rules(env: Path) -> None:
    with pytest.raises(ConfigError, match="model.base_checkpoint must be null"):
        load_experiment_config(TOY, [f"model.base_checkpoint={env}"])
    with pytest.raises(ConfigError, match="exactly one of data.extract and data.toy"):
        load_experiment_config(TOY, [f"data.extract={env}"])


def test_models_are_frozen(env: Path) -> None:
    config = load_experiment_config(TOY)
    with pytest.raises(ValidationError):
        config.experiment.seed = 1  # pyright: ignore[reportAttributeAccessIssue]


def test_plain_file_layout_and_strict_types(env: Path, tmp_path: Path) -> None:
    whole = tmp_path / "whole.yaml"
    whole.write_text(
        "model: {device: cpu}\n"
        "data: {toy: {subjects: 20}}\n"
        "experiment:\n  name: plain\n  work_dir: ${oc.env:EHRFM_WORK_DIR}/plain\n"
    )
    config = load_experiment_config(whole, ["experiment.seed=4"])
    assert config.experiment.seed == 4 and config.data.toy is not None
    with pytest.raises(ConfigError, match="valid integer"):
        validate_experiment(
            {
                "model": {},
                "data": {"toy": {"subjects": "20"}},
                "experiment": {"name": "x", "work_dir": "w"},
            }
        )


def test_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="is not a file"):
        load_experiment_config(tmp_path / "none.yaml")


def test_malformed_yaml_fails_as_a_config_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("experiment: {name: [\n")
    with pytest.raises(ConfigError, match="cannot read"):
        load_experiment_config(broken)


def test_schema_has_one_field_per_config_group() -> None:
    groups = {path.name for path in CONFIGS.iterdir() if path.is_dir()}
    assert groups == set(ExperimentConfig.model_fields)
