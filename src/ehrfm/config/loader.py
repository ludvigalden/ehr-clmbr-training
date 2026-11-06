"""Compose a Hydra/OmegaConf experiment configuration into an :class:`ExperimentConfig`.

Two layouts are accepted:

- A file in the ``experiment`` group of a config tree, for example
  ``configs/experiment/toy.yaml``: the tree root is the directory holding
  ``main.yaml``, and the file is selected as ``experiment=<stem>``. The
  experiment file may override other groups (``override /data: toy``).
- Any other YAML or JSON file holding the whole tree (the five groups as
  top-level keys).

Overrides use Hydra syntax (``group.key=value``; ``+key=value`` adds a key,
which the strict models then reject unless it exists). Interpolations are
resolved before validation; ``${oc.env:VAR}`` without a default fails when
``VAR`` is unset, and every failure is reported as one
:class:`ConfigError` that names the offending key.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import (
    InterpolationResolutionError,
    MissingMandatoryValue,
    OmegaConfBaseException,
)
from pydantic import ValidationError

from ehrfm.config.experiment import ExperimentConfig

MAIN_CONFIG: Final = "main"
EXPERIMENT_GROUP: Final = "experiment"
_MISSING_ENV = re.compile(r"Environment variable '([^']+)' not found")


class ConfigError(ValueError):
    """The configuration could not be composed, resolved or validated."""


def config_root(path: Path) -> Path | None:
    """The config tree root when ``path`` is an ``experiment`` group file of one."""
    parent = path.parent
    root = parent.parent
    if parent.name == EXPERIMENT_GROUP and (root / f"{MAIN_CONFIG}.yaml").is_file():
        return root
    return None


def _compose(root: Path, overrides: Sequence[str]) -> DictConfig:
    from hydra import compose, initialize_config_dir
    from hydra.errors import HydraException

    try:
        with initialize_config_dir(config_dir=str(root), version_base="1.3", job_name="ehrfm"):
            return compose(config_name=MAIN_CONFIG, overrides=list(overrides))
    except HydraException as exc:
        raise ConfigError(f"cannot compose {root}: {exc}") from exc


def _load_file(path: Path, overrides: Sequence[str]) -> DictConfig:
    import yaml

    try:
        loaded = OmegaConf.load(path)
    except (OSError, OmegaConfBaseException, ValueError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(loaded, DictConfig):
        raise ConfigError(f"{path} does not hold a mapping")
    for override in overrides:
        key, sep, _ = override.lstrip("+").partition("=")
        if not sep or not key:
            raise ConfigError(f"override {override!r} is not of the form key=value")
    try:
        merged = OmegaConf.merge(loaded, OmegaConf.from_dotlist([o.lstrip("+") for o in overrides]))
    except OmegaConfBaseException as exc:
        raise ConfigError(f"cannot apply overrides to {path}: {exc}") from exc
    if not isinstance(merged, DictConfig):
        raise ConfigError(f"{path} does not hold a mapping")
    return merged


def _resolve(tree: DictConfig, source: Path) -> dict[str, Any]:
    try:
        data: Any = OmegaConf.to_container(tree, resolve=True, throw_on_missing=True)
    except InterpolationResolutionError as exc:
        missing = _MISSING_ENV.search(str(exc))
        where = exc.full_key or "?"
        if missing is not None:
            raise ConfigError(
                f"environment variable {missing.group(1)} is not set "
                f"(needed by {where} in {source})"
            ) from exc
        raise ConfigError(f"cannot resolve {where} in {source}: {exc.msg}") from exc
    except MissingMandatoryValue as exc:
        raise ConfigError(f"{exc.full_key} has no value in {source}") from exc
    except OmegaConfBaseException as exc:
        raise ConfigError(f"cannot resolve {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{source} does not hold a mapping")
    return {str(key): value for key, value in data.items()}


def _validation_message(exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"  {where}: {error['msg']}")
    return "\n".join(lines)


def compose_experiment(path: Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Compose and resolve ``path`` with ``overrides`` into a plain mapping (not yet validated)."""
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"{path} is not a file")
    root = config_root(path)
    if root is not None:
        tree = _compose(root, [f"{EXPERIMENT_GROUP}={path.stem}", *overrides])
    else:
        tree = _load_file(path, overrides)
    return _resolve(tree, path)


def validate_experiment(data: dict[str, Any], source: str = "configuration") -> ExperimentConfig:
    """Validate a resolved mapping (strict: unknown keys and wrong types fail)."""
    try:
        # JSON mode: lists validate as tuples and strings as paths, nothing else is coerced.
        return ExperimentConfig.model_validate_json(json.dumps(data))
    except ValidationError as exc:
        raise ConfigError(f"invalid {source}:\n{_validation_message(exc)}") from exc


def load_experiment_config(path: Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Compose, resolve and validate the experiment configuration at ``path``."""
    return validate_experiment(compose_experiment(path, overrides), str(path))


__all__ = [
    "ConfigError",
    "compose_experiment",
    "config_root",
    "load_experiment_config",
    "validate_experiment",
]
