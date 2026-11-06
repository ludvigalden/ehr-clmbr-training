"""Typed, frozen server configuration with environment overrides.

Every value is explicit: there are no site defaults and no ``.env`` files.
:func:`ServerConfig.from_env` reads the ``EHRFM_SERVER_`` prefixed variables
listed in :class:`ServerConfig`'s field descriptions; anything unset keeps its
field default, and every value is validated by the strict, frozen model.

The two serving layouts are mutually exclusive:

- named arguments (or environment variables) pointing at an existing encoder
  checkpoint (``model_dir``) and a head registry (``heads_dir``);
- ``toy=True``, which serves the generated toy checkpoint with a head fitted
  on the fly from the toy label set (see :mod:`ehrfm.server.toy`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, Strict, model_validator

ENV_PREFIX: Final = "EHRFM_SERVER_"
DEFAULT_TINY_SUBJECTS: Final = 20
"""Subjects of the toy extract served by ``--toy`` (kept small so startup fits fast)."""

_TOY_VALUES: Final = {"1", "true", "yes", "on"}

ConfigPath = Annotated[Path, Strict(False)]
"""A path field that also accepts the string form (as every config module here does)."""


def _int(name: str, value: str) -> int:
    """Parse an integer variable, naming the variable on a bad value."""
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def default_toy_work_dir(env: Mapping[str, str] | None = None) -> Path:
    """The default ``--toy`` work directory: ``<XDG cache home>/ehrfm/toy-server``."""
    source: Mapping[str, str] = os.environ if env is None else env
    cache_home = source.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache_home) / "ehrfm" / "toy-server"


class ServerConfig(BaseModel):
    """Everything the serving app needs, frozen once built."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    model_dir: ConfigPath | None = None
    """Checkpoint directory to load (required unless ``toy``)."""
    expected_weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """Weight pin passed to the encoder loader; a mismatch fails startup."""
    heads_dir: ConfigPath | None = None
    """Head registry root (required unless ``toy``)."""
    extract_dir: ConfigPath | None = None
    """Extract whose ``codes.parquet`` parents expand request codes into ancestor
    tokens. Required when ``ontology="extract"`` (the default) and not toy; with
    ``ontology="none"`` codes tokenize with no parents."""
    ontology: Literal["extract", "none"] = "extract"
    """Where the tokenizer's ancestor expansion comes from. ``none`` is an
    explicit opt-out that degrades codes to themselves; it is echoed by
    ``/ready`` and ``/health``."""
    device: str = "cpu"
    tokens_per_batch: int = Field(default=8192, ge=1)
    postprocess: str = "raw"
    """Name of the registered postprocess hook applied to head outputs."""
    unknown_code_limit: int = Field(default=20, ge=1)
    """Most unknown codes a response lists (the counts always cover every event)."""
    toy: bool = False
    toy_subjects: int = Field(default=DEFAULT_TINY_SUBJECTS, ge=10)
    toy_work_dir: ConfigPath | None = None

    @model_validator(mode="after")
    def _one_layout(self) -> Self:
        if self.toy and (self.model_dir is not None or self.heads_dir is not None):
            raise ValueError("toy mode takes no model_dir or heads_dir")
        if not self.toy and (self.model_dir is None or self.heads_dir is None):
            raise ValueError("pass model_dir and heads_dir, or set toy=true")
        if self.ontology == "extract" and not self.toy and self.extract_dir is None:
            raise ValueError(
                'ontology="extract" needs extract_dir (the request codes\' ancestor '
                'source); set ontology="none" to serve codes without ancestors'
            )
        return self

    @classmethod
    def env_kwargs(cls, env: Mapping[str, str] | None = None) -> dict[str, object]:
        """The ``EHRFM_SERVER_`` prefixed variables, unvalidated, for merging with
        command-line flags before validation (see :meth:`from_env`)."""
        source: Mapping[str, str] = os.environ if env is None else env

        def raw(key: str) -> str | None:
            value = source.get(f"{ENV_PREFIX}{key}")
            return value if value not in (None, "") else None

        kwargs: dict[str, object] = {}
        strings = {
            "host": "HOST",
            "model_dir": "MODEL_DIR",
            "expected_weights_sha256": "WEIGHTS_SHA256",
            "heads_dir": "HEADS_DIR",
            "extract_dir": "EXTRACT",
            "ontology": "ONTOLOGY",
            "device": "DEVICE",
            "postprocess": "POSTPROCESS",
            "toy_work_dir": "TOY_WORK_DIR",
        }
        for field_name, suffix in strings.items():
            value = raw(suffix)
            if value is not None:
                kwargs[field_name] = value
        if (port := raw("PORT")) is not None:
            kwargs["port"] = _int(f"{ENV_PREFIX}PORT", port)
        if (tokens := raw("TOKENS_PER_BATCH")) is not None:
            kwargs["tokens_per_batch"] = _int(f"{ENV_PREFIX}TOKENS_PER_BATCH", tokens)
        if (limit := raw("UNKNOWN_CODE_LIMIT")) is not None:
            kwargs["unknown_code_limit"] = _int(f"{ENV_PREFIX}UNKNOWN_CODE_LIMIT", limit)
        if (toy := raw("TOY")) is not None:
            kwargs["toy"] = toy.lower() in _TOY_VALUES
        if (subjects := raw("TOY_SUBJECTS")) is not None:
            kwargs["toy_subjects"] = _int(f"{ENV_PREFIX}TOY_SUBJECTS", subjects)
        return kwargs

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServerConfig:
        """Build a configuration from ``EHRFM_SERVER_`` prefixed variables."""
        return cls.model_validate(cls.env_kwargs(env))


__all__ = ["DEFAULT_TINY_SUBJECTS", "ENV_PREFIX", "ServerConfig", "default_toy_work_dir"]
