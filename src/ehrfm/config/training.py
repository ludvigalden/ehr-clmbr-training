"""Configuration of a continued-pretraining run.

A :class:`TrainingConfig` has three parts, split by what they do to the run's
identity:

``inputs``
    What is trained on and from: the MEDS extract, the base checkpoint
    directory, an optional SHA-256 pin of the base weights and an optional
    concept set the base dictionary must match. Inputs enter the run
    fingerprint by content hash, never by path.
``optimization``
    Every setting that changes the trained weights. All of it enters the run
    fingerprint, so changing any field invalidates a cached result.
``runtime``
    Where and how the run executes: output directory, device, checkpoint
    cadence, worker counts. None of it enters the fingerprint; a rerun on
    another device or with another checkpoint cadence is still a cache hit.

There are no default locations: the extract, the base checkpoint and the
output directory always come from the configuration. Models are strict,
frozen and reject unknown keys; path fields also accept strings so that YAML
and JSON files validate.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, Strict, field_validator, model_validator

ConfigPath = Annotated[Path, Strict(False)]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HOOK_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")


class _Frozen(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class TrainingInputs(_Frozen):
    """What a run trains on and starts from."""

    extract: ConfigPath
    """Verified MEDS extract directory (with its ``artifact_manifest.json``)."""
    base_checkpoint: ConfigPath
    """Checkpoint directory holding ``config.json``, ``dictionary.msgpack`` and
    ``model.safetensors``."""
    expected_weights_sha256: str | None = None
    """Pin of ``model.safetensors``; a different file fails before any tensor is built."""
    concept_set: ConfigPath | None = None
    """Concept-set export whose dictionary hash the base dictionary must match."""

    @field_validator("expected_weights_sha256")
    @classmethod
    def _sha256(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.match(value) is None:
            raise ValueError("expected_weights_sha256 must be a lowercase hex SHA-256 digest")
        return value


class VocabExpansionConfig(_Frozen):
    """Optional vocabulary-expansion step, off by default.

    ``hook`` names a function registered with
    :func:`ehrfm.training.vocab_expansion.register_vocab_expansion`. It may
    replace up to ``n_replace`` dictionary entries before tokenization; the
    embeddings of replaced entries start at zero.
    """

    hook: str | None = None
    n_replace: int = Field(default=0, ge=0)

    @field_validator("hook")
    @classmethod
    def _hook_name(cls, value: str | None) -> str | None:
        if value is not None and _HOOK_NAME.match(value) is None:
            raise ValueError(f"vocabulary-expansion hook name {value!r} is not a valid identifier")
        return value

    @property
    def enabled(self) -> bool:
        """Whether a hook runs."""
        return self.hook is not None


class OptimizationConfig(_Frozen):
    """Every setting that changes the trained weights (all fingerprinted)."""

    seed: int = 0
    tokens_per_batch: int = Field(default=8192, gt=0)
    """Token budget of one packed femr batch; one batch is one optimizer step per process."""
    epochs: int = Field(default=1, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    """Stop after this many optimizer steps, even inside an epoch."""
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=0, ge=0)
    schedule: Literal["constant", "cosine"] = "cosine"
    max_grad_norm: float | None = Field(default=1.0, gt=0)
    validation_fraction: float = Field(default=0.1, ge=0, lt=1)
    """Share of the extract's train subjects held out for the validation loss."""
    clmbr_vocab_size: int | None = Field(default=None, gt=0)
    """Pin of the base checkpoint's CLMBR head size (8192 for CLMBR-T-Base)."""
    precision: Literal["fp32", "bf16"] = "fp32"
    """``bf16`` autocasts the forward pass on CUDA; CPU runs always use fp32."""
    vocab_expansion: VocabExpansionConfig = VocabExpansionConfig()


class RuntimeConfig(_Frozen):
    """Where and how a run executes (never fingerprinted)."""

    output_dir: ConfigPath
    """Run directory: the finished checkpoint lands in ``<output_dir>/model``."""
    device: str = "auto"
    """``auto``, ``cpu``, ``cuda``, ``cuda:<n>`` or ``mps``; distributed runs use their local GPU."""
    database_cache: ConfigPath | None = None
    """meds_reader database cache; defaults to ``<output_dir>/database``."""
    checkpoint_every: int = Field(default=500, gt=0)
    keep_checkpoints: int = Field(default=2, gt=0)
    log_every: int = Field(default=10, gt=0)
    tokenize_workers: int = Field(default=1, gt=0)
    deterministic: bool = True
    """Request deterministic kernels (``torch.use_deterministic_algorithms``, warn only)."""


class TrainingConfig(_Frozen):
    """A complete continued-pretraining run."""

    inputs: TrainingInputs
    optimization: OptimizationConfig = OptimizationConfig()
    runtime: RuntimeConfig

    @model_validator(mode="after")
    def _expansion_bound(self) -> Self:
        expansion = self.optimization.vocab_expansion
        if expansion.n_replace and not expansion.enabled:
            raise ValueError("vocab_expansion.n_replace is set but no hook is named")
        return self

    @property
    def database_cache(self) -> Path:
        """The meds_reader database cache directory."""
        return self.runtime.database_cache or self.runtime.output_dir / "database"


def load_training_config(path: Path) -> TrainingConfig:
    """Read a YAML or JSON training configuration.

    OmegaConf resolves interpolations first, so ``${oc.env:VAR}`` reads paths
    from the environment instead of the file.
    """
    loaded = OmegaConf.load(path)
    data: Any = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a mapping")
    return TrainingConfig.model_validate(data)


__all__ = [
    "OptimizationConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "TrainingInputs",
    "VocabExpansionConfig",
    "load_training_config",
]
