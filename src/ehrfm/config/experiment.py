"""Configuration of an experiment: one pipeline run from inputs to evaluation.

:class:`ExperimentConfig` mirrors the composed Hydra tree under ``configs/``
one-to-one: its five fields are the five config groups (``model``, ``data``,
``tasks``, ``experiment`` and ``eval``), and every nested model is strict,
frozen and rejects unknown keys, so a misspelled key in a YAML file or on the
command line fails before any work starts.

There are no default locations. The extract, the base checkpoint and the work
directory come from the configuration, usually through ``${oc.env:VAR}``
interpolation without a fallback; a missing variable fails composition (see
:mod:`ehrfm.config.loader`). The one exception is the toy mode
(``data.toy``), which generates a synthetic extract, label set and checkpoint
inside the work directory.
"""

from __future__ import annotations

import re
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from ehrfm.config.training import ConfigPath, OptimizationConfig
from ehrfm.inputs.contract import Split

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")

Name = Annotated[str, Field(pattern=_NAME.pattern)]
"""A lowercase identifier usable as a directory and MLflow name."""


class _Frozen(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class ToyInputs(_Frozen):
    """Generate the synthetic extract, label set and toy checkpoint instead of reading inputs."""

    subjects: int = Field(default=60, ge=10)
    seed: int = 7


class DataConfig(_Frozen):
    """The MEDS extract (or the toy generator) and an optional concept set."""

    extract: ConfigPath | None = None
    """Verified MEDS extract directory."""
    concept_set: ConfigPath | None = None
    """Concept-set export the base dictionary must match (checked before pretraining)."""
    toy: ToyInputs | None = None

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if (self.extract is None) == (self.toy is None):
            raise ValueError("set exactly one of data.extract and data.toy")
        if self.toy is not None and self.concept_set is not None:
            raise ValueError("data.concept_set cannot be combined with data.toy")
        return self


class ModelConfig(_Frozen):
    """The base checkpoint and where models run."""

    base_checkpoint: ConfigPath | None = None
    """Checkpoint directory; must be null in toy mode (a toy checkpoint is generated)."""
    expected_weights_sha256: str | None = None
    device: str = "auto"
    """``auto``, ``cpu``, ``cuda``, ``cuda:<n>`` or ``mps``."""

    @field_validator("expected_weights_sha256")
    @classmethod
    def _sha256(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.match(value) is None:
            raise ValueError("expected_weights_sha256 must be a lowercase hex SHA-256 digest")
        return value


class HookSpec(_Frozen):
    """One registered pipeline hook and its options (see :mod:`ehrfm.experiment.hooks`)."""

    name: Name
    options: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    """Passed to the hook verbatim; part of the hook's cache key."""


class TaskSpec(_Frozen):
    """A prediction task: a verified label set over the extract."""

    name: Name
    labels: ConfigPath


class TasksConfig(_Frozen):
    """Tasks whose prediction times get representations, and the head hooks over them."""

    label_sets: tuple[TaskSpec, ...] = ()
    """Empty in toy mode (the generated label set is used); empty otherwise means
    per-subject representations without labels."""
    heads: tuple[HookSpec, ...] = ()

    @model_validator(mode="after")
    def _unique(self) -> Self:
        names = [task.name for task in self.label_sets]
        if len(names) != len(set(names)):
            raise ValueError("tasks.label_sets names must be unique")
        _require_unique_hooks(self.heads, "tasks.heads")
        return self


class EvalConfig(_Frozen):
    """Evaluation hooks, run after the head hooks."""

    hooks: tuple[HookSpec, ...] = ()

    @model_validator(mode="after")
    def _unique(self) -> Self:
        _require_unique_hooks(self.hooks, "eval.hooks")
        return self


def _require_unique_hooks(hooks: tuple[HookSpec, ...], where: str) -> None:
    names = [hook.name for hook in hooks]
    if len(names) != len(set(names)):
        raise ValueError(f"{where} names must be unique")


class CoverageStageConfig(_Frozen):
    """Dictionary coverage of the extract by the base checkpoint."""

    enabled: bool = True
    top_n: int = Field(default=20, ge=0)
    min_events: int = Field(default=10, ge=1)


class PretrainingStageConfig(_Frozen):
    """Optional continued pretraining of the base checkpoint (:mod:`ehrfm.training`)."""

    enabled: bool = False
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint_every: int = Field(default=500, gt=0)
    keep_checkpoints: int = Field(default=2, gt=0)
    log_every: int = Field(default=10, gt=0)
    tokenize_workers: int = Field(default=1, gt=0)
    deterministic: bool = True


class RepresentationStageConfig(_Frozen):
    """Representation extraction with the (continued-pretrained or base) model."""

    splits: tuple[Split, ...] = ("train", "tuning", "held_out")
    tokens_per_batch: int = Field(default=8192, gt=0)

    @field_validator("splits")
    @classmethod
    def _nonempty_unique(cls, value: tuple[Split, ...]) -> tuple[Split, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("representations.splits must be non-empty and unique")
        return value


class TrackingConfig(_Frozen):
    """MLflow tracking."""

    enabled: bool = True
    uri: str | None = None
    """Tracking URI; null means a local file store at ``<work_dir>/mlruns``."""
    experiment_name: Name | None = None
    """MLflow experiment; defaults to ``experiment.name``."""


DEFAULT_COUNT_FIELDS: tuple[str, ...] = (
    "n",
    "n_*",
    "count",
    "counts",
    "*_count",
    "*_counts",
    "events",
    "events_*",
    "*_events",
    "subjects",
    "subjects_*",
    "*_subjects",
    "labels",
    "labels_*",
    "*_labels",
    "positives",
    "negatives",
    "*_positive",
    "*_negative",
    "support",
)
"""Field names (``fnmatch`` patterns on the last key segment) that hold counts of
people, events or labels."""


class SmallCellPolicy(_Frozen):
    """Small-cell suppression applied to everything sent to the tracking store.

    A *count field* is a key whose name matches one of ``count_fields``; its
    value is a cell when it is a number, and every number inside it is a cell
    when it is a list or mapping (a breakdown such as ``subjects_by_split``).
    A cell below ``threshold`` is masked: ``null`` in JSON artifacts, omitted
    from metrics, the string ``suppressed`` in parameters.
    """

    threshold: int = Field(default=5, ge=1)
    """Smallest count that may be published."""
    count_fields: tuple[str, ...] = DEFAULT_COUNT_FIELDS
    suppress_zero: bool = True
    """Treat 0 as a small cell too; off publishes structural zeros."""
    mask_group: bool = True
    """Also mask every other number in a record (or metric group) whose count is
    suppressed, since rates over a small cell disclose it."""
    complementary: bool = True
    """When exactly one cell of a breakdown or of a list of sibling records is
    suppressed, also suppress the next smallest, so a published total cannot
    recover it."""

    @field_validator("count_fields")
    @classmethod
    def _patterns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not pattern for pattern in value):
            raise ValueError("count_fields must be a non-empty list of non-empty patterns")
        return value


class ExperimentSection(_Frozen):
    """The ``experiment`` config group: identity, work directory and stage settings."""

    name: Name
    seed: int = 0
    work_dir: ConfigPath
    """Root of the stage caches, the pretraining run, the database cache and the
    default MLflow store."""
    coverage: CoverageStageConfig = CoverageStageConfig()
    pretraining: PretrainingStageConfig = PretrainingStageConfig()
    representations: RepresentationStageConfig = RepresentationStageConfig()
    tracking: TrackingConfig = TrackingConfig()
    suppression: SmallCellPolicy = SmallCellPolicy()
    hook_modules: tuple[str, ...] = ()
    """Modules imported before the pipeline runs so they can register hooks."""


class ExperimentConfig(_Frozen):
    """A complete experiment, one field per Hydra config group."""

    model: ModelConfig
    data: DataConfig
    tasks: TasksConfig = TasksConfig()
    experiment: ExperimentSection
    eval: EvalConfig = EvalConfig()

    @model_validator(mode="after")
    def _toy_consistency(self) -> Self:
        if self.data.toy is not None:
            if self.model.base_checkpoint is not None:
                raise ValueError("model.base_checkpoint must be null when data.toy is set")
            if self.tasks.label_sets:
                raise ValueError("tasks.label_sets must be empty when data.toy is set")
        elif self.model.base_checkpoint is None:
            raise ValueError("model.base_checkpoint is required unless data.toy is set")
        return self

    @property
    def tracking_experiment(self) -> str:
        """The MLflow experiment name."""
        return self.experiment.tracking.experiment_name or self.experiment.name


__all__ = [
    "DEFAULT_COUNT_FIELDS",
    "CoverageStageConfig",
    "DataConfig",
    "EvalConfig",
    "ExperimentConfig",
    "ExperimentSection",
    "HookSpec",
    "ModelConfig",
    "PretrainingStageConfig",
    "RepresentationStageConfig",
    "SmallCellPolicy",
    "TaskSpec",
    "TasksConfig",
    "ToyInputs",
    "TrackingConfig",
]
