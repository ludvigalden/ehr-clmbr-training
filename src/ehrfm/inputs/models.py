"""Pydantic models of the contract's JSON files, as read by this package.

They mirror the upstream contract field for field and are frozen, strict and
reject unknown keys, so a manifest this package accepts is one the contract
defines.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from ehrfm.inputs.contract import CARE_SITE_VOCABULARY, CODE_PATTERN, SPLITS, Split

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Timestamp = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]
SemVer = Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]
TaskName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_]*$")]
VocabularyCode = Annotated[str, StringConstraints(pattern=CODE_PATTERN)]
LabelType = Literal["boolean", "integer", "float", "categorical"]
EntryType = Literal["code", "numeric", "text", "unused"]
ArtifactType = Literal["meds_extract", "label_set", "clmbr_concept_set"]


def _relative_posix(value: str) -> str:
    parts = value.split("/")
    if not value or "\\" in value or "\n" in value or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"{value!r} is not a normalised relative POSIX path")
    return value


RelPath = Annotated[str, AfterValidator(_relative_posix)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def _require_sorted_unique(values: list[str], what: str) -> None:
    if values != sorted(set(values)):
        raise ValueError(f"{what} must be sorted and unique")


def _require_all_splits(counts: dict[Split, int], what: str) -> None:
    if set(counts) != set(SPLITS):
        raise ValueError(f"{what} must have exactly the keys {list(SPLITS)}")
    if any(v < 0 for v in counts.values()):
        raise ValueError(f"{what} values must be non-negative")


class Producer(_Frozen):
    """The tool and version that wrote an artifact."""

    name: Literal["omopmap"]
    version: SemVer


class InputRecord(_Frozen):
    """A hashed input the artifact was derived from."""

    name: str = Field(min_length=1)
    sha256: Sha256


class FileRecord(_Frozen):
    """One file of an artifact directory."""

    path: RelPath
    sha256: Sha256
    bytes: int = Field(ge=0)
    rows: int | None = Field(ge=0)


class MedsExtractSummary(_Frozen):
    """Counts recorded for a MEDS extract."""

    artifact_type: Literal["meds_extract"]
    meds_version: Literal["0.3.3"]
    subjects: int = Field(ge=0)
    events: int = Field(ge=0)
    distinct_codes: int = Field(ge=0)
    subjects_by_split: dict[Split, int]
    events_by_split: dict[Split, int]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        _require_all_splits(self.subjects_by_split, "subjects_by_split")
        _require_all_splits(self.events_by_split, "events_by_split")
        if sum(self.subjects_by_split.values()) != self.subjects:
            raise ValueError("subjects_by_split does not sum to subjects")
        if sum(self.events_by_split.values()) != self.events:
            raise ValueError("events_by_split does not sum to events")
        return self


class LabelSetSummary(_Frozen):
    """Counts recorded for a label set."""

    artifact_type: Literal["label_set"]
    task_name: TaskName
    label_type: LabelType
    extract_manifest_sha256: Sha256
    labels: int = Field(ge=0)
    subjects: int = Field(ge=0)
    labels_by_split: dict[Split, int]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        _require_all_splits(self.labels_by_split, "labels_by_split")
        if sum(self.labels_by_split.values()) != self.labels:
            raise ValueError("labels_by_split does not sum to labels")
        return self


class ConceptSetSummary(_Frozen):
    """Counts recorded for a CLMBR concept-set export."""

    artifact_type: Literal["clmbr_concept_set"]
    dictionary_sha256: Sha256
    concepts: int = Field(ge=0)
    concept_list_sha256: Sha256
    vocabulary_counts: dict[str, int]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if sum(self.vocabulary_counts.values()) != self.concepts:
            raise ValueError("vocabulary_counts does not sum to concepts")
        if CARE_SITE_VOCABULARY in self.vocabulary_counts:
            raise ValueError(f"{CARE_SITE_VOCABULARY} entries are excluded from the concept set")
        return self


ArtifactSummary = Annotated[
    MedsExtractSummary | LabelSetSummary | ConceptSetSummary,
    Field(discriminator="artifact_type"),
]


class ArtifactManifest(_Frozen):
    """``artifact_manifest.json``: the integrity record at the root of every artifact."""

    contract_version: SemVer
    artifact_type: ArtifactType
    producer: Producer
    created_at: Timestamp
    inputs: tuple[InputRecord, ...]
    files: tuple[FileRecord, ...]
    summary: ArtifactSummary

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.summary.artifact_type != self.artifact_type:
            raise ValueError("artifact_type differs from summary.artifact_type")
        _require_sorted_unique([i.name for i in self.inputs], "inputs (by name)")
        _require_sorted_unique([f.path for f in self.files], "files (by path)")
        return self


class DatasetMetadata(_Frozen):
    """``metadata/dataset.json`` of a MEDS extract."""

    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    etl_name: Literal["omopmap"]
    etl_version: SemVer
    meds_version: Literal["0.3.3"]
    created_at: Timestamp


class TaskDefinition(_Frozen):
    """``task.json`` of a label set."""

    name: TaskName
    label_type: LabelType
    description: str = Field(min_length=1)
    categories: tuple[str, ...] | None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.label_type == "categorical":
            if not self.categories:
                raise ValueError("categorical tasks must list their categories")
            _require_sorted_unique(list(self.categories), "categories")
        elif self.categories is not None:
            raise ValueError("categories are only allowed for categorical tasks")
        return self


class ClmbrManifest(_Frozen):
    """The producer's committed CLMBR dictionary manifest: counts and hashes only."""

    contract_version: SemVer
    source_repo: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    filename: str = Field(min_length=1)
    file_sha256: Sha256
    entry_type_counts: dict[EntryType, int]
    vocabulary_counts: dict[str, int]
    total: int = Field(ge=0)
    care_site_excluded: int = Field(ge=0)
    concept_list_sha256: Sha256

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if set(self.entry_type_counts) != {"code", "numeric", "text", "unused"}:
            raise ValueError("entry_type_counts must list all four entry types")
        if sum(self.vocabulary_counts.values()) != self.total:
            raise ValueError("vocabulary_counts does not sum to total")
        if CARE_SITE_VOCABULARY in self.vocabulary_counts:
            raise ValueError(f"{CARE_SITE_VOCABULARY} must not appear in vocabulary_counts")
        return self


__all__ = [
    "ArtifactManifest",
    "ArtifactSummary",
    "ArtifactType",
    "ClmbrManifest",
    "ConceptSetSummary",
    "DatasetMetadata",
    "FileRecord",
    "InputRecord",
    "LabelSetSummary",
    "LabelType",
    "MedsExtractSummary",
    "Producer",
    "TaskDefinition",
    "VocabularyCode",
]
