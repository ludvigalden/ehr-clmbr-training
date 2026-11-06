"""Open and verify the consumed artifacts: MEDS extracts, label sets, concept sets.

Each ``open_*`` function verifies the artifact before returning a handle:
the manifest parses and has a readable contract version and the expected type;
the file set equals the manifest's list; every size, SHA-256 and Parquet row
count matches; every Parquet schema equals the contract's; the layout rules
hold; and the recorded summary agrees with counts this package can derive
cheaply (from Parquet footers and the metadata tables). Row-level content
rules (sort order, birth events, code format in every row) are the producer's
validator's responsibility and are not re-checked here; the hash check proves
the files are the ones that validator accepted.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, ValidationError

from ehrfm.inputs.contract import (
    CODES_PARQUET,
    CONCEPT_SET_PARQUET,
    DATA_DIR,
    DATASET_JSON,
    LABELS_PARQUET,
    SHARD_PATTERN,
    SPLITS,
    SUBJECT_SPLITS_PARQUET,
    TASK_JSON,
    Split,
)
from ehrfm.inputs.errors import (
    IntegrityError,
    LinkageError,
    ManifestFormatError,
    MissingArtifactError,
    UnexpectedFileError,
)
from ehrfm.inputs.manifest import concept_list_sha256, sha256_file, verified_manifest
from ehrfm.inputs.models import (
    ArtifactManifest,
    ClmbrManifest,
    ConceptSetSummary,
    DatasetMetadata,
    LabelSetSummary,
    MedsExtractSummary,
    TaskDefinition,
)
from ehrfm.inputs.schemas import (
    CONCEPT_SET_SCHEMA,
    LABEL_SCHEMA,
    MEDS_CODES_SCHEMA,
    MEDS_DATA_SCHEMA,
    SUBJECT_SPLITS_SCHEMA,
    check_schema,
    column_values,
)

_VALUE_COLUMNS: Final = {
    "boolean": "boolean_value",
    "integer": "integer_value",
    "float": "float_value",
    "categorical": "categorical_value",
}


class _Handle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class MedsExtract(_Handle):
    """A verified MEDS 0.3.3 extract directory."""

    root: Path
    manifest: ArtifactManifest
    manifest_sha256: str
    summary: MedsExtractSummary
    dataset: DatasetMetadata
    shards: dict[Split, tuple[str, ...]]

    def subject_splits(self) -> dict[int, Split]:
        """Map every subject id to its split, from ``metadata/subject_splits.parquet``."""
        table = pq.read_table(self.root / SUBJECT_SPLITS_PARQUET)
        ids: list[int] = column_values(table, "subject_id")
        names: list[str] = column_values(table, "split")
        return {sid: _as_split(self.root, name) for sid, name in zip(ids, names, strict=True)}


class LabelSet(_Handle):
    """A verified label set for one task."""

    root: Path
    manifest: ArtifactManifest
    manifest_sha256: str
    summary: LabelSetSummary
    task: TaskDefinition

    @property
    def labels_path(self) -> Path:
        """Path of ``labels.parquet``."""
        return self.root / LABELS_PARQUET


class ConceptSet(_Handle):
    """A verified CLMBR concept-set export."""

    root: Path
    manifest: ArtifactManifest
    manifest_sha256: str
    summary: ConceptSetSummary

    def codes(self) -> frozenset[str]:
        """The ``<vocabulary_id>/<concept_code>`` strings of the concept set."""
        column: list[str] = column_values(pq.read_table(self.root / CONCEPT_SET_PARQUET), "code")
        return frozenset(column)


def _as_split(root: Path, name: str) -> Split:
    for split in SPLITS:
        if name == split:
            return split
    raise IntegrityError(root, f"unknown split name {name!r}")


def _read_json_model[M: BaseModel](root: Path, rel: str, model: type[M]) -> M:
    try:
        return model.model_validate_json((root / rel).read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ManifestFormatError(root, f"{rel} is invalid: {exc}") from exc


def _require_files(root: Path, manifest: ArtifactManifest, required: set[str]) -> None:
    listed = {record.path for record in manifest.files}
    missing = sorted(required - listed)
    if missing:
        raise MissingArtifactError(root, f"required files are missing: {missing}")


def open_extract(root: Path) -> MedsExtract:
    """Verify and open the MEDS extract at ``root``."""
    manifest, manifest_sha = verified_manifest(root, "meds_extract")
    summary = manifest.summary
    if not isinstance(summary, MedsExtractSummary):
        raise ManifestFormatError(root, "summary is not a MEDS extract summary")
    _require_files(root, manifest, {DATASET_JSON, CODES_PARQUET, SUBJECT_SPLITS_PARQUET})

    shard_numbers: dict[Split, list[int]] = {split: [] for split in SPLITS}
    rows_by_split: dict[Split, int] = dict.fromkeys(SPLITS, 0)
    for record in manifest.files:
        if record.path in (DATASET_JSON, CODES_PARQUET, SUBJECT_SPLITS_PARQUET):
            continue
        match = SHARD_PATTERN.match(record.path)
        if match is None:
            raise UnexpectedFileError(root, f"{record.path} is not part of a MEDS extract")
        split = _as_split(root, match.group(1))
        shard_numbers[split].append(int(match.group(2)))
        rows_by_split[split] += record.rows or 0
        check_schema(root, record.path, MEDS_DATA_SCHEMA)
    for split in SPLITS:
        if not (root / DATA_DIR / split).is_dir():
            raise MissingArtifactError(root, f"split directory {DATA_DIR}/{split} is missing")
        numbers = sorted(shard_numbers[split])
        if numbers != list(range(len(numbers))):
            raise UnexpectedFileError(
                root, f"shards of {split} are numbered {numbers}, expected 0..n-1 without gaps"
            )

    check_schema(root, CODES_PARQUET, MEDS_CODES_SCHEMA)
    check_schema(root, SUBJECT_SPLITS_PARQUET, SUBJECT_SPLITS_SCHEMA)
    dataset = _read_json_model(root, DATASET_JSON, DatasetMetadata)

    if rows_by_split != summary.events_by_split:
        raise IntegrityError(
            root,
            f"events per split are {rows_by_split}, the summary records {summary.events_by_split}",
        )
    splits_table = pq.read_table(root / SUBJECT_SPLITS_PARQUET)
    split_names: list[str] = column_values(splits_table, "split")
    counted = Counter(_as_split(root, name) for name in split_names)
    subjects_by_split: dict[Split, int] = {split: counted[split] for split in SPLITS}
    if subjects_by_split != summary.subjects_by_split:
        raise IntegrityError(
            root,
            f"subjects per split are {subjects_by_split}, "
            f"the summary records {summary.subjects_by_split}",
        )
    codes_rows = pq.ParquetFile(root / CODES_PARQUET).metadata.num_rows
    if codes_rows != summary.distinct_codes:
        raise IntegrityError(
            root,
            f"{CODES_PARQUET} has {codes_rows} codes, the summary records "
            f"{summary.distinct_codes} distinct codes",
        )

    shards: dict[Split, tuple[str, ...]] = {
        split: tuple(f"{DATA_DIR}/{split}/{n}.parquet" for n in sorted(shard_numbers[split]))
        for split in SPLITS
    }
    return MedsExtract(
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        summary=summary,
        dataset=dataset,
        shards=shards,
    )


def open_label_set(root: Path, extract: MedsExtract | None = None) -> LabelSet:
    """Verify and open the label set at ``root``.

    With ``extract``, the label set must have been built against that extract
    (its recorded manifest hash must match), and the labels per split are
    re-derived from the extract's subject splits.
    """
    manifest, manifest_sha = verified_manifest(root, "label_set")
    summary = manifest.summary
    if not isinstance(summary, LabelSetSummary):
        raise ManifestFormatError(root, "summary is not a label-set summary")
    listed = {record.path for record in manifest.files}
    if listed != {LABELS_PARQUET, TASK_JSON}:
        raise UnexpectedFileError(
            root,
            f"a label set holds exactly {sorted({LABELS_PARQUET, TASK_JSON})}, found {sorted(listed)}",
        )
    check_schema(root, LABELS_PARQUET, LABEL_SCHEMA)
    task = _read_json_model(root, TASK_JSON, TaskDefinition)
    if task.name != summary.task_name or task.name != root.name:
        raise IntegrityError(
            root,
            f"task name {task.name!r} differs from the summary's {summary.task_name!r} "
            f"or the directory name {root.name!r}",
        )
    if task.label_type != summary.label_type:
        raise IntegrityError(
            root,
            f"task label_type {task.label_type} differs from the summary's {summary.label_type}",
        )

    table = pq.read_table(root / LABELS_PARQUET)
    if table.num_rows != summary.labels:
        raise IntegrityError(root, f"{table.num_rows} labels, the summary records {summary.labels}")
    set_column = _VALUE_COLUMNS[task.label_type]
    for column in _VALUE_COLUMNS.values():
        nulls = table.column(column).null_count
        expected_nulls = 0 if column == set_column else table.num_rows
        if nulls != expected_nulls:
            raise IntegrityError(
                root,
                f"{column} has {nulls} nulls; a {task.label_type} task sets exactly {set_column}",
            )
    subject_ids: list[int] = column_values(table, "subject_id")
    distinct = len(pc.unique(table.column("subject_id")))
    if distinct != summary.subjects:
        raise IntegrityError(
            root, f"{distinct} labelled subjects, the summary records {summary.subjects}"
        )

    if extract is not None:
        if summary.extract_manifest_sha256 != extract.manifest_sha256:
            raise LinkageError(
                root,
                f"labels were built against extract manifest {summary.extract_manifest_sha256}, "
                f"but {extract.root} has manifest {extract.manifest_sha256}",
            )
        splits = extract.subject_splits()
        unknown = sorted({sid for sid in subject_ids if sid not in splits})
        if unknown:
            raise LinkageError(root, f"labelled subjects absent from the extract: {unknown[:10]}")
        counted = Counter(splits[sid] for sid in subject_ids)
        labels_by_split: dict[Split, int] = {split: counted[split] for split in SPLITS}
        if labels_by_split != summary.labels_by_split:
            raise IntegrityError(
                root,
                f"labels per split are {labels_by_split}, "
                f"the summary records {summary.labels_by_split}",
            )

    return LabelSet(
        root=root, manifest=manifest, manifest_sha256=manifest_sha, summary=summary, task=task
    )


def open_concept_set(root: Path, clmbr_manifest: Path | None = None) -> ConceptSet:
    """Verify and open the CLMBR concept-set export at ``root``.

    With ``clmbr_manifest`` (the producer's committed dictionary manifest), its
    concept-list hash and dictionary hash must match the export's.
    """
    manifest, manifest_sha = verified_manifest(root, "clmbr_concept_set")
    summary = manifest.summary
    if not isinstance(summary, ConceptSetSummary):
        raise ManifestFormatError(root, "summary is not a concept-set summary")
    listed = {record.path for record in manifest.files}
    if listed != {CONCEPT_SET_PARQUET}:
        raise UnexpectedFileError(
            root, f"a concept set holds exactly {[CONCEPT_SET_PARQUET]}, found {sorted(listed)}"
        )
    check_schema(root, CONCEPT_SET_PARQUET, CONCEPT_SET_SCHEMA)
    codes: list[str] = column_values(pq.read_table(root / CONCEPT_SET_PARQUET), "code")
    if len(codes) != summary.concepts:
        raise IntegrityError(root, f"{len(codes)} concepts, the summary records {summary.concepts}")
    digest = concept_list_sha256(codes)
    if digest != summary.concept_list_sha256:
        raise IntegrityError(
            root,
            f"concept-list hash {digest} differs from the summary's {summary.concept_list_sha256}",
        )
    if clmbr_manifest is not None:
        if not clmbr_manifest.is_file():
            raise MissingArtifactError(root, f"CLMBR manifest {clmbr_manifest} does not exist")
        try:
            pinned = ClmbrManifest.model_validate_json(clmbr_manifest.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise ManifestFormatError(root, f"{clmbr_manifest} is invalid: {exc}") from exc
        if (
            pinned.concept_list_sha256 != summary.concept_list_sha256
            or pinned.file_sha256 != summary.dictionary_sha256
        ):
            raise LinkageError(
                root,
                f"concept set does not match {clmbr_manifest}: concept-list hash "
                f"{summary.concept_list_sha256} vs {pinned.concept_list_sha256}, dictionary hash "
                f"{summary.dictionary_sha256} vs {pinned.file_sha256}",
            )
    return ConceptSet(root=root, manifest=manifest, manifest_sha256=manifest_sha, summary=summary)


def dictionary_matches(concept_set: ConceptSet, dictionary_path: Path) -> bool:
    """Return True when ``dictionary_path`` is the dictionary the concept set came from."""
    return sha256_file(dictionary_path) == concept_set.summary.dictionary_sha256


__all__ = [
    "ConceptSet",
    "LabelSet",
    "MedsExtract",
    "dictionary_matches",
    "open_concept_set",
    "open_extract",
    "open_label_set",
]
