"""Tiny synthetic artifacts that follow the upstream artifact contract.

Everything here is generated at run time from a seed: a MEDS 0.3.3 extract
of about twenty subjects with its ``artifact_manifest.json``, a boolean label
set built against that extract, and (from a toy dictionary) a CLMBR
concept-set export. Nothing is committed as a binary file, and the same seed
writes byte-identical files.

All codes are invented. ``DEMOCOND/``, ``DEMODRUG/``, ``DEMOLAB/``,
``DEMOPROC/`` and ``DEMOOBS/`` are made-up vocabularies; ``Gender/`` and
``Visit/`` follow the style of OMOP-generated vocabularies; the ``LOINC/``
and ``RxNorm/`` entries carry made-up concept codes prefixed with ``X``,
which no real LOINC or RxNorm code uses. No code, description or value
refers to a real concept.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from ehrfm.inputs.contract import (
    CODES_PARQUET,
    CONCEPT_SET_PARQUET,
    CONTRACT_VERSION,
    DATASET_JSON,
    LABELS_PARQUET,
    MANIFEST_NAME,
    SPLITS,
    SUBJECT_SPLITS_PARQUET,
    TASK_JSON,
)
from ehrfm.inputs.manifest import artifact_files, concept_list_sha256, sha256_file
from ehrfm.inputs.schemas import (
    CONCEPT_SET_SCHEMA,
    LABEL_SCHEMA,
    MEDS_CODES_SCHEMA,
    MEDS_DATA_SCHEMA,
    SUBJECT_SPLITS_SCHEMA,
    column_values,
)

#: Fixed creation time so repeated builds are byte-identical.
CREATED_AT: Final = "2026-01-01T00:00:00Z"
PRODUCER: Final = {"name": "omopmap", "version": "0.1.0"}
DATASET_NAME: Final = "synthetic_demo"
TASK_NAME: Final = "demo_outcome_30d"
OUTCOME_CODE: Final = "DEMOCOND/C07"

SEX_CODES: Final = ("Gender/F", "Gender/M")
VISIT_CODES: Final = ("Visit/ER", "Visit/OP")
CONDITION_CODES: Final = tuple(f"DEMOCOND/C{n:02d}" for n in range(1, 9))
DRUG_CODES: Final = (*(f"DEMODRUG/D{n:02d}" for n in range(1, 5)), "RxNorm/X900001")
LAB_CODES: Final = (*(f"DEMOLAB/L{n:02d}" for n in range(1, 4)), "LOINC/X0001-0")
PROCEDURE_CODES: Final = tuple(f"DEMOPROC/P{n:02d}" for n in range(1, 4))
OBSERVATION_CODES: Final = ("DEMOOBS/O01",)
OBSERVATION_TEXTS: Final = ("low", "medium", "high")
#: Invented parent groups, written to ``codes.parquet`` ``parent_codes``.
PARENT_GROUPS: Final = {
    **dict.fromkeys(CONDITION_CODES[:4], "DEMOCOND/GROUP-A"),
    **dict.fromkeys(CONDITION_CODES[4:], "DEMOCOND/GROUP-B"),
    **dict.fromkeys(DRUG_CODES, "DEMODRUG/GROUP-A"),
}

_Row = tuple[int, dt.datetime | None, str, float | None, str | None]


@dataclass(frozen=True, slots=True)
class SyntheticArtifacts:
    """Paths of one generated artifact set."""

    extract: Path
    labels: Path
    codes: tuple[str, ...]
    covariates: Path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_table(path: Path, columns: dict[str, pa.Array[Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), path)


def _file_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for rel in artifact_files(root):
        path = root / rel
        rows = pq.ParquetFile(path).metadata.num_rows if rel.endswith(".parquet") else None
        records.append(
            {"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size, "rows": rows}
        )
    return records


def write_manifest(
    root: Path,
    summary: dict[str, object],
    inputs: Sequence[tuple[str, str]] = (),
) -> Path:
    """Hash every file under ``root`` and write its ``artifact_manifest.json`` last."""
    payload = {
        "artifact_type": summary["artifact_type"],
        "contract_version": CONTRACT_VERSION,
        "created_at": CREATED_AT,
        "files": _file_records(root),
        "inputs": [{"name": n, "sha256": s} for n, s in sorted(inputs)],
        "producer": PRODUCER,
        "summary": summary,
    }
    path = root / MANIFEST_NAME
    _write_json(path, payload)
    return path


def _split_of(subject_id: int) -> str:
    bucket = subject_id % 10
    return "held_out" if bucket == 0 else "tuning" if bucket == 5 else "train"


def _subject_rows(rng: random.Random, subject_id: int) -> list[_Row]:
    birth = dt.datetime(1950, 1, 1) + dt.timedelta(days=rng.randrange(0, 365 * 50))
    rows: list[_Row] = [
        (subject_id, None, rng.choice(SEX_CODES), None, None),
        (subject_id, birth, "MEDS_BIRTH", None, None),
    ]
    visit = birth + dt.timedelta(days=365 * 18 + rng.randrange(0, 365 * 10))
    for _ in range(rng.randint(2, 5)):
        visit += dt.timedelta(days=rng.randint(5, 120), minutes=rng.randrange(0, 24 * 60))
        rows.append((subject_id, visit, rng.choice(VISIT_CODES), None, None))
        for offset in range(rng.randint(2, 5)):
            when = visit + dt.timedelta(minutes=10 * (offset + 1))
            family = rng.randrange(5)
            if family == 0:
                rows.append((subject_id, when, rng.choice(CONDITION_CODES), None, None))
            elif family == 1:
                rows.append((subject_id, when, rng.choice(DRUG_CODES), None, None))
            elif family == 2:
                value = round(rng.uniform(0.5, 9.5), 2)
                rows.append((subject_id, when, rng.choice(LAB_CODES), value, None))
            elif family == 3:
                rows.append((subject_id, when, rng.choice(PROCEDURE_CODES), None, None))
            else:
                text = rng.choice(OBSERVATION_TEXTS)
                rows.append((subject_id, when, OBSERVATION_CODES[0], None, text))
    return rows


def _sort_key(row: _Row) -> tuple[int, bool, dt.datetime, str]:
    subject_id, time, code, _, _ = row
    return (subject_id, time is not None, time or dt.datetime.min, code)


def _data_columns(rows: Sequence[_Row]) -> dict[str, pa.Array[Any]]:
    return {
        "subject_id": pa.array([r[0] for r in rows], pa.int64()),
        "time": pa.array([r[1] for r in rows], pa.timestamp("us")),
        "code": pa.array([r[2] for r in rows], pa.string()),
        "numeric_value": pa.array([r[3] for r in rows], pa.float32()),
        "text_value": pa.array([r[4] for r in rows], pa.string()),
    }


def write_extract(root: Path, n_subjects: int = 20, seed: int = 7) -> tuple[str, ...]:
    """Write a seeded MEDS extract with its manifest; return its sorted distinct codes."""
    by_split: dict[str, list[_Row]] = {split: [] for split in SPLITS}
    for subject_id in range(1, n_subjects + 1):
        rng = random.Random(f"{seed}:{subject_id}")
        by_split[_split_of(subject_id)].extend(_subject_rows(rng, subject_id))

    events_by_split: dict[str, int] = {}
    subjects_by_split: dict[str, int] = {}
    all_codes: set[str] = set()
    for split in SPLITS:
        rows = sorted(by_split[split], key=_sort_key)
        subject_ids = sorted({r[0] for r in rows})
        # Two shards for train, one for the others, so shard numbering is exercised.
        n_shards = 2 if split == "train" else 1
        for shard in range(n_shards):
            members = set(subject_ids[shard::n_shards])
            shard_rows = [r for r in rows if r[0] in members]
            _write_table(
                root / "data" / split / f"{shard}.parquet",
                _data_columns(shard_rows),
                MEDS_DATA_SCHEMA,
            )
        (root / "data" / split).mkdir(parents=True, exist_ok=True)
        events_by_split[split] = len(rows)
        subjects_by_split[split] = len(subject_ids)
        all_codes.update(r[2] for r in rows)

    codes = tuple(sorted(all_codes))
    _write_table(
        root / CODES_PARQUET,
        {
            "code": pa.array(codes, pa.string()),
            "description": pa.array([None] * len(codes), pa.string()),
            "parent_codes": pa.array(
                [[PARENT_GROUPS[c]] if c in PARENT_GROUPS else None for c in codes],
                pa.list_(pa.string()),
            ),
        },
        MEDS_CODES_SCHEMA,
    )
    ids = list(range(1, n_subjects + 1))
    _write_table(
        root / SUBJECT_SPLITS_PARQUET,
        {
            "subject_id": pa.array(ids, pa.int64()),
            "split": pa.array([_split_of(i) for i in ids], pa.string()),
        },
        SUBJECT_SPLITS_SCHEMA,
    )
    _write_json(
        root / DATASET_JSON,
        {
            "created_at": CREATED_AT,
            "dataset_name": root.name,
            "dataset_version": "1",
            "etl_name": "omopmap",
            "etl_version": "0.1.0",
            "meds_version": "0.3.3",
        },
    )
    generator = json.dumps({"n_subjects": n_subjects, "seed": seed}, sort_keys=True)
    write_manifest(
        root,
        {
            "artifact_type": "meds_extract",
            "distinct_codes": len(codes),
            "events": sum(events_by_split.values()),
            "events_by_split": events_by_split,
            "meds_version": "0.3.3",
            "subjects": n_subjects,
            "subjects_by_split": subjects_by_split,
        },
        inputs=[(f"synthetic:{seed}", hashlib.sha256(generator.encode()).hexdigest())],
    )
    return codes


def write_label_set(root: Path, extract_root: Path, seed: int = 7) -> None:
    """Write a boolean label set over ``extract_root``: one prediction time per subject."""
    splits = pq.read_table(extract_root / SUBJECT_SPLITS_PARQUET)
    split_of: dict[int, str] = dict(
        zip(column_values(splits, "subject_id"), column_values(splits, "split"), strict=True)
    )
    first_visit: dict[int, dt.datetime] = {}
    outcome: dict[int, bool] = {}
    for shard in sorted((extract_root / "data").rglob("*.parquet")):
        table = pq.read_table(shard)
        for sid, time, code in zip(
            column_values(table, "subject_id"),
            column_values(table, "time"),
            column_values(table, "code"),
            strict=True,
        ):
            if isinstance(code, str) and code.startswith("Visit/") and sid not in first_visit:
                first_visit[sid] = time
            if code == OUTCOME_CODE:
                outcome[sid] = True
    rng = random.Random(f"{seed}:labels")
    labelled = sorted(sid for sid in first_visit if rng.random() < 0.8)
    _write_table(
        root / LABELS_PARQUET,
        {
            "subject_id": pa.array(labelled, pa.int64()),
            "prediction_time": pa.array([first_visit[s] for s in labelled], pa.timestamp("us")),
            "boolean_value": pa.array([outcome.get(s, False) for s in labelled], pa.bool_()),
            "integer_value": pa.array([None] * len(labelled), pa.int64()),
            "float_value": pa.array([None] * len(labelled), pa.float64()),
            "categorical_value": pa.array([None] * len(labelled), pa.string()),
        },
        LABEL_SCHEMA,
    )
    _write_json(
        root / TASK_JSON,
        {
            "categories": None,
            "description": "Invented demonstration outcome: DEMOCOND/C07 recorded for the subject.",
            "label_type": "boolean",
            "name": root.name,
        },
    )
    labels_by_split = {split: sum(1 for s in labelled if split_of[s] == split) for split in SPLITS}
    write_manifest(
        root,
        {
            "artifact_type": "label_set",
            "extract_manifest_sha256": sha256_file(extract_root / MANIFEST_NAME),
            "label_type": "boolean",
            "labels": len(labelled),
            "labels_by_split": labels_by_split,
            "subjects": len(labelled),
            "task_name": root.name,
        },
    )


COVARIATES_CSV: Final = "covariates.csv"


def write_covariates(path: Path, extract_root: Path) -> None:
    """Write a per-subject covariate table (``subject_id,sex,birth_time``) for ``extract_root``.

    The values are read back from the extract's own birth and sex events, so
    the table is as synthetic as the extract. It stands in for the covariate
    table a user supplies to stratify evaluations and fit baselines.
    """
    birth: dict[int, dt.datetime] = {}
    sex: dict[int, str] = {}
    for shard in sorted((extract_root / "data").rglob("*.parquet")):
        table = pq.read_table(shard)
        for sid, time, code in zip(
            column_values(table, "subject_id"),
            column_values(table, "time"),
            column_values(table, "code"),
            strict=True,
        ):
            if code == "MEDS_BIRTH":
                birth[sid] = time
            elif isinstance(code, str) and code in SEX_CODES:
                sex[sid] = code.split("/", 1)[1]
    lines = ["subject_id,sex,birth_time"]
    for sid in sorted(birth):
        lines.append(f"{sid},{sex.get(sid, '')},{birth[sid].isoformat()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_concept_set(root: Path, dictionary_codes: Sequence[str], dictionary_sha256: str) -> str:
    """Write a CLMBR concept-set export for the vocabulary codes of a dictionary.

    Returns the concept-list hash.
    """
    codes = sorted({c for c in dictionary_codes if "/" in c})
    vocabulary_ids = [c.split("/", 1)[0] for c in codes]
    _write_table(
        root / CONCEPT_SET_PARQUET,
        {
            "vocabulary_id": pa.array(vocabulary_ids, pa.string()),
            "concept_code": pa.array([c.split("/", 1)[1] for c in codes], pa.string()),
            "code": pa.array(codes, pa.string()),
            "entry_types": pa.array([["code"] for _ in codes], pa.list_(pa.string())),
        },
        CONCEPT_SET_SCHEMA,
    )
    counts: dict[str, int] = {}
    for vocabulary_id in vocabulary_ids:
        counts[vocabulary_id] = counts.get(vocabulary_id, 0) + 1
    digest = concept_list_sha256(codes)
    write_manifest(
        root,
        {
            "artifact_type": "clmbr_concept_set",
            "concept_list_sha256": digest,
            "concepts": len(codes),
            "dictionary_sha256": dictionary_sha256,
            "vocabulary_counts": counts,
        },
        inputs=[("dictionary", dictionary_sha256)],
    )
    return digest


def build_artifacts(base: Path, n_subjects: int = 20, seed: int = 7) -> SyntheticArtifacts:
    """Write an extract and a label set under ``base``."""
    extract = base / "meds" / DATASET_NAME
    labels = base / "labels" / DATASET_NAME / TASK_NAME
    codes = write_extract(extract, n_subjects=n_subjects, seed=seed)
    write_label_set(labels, extract, seed=seed)
    covariates = base / COVARIATES_CSV
    write_covariates(covariates, extract)
    return SyntheticArtifacts(extract=extract, labels=labels, codes=codes, covariates=covariates)
