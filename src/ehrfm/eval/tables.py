"""Per-subject covariate tables, pinned by SHA-256.

Stratum columns and baseline features come from a table the user supplies:
one row per subject, an integer ``subject_id`` column and any named columns,
as Parquet or CSV. The table is pinned by its SHA-256 (the hook options carry
it), so a changed file fails loudly instead of being mistaken for a cached
result.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

MISSING: Final = "(missing)"
"""String value of a cell that is empty, absent or belongs to a subject the table lacks."""

SUBJECT_KEY: Final = "subject_id"


class TableError(ValueError):
    """A covariate table is missing, changed, malformed or lacks a column."""


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing(value: object) -> bool:
    return value is None or value == "" or (isinstance(value, float) and np.isnan(value))


@dataclass(frozen=True, slots=True)
class SubjectTable:
    """Rows of a pinned covariate table by subject."""

    path: Path
    sha256: str
    rows: Mapping[int, Mapping[str, object]]
    columns: tuple[str, ...]

    def _require(self, columns: Sequence[str]) -> None:
        missing = [name for name in columns if name not in self.columns]
        if missing:
            raise TableError(f"covariate table {self.path} has no column(s) {missing}")

    def strings(
        self, columns: Sequence[str], subject_ids: NDArray[np.int64]
    ) -> dict[str, NDArray[np.str_]]:
        """``columns`` as strings aligned to ``subject_ids`` (missing cells are :data:`MISSING`)."""
        self._require(columns)
        result: dict[str, NDArray[np.str_]] = {}
        for name in columns:
            values: list[str] = []
            for subject in subject_ids:
                value = self.rows.get(int(subject), {}).get(name)
                values.append(MISSING if _is_missing(value) else str(value))
            result[name] = np.asarray(values, dtype=np.str_)
        return result

    def numbers(
        self, columns: Sequence[str], subject_ids: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """``columns`` as a float matrix aligned to ``subject_ids``; any missing cell raises."""
        self._require(columns)
        matrix = np.empty((subject_ids.shape[0], len(columns)), dtype=np.float64)
        for row, subject in enumerate(subject_ids):
            record = self.rows.get(int(subject))
            if record is None:
                raise TableError(f"covariate table {self.path} has no row for subject {subject}")
            for column, name in enumerate(columns):
                value = record.get(name)
                if _is_missing(value):
                    raise TableError(f"{self.path}: subject {subject} has no value for {name!r}")
                try:
                    matrix[row, column] = float(str(value))
                except ValueError:
                    raise TableError(
                        f"{self.path}: {name!r} of subject {subject} is not a number"
                    ) from None
        return matrix

    def epoch_seconds(self, column: str, subject_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        """A time column (epoch seconds, or an ISO-8601 date/time read as UTC) per subject."""
        self._require([column])
        seconds = np.empty(subject_ids.shape[0], dtype=np.float64)
        for row, subject in enumerate(subject_ids):
            value = self.rows.get(int(subject), {}).get(column)
            if _is_missing(value):
                raise TableError(f"{self.path}: subject {subject} has no value for {column!r}")
            seconds[row] = _as_epoch_seconds(value, self.path, column)
        return seconds


def _as_epoch_seconds(value: object, path: Path, column: str) -> float:
    if isinstance(value, dt.datetime):
        moment = value
    elif isinstance(value, dt.date):
        moment = dt.datetime(value.year, value.month, value.day)
    elif isinstance(value, int | float):
        return float(value)
    else:
        text = str(value)
        try:
            return float(text)
        except ValueError:
            pass
        try:
            moment = dt.datetime.fromisoformat(text)
        except ValueError:
            raise TableError(f"{path}: {column!r} value {text!r} is not a time") from None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.UTC)
    return moment.timestamp()


def read_subject_table(path: Path, *, sha256: str) -> SubjectTable:
    """Read the covariate table at ``path``, which must have SHA-256 ``sha256``."""
    if not path.is_file():
        raise TableError(f"covariate table {path} is not a file")
    actual = file_sha256(path)
    if actual != sha256:
        raise TableError(f"covariate table {path} has SHA-256 {actual}, expected {sha256}")
    records: list[dict[str, object]]
    if path.suffix == ".parquet":
        table = pq.read_table(path)
        columns = tuple(table.column_names)
        records = table.to_pylist()
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            records = [dict(row) for row in reader]
    else:
        raise TableError(f"covariate table {path} must be .parquet or .csv")
    if SUBJECT_KEY not in columns:
        raise TableError(f"covariate table {path} has no {SUBJECT_KEY!r} column")
    rows: dict[int, dict[str, object]] = {}
    for record in records:
        try:
            subject = int(str(record[SUBJECT_KEY]))
        except ValueError:
            raise TableError(f"{path}: {record[SUBJECT_KEY]!r} is not a subject id") from None
        if subject in rows:
            raise TableError(f"covariate table {path} repeats subject {subject}")
        rows[subject] = record
    return SubjectTable(path=path, sha256=actual, rows=rows, columns=columns)


__all__ = [
    "MISSING",
    "SUBJECT_KEY",
    "SubjectTable",
    "TableError",
    "file_sha256",
    "read_subject_table",
]
