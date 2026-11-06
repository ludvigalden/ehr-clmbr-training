"""Prediction times from an upstream label set.

Prediction times are never generated here: they are the ``(subject_id,
prediction_time)`` pairs of a verified label set, optionally restricted to some
of the extract's splits. By the contract the model may use events with
``time <= prediction_time``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pyarrow.parquet as pq

from ehrfm.inputs.artifacts import LabelSet, MedsExtract
from ehrfm.inputs.contract import Split
from ehrfm.inputs.schemas import column_values


@dataclass(frozen=True, slots=True, order=True)
class PredictionTime:
    """One point at which a representation is wanted, ordered by subject then time."""

    subject_id: int
    time: dt.datetime

    def __post_init__(self) -> None:
        if self.time.tzinfo is not None:
            raise ValueError(f"prediction times are timezone-naive, got {self.time!r}")

    @property
    def epoch_seconds(self) -> int:
        """The time as whole seconds since the epoch, reading the naive time as UTC.

        femr records event times the same way, so the two compare directly.
        """
        return epoch_seconds(self.time)


def epoch_seconds(time: dt.datetime) -> int:
    """Seconds since the epoch of a naive time read as UTC, truncated toward zero.

    This matches femr, which stores ``timestamp()`` as float and casts it to int64.
    """
    return int(time.replace(tzinfo=dt.UTC).timestamp())


@dataclass(frozen=True, slots=True)
class LabelledTime:
    """A prediction time and the label value recorded for it."""

    at: PredictionTime
    value: bool | int | float | str


_VALUE_COLUMN = {
    "boolean": "boolean_value",
    "integer": "integer_value",
    "float": "float_value",
    "categorical": "categorical_value",
}


def labelled_times(
    label_set: LabelSet,
    extract: MedsExtract,
    splits: Collection[Split] | None = None,
) -> tuple[LabelledTime, ...]:
    """The label set's rows as sorted ``(prediction time, value)`` pairs.

    ``extract`` must be the extract the label set was verified against (its
    manifest hash is checked again). ``splits`` keeps only subjects in those
    MEDS splits of the extract.
    """
    if label_set.summary.extract_manifest_sha256 != extract.manifest_sha256:
        raise ValueError(
            f"label set {label_set.root} was built against another extract than {extract.root}"
        )
    column = _VALUE_COLUMN[label_set.task.label_type]
    table = pq.read_table(label_set.labels_path, columns=["subject_id", "prediction_time", column])
    subject_ids: list[int] = column_values(table, "subject_id")
    times: list[dt.datetime] = column_values(table, "prediction_time")
    values: list[Any] = column_values(table, column)
    keep: set[int] | None = None
    if splits is not None:
        membership = extract.subject_splits()
        keep = {sid for sid, name in membership.items() if name in splits}
    rows = [
        LabelledTime(at=PredictionTime(sid, time), value=value)
        for sid, time, value in zip(subject_ids, times, values, strict=True)
        if keep is None or sid in keep
    ]
    rows.sort(key=lambda row: row.at)
    return tuple(rows)


def prediction_times(
    label_set: LabelSet,
    extract: MedsExtract,
    splits: Collection[Split] | None = None,
) -> tuple[PredictionTime, ...]:
    """The label set's prediction times, sorted, optionally restricted to ``splits``."""
    return tuple(row.at for row in labelled_times(label_set, extract, splits))


def by_subject(times: Iterable[PredictionTime]) -> dict[int, tuple[PredictionTime, ...]]:
    """Group prediction times per subject, each group sorted by time."""
    grouped: dict[int, list[PredictionTime]] = {}
    for item in times:
        grouped.setdefault(item.subject_id, []).append(item)
    return {sid: tuple(sorted(group)) for sid, group in sorted(grouped.items())}


def subjects_of(times: Sequence[PredictionTime]) -> tuple[int, ...]:
    """The distinct subjects of ``times``, sorted."""
    return tuple(sorted({item.subject_id for item in times}))


__all__ = [
    "LabelledTime",
    "PredictionTime",
    "by_subject",
    "epoch_seconds",
    "labelled_times",
    "prediction_times",
    "subjects_of",
]
