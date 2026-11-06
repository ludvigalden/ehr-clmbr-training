"""Predictions: how a head's output reaches the evaluator.

A head hook writes one prediction set per (task, split) into its
``output_dir`` with :func:`write_predictions`: the positive-class probability
of every row of the representation set it scored, in the same row order. An
evaluation hook reads them back with :func:`load_predictions`, which joins
them to the representation set and takes the labels from there, so a head can
never supply or alter labels.

On disk (inside the head's ``output_dir``)::

    predictions/<task>.<split>.json   PredictionMeta
    predictions/<task>.<split>.npy    float64 probabilities, one per row

The meta records the representation set's row count and a SHA-256 over its
``(subject_id, time)`` rows; loading fails unless both match the set the
evaluator joins against.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ehrfm.experiment.representations import RepresentationSet
from ehrfm.inputs.contract import Split

PREDICTIONS_DIR: Final = "predictions"
PREDICTIONS_FORMAT: Final = 1


class PredictionError(ValueError):
    """Predictions are missing, malformed or do not match their representation set."""


class PredictionMeta(BaseModel):
    """Description of one stored prediction set."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    format_version: int = PREDICTIONS_FORMAT
    head: str
    task: str
    split: Split
    rows: int = Field(ge=0)
    rows_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class BinaryPredictions:
    """A head's probabilities joined to the labels of its representation set."""

    head: str
    task: str
    split: Split
    subject_ids: NDArray[np.int64]
    times: NDArray[np.int64]
    y_true: NDArray[np.float64]
    y_prob: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.y_true.shape[0])


def rows_sha256(subject_ids: NDArray[np.int64], times: NDArray[np.int64]) -> str:
    """SHA-256 identifying the ordered ``(subject_id, time)`` rows of a set."""
    stacked = np.stack([subject_ids.astype("<i8"), times.astype("<i8")], axis=1)
    return hashlib.sha256(np.ascontiguousarray(stacked).tobytes()).hexdigest()


def _stem(task: str, split: Split) -> str:
    return f"{task}.{split}"


def write_predictions(
    output_dir: Path,
    head: str,
    representations: RepresentationSet,
    y_prob: NDArray[np.float64],
) -> Path:
    """Write ``y_prob`` (one probability per row of ``representations``); return the .npy path."""
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if probabilities.shape != (len(representations),):
        raise PredictionError(
            f"{probabilities.shape} probabilities for {len(representations)} representation rows"
        )
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise PredictionError("probabilities must be finite and within [0, 1]")
    meta = PredictionMeta(
        head=head,
        task=representations.meta.task,
        split=representations.meta.split,
        rows=len(representations),
        rows_sha256=rows_sha256(representations.subject_ids, representations.times),
    )
    directory = output_dir / PREDICTIONS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stem = _stem(meta.task, meta.split)
    np.save(directory / f"{stem}.npy", probabilities, allow_pickle=False)
    (directory / f"{stem}.json").write_text(
        json.dumps(meta.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return directory / f"{stem}.npy"


def has_predictions(head_output_dir: Path, task: str, split: Split) -> bool:
    """Whether a head published predictions for ``task`` and ``split``."""
    return (head_output_dir / PREDICTIONS_DIR / f"{_stem(task, split)}.json").is_file()


def load_predictions(
    head_output_dir: Path, representations: RepresentationSet
) -> BinaryPredictions:
    """A head's predictions for the set ``representations``, joined to its boolean labels."""
    task, split = representations.meta.task, representations.meta.split
    directory = head_output_dir / PREDICTIONS_DIR
    stem = _stem(task, split)
    meta_path = directory / f"{stem}.json"
    if not meta_path.is_file():
        raise PredictionError(f"no predictions for task {task!r}, split {split!r} in {directory}")
    meta = PredictionMeta.model_validate_json(meta_path.read_bytes())
    if meta.format_version != PREDICTIONS_FORMAT:
        raise PredictionError(f"{meta_path}: unsupported format version {meta.format_version}")
    if representations.meta.label_type != "boolean" or representations.labels is None:
        raise PredictionError(f"task {task!r} is not a boolean-labelled task")
    expected = rows_sha256(representations.subject_ids, representations.times)
    if meta.rows != len(representations) or meta.rows_sha256 != expected:
        raise PredictionError(
            f"{meta_path} was written for other rows than the {task}/{split} representations"
        )
    probabilities = np.load(directory / f"{stem}.npy", allow_pickle=False)
    if probabilities.dtype != np.float64 or probabilities.shape != (meta.rows,):
        raise PredictionError(f"{directory / f'{stem}.npy'} has shape {probabilities.shape}")
    return BinaryPredictions(
        head=meta.head,
        task=task,
        split=split,
        subject_ids=representations.subject_ids,
        times=representations.times,
        y_true=representations.labels,
        y_prob=probabilities,
    )


__all__ = [
    "PREDICTIONS_DIR",
    "BinaryPredictions",
    "PredictionError",
    "PredictionMeta",
    "has_predictions",
    "load_predictions",
    "rows_sha256",
    "write_predictions",
]
