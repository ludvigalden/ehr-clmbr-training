"""Representation sets: the extracted embeddings the head and evaluation hooks consume.

One set is written per (task, split): ``<name>.safetensors`` holds
``embeddings`` (float32, n x hidden), ``subject_ids`` (int64), ``times``
(int64 seconds since the epoch, UTC) and, for a labelled task, ``labels``
(float64). ``<name>.json`` holds the :class:`RepresentationMeta`. Boolean
labels are 0/1; categorical labels are indices into ``meta.categories``.
Without label sets the task name is :data:`SUBJECTS_TASK` and each subject has
one row (its last position).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file, save_file

from ehrfm.inputs.contract import Split
from ehrfm.models.representations import Representations

SUBJECTS_TASK: Final = "subjects"
LabelType = Literal["boolean", "integer", "float", "categorical"]


class RepresentationMeta(BaseModel):
    """Description of one stored representation set."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    task: str
    split: Split
    n: int = Field(ge=0)
    hidden_size: int = Field(gt=0)
    label_type: LabelType | None = None
    categories: tuple[str, ...] | None = None
    model_weights_sha256: str


@dataclass(frozen=True, slots=True)
class RepresentationSet:
    """One task and split: rows of (subject, time, embedding[, label])."""

    meta: RepresentationMeta
    subject_ids: NDArray[np.int64]
    times: NDArray[np.int64]
    embeddings: NDArray[np.float32]
    labels: NDArray[np.float64] | None

    def __len__(self) -> int:
        return int(self.subject_ids.shape[0])


def set_name(task: str, split: Split) -> str:
    """File stem of the set for ``task`` and ``split``."""
    return f"{task}.{split}"


def encode_labels(
    values: list[bool | int | float | str],
    label_type: LabelType,
    categories: tuple[str, ...] | None,
) -> NDArray[np.float64]:
    """Labels as float64 (booleans as 0/1, categories as indices)."""
    if label_type == "categorical":
        index = {name: position for position, name in enumerate(categories or ())}
        return np.asarray([index[str(value)] for value in values], dtype=np.float64)
    return np.asarray([float(value) for value in values], dtype=np.float64)


def save_set(
    directory: Path,
    meta: RepresentationMeta,
    representations: Representations,
    labels: NDArray[np.float64] | None,
) -> None:
    """Write one set under ``directory``."""
    tensors = {
        "embeddings": representations.embeddings.to(torch.float32).contiguous(),
        "subject_ids": torch.from_numpy(representations.subject_ids.copy()),
        "times": torch.from_numpy(representations.times.copy()),
    }
    if labels is not None:
        if labels.shape[0] != len(representations):
            raise ValueError(f"{labels.shape[0]} labels for {len(representations)} representations")
        tensors["labels"] = torch.from_numpy(labels.copy())
    name = set_name(meta.task, meta.split)
    save_file(tensors, str(directory / f"{name}.safetensors"))
    (directory / f"{name}.json").write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_set(directory: Path, task: str, split: Split) -> RepresentationSet:
    """Read the set for ``task`` and ``split`` from ``directory``."""
    name = set_name(task, split)
    meta = RepresentationMeta.model_validate_json((directory / f"{name}.json").read_bytes())
    tensors = load_file(str(directory / f"{name}.safetensors"))
    labels = tensors.get("labels")
    return RepresentationSet(
        meta=meta,
        subject_ids=tensors["subject_ids"].numpy().astype(np.int64),
        times=tensors["times"].numpy().astype(np.int64),
        embeddings=tensors["embeddings"].numpy().astype(np.float32),
        labels=None if labels is None else labels.numpy().astype(np.float64),
    )


__all__ = [
    "SUBJECTS_TASK",
    "LabelType",
    "RepresentationMeta",
    "RepresentationSet",
    "encode_labels",
    "load_set",
    "save_set",
    "set_name",
]
