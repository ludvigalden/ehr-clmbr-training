"""Representations from a CLMBR checkpoint: per subject or per prediction time.

:class:`ClmbrEncoder` loads a checkpoint through :mod:`ehrfm.models.checkpoint`
(with its surgery and weight provenance), places it on a device and runs
femr's transformer over batches from :mod:`ehrfm.data.batches`. The model
returns one hidden state per token position, with the subject and timestamp
of that position. Two selections are offered:

- per subject: the hidden state of the subject's last position, which has
  attended to the whole record;
- per prediction time: the hidden state of the subject's last position whose
  timestamp is at or before the prediction time, so no later event can leak
  in. This is the position femr's own labelled-subject task selects.

femr 0.2.4 gives every subject a birth position first and stamps it with the
time of the subject's *last* event, so a subject's position timestamps are not
sorted. Selection by time therefore skips the birth position and searches only
the post-birth positions, after checking that they are sorted. femr's
labelled-subject task never assigns a label to the birth position either, so a
prediction time before the first post-birth event has no representation.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import meds_reader
import numpy as np
import torch
from femr.models.transformer import FEMRModel, to_device
from numpy.typing import NDArray
from torch import Tensor, nn

from ehrfm.data.batches import BatchBuilder, ModelBatch
from ehrfm.data.prediction_times import PredictionTime, by_subject
from ehrfm.models.checkpoint import LoadedCheckpoint, SurgeryReport, load_checkpoint
from ehrfm.models.device import resolve_device
from ehrfm.models.weights import WeightProvenance


class RepresentationError(RuntimeError):
    """A representation could not be selected for a requested subject or time."""


@dataclass(frozen=True, slots=True)
class TokenRepresentations:
    """Hidden states of every token position of one batch, on the CPU."""

    subject_ids: NDArray[np.int64]
    timestamps: NDArray[np.int64]
    hidden: Tensor


@dataclass(frozen=True, slots=True)
class Representations:
    """One row per requested subject or prediction time, on the CPU as float32.

    ``times`` holds the prediction times for per-time extraction and the
    timestamp (seconds since the epoch, UTC) of the selected position for
    per-subject extraction.
    """

    subject_ids: NDArray[np.int64]
    times: NDArray[np.int64]
    embeddings: Tensor

    def __len__(self) -> int:
        return int(self.subject_ids.shape[0])


class ClmbrEncoder:
    """A loaded CLMBR checkpoint that turns MEDS subjects into representations."""

    def __init__(self, loaded: LoadedCheckpoint, device: torch.device) -> None:
        self.model: FEMRModel = loaded.model
        # transformers' decorated ``to`` hides nn.Module's signature from the type checker.
        cast(nn.Module, self.model).to(device)
        self.model.eval()
        self.surgery: SurgeryReport = loaded.surgery
        self.weights: WeightProvenance = loaded.weights
        self.device = device

    @classmethod
    def load(
        cls,
        model_dir: Path,
        *,
        device: str | None = None,
        expected_weights_sha256: str | None = None,
    ) -> ClmbrEncoder:
        """Load ``model_dir`` onto ``device`` (see :func:`ehrfm.models.device.resolve_device`)."""
        loaded = load_checkpoint(model_dir, expected_weights_sha256=expected_weights_sha256)
        return cls(loaded, resolve_device(device))

    @property
    def hidden_size(self) -> int:
        """Width of one representation."""
        return int(self.model.config.transformer_config.hidden_size)

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def encode(self, batch: ModelBatch) -> TokenRepresentations:
        """Run the transformer over one batch and return every position's hidden state."""
        model_input = to_device(dict(batch.model_input), self.device)
        with torch.no_grad(), self._autocast():
            _, result = self.model(**model_input, return_reprs=True)
        subject_ids = result["subject_ids"].detach().cpu().numpy().astype(np.int64)
        timestamps = result["timestamps"].detach().cpu().numpy().astype(np.int64)
        hidden = result["representations"].detach().to("cpu", torch.float32)
        if hidden.shape[0] != subject_ids.shape[0] or hidden.shape[0] != timestamps.shape[0]:
            raise RepresentationError(
                f"model returned {hidden.shape[0]} hidden states for "
                f"{subject_ids.shape[0]} positions"
            )
        return TokenRepresentations(subject_ids=subject_ids, timestamps=timestamps, hidden=hidden)

    def _encoded(
        self,
        database: meds_reader.SubjectDatabase,
        subject_ids: Sequence[int],
        builder: BatchBuilder,
    ) -> Iterator[TokenRepresentations]:
        for batch in builder.batches(database, subject_ids):
            yield self.encode(batch)

    def subject_representations(
        self,
        database: meds_reader.SubjectDatabase,
        subject_ids: Sequence[int],
        builder: BatchBuilder,
    ) -> Representations:
        """The last-position representation of each subject, in ascending subject order."""
        ids: list[int] = []
        times: list[int] = []
        rows: list[Tensor] = []
        for tokens in self._encoded(database, subject_ids, builder):
            for subject_id in np.unique(tokens.subject_ids):
                last = int(np.flatnonzero(tokens.subject_ids == subject_id)[-1])
                ids.append(int(subject_id))
                times.append(int(tokens.timestamps[last]))
                rows.append(tokens.hidden[last])
        return _stack(ids, times, rows, self.hidden_size)

    def representations_at(
        self,
        database: meds_reader.SubjectDatabase,
        times: Sequence[PredictionTime],
        builder: BatchBuilder,
    ) -> Representations:
        """One representation per prediction time, in the order of ``times``.

        Each row is the hidden state of the subject's last position at or
        before the prediction time. A prediction time before the subject's
        first position raises :class:`RepresentationError`.
        """
        grouped = by_subject(times)
        selected: dict[PredictionTime, Tensor] = {}
        for tokens in self._encoded(database, sorted(grouped), builder):
            for subject_id in np.unique(tokens.subject_ids):
                positions = np.flatnonzero(tokens.subject_ids == subject_id)
                stamps = tokens.timestamps[positions]
                for item in grouped.get(int(subject_id), ()):
                    index = position_at(stamps, item.epoch_seconds)
                    if index is None:
                        raise RepresentationError(
                            f"prediction time {item.time} of subject {item.subject_id} "
                            "precedes the subject's first event after birth"
                        )
                    selected[item] = tokens.hidden[int(positions[index])]
        missing = [item for item in times if item not in selected]
        if missing:
            raise RepresentationError(
                f"no events for {len(missing)} prediction times: {missing[:3]}"
            )
        return _stack(
            [item.subject_id for item in times],
            [item.epoch_seconds for item in times],
            [selected[item] for item in times],
            self.hidden_size,
        )


def position_at(stamps: NDArray[np.int64], epoch_seconds: int) -> int | None:
    """Index of one subject's last post-birth position at or before ``epoch_seconds``.

    ``stamps`` are the subject's position timestamps as femr returns them:
    position 0 is the birth position (stamped with the last event's time) and
    is never selected. Returns ``None`` when ``epoch_seconds`` precedes the
    first post-birth position. Raises :class:`RepresentationError` when the
    post-birth timestamps are not sorted, since a binary search over them
    would then select an arbitrary position.
    """
    post_birth = stamps[1:]
    if post_birth.shape[0] > 1 and bool(np.any(np.diff(post_birth) < 0)):
        raise RepresentationError("post-birth position timestamps are not sorted")
    index = int(np.searchsorted(post_birth, epoch_seconds, side="right"))
    if index == 0:
        return None
    return index


def _stack(ids: list[int], times: list[int], rows: list[Tensor], width: int) -> Representations:
    embeddings = torch.stack(rows) if rows else torch.empty((0, width), dtype=torch.float32)
    return Representations(
        subject_ids=np.asarray(ids, dtype=np.int64),
        times=np.asarray(times, dtype=np.int64),
        embeddings=embeddings,
    )


__all__ = [
    "ClmbrEncoder",
    "RepresentationError",
    "Representations",
    "TokenRepresentations",
    "position_at",
]
