"""Batches for femr 0.2.4 models, built from a meds_reader database.

Subjects are packed back to back into one femr batch until a token budget is
reached; femr's block-diagonal attention mask keeps them apart. A subject
longer than the budget gets a batch of its own and is never truncated. Each
batch is tensorized and collated exactly as femr's own
``FEMRBatchProcessor.convert_subject`` and ``collate`` do, so it can be passed
straight to ``FEMRModel(**batch.model_input)``.

Batches built here carry no task: the model returns one representation per
token, with the token's subject and timestamp, and the caller selects rows.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import datasets.formatting
import meds_reader
from femr.models.processor import BatchCreator, FEMRBatchProcessor
from femr.models.tokenizer import HierarchicalTokenizer

DEFAULT_TOKENS_PER_BATCH: Final = 8192


@dataclass(frozen=True, slots=True)
class ModelBatch:
    """One collated femr batch and the subjects packed into it."""

    model_input: Mapping[str, Any]
    subject_ids: tuple[int, ...]
    n_tokens: int


class BatchBuilder:
    """Packs subjects into femr batches with one checkpoint's tokenizer."""

    def __init__(
        self,
        tokenizer: HierarchicalTokenizer,
        tokens_per_batch: int = DEFAULT_TOKENS_PER_BATCH,
    ) -> None:
        if tokens_per_batch < 1:
            raise ValueError(f"tokens_per_batch must be positive, got {tokens_per_batch}")
        self.tokenizer = tokenizer
        self.tokens_per_batch = tokens_per_batch
        self._processor = FEMRBatchProcessor(tokenizer)
        self._formatter: Any = datasets.formatting.get_formatter("pt")

    def timeline_length(self, subject: meds_reader.Subject) -> int:
        """Number of token positions femr gives the subject (its birth position included)."""
        creator = BatchCreator(self.tokenizer)
        creator.start_batch()
        creator.add_subject(subject)
        return len(creator.ages)

    def build(self, subjects: Sequence[meds_reader.Subject]) -> ModelBatch:
        """Pack ``subjects``, in order, into one collated batch."""
        if not subjects:
            raise ValueError("a batch needs at least one subject")
        creator: BatchCreator = self._processor.creator
        creator.start_batch()
        for subject in subjects:
            creator.add_subject(subject)
        data = creator.get_batch_data()
        tensors = self._formatter.recursive_tensorize(data)
        collated = self._processor.collate([tensors])
        return ModelBatch(
            model_input=collated,
            subject_ids=tuple(subject.subject_id for subject in subjects),
            n_tokens=len(creator.ages),
        )

    def batches(
        self, database: meds_reader.SubjectDatabase, subject_ids: Sequence[int]
    ) -> Iterator[ModelBatch]:
        """Yield batches covering ``subject_ids`` (deduplicated, in ascending order)."""
        pending: list[meds_reader.Subject] = []
        pending_tokens = 0
        for subject_id in sorted(set(subject_ids)):
            subject = database[subject_id]
            length = self.timeline_length(subject)
            if pending and pending_tokens + length > self.tokens_per_batch:
                yield self.build(pending)
                pending, pending_tokens = [], 0
            pending.append(subject)
            pending_tokens += length
        if pending:
            yield self.build(pending)


__all__ = ["DEFAULT_TOKENS_PER_BATCH", "BatchBuilder", "ModelBatch"]
