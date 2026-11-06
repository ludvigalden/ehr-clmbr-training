"""Held-out next-code loss of a CLMBR checkpoint, and the base-versus-continued comparison.

:func:`next_token_loss` packs the given subjects of an extract exactly as
continued pretraining does (:mod:`ehrfm.training.pretrain`: the hierarchical
tokenizer over the checkpoint's dictionary, the CLMBR next-code task over its
head) and returns the label-weighted mean cross-entropy, the quantity
pretraining minimises, on data neither model was trained on.

:func:`compare_next_token_loss` computes it for a base checkpoint and for the
continued-pretrained checkpoint on the same subjects. The two must share one
dictionary and head width, otherwise the head positions name different codes
and the losses are not comparable. The difference ``continued - base`` is the
forgetting signal: positive means the continued model predicts the held-out
timelines worse than the base model did.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from femr.models.processor import FEMRBatchProcessor

from ehrfm.data.tokenization import ontology_from_extract
from ehrfm.inputs import MedsExtract
from ehrfm.inputs.database import ensure_database
from ehrfm.inputs.manifest import sha256_file
from ehrfm.models.checkpoint import DICTIONARY_FILE
from ehrfm.models.device import resolve_device
from ehrfm.training.pretrain import (
    ClmbrPretrainingTask,
    packed_batches,
    tokenization_key,
    training_tokenizer,
    validation_loss,
)
from ehrfm.training.surgery import prepare_base_model


class NextTokenError(ValueError):
    """The checkpoints cannot be compared, or the subjects give no labels."""


@dataclass(frozen=True, slots=True)
class NextTokenLoss:
    """Mean next-code loss of one checkpoint over a set of subjects."""

    model_weights_sha256: str
    dictionary_sha256: str
    clmbr_vocab_size: int
    loss: float
    n_subjects: int


@dataclass(frozen=True, slots=True)
class NextTokenComparison:
    """Base and continued-pretrained next-code loss on the same subjects."""

    base: NextTokenLoss
    continued: NextTokenLoss

    @property
    def loss_increase(self) -> float:
        """``continued - base``; positive means the continued model got worse."""
        return self.continued.loss - self.base.loss


def next_token_loss(
    model_dir: Path,
    extract: MedsExtract,
    subject_ids: Sequence[int],
    *,
    database_cache: Path,
    tokens_per_batch: int,
    device: str = "cpu",
    expected_weights_sha256: str | None = None,
    femr_version: str = "",
) -> NextTokenLoss:
    """The label-weighted mean next-code loss of ``model_dir`` over ``subject_ids``."""
    if not subject_ids:
        raise NextTokenError("no subjects to evaluate the next-code loss on")
    prepared = prepare_base_model(model_dir, expected_weights_sha256=expected_weights_sha256)
    dictionary_sha256 = sha256_file(model_dir / DICTIONARY_FILE)
    tokenizer = training_tokenizer(prepared.dictionary, ontology_from_extract(extract))
    processor = FEMRBatchProcessor(
        tokenizer, ClmbrPretrainingTask(clmbr_vocab_size=prepared.clmbr_vocab_size)
    )
    database = ensure_database(extract, database_cache)
    key = tokenization_key(
        extract_manifest_sha256=extract.manifest_sha256,
        dictionary_sha256=dictionary_sha256,
        clmbr_vocab_size=prepared.clmbr_vocab_size,
        tokens_per_batch=tokens_per_batch,
        subject_ids=subject_ids,
        femr_version=femr_version,
    )
    target = resolve_device(device)
    with tempfile.TemporaryDirectory(prefix="next-token-") as scratch:
        data = packed_batches(
            database,
            subject_ids,
            processor,
            tokens_per_batch=tokens_per_batch,
            store=Path(scratch),
            key=key,
        )
        data.set_format("pt")
        model = prepared.model
        cast(Any, model).to(target)
        loss = validation_loss(model, data, processor, target, "fp32")
    return NextTokenLoss(
        model_weights_sha256=prepared.weights.sha256,
        dictionary_sha256=dictionary_sha256,
        clmbr_vocab_size=prepared.clmbr_vocab_size,
        loss=loss,
        n_subjects=len(subject_ids),
    )


def compare_next_token_loss(
    base_dir: Path,
    continued_dir: Path,
    extract: MedsExtract,
    subject_ids: Sequence[int],
    *,
    database_cache: Path,
    tokens_per_batch: int,
    device: str = "cpu",
) -> NextTokenComparison:
    """Next-code loss of the base and the continued checkpoint on the same subjects."""
    base_dictionary = sha256_file(base_dir / DICTIONARY_FILE)
    continued_dictionary = sha256_file(continued_dir / DICTIONARY_FILE)
    if base_dictionary != continued_dictionary:
        raise NextTokenError(
            "the base and continued checkpoints have different dictionaries; "
            "their next-code losses are not comparable"
        )
    base = next_token_loss(
        base_dir,
        extract,
        subject_ids,
        database_cache=database_cache,
        tokens_per_batch=tokens_per_batch,
        device=device,
    )
    continued = next_token_loss(
        continued_dir,
        extract,
        subject_ids,
        database_cache=database_cache,
        tokens_per_batch=tokens_per_batch,
        device=device,
    )
    if base.clmbr_vocab_size != continued.clmbr_vocab_size:
        raise NextTokenError(
            f"head widths differ ({base.clmbr_vocab_size} vs {continued.clmbr_vocab_size})"
        )
    return NextTokenComparison(base=base, continued=continued)


__all__ = [
    "NextTokenComparison",
    "NextTokenError",
    "NextTokenLoss",
    "compare_next_token_loss",
    "next_token_loss",
]
