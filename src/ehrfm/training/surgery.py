"""Checkpoint surgery for continued pretraining, and the invariants that prove it.

The released CLMBR-T-Base checkpoint does not load into femr 0.2.4 as-is.
:func:`ehrfm.models.checkpoint.load_checkpoint` applies three fixes; continued
pretraining additionally needs the CLMBR next-code head. Each fix has a
failure mode that does not raise on its own, so :func:`check_surgery` asserts
the result and :func:`prepare_base_model` refuses to train on a model that
fails any check:

hierarchical flag
    The release's ``config.json`` leaves ``is_hierarchical`` false although
    the dictionary is hierarchical. Unpatched, femr builds a flat
    ``nn.Embedding`` and reads ``batch["tokens"]``, which a hierarchical batch
    does not have. Check: the flag is set, the transformer embeds through an
    ``nn.EmbeddingBag`` and has no flat ``embed``.
hawk to uniform attention
    femr 0.2.4 builds a recurrent hawk block in every even layer; the
    checkpoint was trained with uniform attention everywhere. A leftover hawk
    layer either raises in the torch-hawk shim or, with the real package,
    trains a randomly initialised recurrent block. Check: no layer has
    ``use_hawk`` and no recurrent module remains anywhere in the model.
embedding remap
    The weights store ``transformer.embed.weight``; the bag needs
    ``transformer.embed_bag.weight``. A loader that tolerates missing keys
    would leave the embedding at its random initialisation. The loader loads
    strictly; the check additionally pins the bag's shape to the
    dictionary: ``num_embeddings`` equals the dictionary's entry count and
    the transformer's ``vocab_size``.
CLMBR head
    Next-code pretraining needs a ``clmbr`` task head whose output size is
    the checkpoint's ``clmbr_vocab_size`` (8192 for CLMBR-T-Base). Without a
    task config femr returns a constant zero loss, so training silently does
    nothing. Check: the head exists, is a ``CLMBRTaskHead``, its width
    matches the checkpoint (and the configured pin, when set) and fits inside
    the dictionary.

The base dictionary can also be checked against a concept-set export
(:func:`require_concept_set_match`), which links a training run to the
vocabulary mapping that produced its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import msgpack
from femr.models.transformer import CLMBRTaskHead, FEMRModel
from torch import nn

from ehrfm.inputs.artifacts import dictionary_matches, open_concept_set
from ehrfm.models import _compat
from ehrfm.models.checkpoint import (
    DICTIONARY_FILE,
    CheckpointError,
    SurgeryReport,
    load_checkpoint,
)
from ehrfm.models.weights import WeightProvenance


class SurgeryError(CheckpointError):
    """A loaded checkpoint does not satisfy a continued-pretraining invariant."""


@dataclass(frozen=True, slots=True)
class BaseCheckpoint:
    """A base checkpoint loaded, patched and checked for continued pretraining."""

    model: FEMRModel
    surgery: SurgeryReport
    weights: WeightProvenance
    dictionary: dict[str, Any]
    clmbr_vocab_size: int


def read_dictionary(model_dir: Path) -> dict[str, Any]:
    """Read a checkpoint's ``dictionary.msgpack`` payload."""
    with (model_dir / DICTIONARY_FILE).open("rb") as handle:
        payload = msgpack.unpack(handle, strict_map_key=False)
    if not isinstance(payload, dict) or "vocab" not in payload:
        raise CheckpointError(f"{model_dir / DICTIONARY_FILE} is not a CLMBR dictionary")
    return cast(dict[str, Any], payload)


def check_hierarchical(model: FEMRModel) -> None:
    """The model embeds hierarchically: flag set, an ``EmbeddingBag``, no flat embedding."""
    config = model.config.transformer_config
    if not config.is_hierarchical:
        raise SurgeryError("transformer_config.is_hierarchical is false; the patch was not applied")
    transformer = model.transformer
    if not isinstance(getattr(transformer, "embed_bag", None), nn.EmbeddingBag):
        raise SurgeryError("the transformer has no embed_bag; it would embed flat tokens")
    if hasattr(transformer, "embed"):
        raise SurgeryError("the transformer still has a flat embed next to embed_bag")


def check_uniform_attention(model: FEMRModel) -> None:
    """No hawk layer and no recurrent module remain in the model."""
    hawk_layers = [
        index for index, layer in enumerate(model.transformer.layers) if bool(layer.use_hawk)
    ]
    if hawk_layers:
        raise SurgeryError(f"layers {hawk_layers} still use hawk instead of uniform attention")
    recurrent = [
        name
        for name, module in model.named_modules()
        if name.endswith("hawk_module") or isinstance(module, _compat.RecurrentBlock)
    ]
    if recurrent:
        raise SurgeryError(f"recurrent modules remain: {recurrent}")


def check_embedding(model: FEMRModel, dictionary_entries: int) -> None:
    """The embedding bag covers exactly the dictionary's entries."""
    check_hierarchical(model)
    bag = model.transformer.embed_bag
    vocab_size = int(model.config.transformer_config.vocab_size)
    if bag.num_embeddings != dictionary_entries or vocab_size != dictionary_entries:
        raise SurgeryError(
            f"embedding has {bag.num_embeddings} rows and the config {vocab_size}, but the "
            f"dictionary has {dictionary_entries} entries"
        )


def check_clmbr_head(
    model: FEMRModel, dictionary_entries: int, expected_vocab_size: int | None = None
) -> int:
    """The model has a CLMBR head of the checkpoint's width; returns that width."""
    task_config = model.config.task_config
    if task_config is None or task_config.task_type != "clmbr":
        kind = None if task_config is None else task_config.task_type
        raise SurgeryError(
            f"the checkpoint has task head {kind!r}, not 'clmbr'; next-code pretraining "
            "would compute no loss"
        )
    head = getattr(model, "task_model", None)
    if not isinstance(head, CLMBRTaskHead):
        raise SurgeryError("the model has no CLMBR task head module")
    configured = int(task_config.task_kwargs["clmbr_vocab_size"])
    width = int(head.final_layer.out_features)
    if width != configured:
        raise SurgeryError(f"CLMBR head has {width} outputs but the config records {configured}")
    if expected_vocab_size is not None and width != expected_vocab_size:
        raise SurgeryError(f"CLMBR head has {width} outputs, {expected_vocab_size} were pinned")
    if width > dictionary_entries:
        raise SurgeryError(
            f"CLMBR head has {width} outputs but the dictionary only {dictionary_entries} entries"
        )
    return width


def check_surgery(
    model: FEMRModel, dictionary_entries: int, expected_vocab_size: int | None = None
) -> int:
    """Assert every continued-pretraining invariant; return the CLMBR head width."""
    check_hierarchical(model)
    check_uniform_attention(model)
    check_embedding(model, dictionary_entries)
    return check_clmbr_head(model, dictionary_entries, expected_vocab_size)


def require_concept_set_match(concept_set: Path, model_dir: Path) -> None:
    """Fail unless the checkpoint's dictionary is the one the concept set was exported from."""
    opened = open_concept_set(concept_set)
    if not dictionary_matches(opened, model_dir / DICTIONARY_FILE):
        raise SurgeryError(
            f"{model_dir / DICTIONARY_FILE} is not the dictionary concept set {concept_set} was "
            f"exported from (dictionary hash {opened.summary.dictionary_sha256})"
        )


def prepare_base_model(
    model_dir: Path,
    *,
    expected_weights_sha256: str | None = None,
    clmbr_vocab_size: int | None = None,
    concept_set: Path | None = None,
) -> BaseCheckpoint:
    """Load ``model_dir`` for continued pretraining and check every surgery invariant.

    The checkpoint keeps its own CLMBR head (and its weights). The model is
    returned in training mode.
    """
    if concept_set is not None:
        require_concept_set_match(concept_set, model_dir)
    loaded = load_checkpoint(model_dir, expected_weights_sha256=expected_weights_sha256)
    dictionary = read_dictionary(model_dir)
    width = check_surgery(loaded.model, len(dictionary["vocab"]), clmbr_vocab_size)
    loaded.model.train()
    return BaseCheckpoint(
        model=loaded.model,
        surgery=loaded.surgery,
        weights=loaded.weights,
        dictionary=dictionary,
        clmbr_vocab_size=width,
    )


__all__ = [
    "BaseCheckpoint",
    "SurgeryError",
    "check_clmbr_head",
    "check_embedding",
    "check_hierarchical",
    "check_surgery",
    "check_uniform_attention",
    "prepare_base_model",
    "read_dictionary",
    "require_concept_set_match",
]
