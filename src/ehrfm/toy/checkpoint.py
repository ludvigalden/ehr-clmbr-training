"""Toy CLMBR checkpoint factory: a tiny FEMR 0.2.4 model directory built on CPU.

The directory mirrors the layout of the released CLMBR-T-Base checkpoint so
that the production loader (:mod:`ehrfm.models.checkpoint`) and its surgery run
on it unchanged:

``config.json``
    A ``FEMRModelConfig``: ``architectures``, ``task_config`` (a ``clmbr``
    head with ``task_kwargs.clmbr_vocab_size``), ``torch_dtype`` and a sparse
    ``transformer_config``. Like the release, ``transformer_config`` omits
    ``is_hierarchical``, so the loader has to patch it.
``dictionary.msgpack``
    The CLMBR V2 dictionary schema: ``{age_stats, is_hierarchical, vocab}``.
    ``age_stats`` is the flat ``{mean, std}`` of age in seconds;
    ``is_hierarchical`` is true; ``vocab`` lists ``code``, ``numeric``,
    ``text`` and ``unused`` entries, each with ``code_string``,
    ``text_string``, ``type``, ``val_start``, ``val_end`` and ``weight``, and
    ``property`` on numeric and text entries (the code the value belongs to).
    Code entries come first, sorted by weight, as femr's dictionary builder
    orders them.
``model.safetensors``
    Seeded random weights for a model whose every layer uses uniform
    attention, with the flat ``transformer.embed.weight`` key, plus the
    ``task_model.final_layer`` CLMBR head.

The dictionary is synthesised from the synthetic extract's codes
(:mod:`ehrfm.toy.artifacts`): every invented code there is
made up, and part of each family is deliberately left out so that dictionary
coverage of the extract lies strictly between 0 and 1. Nothing is committed;
the same seed writes byte-identical files.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import msgpack
import safetensors.torch
import torch
import transformers
from femr.models.config import FEMRModelConfig

from ehrfm.models.checkpoint import (
    CONFIG_FILE,
    DICTIONARY_FILE,
    WEIGHTS_FILE,
    read_config,
    uniform_attention_model,
)

HIDDEN_SIZE: Final = 32
INTERMEDIATE_SIZE: Final = 64
N_HEADS: Final = 4
N_LAYERS: Final = 2
CLMBR_VOCAB_SIZE: Final = 16
N_UNUSED: Final = 4
#: Age statistics in seconds (flat, as in the release): mean of about 45 years.
AGE_STATS: Final = {"mean": 1.42e9, "std": 5.0e8}
NUMERIC_BINS: Final = (float("-inf"), 3.0, 6.0, float("inf"))


@dataclass(frozen=True, slots=True)
class ToyCheckpoint:
    """A built toy checkpoint directory and what it was built from."""

    path: Path
    vocab_size: int
    dictionary_codes: tuple[str, ...]


def _entry(
    entry_type: str,
    weight: float,
    code_string: str = "",
    text_string: str = "",
    val_start: float = 0.0,
    val_end: float = 0.0,
    prop: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "code_string": code_string,
        "text_string": text_string,
        "type": entry_type,
        "val_end": val_end,
        "val_start": val_start,
        "weight": weight,
    }
    if prop is not None:
        entry["property"] = prop
    return entry


def dictionary_codes(extract_codes: Iterable[str], parent_codes: Iterable[str] = ()) -> list[str]:
    """Pick the code entries of the toy dictionary from the extract's codes.

    Reserved and OMOP-generated codes are always kept; of each invented family
    every second code is kept, so some extract codes stay out of vocabulary.
    Parent group codes are kept so hierarchical tokenisation has parents to hit.
    """
    kept: list[str] = []
    by_family: dict[str, list[str]] = {}
    for code in sorted(set(extract_codes)):
        family = code.split("/", 1)[0]
        if "/" not in code or family in ("Gender", "Visit", "LOINC", "RxNorm"):
            kept.append(code)
        else:
            by_family.setdefault(family, []).append(code)
    for codes in by_family.values():
        kept.extend(codes[::2])
    kept.extend(sorted(set(parent_codes)))
    return sorted(set(kept))


def build_dictionary(
    codes: Sequence[str],
    numeric_properties: Sequence[str] = (),
    text_properties: Sequence[tuple[str, Sequence[str]]] = (),
) -> dict[str, Any]:
    """Assemble a CLMBR V2 dictionary payload.

    Weights are deterministic negative numbers so the order matches femr's
    weight-sorted layout; they carry no statistical meaning.
    """
    vocab: list[dict[str, Any]] = []
    for index, code in enumerate(codes):
        vocab.append(_entry("code", weight=-1.0 + index / (4 * len(codes)), code_string=code))
    for prop in numeric_properties:
        for start, end in zip(NUMERIC_BINS, NUMERIC_BINS[1:], strict=False):
            vocab.append(_entry("numeric", weight=-0.5, val_start=start, val_end=end, prop=prop))
    for prop, texts in text_properties:
        for text in texts:
            vocab.append(_entry("text", weight=-0.25, text_string=text, prop=prop))
    vocab.extend(_entry("unused", weight=0.0) for _ in range(N_UNUSED))
    return {"age_stats": dict(AGE_STATS), "is_hierarchical": True, "vocab": vocab}


def _config_payload(vocab_size: int) -> dict[str, Any]:
    return {
        "architectures": ["FEMRModel"],
        "task_config": {
            "task_kwargs": {"clmbr_vocab_size": CLMBR_VOCAB_SIZE},
            "task_type": "clmbr",
        },
        "torch_dtype": "float32",
        "transformer_config": {
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "model_type": "",
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "vocab_size": vocab_size,
        },
        "transformers_version": transformers.__version__,
    }


def _uniform_state_dict(config: FEMRModelConfig, seed: int) -> dict[str, torch.Tensor]:
    """Seeded weights for a flat-embedding model whose every layer is uniform attention."""
    model, _ = uniform_attention_model(config)
    generator = torch.Generator().manual_seed(seed)
    state: dict[str, torch.Tensor] = {}
    for name, tensor in sorted(model.state_dict().items()):
        if name.endswith("norm.weight"):
            state[name] = torch.ones_like(tensor)
        else:
            state[name] = torch.randn(tensor.shape, generator=generator) * 0.02
    return state


def build_toy_checkpoint(
    out_dir: Path,
    extract_codes: Iterable[str],
    parent_codes: Iterable[str] = (),
    numeric_properties: Sequence[str] = (),
    text_properties: Sequence[tuple[str, Sequence[str]]] = (),
    seed: int = 0,
) -> ToyCheckpoint:
    """Write a toy CLMBR checkpoint directory to ``out_dir`` (created; must not exist)."""
    out_dir.mkdir(parents=True)
    codes = dictionary_codes(extract_codes, parent_codes)
    dictionary = build_dictionary(codes, numeric_properties, text_properties)
    vocab_size = len(dictionary["vocab"])
    with (out_dir / DICTIONARY_FILE).open("wb") as handle:
        msgpack.pack(dictionary, handle)

    payload = _config_payload(vocab_size)
    (out_dir / CONFIG_FILE).write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    config = read_config(out_dir)
    state = _uniform_state_dict(config, seed)
    safetensors.torch.save_file(state, str(out_dir / WEIGHTS_FILE), metadata={"format": "pt"})
    return ToyCheckpoint(path=out_dir, vocab_size=vocab_size, dictionary_codes=tuple(codes))
