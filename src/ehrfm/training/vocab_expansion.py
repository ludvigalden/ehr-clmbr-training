"""Optional vocabulary expansion before continued pretraining.

A base dictionary built on one population can leave many codes of another
out of vocabulary. Expansion replaces some base dictionary entries with
entries for codes the extract needs, before tokenization, and starts the
embeddings of the replaced entries at zero so they carry no meaning from the
codes they used to stand for.

The package ships no expansion strategy. Expansion runs only when the
training configuration names a hook registered here
(``optimization.vocab_expansion.hook``); with no hook the base dictionary is
used unchanged and the output dictionary is byte-identical to it.

A hook receives a :class:`VocabExpansionContext` and returns at most
``n_replace`` :class:`Replacement` items. The hook name and ``n_replace`` are
part of the run fingerprint and the expanded dictionary's hash is recorded
with the run, so a hook's behaviour should change only together with its
name.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import femr.ontology
import torch
from femr.models.transformer import FEMRModel

from ehrfm.inputs.artifacts import MedsExtract

ENTRY_TYPES: Final = frozenset({"code", "numeric", "text", "unused"})
_ENTRY_KEYS: Final = frozenset({"code_string", "text_string", "type", "val_start", "val_end"})


class VocabExpansionError(ValueError):
    """A hook is unknown, or its replacements do not fit the dictionary."""


@dataclass(frozen=True, slots=True)
class VocabExpansionContext:
    """What a hook may look at. ``dictionary`` is a private copy."""

    dictionary: Mapping[str, Any]
    extract: MedsExtract
    ontology: femr.ontology.Ontology
    n_replace: int
    clmbr_vocab_size: int


@dataclass(frozen=True, slots=True)
class Replacement:
    """Put ``entry`` at dictionary position ``index``."""

    index: int
    entry: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ExpansionResult:
    """The dictionary to train with and the positions that were replaced."""

    dictionary: dict[str, Any]
    replaced: tuple[int, ...]


VocabExpansionHook = Callable[[VocabExpansionContext], Sequence[Replacement]]

_HOOKS: dict[str, VocabExpansionHook] = {}


def register_vocab_expansion(name: str, hook: VocabExpansionHook) -> None:
    """Register ``hook`` under ``name``; a name can be registered once."""
    if name in _HOOKS and _HOOKS[name] is not hook:
        raise VocabExpansionError(f"vocabulary-expansion hook {name!r} is already registered")
    _HOOKS[name] = hook


def unregister_vocab_expansion(name: str) -> None:
    """Remove a registered hook (no-op when absent)."""
    _HOOKS.pop(name, None)


def registered_hooks() -> tuple[str, ...]:
    """Names of the registered hooks, sorted."""
    return tuple(sorted(_HOOKS))


def _check_entry(index: int, entry: Mapping[str, Any]) -> None:
    missing = _ENTRY_KEYS - entry.keys()
    if missing:
        raise VocabExpansionError(f"replacement for entry {index} lacks {sorted(missing)}")
    if entry["type"] not in ENTRY_TYPES:
        raise VocabExpansionError(f"replacement for entry {index} has type {entry['type']!r}")
    if entry["type"] in ("numeric", "text") and "property" not in entry:
        raise VocabExpansionError(f"{entry['type']} replacement for entry {index} lacks property")


def apply_replacements(
    dictionary: Mapping[str, Any], replacements: Sequence[Replacement], n_replace: int
) -> ExpansionResult:
    """Return a copy of ``dictionary`` with ``replacements`` applied, after checking them.

    Positions must be unique and inside the dictionary, there may be at most
    ``n_replace`` of them, and a new code entry may not duplicate a code that
    keeps its entry (femr would make one of the two unreachable).
    """
    if len(replacements) > n_replace:
        raise VocabExpansionError(
            f"hook returned {len(replacements)} replacements, at most {n_replace} are allowed"
        )
    expanded = copy.deepcopy(dict(dictionary))
    vocab = cast(list[dict[str, Any]], expanded["vocab"])
    indices = [item.index for item in replacements]
    if len(set(indices)) != len(indices):
        raise VocabExpansionError("hook replaced a dictionary position more than once")
    for item in replacements:
        if not 0 <= item.index < len(vocab):
            raise VocabExpansionError(
                f"replacement position {item.index} is outside the dictionary ({len(vocab)})"
            )
        _check_entry(item.index, item.entry)
    replaced = set(indices)
    kept_codes = {
        entry["code_string"]
        for position, entry in enumerate(vocab)
        if position not in replaced and entry["type"] == "code"
    }
    new_codes: set[str] = set()
    for item in replacements:
        if item.entry["type"] != "code":
            continue
        code = str(item.entry["code_string"])
        if code in kept_codes or code in new_codes:
            raise VocabExpansionError(f"code {code!r} would have two dictionary entries")
        new_codes.add(code)
    for item in replacements:
        vocab[item.index] = dict(item.entry)
    return ExpansionResult(dictionary=expanded, replaced=tuple(sorted(indices)))


def expand_vocabulary(
    hook_name: str | None,
    n_replace: int,
    dictionary: Mapping[str, Any],
    *,
    extract: MedsExtract,
    ontology: femr.ontology.Ontology,
    clmbr_vocab_size: int,
) -> ExpansionResult:
    """Run the named hook, or return the dictionary unchanged when ``hook_name`` is None."""
    if hook_name is None:
        return ExpansionResult(dictionary=copy.deepcopy(dict(dictionary)), replaced=())
    hook = _HOOKS.get(hook_name)
    if hook is None:
        raise VocabExpansionError(
            f"no vocabulary-expansion hook {hook_name!r} is registered "
            f"(registered: {list(registered_hooks())})"
        )
    context = VocabExpansionContext(
        dictionary=copy.deepcopy(dict(dictionary)),
        extract=extract,
        ontology=ontology,
        n_replace=n_replace,
        clmbr_vocab_size=clmbr_vocab_size,
    )
    return apply_replacements(dictionary, list(hook(context)), n_replace)


def zero_replaced_embeddings(model: FEMRModel, replaced: Sequence[int]) -> None:
    """Set the embedding rows of replaced dictionary positions to zero."""
    if not replaced:
        return
    bag = model.transformer.embed_bag
    with torch.no_grad():
        bag.weight[list(replaced)] = 0.0


__all__ = [
    "ExpansionResult",
    "Replacement",
    "VocabExpansionContext",
    "VocabExpansionError",
    "VocabExpansionHook",
    "apply_replacements",
    "expand_vocabulary",
    "register_vocab_expansion",
    "registered_hooks",
    "unregister_vocab_expansion",
    "zero_replaced_embeddings",
]
