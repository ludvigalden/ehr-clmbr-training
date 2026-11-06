"""The vocabulary-expansion extension point: a no-op by default, checked when used."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import femr.ontology
import pytest
import torch

from ehrfm.inputs import MedsExtract
from ehrfm.toy.checkpoint import CLMBR_VOCAB_SIZE, ToyCheckpoint
from ehrfm.training.surgery import prepare_base_model, read_dictionary
from ehrfm.training.vocab_expansion import (
    Replacement,
    VocabExpansionContext,
    VocabExpansionError,
    apply_replacements,
    expand_vocabulary,
    register_vocab_expansion,
    registered_hooks,
    unregister_vocab_expansion,
    zero_replaced_embeddings,
)

NEW_CODE = "DEMOCOND/C02"  # an extract code the toy dictionary leaves out


def _code_entry(code: str) -> dict[str, Any]:
    return {
        "code_string": code,
        "text_string": "",
        "type": "code",
        "val_start": 0.0,
        "val_end": 0.0,
        "weight": -0.1,
    }


def _last_unused(dictionary: dict[str, Any]) -> int:
    return max(i for i, e in enumerate(dictionary["vocab"]) if e["type"] == "unused")


@pytest.fixture
def hook_name() -> Iterator[str]:
    name = "test.add-one-code"

    def hook(context: VocabExpansionContext) -> Sequence[Replacement]:
        assert context.n_replace == 1
        return [Replacement(_last_unused(dict(context.dictionary)), _code_entry(NEW_CODE))]

    register_vocab_expansion(name, hook)
    yield name
    unregister_vocab_expansion(name)


def test_no_hook_leaves_the_dictionary_unchanged(
    toy_checkpoint: ToyCheckpoint, extract: MedsExtract, ontology: femr.ontology.Ontology
) -> None:
    dictionary = read_dictionary(toy_checkpoint.path)
    result = expand_vocabulary(
        None,
        0,
        dictionary,
        extract=extract,
        ontology=ontology,
        clmbr_vocab_size=CLMBR_VOCAB_SIZE,
    )
    assert result.replaced == ()
    assert result.dictionary == dictionary
    assert result.dictionary is not dictionary


def test_registered_hook_replaces_and_zeroes(
    hook_name: str,
    toy_checkpoint: ToyCheckpoint,
    extract: MedsExtract,
    ontology: femr.ontology.Ontology,
) -> None:
    assert hook_name in registered_hooks()
    prepared = prepare_base_model(toy_checkpoint.path)
    result = expand_vocabulary(
        hook_name,
        1,
        prepared.dictionary,
        extract=extract,
        ontology=ontology,
        clmbr_vocab_size=CLMBR_VOCAB_SIZE,
    )
    index = _last_unused(prepared.dictionary)
    assert result.replaced == (index,)
    assert result.dictionary["vocab"][index]["code_string"] == NEW_CODE
    assert prepared.dictionary["vocab"][index]["type"] == "unused"  # input untouched
    zero_replaced_embeddings(prepared.model, result.replaced)
    weight = prepared.model.transformer.embed_bag.weight
    assert torch.count_nonzero(weight[index]) == 0
    assert torch.count_nonzero(weight[index - 1]) > 0


def test_unknown_hook_fails(
    toy_checkpoint: ToyCheckpoint, extract: MedsExtract, ontology: femr.ontology.Ontology
) -> None:
    with pytest.raises(VocabExpansionError, match="no vocabulary-expansion hook"):
        expand_vocabulary(
            "absent",
            1,
            read_dictionary(toy_checkpoint.path),
            extract=extract,
            ontology=ontology,
            clmbr_vocab_size=CLMBR_VOCAB_SIZE,
        )


def test_hook_name_registers_once() -> None:
    def first(context: VocabExpansionContext) -> Sequence[Replacement]:
        del context
        return []

    def second(context: VocabExpansionContext) -> Sequence[Replacement]:
        del context
        return []

    register_vocab_expansion("test.once", first)
    try:
        with pytest.raises(VocabExpansionError, match="already registered"):
            register_vocab_expansion("test.once", second)
    finally:
        unregister_vocab_expansion("test.once")


@pytest.mark.parametrize(
    ("replacements", "message"),
    [
        ([Replacement(0, _code_entry("X/1")), Replacement(1, _code_entry("X/2"))], "at most 1"),
        ([Replacement(10_000, _code_entry("X/1"))], "outside the dictionary"),
        ([Replacement(0, {"type": "code"})], "lacks"),
        ([Replacement(0, {**_code_entry("X/1"), "type": "weird"})], "has type"),
    ],
)
def test_bad_replacements_are_rejected(
    toy_checkpoint: ToyCheckpoint, replacements: list[Replacement], message: str
) -> None:
    with pytest.raises(VocabExpansionError, match=message):
        apply_replacements(read_dictionary(toy_checkpoint.path), replacements, 1)


def test_duplicate_code_is_rejected(toy_checkpoint: ToyCheckpoint) -> None:
    dictionary = read_dictionary(toy_checkpoint.path)
    kept = dictionary["vocab"][0]["code_string"]
    index = _last_unused(dictionary)
    with pytest.raises(VocabExpansionError, match="two dictionary entries"):
        apply_replacements(dictionary, [Replacement(index, _code_entry(kept))], 1)
    with pytest.raises(VocabExpansionError, match="more than once"):
        apply_replacements(
            dictionary,
            [Replacement(index, _code_entry("X/1")), Replacement(index, _code_entry("X/2"))],
            2,
        )
