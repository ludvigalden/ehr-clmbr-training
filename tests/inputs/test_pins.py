"""Well-formedness of the pinned CLMBR-T-Base dictionary values."""

from __future__ import annotations

import dataclasses
import re

import pytest

from ehrfm.inputs import CLMBR_T_BASE_PIN, DictionaryPin


def test_revision_is_a_git_commit_sha() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", CLMBR_T_BASE_PIN.revision)


def test_hashes_are_sha256_digests() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", CLMBR_T_BASE_PIN.file_sha256)
    assert re.fullmatch(r"[0-9a-f]{64}", CLMBR_T_BASE_PIN.concept_list_sha256)
    assert CLMBR_T_BASE_PIN.file_sha256 != CLMBR_T_BASE_PIN.concept_list_sha256


def test_identifying_fields_are_present() -> None:
    assert CLMBR_T_BASE_PIN.repo == "StanfordShahLab/clmbr-t-base"
    assert CLMBR_T_BASE_PIN.filename == "dictionary.msgpack"
    assert CLMBR_T_BASE_PIN.licence


def test_pin_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        CLMBR_T_BASE_PIN.revision = "0" * 40  # type: ignore[misc]


def test_pin_is_a_dictionary_pin() -> None:
    assert isinstance(CLMBR_T_BASE_PIN, DictionaryPin)
