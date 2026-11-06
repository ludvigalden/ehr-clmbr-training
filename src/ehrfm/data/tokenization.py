"""Tokenization of a MEDS extract with a checkpoint's dictionary.

femr 0.2.4 tokenizes hierarchically: an event's code and all its ancestors that
appear in the dictionary become (weighted) tokens. The ancestors come from a
femr ``Ontology``. Here the ontology is read from the extract's own
``metadata/codes.parquet`` ``parent_codes`` column, which the producer fills;
no vocabulary download is needed on the consumer side.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Mapping
from pathlib import Path

import femr.ontology
import pyarrow.parquet as pq
from femr.models.tokenizer import HierarchicalTokenizer

from ehrfm.inputs.artifacts import MedsExtract
from ehrfm.inputs.contract import CODES_PARQUET
from ehrfm.inputs.schemas import column_values
from ehrfm.models.checkpoint import load_tokenizer


class CodeOntology(femr.ontology.Ontology):
    """A femr ontology built from explicit ``code -> parent codes`` pairs.

    It has no descriptions. Ancestors are resolved transitively, as in femr.
    """

    def __init__(self, parents: Mapping[str, Iterable[str]]) -> None:
        # femr's constructor reads a vocabulary download; this one only needs parents.
        self.description_map: dict[str, str] = {}
        self.parents_map: dict[str, set[str]] = collections.defaultdict(set)
        self.children_map: dict[str, set[str]] = collections.defaultdict(set)
        for code, code_parents in parents.items():
            for parent in code_parents:
                if parent == code:
                    continue
                self.parents_map[code].add(parent)
                self.children_map[parent].add(code)
        self.all_parents_map: dict[str, set[str]] = {}
        self.all_children_map: dict[str, set[str]] = {}

    def ancestors(self, code: str) -> frozenset[str]:
        """The code itself and every ancestor of it."""
        return frozenset(self.get_all_parents(code))


def ontology_from_extract(extract: MedsExtract) -> CodeOntology:
    """The ontology recorded in the extract's ``codes.parquet`` ``parent_codes``."""
    table = pq.read_table(extract.root / CODES_PARQUET)
    codes: list[str] = column_values(table, "code")
    parent_lists: list[list[str | None] | None] = column_values(table, "parent_codes")
    parents: dict[str, list[str]] = {}
    for code, code_parents in zip(codes, parent_lists, strict=True):
        parents[code] = [p for p in code_parents or () if p is not None]
    return CodeOntology(parents)


def extract_tokenizer(model_dir: Path, extract: MedsExtract) -> HierarchicalTokenizer:
    """The checkpoint's tokenizer, resolving ancestors through the extract's ontology."""
    return load_tokenizer(model_dir, ontology_from_extract(extract))


def dictionary_code_entries(tokenizer: HierarchicalTokenizer) -> frozenset[str]:
    """The codes that have a ``code`` entry in the tokenizer's dictionary."""
    return frozenset(tokenizer.code_lookup)


__all__ = [
    "CodeOntology",
    "dictionary_code_entries",
    "extract_tokenizer",
    "ontology_from_extract",
]
