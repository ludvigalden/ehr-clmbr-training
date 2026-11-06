"""Dictionary coverage of an extract: how much of it the model's vocabulary knows.

A code is *recognised* when the checkpoint dictionary has a ``code`` entry for
it. It is *tokenized* when the code or any of its ancestors has one, because
femr's hierarchical tokenizer then still emits a token for its events. A code
that is not tokenized is out of vocabulary: femr drops its events silently.

Coverage is measured over distinct vocabulary codes (the reserved ``MEDS_*``
codes are excluded) and over events. The report lists codes only, never
descriptions, and never lists a code seen in fewer than ``min_events`` events.
"""

from __future__ import annotations

import collections
import hashlib
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Final

import msgpack
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from ehrfm.data.tokenization import CodeOntology, ontology_from_extract
from ehrfm.inputs.artifacts import MedsExtract
from ehrfm.inputs.contract import RESERVED_CODES, SPLITS
from ehrfm.inputs.schemas import column_values
from ehrfm.models.checkpoint import DICTIONARY_FILE, require_layout

DEFAULT_TOP_N: Final = 20
DEFAULT_MIN_EVENTS: Final = 10
_NO_VOCABULARY: Final = ""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class VocabularyCoverage(_Frozen):
    """Coverage of the codes of one vocabulary (the part before the first ``/``)."""

    vocabulary_id: str
    distinct_codes: int = Field(ge=0)
    recognised_codes: int = Field(ge=0)
    tokenized_codes: int = Field(ge=0)
    events: int = Field(ge=0)
    tokenized_events: int = Field(ge=0)


class UnrecognisedCode(_Frozen):
    """A frequent code with no token at all (out of vocabulary)."""

    code: str
    events: int = Field(ge=0)


class CoverageReport(_Frozen):
    """Coverage of one extract by one checkpoint dictionary."""

    extract_manifest_sha256: str
    dictionary_sha256: str
    distinct_codes: int = Field(ge=0)
    recognised_codes: int = Field(ge=0)
    tokenized_codes: int = Field(ge=0)
    events: int = Field(ge=0)
    recognised_events: int = Field(ge=0)
    tokenized_events: int = Field(ge=0)
    by_vocabulary: tuple[VocabularyCoverage, ...]
    unrecognised_min_events: int = Field(ge=1)
    unrecognised_top_n: int = Field(ge=0)
    unrecognised: tuple[UnrecognisedCode, ...]

    @property
    def recognised_share(self) -> float:
        """Share of distinct codes with their own dictionary entry."""
        return _share(self.recognised_codes, self.distinct_codes)

    @property
    def tokenized_share(self) -> float:
        """Share of distinct codes that yield a token (directly or through an ancestor)."""
        return _share(self.tokenized_codes, self.distinct_codes)

    @property
    def tokenized_event_share(self) -> float:
        """Share of events that yield a token."""
        return _share(self.tokenized_events, self.events)


def _share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def vocabulary_id(code: str) -> str:
    """The vocabulary of a ``<vocabulary_id>/<concept_code>`` string ("" when there is none)."""
    head, sep, _ = code.partition("/")
    return head if sep else _NO_VOCABULARY


def extract_code_events(extract: MedsExtract) -> dict[str, int]:
    """Number of rows per code over every shard of the extract."""
    counts: collections.Counter[str] = collections.Counter()
    for split in SPLITS:
        for rel in extract.shards[split]:
            column = pq.read_table(extract.root / rel, columns=["code"]).column("code")
            tallied = pc.value_counts(column)
            values: list[str] = column_values(pa.table({"v": tallied.field("values")}), "v")
            totals: list[int] = column_values(pa.table({"c": tallied.field("counts")}), "c")
            for code, count in zip(values, totals, strict=True):
                counts[code] += count
    return dict(counts)


def dictionary_codes(model_dir: Path) -> tuple[frozenset[str], str]:
    """The ``code`` entries of a checkpoint dictionary and the SHA-256 of its file."""
    require_layout(model_dir)
    data = (model_dir / DICTIONARY_FILE).read_bytes()
    payload = msgpack.unpackb(data)
    if not isinstance(payload, dict) or not isinstance(payload.get("vocab"), list):
        raise ValueError(f"{model_dir / DICTIONARY_FILE} has no vocab list")
    codes: set[str] = set()
    for entry in payload["vocab"]:
        if isinstance(entry, dict) and entry.get("type") == "code":
            code = entry.get("code_string")
            if isinstance(code, str):
                codes.add(code)
    return frozenset(codes), hashlib.sha256(data).hexdigest()


def measure_coverage(
    code_events: Mapping[str, int],
    dictionary: Collection[str],
    ontology: CodeOntology,
    *,
    extract_manifest_sha256: str,
    dictionary_sha256: str,
    top_n: int = DEFAULT_TOP_N,
    min_events: int = DEFAULT_MIN_EVENTS,
) -> CoverageReport:
    """Coverage of ``code_events`` (code -> event count) by ``dictionary`` code entries."""
    known = frozenset(dictionary)
    per_vocab: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    totals = [0, 0, 0, 0, 0, 0]
    out_of_vocabulary: list[UnrecognisedCode] = []
    for code, events in code_events.items():
        if code in RESERVED_CODES:
            continue
        recognised = code in known
        tokenized = recognised or not ontology.ancestors(code).isdisjoint(known)
        row = per_vocab[vocabulary_id(code)]
        row[0] += 1
        row[1] += recognised
        row[2] += tokenized
        row[3] += events
        row[4] += events if tokenized else 0
        totals[0] += 1
        totals[1] += recognised
        totals[2] += tokenized
        totals[3] += events
        totals[4] += events if recognised else 0
        totals[5] += events if tokenized else 0
        if not tokenized and events >= min_events:
            out_of_vocabulary.append(UnrecognisedCode(code=code, events=events))
    out_of_vocabulary.sort(key=lambda item: (-item.events, item.code))
    return CoverageReport(
        extract_manifest_sha256=extract_manifest_sha256,
        dictionary_sha256=dictionary_sha256,
        distinct_codes=totals[0],
        recognised_codes=totals[1],
        tokenized_codes=totals[2],
        events=totals[3],
        recognised_events=totals[4],
        tokenized_events=totals[5],
        by_vocabulary=tuple(
            VocabularyCoverage(
                vocabulary_id=vocab,
                distinct_codes=row[0],
                recognised_codes=row[1],
                tokenized_codes=row[2],
                events=row[3],
                tokenized_events=row[4],
            )
            for vocab, row in sorted(per_vocab.items())
        ),
        unrecognised_min_events=min_events,
        unrecognised_top_n=top_n,
        unrecognised=tuple(out_of_vocabulary[:top_n]),
    )


def extract_coverage(
    extract: MedsExtract,
    model_dir: Path,
    *,
    top_n: int = DEFAULT_TOP_N,
    min_events: int = DEFAULT_MIN_EVENTS,
) -> CoverageReport:
    """Coverage of a verified extract by the dictionary of the checkpoint in ``model_dir``."""
    codes, dictionary_sha256 = dictionary_codes(model_dir)
    return measure_coverage(
        extract_code_events(extract),
        codes,
        ontology_from_extract(extract),
        extract_manifest_sha256=extract.manifest_sha256,
        dictionary_sha256=dictionary_sha256,
        top_n=top_n,
        min_events=min_events,
    )


__all__ = [
    "DEFAULT_MIN_EVENTS",
    "DEFAULT_TOP_N",
    "CoverageReport",
    "UnrecognisedCode",
    "VocabularyCoverage",
    "dictionary_codes",
    "extract_code_events",
    "extract_coverage",
    "measure_coverage",
    "vocabulary_id",
]
