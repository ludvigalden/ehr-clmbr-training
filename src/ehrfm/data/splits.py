"""Seeded, hash-based subject splits.

A subject's split depends only on the seed and its own ``subject_id``: the
SHA-256 of ``"<seed>:<subject_id>"`` is mapped to a number in ``[0, 1)`` and
placed against the cumulative split fractions. Assignments therefore do not
depend on the order of subjects or on which other subjects are present, and
adding subjects never moves an existing one. Splits are disjoint by
construction.

The extract's own MEDS splits (``train``, ``tuning``, ``held_out``) are the
primary partition; :func:`resplit` divides one of them further, for example
the training split into fitting and early-stopping subsets.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from ehrfm.inputs.artifacts import MedsExtract
from ehrfm.inputs.contract import Split

_SCALE: Final = float(1 << 64)


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """Named split fractions and the seed that places subjects among them."""

    fractions: tuple[tuple[str, float], ...]
    seed: str

    def __post_init__(self) -> None:
        names = [name for name, _ in self.fractions]
        if not names:
            raise ValueError("a split spec needs at least one split")
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError(f"split names must be unique and non-empty, got {names}")
        if any(not (fraction > 0) or not math.isfinite(fraction) for _, fraction in self.fractions):
            raise ValueError(f"split fractions must be positive, got {self.fractions}")
        total = math.fsum(fraction for _, fraction in self.fractions)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"split fractions must sum to 1, got {total}")

    @classmethod
    def of(cls, fractions: Mapping[str, float], seed: str | int) -> SplitSpec:
        """Build a spec from a ``name -> fraction`` mapping (its order is kept)."""
        return cls(fractions=tuple(fractions.items()), seed=str(seed))

    @property
    def names(self) -> tuple[str, ...]:
        """Split names, in order."""
        return tuple(name for name, _ in self.fractions)


def unit_hash(seed: str, subject_id: int) -> float:
    """Map ``(seed, subject_id)`` to a number in ``[0, 1)`` through SHA-256."""
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / _SCALE


def assign_split(subject_id: int, spec: SplitSpec) -> str:
    """The split ``subject_id`` falls in under ``spec``."""
    point = unit_hash(spec.seed, subject_id)
    upper = 0.0
    for name, fraction in spec.fractions:
        upper += fraction
        if point < upper:
            return name
    return spec.fractions[-1][0]


def split_subjects(subject_ids: Iterable[int], spec: SplitSpec) -> dict[str, tuple[int, ...]]:
    """Assign every subject to one split; each split's ids are sorted. Every split is present."""
    buckets: dict[str, list[int]] = {name: [] for name in spec.names}
    for subject_id in set(subject_ids):
        buckets[assign_split(subject_id, spec)].append(subject_id)
    return {name: tuple(sorted(ids)) for name, ids in buckets.items()}


def extract_split(extract: MedsExtract, split: Split) -> tuple[int, ...]:
    """The sorted subject ids of one of the extract's MEDS splits."""
    return tuple(sorted(sid for sid, name in extract.subject_splits().items() if name == split))


def resplit(extract: MedsExtract, split: Split, spec: SplitSpec) -> dict[str, tuple[int, ...]]:
    """Divide one MEDS split of the extract according to ``spec``."""
    return split_subjects(extract_split(extract, split), spec)


__all__ = [
    "SplitSpec",
    "assign_split",
    "extract_split",
    "resplit",
    "split_subjects",
    "unit_hash",
]
