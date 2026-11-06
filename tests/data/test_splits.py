"""Seeded hash splits: determinism, disjointness, stability, validation."""

from __future__ import annotations

import pytest

from ehrfm.data.splits import (
    SplitSpec,
    assign_split,
    extract_split,
    resplit,
    split_subjects,
    unit_hash,
)
from ehrfm.inputs import MedsExtract

SPEC = SplitSpec.of({"fit": 0.7, "early_stop": 0.15, "calibrate": 0.15}, seed=11)


def test_same_seed_same_assignment() -> None:
    ids = range(1, 2001)
    assert split_subjects(ids, SPEC) == split_subjects(reversed(ids), SPEC)


def test_splits_are_disjoint_and_cover_every_subject() -> None:
    ids = set(range(1, 2001))
    parts = split_subjects(ids, SPEC)
    assert list(parts) == ["fit", "early_stop", "calibrate"]
    seen: set[int] = set()
    for members in parts.values():
        assert seen.isdisjoint(members)
        seen.update(members)
    assert seen == ids


def test_fractions_are_respected_approximately() -> None:
    parts = split_subjects(range(20_000), SPEC)
    assert len(parts["fit"]) / 20_000 == pytest.approx(0.7, abs=0.02)
    assert len(parts["early_stop"]) / 20_000 == pytest.approx(0.15, abs=0.02)


def test_assignment_ignores_other_subjects() -> None:
    small = split_subjects(range(100), SPEC)
    large = split_subjects(range(1000), SPEC)
    for name, members in small.items():
        assert set(members) <= set(large[name])


def test_seed_changes_assignment() -> None:
    other = SplitSpec.of({"fit": 0.7, "early_stop": 0.15, "calibrate": 0.15}, seed=12)
    ids = range(500)
    assert split_subjects(ids, SPEC) != split_subjects(ids, other)


def test_known_assignments_are_stable() -> None:
    """Pinned values: a change to the hashing scheme must be deliberate."""
    assert unit_hash("11", 1) == int("500547957dcbeaff", 16) / 2**64
    assert [assign_split(i, SPEC) for i in range(11, 22)] == [
        "fit",
        "calibrate",
        "early_stop",
        "fit",
        "fit",
        "fit",
        "fit",
        "early_stop",
        "early_stop",
        "early_stop",
        "calibrate",
    ]


def test_unit_hash_is_in_unit_interval() -> None:
    values = [unit_hash("s", i) for i in range(1000)]
    assert all(0.0 <= v < 1.0 for v in values)


@pytest.mark.parametrize(
    "fractions",
    [{}, {"a": 0.5, "b": 0.4}, {"a": 1.2, "b": -0.2}, {"a": 0.5, "": 0.5}],
)
def test_invalid_specs_are_rejected(fractions: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SplitSpec.of(fractions, seed=0)


def test_resplit_divides_an_extract_split(extract: MedsExtract) -> None:
    train = extract_split(extract, "train")
    assert len(train) == extract.summary.subjects_by_split["train"] == 16
    parts = resplit(extract, "train", SplitSpec.of({"fit": 0.75, "early_stop": 0.25}, seed=1))
    assert sorted(parts["fit"] + parts["early_stop"]) == list(train)
    assert set(parts["fit"]).isdisjoint(parts["early_stop"])
    assert parts == resplit(
        extract, "train", SplitSpec.of({"fit": 0.75, "early_stop": 0.25}, seed=1)
    )
