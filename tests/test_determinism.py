"""Determinism: the same seed gives byte-identical fixtures and identical outputs."""

from __future__ import annotations

from pathlib import Path

import femr.ontology
import meds_reader
import torch

from ehrfm.inputs import ensure_database, open_extract
from ehrfm.toy.artifacts import (
    LAB_CODES,
    OBSERVATION_CODES,
    OBSERVATION_TEXTS,
    PARENT_GROUPS,
    build_artifacts,
)
from ehrfm.toy.checkpoint import build_toy_checkpoint
from tests.conftest import codes_ontology
from tests.models.test_toy_checkpoint import subject_representations


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _build(base: Path, seed: int) -> tuple[Path, Path]:
    built = build_artifacts(base / "artifacts", seed=seed)
    checkpoint = build_toy_checkpoint(
        base / "checkpoint",
        extract_codes=built.codes,
        parent_codes=sorted(set(PARENT_GROUPS.values())),
        numeric_properties=LAB_CODES,
        text_properties=[(OBSERVATION_CODES[0], OBSERVATION_TEXTS)],
        seed=seed,
    )
    return base / "artifacts", checkpoint.path


def test_same_seed_gives_identical_bytes(tmp_path: Path) -> None:
    a_artifacts, a_checkpoint = _build(tmp_path / "a", seed=3)
    b_artifacts, b_checkpoint = _build(tmp_path / "b", seed=3)
    assert _tree_bytes(a_artifacts) == _tree_bytes(b_artifacts)
    assert _tree_bytes(a_checkpoint) == _tree_bytes(b_checkpoint)


def test_different_seed_differs(tmp_path: Path) -> None:
    a_artifacts, a_checkpoint = _build(tmp_path / "a", seed=3)
    b_artifacts, b_checkpoint = _build(tmp_path / "b", seed=4)
    assert _tree_bytes(a_artifacts) != _tree_bytes(b_artifacts)
    assert _tree_bytes(a_checkpoint) != _tree_bytes(b_checkpoint)


def test_same_seed_gives_identical_representations(tmp_path: Path) -> None:
    outputs: list[dict[int, torch.Tensor]] = []
    for name in ("a", "b"):
        artifacts_root, checkpoint = _build(tmp_path / name, seed=5)
        extract_root = artifacts_root / "meds" / "synthetic_demo"
        extract = open_extract(extract_root)
        ontology: femr.ontology.Ontology = codes_ontology(extract_root)
        database_path = ensure_database(extract, tmp_path / name / "db")
        with meds_reader.SubjectDatabase(str(database_path)) as database:
            outputs.append(subject_representations(checkpoint, database, ontology))
    first, second = outputs
    assert first.keys() == second.keys()
    for subject_id in first:
        assert torch.equal(first[subject_id], second[subject_id])
