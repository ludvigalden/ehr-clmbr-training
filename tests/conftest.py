"""Shared fixtures: synthetic contract artifacts and a toy CLMBR checkpoint.

Importing :mod:`ehrfm.models` installs the package's torch-hawk compatibility
shim (the same mechanism production entrypoints use), so femr's model modules
import on hosts without torch-hawk.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import femr.ontology
import meds_reader
import pytest

from ehrfm.data.tokenization import ontology_from_extract
from ehrfm.inputs import MedsExtract, ensure_database, open_extract
from ehrfm.models import _compat
from ehrfm.toy.artifacts import (
    LAB_CODES,
    OBSERVATION_CODES,
    OBSERVATION_TEXTS,
    PARENT_GROUPS,
    SyntheticArtifacts,
    build_artifacts,
)
from ehrfm.toy.checkpoint import ToyCheckpoint, build_toy_checkpoint

_compat.install()


@pytest.fixture(autouse=True)
def _no_cli_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force console output uncoloured for every test.

    typer/rich emit ANSI codes when a colour-emitting environment is
    detected (``GITHUB_ACTIONS`` on CI runners, ``FORCE_COLOR``, a real
    ``TERM``), and the codes split the phrases and flag names the tests
    assert on. The autouse fixture neutralises those variables in every
    test's environment, so output assertions hold regardless of the
    environment the suite runs in.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    for variable in ("GITHUB_ACTIONS", "FORCE_COLOR", "CLICOLOR_FORCE"):
        monkeypatch.delenv(variable, raising=False)


def codes_ontology(extract: Path) -> femr.ontology.Ontology:
    """The ontology recorded in the extract's ``codes.parquet`` (its invented parent codes)."""
    return ontology_from_extract(open_extract(extract))


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> SyntheticArtifacts:
    """A synthetic extract (20 subjects) and label set, built once per session."""
    return build_artifacts(tmp_path_factory.mktemp("artifacts"))


@pytest.fixture(scope="session")
def toy_checkpoint(
    tmp_path_factory: pytest.TempPathFactory, artifacts: SyntheticArtifacts
) -> ToyCheckpoint:
    """A toy CLMBR checkpoint whose dictionary is drawn from the synthetic extract."""
    return build_toy_checkpoint(
        tmp_path_factory.mktemp("checkpoint") / "toy_clmbr",
        extract_codes=artifacts.codes,
        parent_codes=sorted(set(PARENT_GROUPS.values())),
        numeric_properties=LAB_CODES,
        text_properties=[(OBSERVATION_CODES[0], OBSERVATION_TEXTS)],
    )


@pytest.fixture(scope="session")
def ontology(artifacts: SyntheticArtifacts) -> femr.ontology.Ontology:
    """The invented ontology of the synthetic extract."""
    return codes_ontology(artifacts.extract)


@pytest.fixture
def database_cache(tmp_path: Path) -> Iterator[Path]:
    """A fresh meds_reader database cache directory."""
    yield tmp_path / "db-cache"


@pytest.fixture(scope="session")
def extract(artifacts: SyntheticArtifacts) -> MedsExtract:
    """The verified synthetic extract."""
    return open_extract(artifacts.extract)


@pytest.fixture(scope="session")
def database(
    tmp_path_factory: pytest.TempPathFactory, extract: MedsExtract
) -> Iterator[meds_reader.SubjectDatabase]:
    """An open meds_reader database of the synthetic extract, shared by the session."""
    path = ensure_database(extract, tmp_path_factory.mktemp("session-db"))
    with meds_reader.SubjectDatabase(str(path)) as opened:
        yield opened
