"""meds_reader database build from the synthetic extract, cached by manifest hash."""

from __future__ import annotations

from pathlib import Path

import meds_reader
import pytest

from ehrfm.inputs import ensure_database, open_database, open_extract
from ehrfm.inputs.database import database_dir
from ehrfm.inputs.errors import DatabaseBuildError
from ehrfm.toy.artifacts import SyntheticArtifacts


def test_database_builds_and_holds_every_subject(
    artifacts: SyntheticArtifacts, database_cache: Path
) -> None:
    extract = open_extract(artifacts.extract)
    path = ensure_database(extract, database_cache)
    assert path == database_cache / extract.manifest_sha256
    with meds_reader.SubjectDatabase(str(path)) as database:
        assert sorted(database) == list(range(1, 21))
        subject = database[1]
        codes = [event.code for event in subject.events]
        assert "MEDS_BIRTH" in codes
        assert set(database.properties) >= {"code", "time", "numeric_value", "text_value"}


def test_database_is_reused(artifacts: SyntheticArtifacts, database_cache: Path) -> None:
    extract = open_extract(artifacts.extract)
    first = ensure_database(extract, database_cache)
    marker = first / "ehrfm_extract_manifest.sha256"
    stamp = marker.stat().st_mtime_ns
    assert ensure_database(extract, database_cache) == first
    assert marker.stat().st_mtime_ns == stamp
    assert [p.name for p in database_cache.iterdir()] == [extract.manifest_sha256]


def test_open_database(artifacts: SyntheticArtifacts, database_cache: Path) -> None:
    extract = open_extract(artifacts.extract)
    with open_database(extract, database_cache) as database:
        assert len(database) == extract.summary.subjects


def test_incomplete_cache_entry_fails_closed(
    artifacts: SyntheticArtifacts, database_cache: Path
) -> None:
    extract = open_extract(artifacts.extract)
    database_dir(extract, database_cache).mkdir(parents=True)
    with pytest.raises(DatabaseBuildError, match="not a complete database"):
        ensure_database(extract, database_cache)
