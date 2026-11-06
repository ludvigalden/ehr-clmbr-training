"""Build or open the ``meds_reader`` database of a verified MEDS extract.

The database is a binary index whose format belongs to the installed
``meds-reader`` release, so this package builds it from the extract with
``meds_reader_convert``. The producing repository deliberately leaves this
conversion to the consumer (its ``artifacts.md``, section "FEMR side: the
``meds_reader`` database"); this package is that consumer and owns the step.
Databases are cached under
``<cache_dir>/<extract manifest sha256>/``: the manifest hash covers every
file of the extract, so a changed extract can never reuse a stale database. A
build writes to a temporary sibling directory and is renamed into place only
after it succeeds and its subject count matches the extract's manifest.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

import meds_reader

from ehrfm.inputs.artifacts import MedsExtract
from ehrfm.inputs.errors import DatabaseBuildError

CONVERT_EXECUTABLE: Final = "meds_reader_convert"
_COMPLETE_MARKER: Final = "ehrfm_extract_manifest.sha256"


def convert_executable() -> Path:
    """Locate ``meds_reader_convert``: next to the interpreter first, then on PATH."""
    beside = Path(sys.executable).parent / CONVERT_EXECUTABLE
    if beside.is_file() and os.access(beside, os.X_OK):
        return beside
    found = shutil.which(CONVERT_EXECUTABLE)
    if found is None:
        raise DatabaseBuildError(
            Path(CONVERT_EXECUTABLE),
            "meds_reader_convert is not installed; it ships with the meds-reader package",
        )
    return Path(found)


def database_dir(extract: MedsExtract, cache_dir: Path) -> Path:
    """The cache location of ``extract``'s database."""
    return cache_dir / extract.manifest_sha256


def _is_complete(path: Path, extract: MedsExtract) -> bool:
    marker = path / _COMPLETE_MARKER
    return (
        marker.is_file() and marker.read_text(encoding="utf-8").strip() == extract.manifest_sha256
    )


def ensure_database(extract: MedsExtract, cache_dir: Path, num_threads: int = 1) -> Path:
    """Return the database directory for ``extract``, building it when absent."""
    target = database_dir(extract, cache_dir)
    if _is_complete(target, extract):
        return target
    if target.exists():
        raise DatabaseBuildError(
            extract.root,
            f"{target} exists but is not a complete database for this extract; remove it and retry",
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{extract.manifest_sha256[:12]}-", dir=cache_dir))
    output = staging / "database"
    try:
        command = [
            str(convert_executable()),
            str(extract.root),
            str(output),
            "--num_threads",
            str(num_threads),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise DatabaseBuildError(
                extract.root,
                f"meds_reader_convert exited {result.returncode}: "
                f"{(result.stderr or result.stdout).strip()[-2000:]}",
            )
        with meds_reader.SubjectDatabase(str(output)) as database:
            subjects = len(database)
        if subjects != extract.summary.subjects:
            raise DatabaseBuildError(
                extract.root,
                f"database holds {subjects} subjects, the extract manifest records "
                f"{extract.summary.subjects}",
            )
        (output / _COMPLETE_MARKER).write_text(extract.manifest_sha256 + "\n", encoding="utf-8")
        output.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def open_database(
    extract: MedsExtract, cache_dir: Path, num_threads: int = 1
) -> meds_reader.SubjectDatabase:
    """Build the database when needed and open it."""
    path = ensure_database(extract, cache_dir, num_threads=num_threads)
    return meds_reader.SubjectDatabase(str(path), num_threads=num_threads)


__all__ = ["convert_executable", "database_dir", "ensure_database", "open_database"]
