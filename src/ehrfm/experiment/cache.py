"""Stage-level cache of the experiment pipeline.

Every stage computes a *key*: the SHA-256 of the canonical JSON of what
determines its output (the stage name, a logic version, the keys or content
hashes of what it consumes, and its own configuration). A completed stage is
published atomically as ``<work_dir>/stages/<stage>/<key[:16]>/`` holding
``stage.json`` (a :class:`StageRecord`) and any files the stage wrote. Another
run with the same key reads the record instead of recomputing.

Runtime settings (devices, worker counts, output locations) never enter a key.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from ehrfm.training.cache import staged_directory, write_json_atomic

STAGES_DIR: Final = "stages"
STAGE_RECORD: Final = "stage.json"
STAGE_FORMAT: Final = 1
KEY_PREFIX: Final = 16

StageStatus = Literal["computed", "cached"]


class StageRecord(BaseModel):
    """What a completed stage produced, keyed by its inputs."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    format: int = STAGE_FORMAT
    stage: str
    key: str
    key_inputs: dict[str, JsonValue]
    """The canonical inputs the key was computed from (hashes and configuration)."""
    outputs: dict[str, JsonValue]
    """Stage outputs other stages and tracking read (paths are relative to the stage dir)."""
    created_at: str
    duration_seconds: float


def stage_key(key_inputs: dict[str, JsonValue]) -> str:
    """SHA-256 of the canonical JSON form of ``key_inputs``."""
    canonical = json.dumps(key_inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stage_dir(work_dir: Path, stage: str, key: str) -> Path:
    """Directory of the stage result with ``key``."""
    return work_dir / STAGES_DIR / stage / key[:KEY_PREFIX]


def read_stage(path: Path) -> StageRecord | None:
    """The record in ``path``, or None when absent or unreadable."""
    try:
        return StageRecord.model_validate_json((path / STAGE_RECORD).read_bytes())
    except (OSError, ValueError, ValidationError):
        return None


def lookup(work_dir: Path, stage: str, key: str) -> StageRecord | None:
    """The completed record of ``stage`` with exactly ``key``, if any."""
    record = read_stage(stage_dir(work_dir, stage, key))
    if record is None or record.key != key or record.stage != stage:
        return None
    return record


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


@contextmanager
def publishing(work_dir: Path, stage: str, key: str) -> Generator[Path]:
    """Yield an empty staging directory that becomes the stage dir on success."""
    with staged_directory(stage_dir(work_dir, stage, key)) as staging:
        yield staging


def run_stage(
    work_dir: Path,
    stage: str,
    key_inputs: dict[str, JsonValue],
    compute: Callable[[Path], dict[str, JsonValue]],
    *,
    valid: Callable[[StageRecord], bool] | None = None,
) -> tuple[StageRecord, StageStatus, Path]:
    """Return the cached record for ``key_inputs`` or compute and publish a new one.

    ``compute`` receives an empty directory, writes its files there and returns
    the outputs; nothing is published when it raises. ``valid`` can reject a
    cached record whose external outputs went missing; it is then recomputed.
    """
    key = stage_key({"stage": stage, "format": STAGE_FORMAT, **key_inputs})
    final = stage_dir(work_dir, stage, key)
    cached = lookup(work_dir, stage, key)
    if cached is not None and (valid is None or valid(cached)):
        return cached, "cached", final
    started = dt.datetime.now(dt.UTC)
    with publishing(work_dir, stage, key) as staging:
        outputs = compute(staging)
        record = StageRecord(
            stage=stage,
            key=key,
            key_inputs=key_inputs,
            outputs=outputs,
            created_at=_now(),
            duration_seconds=(dt.datetime.now(dt.UTC) - started).total_seconds(),
        )
        write_json_atomic(staging / STAGE_RECORD, record.model_dump(mode="json"))
    return record, "computed", final


def list_stages(work_dir: Path) -> list[tuple[Path, StageRecord]]:
    """Every readable stage record under ``work_dir``, oldest first."""
    found: list[tuple[Path, StageRecord]] = []
    root = work_dir / STAGES_DIR
    if not root.is_dir():
        return found
    for record_path in sorted(root.glob(f"*/*/{STAGE_RECORD}")):
        record = read_stage(record_path.parent)
        if record is not None:
            found.append((record_path.parent, record))
    found.sort(key=lambda item: (item[1].created_at, item[1].stage))
    return found


__all__ = [
    "KEY_PREFIX",
    "STAGES_DIR",
    "STAGE_RECORD",
    "StageRecord",
    "StageStatus",
    "list_stages",
    "lookup",
    "read_stage",
    "run_stage",
    "stage_dir",
    "stage_key",
]
