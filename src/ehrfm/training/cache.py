"""Run fingerprint, run record, resumable checkpoints and atomic publication.

Layout of a run directory (``runtime.output_dir``)::

    run.json            the run record (fingerprint, status, losses)
    run.lock            held while a process trains in this directory
    model/              the finished checkpoint, in the loader's layout
    checkpoints/step-N/ resumable training state every ``checkpoint_every`` steps
    tokenized/<key>/    the packed training batches, keyed by what produced them
    database/           the meds_reader database (unless configured elsewhere)

**Fingerprint.** A run is identified by the SHA-256 of a canonical JSON of
:class:`FingerprintInputs`: the extract's manifest hash, the base checkpoint's
weight, dictionary and config hashes, every optimization setting, the
number of training processes, and the versions of the code that shapes the
result (this package, femr, torch, transformers, meds-reader). Runtime
settings (output directory, device, checkpoint cadence, worker counts) are
not part of it.

**Idempotence.** A run is complete only when ``run.json`` says ``completed``
with the current fingerprint *and* ``model/model.safetensors`` hashes to the
weight hash recorded there. Anything else trains: a different fingerprint
starts over, a matching unfinished run resumes from its newest checkpoint.

**Atomicity.** Every file is written to a temporary sibling, flushed and
renamed into place. The model directory and each checkpoint are assembled in
a temporary directory and renamed as a whole, and ``run.json`` is marked
``completed`` only after the model is in place, so an interrupted run can
never look complete.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import torch
from pydantic import BaseModel, ConfigDict, ValidationError

from ehrfm.config.training import OptimizationConfig, TrainingConfig
from ehrfm.inputs.manifest import sha256_file
from ehrfm.models.checkpoint import CONFIG_FILE, DICTIONARY_FILE, WEIGHTS_FILE, require_layout

#: Bumped whenever the training procedure changes what a fingerprint produces.
TRAINING_LOGIC_VERSION: Final = 1
FINGERPRINT_FORMAT: Final = 1
RUN_RECORD: Final = "run.json"
RUN_LOCK: Final = "run.lock"
MODEL_DIR: Final = "model"
CHECKPOINTS_DIR: Final = "checkpoints"
TOKENIZED_DIR: Final = "tokenized"
TRAINING_PROVENANCE: Final = "training.json"
CHECKPOINT_STATE: Final = "state.pt"
CHECKPOINT_META: Final = "checkpoint.json"
VERSIONED_DISTRIBUTIONS: Final = (
    "ehr-clmbr-training",
    "femr",
    "torch",
    "transformers",
    "meds-reader",
)
_STEP_DIR: Final = re.compile(r"^step-(\d{9})$")

RunStatus = Literal["in_progress", "interrupted", "failed", "completed"]


class RunLockedError(RuntimeError):
    """Another process is training in the same run directory."""


class _Frozen(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class FingerprintInputs(_Frozen):
    """Everything that determines a run's trained weights."""

    format: int = FINGERPRINT_FORMAT
    logic_version: int = TRAINING_LOGIC_VERSION
    extract_manifest_sha256: str
    base_weights_sha256: str
    base_dictionary_sha256: str
    base_config_sha256: str
    optimization: OptimizationConfig
    world_size: int
    versions: dict[str, str]

    def digest(self) -> str:
        """SHA-256 of the canonical JSON form."""
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RunRecord(_Frozen):
    """The persistent state of one run directory."""

    fingerprint: str
    inputs: FingerprintInputs
    status: RunStatus
    step: int
    total_steps: int
    started_at: str
    finished_at: str | None = None
    train_losses: tuple[float, ...] = ()
    validation_loss: float | None = None
    model_weights_sha256: str | None = None
    dictionary_sha256: str | None = None
    replaced_dictionary_entries: tuple[int, ...] = ()
    resumed_from_step: int | None = None
    error: str | None = None


class CheckpointMeta(_Frozen):
    """What a resumable checkpoint belongs to."""

    fingerprint: str
    step: int
    epoch: int
    train_losses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Locations inside one run directory."""

    root: Path

    @property
    def record(self) -> Path:
        return self.root / RUN_RECORD

    @property
    def lock(self) -> Path:
        return self.root / RUN_LOCK

    @property
    def model(self) -> Path:
        return self.root / MODEL_DIR

    @property
    def checkpoints(self) -> Path:
        return self.root / CHECKPOINTS_DIR

    @property
    def tokenized(self) -> Path:
        return self.root / TOKENIZED_DIR


@dataclass(frozen=True, slots=True)
class CacheDecision:
    """What a run should do: return the cached model, resume, or start fresh."""

    action: Literal["cached", "resume", "fresh"]
    reason: str
    record: RunRecord | None = None
    checkpoint: Path | None = None


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def now_iso() -> str:
    """Current UTC time, ISO-8601, second precision."""
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def code_versions() -> dict[str, str]:
    """Installed versions of the distributions that shape a trained model."""
    return {name: importlib.metadata.version(name) for name in VERSIONED_DISTRIBUTIONS}


def fingerprint_inputs(
    config: TrainingConfig,
    *,
    extract_manifest_sha256: str,
    base_weights_sha256: str,
    world_size: int = 1,
    versions: dict[str, str] | None = None,
) -> FingerprintInputs:
    """Collect the fingerprint inputs of ``config``; the base checkpoint files are hashed here."""
    base = config.inputs.base_checkpoint
    require_layout(base)
    return FingerprintInputs(
        extract_manifest_sha256=extract_manifest_sha256,
        base_weights_sha256=base_weights_sha256,
        base_dictionary_sha256=sha256_file(base / DICTIONARY_FILE),
        base_config_sha256=sha256_file(base / CONFIG_FILE),
        optimization=config.optimization,
        world_size=world_size,
        versions=code_versions() if versions is None else versions,
    )


def _fsync_dir(path: Path) -> None:
    with contextlib.suppress(OSError):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a flushed temporary sibling and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    _fsync_dir(path.parent)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write ``payload`` as sorted, indented JSON, atomically."""
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    write_bytes_atomic(path, text.encode("utf-8"))


def write_record(paths: RunPaths, record: RunRecord) -> None:
    """Atomically replace the run record."""
    write_json_atomic(paths.record, record.model_dump(mode="json"))


def read_record(paths: RunPaths) -> RunRecord | None:
    """The run record, or None when it is absent or unreadable (treated as no record)."""
    try:
        raw = json.loads(paths.record.read_text(encoding="utf-8"))
        return RunRecord.model_validate(raw, strict=False)
    except (OSError, ValueError, ValidationError):
        return None


def published_model_matches(paths: RunPaths, record: RunRecord) -> bool:
    """Whether ``model/`` is complete and its weights hash to what the record says."""
    if record.model_weights_sha256 is None:
        return False
    try:
        require_layout(paths.model)
    except RuntimeError:
        return False
    return sha256_file(paths.model / WEIGHTS_FILE) == record.model_weights_sha256


def checkpoint_dirs(paths: RunPaths) -> list[tuple[int, Path]]:
    """Complete checkpoints, oldest first (temporary directories are ignored)."""
    if not paths.checkpoints.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for child in paths.checkpoints.iterdir():
        match = _STEP_DIR.match(child.name)
        if match and (child / CHECKPOINT_META).is_file() and (child / CHECKPOINT_STATE).is_file():
            found.append((int(match.group(1)), child))
    return sorted(found)


def read_checkpoint_meta(path: Path) -> CheckpointMeta | None:
    """The checkpoint's metadata, or None when unreadable."""
    try:
        raw = json.loads((path / CHECKPOINT_META).read_text(encoding="utf-8"))
        return CheckpointMeta.model_validate(raw, strict=False)
    except (OSError, ValueError, ValidationError):
        return None


def latest_checkpoint(paths: RunPaths, fingerprint: str) -> Path | None:
    """The newest complete checkpoint that belongs to ``fingerprint``."""
    for _, path in reversed(checkpoint_dirs(paths)):
        meta = read_checkpoint_meta(path)
        if meta is not None and meta.fingerprint == fingerprint:
            return path
    return None


def decide(paths: RunPaths, fingerprint: str) -> CacheDecision:
    """Decide between the cached model, resuming and a fresh start."""
    record = read_record(paths)
    if record is None:
        return CacheDecision(action="fresh", reason="no run record")
    if record.fingerprint != fingerprint:
        return CacheDecision(action="fresh", reason="fingerprint changed", record=record)
    if record.status == "completed":
        if published_model_matches(paths, record):
            return CacheDecision(action="cached", reason="completed run", record=record)
        return CacheDecision(
            action="fresh", reason="completed record but model missing or altered", record=record
        )
    checkpoint = latest_checkpoint(paths, fingerprint)
    if checkpoint is None:
        return CacheDecision(
            action="fresh", reason=f"{record.status} run without a checkpoint", record=record
        )
    return CacheDecision(
        action="resume", reason=f"{record.status} run", record=record, checkpoint=checkpoint
    )


def reset_run(paths: RunPaths) -> None:
    """Remove the model and checkpoints of a superseded run (the tokenized cache stays)."""
    shutil.rmtree(paths.model, ignore_errors=True)
    shutil.rmtree(paths.checkpoints, ignore_errors=True)


def save_checkpoint(
    paths: RunPaths, meta: CheckpointMeta, state: dict[str, Any], keep: int
) -> Path:
    """Write a resumable checkpoint atomically and prune all but the newest ``keep``."""
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    final = paths.checkpoints / f"step-{meta.step:09d}"
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=paths.checkpoints))
    try:
        torch.save(state, staging / CHECKPOINT_STATE)
        write_json_atomic(staging / CHECKPOINT_META, meta.model_dump(mode="json"))
        _fsync_dir(staging)
        if final.exists():
            shutil.rmtree(final)
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _fsync_dir(paths.checkpoints)
    for _, old in checkpoint_dirs(paths)[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return final


def load_checkpoint_state(path: Path) -> tuple[CheckpointMeta, dict[str, Any]]:
    """Read a checkpoint written by :func:`save_checkpoint`."""
    meta = read_checkpoint_meta(path)
    if meta is None:
        raise RuntimeError(f"checkpoint {path} has no readable {CHECKPOINT_META}")
    state: dict[str, Any] = torch.load(
        path / CHECKPOINT_STATE, map_location="cpu", weights_only=True
    )
    return meta, state


@contextlib.contextmanager
def staged_directory(final: Path) -> Generator[Path]:
    """Yield an empty sibling directory that replaces ``final`` as a whole on success."""
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        yield staging
        _fsync_dir(staging)
        if final.exists():
            shutil.rmtree(final)
        os.rename(staging, final)
        _fsync_dir(final.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@contextlib.contextmanager
def run_lock(paths: RunPaths) -> Generator[None]:
    """Hold an exclusive lock on the run directory; raise when another process holds it."""
    paths.root.mkdir(parents=True, exist_ok=True)
    with paths.lock.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunLockedError(f"another process is training in {paths.root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = [
    "CHECKPOINTS_DIR",
    "MODEL_DIR",
    "RUN_RECORD",
    "TOKENIZED_DIR",
    "TRAINING_LOGIC_VERSION",
    "TRAINING_PROVENANCE",
    "CacheDecision",
    "CheckpointMeta",
    "FingerprintInputs",
    "RunLockedError",
    "RunPaths",
    "RunRecord",
    "RunStatus",
    "checkpoint_dirs",
    "code_versions",
    "decide",
    "fingerprint_inputs",
    "latest_checkpoint",
    "load_checkpoint_state",
    "now_iso",
    "published_model_matches",
    "read_record",
    "reset_run",
    "run_lock",
    "save_checkpoint",
    "sha256_bytes",
    "staged_directory",
    "write_bytes_atomic",
    "write_json_atomic",
    "write_record",
]
