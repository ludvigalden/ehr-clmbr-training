"""Finding the heads an experiment trained in its work directory.

A head hook that saves a servable head writes it to ``output/head`` inside its
stage directory (``<work_dir>/stages/head-<name>/<key>/``). :func:`trained_heads`
lists those, newest first, so one can be published to a
:class:`~ehrfm.inference.registry.HeadRegistry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ehrfm.experiment.cache import list_stages
from ehrfm.inference.head_loader import HEAD_JSON, HeadRecord, read_head_record

HEAD_STAGE_PREFIX: Final = "head-"
HEAD_SUBDIR: Final = Path("output") / "head"


@dataclass(frozen=True, slots=True)
class TrainedHead:
    """A saved head inside a completed head stage."""

    hook: str
    stage_key: str
    created_at: str
    path: Path
    record: HeadRecord


def trained_heads(work_dir: Path) -> list[TrainedHead]:
    """Every saved head in ``work_dir``'s head stages, newest first."""
    found: list[TrainedHead] = []
    for stage_dir, record in list_stages(work_dir):
        if not record.stage.startswith(HEAD_STAGE_PREFIX):
            continue
        head_dir = stage_dir / HEAD_SUBDIR
        if not (head_dir / HEAD_JSON).is_file():
            continue
        found.append(
            TrainedHead(
                hook=record.stage.removeprefix(HEAD_STAGE_PREFIX),
                stage_key=record.key,
                created_at=record.created_at,
                path=head_dir,
                record=read_head_record(head_dir),
            )
        )
    found.reverse()
    return found


def find_trained_head(work_dir: Path, ref: str) -> TrainedHead:
    """The newest head of hook ``ref``, or the head whose stage key starts with ``ref``."""
    heads = trained_heads(work_dir)
    by_hook = [head for head in heads if head.hook == ref]
    if by_hook:
        return by_hook[0]
    by_key = [head for head in heads if head.stage_key.startswith(ref)]
    if len(by_key) == 1:
        return by_key[0]
    if by_key:
        raise LookupError(f"stage key prefix {ref!r} matches {len(by_key)} heads")
    raise LookupError(f"no trained head {ref!r} in {work_dir}")


__all__ = ["HEAD_STAGE_PREFIX", "TrainedHead", "find_trained_head", "trained_heads"]
