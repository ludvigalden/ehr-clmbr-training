"""Save and load task heads with integrity checks.

A saved head is a directory with two files:

``head.json``
    ``format_version``, the :class:`~ehrfm.models.head.HeadSpec`, the SHA-256
    of the weight file, the SHA-256 of the encoder weights the head was fitted
    on (when known), and free-form ``metadata`` (training details such as the
    regularisation strength or tuning metrics; scalars only).
``weights.safetensors``
    The head's state dict.

Loading verifies the format version, the weight hash, that the head's input
width fits the encoder, and (when both sides know it) that the head was
fitted on the same encoder weights. Any mismatch raises :class:`HeadLoadError`;
there is no fallback head.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import safetensors.torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ehrfm.inputs.manifest import sha256_file
from ehrfm.models.head import HEAD_FORMAT_VERSION, HeadSpec, TaskHead, build_head
from ehrfm.models.weights import require_sha256

HEAD_JSON: Final = "head.json"
HEAD_WEIGHTS: Final = "weights.safetensors"

type MetadataValue = str | int | float | bool | None


class HeadLoadError(RuntimeError):
    """A saved head is missing, malformed, tampered with or does not fit the encoder."""


class HeadRecord(BaseModel):
    """Contents of ``head.json``."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    format_version: int = Field(ge=1)
    spec: HeadSpec
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadedHead:
    """A verified, evaluation-mode head and its record."""

    head: TaskHead
    record: HeadRecord


def save_head(
    head: TaskHead,
    out_dir: Path,
    *,
    encoder_weights_sha256: str | None = None,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> HeadRecord:
    """Write ``head`` to the new directory ``out_dir`` and return its record."""
    if encoder_weights_sha256 is not None:
        require_sha256(encoder_weights_sha256)
    out_dir.mkdir(parents=True)
    weights_path = out_dir / HEAD_WEIGHTS
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in head.state_dict().items()}
    safetensors.torch.save_file(state, str(weights_path), metadata={"format": "pt"})
    record = HeadRecord(
        format_version=HEAD_FORMAT_VERSION,
        spec=head.spec,
        weights_sha256=sha256_file(weights_path),
        encoder_weights_sha256=encoder_weights_sha256,
        metadata=dict(metadata or {}),
    )
    payload = record.model_dump(mode="json")
    (out_dir / HEAD_JSON).write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return record


def read_head_record(head_dir: Path) -> HeadRecord:
    """Read and validate ``head.json`` without loading weights."""
    path = head_dir / HEAD_JSON
    if not path.is_file():
        raise HeadLoadError(f"{head_dir} has no {HEAD_JSON}")
    try:
        record = HeadRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise HeadLoadError(f"{path} is not a valid head record: {exc}") from exc
    if record.format_version != HEAD_FORMAT_VERSION:
        raise HeadLoadError(
            f"{path} has format version {record.format_version}; "
            f"this package reads version {HEAD_FORMAT_VERSION}"
        )
    return record


def load_head(
    head_dir: Path,
    *,
    encoder_hidden_size: int | None = None,
    encoder_weights_sha256: str | None = None,
) -> LoadedHead:
    """Load and verify the head saved in ``head_dir``.

    ``encoder_hidden_size`` and ``encoder_weights_sha256`` describe the encoder
    the head will run on; when given they must match the head's record.
    """
    record = read_head_record(head_dir)
    weights_path = head_dir / HEAD_WEIGHTS
    if not weights_path.is_file():
        raise HeadLoadError(f"{head_dir} has no {HEAD_WEIGHTS}")
    actual = sha256_file(weights_path)
    if actual != record.weights_sha256:
        raise HeadLoadError(
            f"{weights_path} has SHA-256 {actual}, the record pins {record.weights_sha256}"
        )
    if encoder_hidden_size is not None and encoder_hidden_size != record.spec.input_dim:
        raise HeadLoadError(
            f"head {record.spec.task_name!r} takes {record.spec.input_dim}-wide inputs, "
            f"the encoder produces {encoder_hidden_size}"
        )
    if (
        encoder_weights_sha256 is not None
        and record.encoder_weights_sha256 is not None
        and encoder_weights_sha256 != record.encoder_weights_sha256
    ):
        raise HeadLoadError(
            f"head {record.spec.task_name!r} was fitted on encoder weights "
            f"{record.encoder_weights_sha256}, not {encoder_weights_sha256}"
        )
    try:
        head = build_head(record.spec)
    except ValueError as exc:
        raise HeadLoadError(str(exc)) from exc
    state = safetensors.torch.load_file(str(weights_path))
    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise HeadLoadError(
            f"{weights_path} does not fit a {record.spec.kind} head: "
            f"missing {sorted(missing)}, unexpected {sorted(unexpected)}"
        )
    head.eval()
    return LoadedHead(head=head, record=record)


__all__ = [
    "HEAD_JSON",
    "HEAD_WEIGHTS",
    "HeadLoadError",
    "HeadRecord",
    "LoadedHead",
    "load_head",
    "read_head_record",
    "save_head",
]
