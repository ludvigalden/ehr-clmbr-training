"""SHA-256 provenance for checkpoint weights.

Weights are read into memory once, hashed, and only then deserialised from
the same bytes, so the recorded hash always describes the tensors that were
loaded. When an expected hash is pinned, a mismatch fails closed before any
tensor is built.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import safetensors.torch
from torch import Tensor

from ehrfm.inputs.manifest import sha256_file

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


class WeightIntegrityError(RuntimeError):
    """Weights on disk do not match the pinned SHA-256."""


@dataclass(frozen=True, slots=True)
class WeightProvenance:
    """Which weight file was loaded, and the SHA-256 of its exact bytes."""

    path: Path
    sha256: str
    size: int


def require_sha256(value: str) -> str:
    """Return ``value`` when it is a lowercase hex SHA-256; raise ValueError otherwise."""
    if _SHA256_RE.match(value) is None:
        raise ValueError(f"{value!r} is not a lowercase hex SHA-256 digest")
    return value


def weights_sha256(path: Path) -> str:
    """SHA-256 of a weight file, for pinning it in configuration."""
    return sha256_file(path)


def read_weights(
    path: Path, expected_sha256: str | None = None
) -> tuple[dict[str, Tensor], WeightProvenance]:
    """Read a safetensors file, hash its bytes and return the tensors with their provenance.

    With ``expected_sha256``, a file whose hash differs raises
    :class:`WeightIntegrityError` and nothing is deserialised.
    """
    if expected_sha256 is not None:
        require_sha256(expected_sha256)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise WeightIntegrityError(
            f"{path} has SHA-256 {digest}, but {expected_sha256} was pinned; "
            "the weights changed since they were pinned"
        )
    state = safetensors.torch.load(data)
    return state, WeightProvenance(path=path, sha256=digest, size=len(data))


__all__ = [
    "WeightIntegrityError",
    "WeightProvenance",
    "read_weights",
    "require_sha256",
    "weights_sha256",
]
