"""A directory of named, verified task heads: what a server lists and loads.

Layout::

    <root>/<name>/head.json            the head record (see head_loader)
    <root>/<name>/weights.safetensors  the head's weights

A name is a lowercase identifier (``^[a-z0-9][a-z0-9_.-]*$``), one directory
level, never a path. :meth:`HeadRegistry.publish` copies a saved head into the
registry atomically after verifying it; nothing else writes there.

Loading for serving is strict. :meth:`HeadRegistry.load` takes the identity of
the encoder that will produce the representations and refuses, with
:class:`~ehrfm.inference.head_loader.HeadLoadError`, a head that

- has a weight file whose SHA-256 differs from its record (tampered),
- takes inputs of another width than the encoder produces,
- was fitted on other encoder weights, or records no encoder at all,
- has a ``head.json`` whose SHA-256 differs from a caller-pinned digest.

There is no fallback head.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ehrfm.inference.head_loader import (
    HEAD_JSON,
    HEAD_WEIGHTS,
    HeadLoadError,
    HeadRecord,
    LoadedHead,
    load_head,
    read_head_record,
)
from ehrfm.inputs.manifest import sha256_file
from ehrfm.models.weights import require_sha256
from ehrfm.training.cache import staged_directory

HEAD_NAME: Final = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class EncoderIdentity:
    """The encoder a head will run on: its output width and weight digest."""

    hidden_size: int
    weights_sha256: str

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        require_sha256(self.weights_sha256)


@dataclass(frozen=True, slots=True)
class HeadEntry:
    """A registered head, described without loading its weights."""

    name: str
    path: Path
    record: HeadRecord
    record_sha256: str
    """SHA-256 of ``head.json``; pin it to detect a changed record."""

    def summary(self) -> dict[str, str | int | None]:
        """A JSON-ready description (for listings)."""
        spec = self.record.spec
        return {
            "name": self.name,
            "kind": spec.kind,
            "task": spec.task_name,
            "label_type": spec.label_type,
            "input_dim": spec.input_dim,
            "encoder_weights_sha256": self.record.encoder_weights_sha256,
            "weights_sha256": self.record.weights_sha256,
            "record_sha256": self.record_sha256,
        }


def require_head_name(name: str) -> str:
    """Return ``name`` if it is a valid registry name, else raise ValueError."""
    if HEAD_NAME.fullmatch(name) is None or ".." in name:
        raise ValueError(f"invalid head name {name!r} (expected {HEAD_NAME.pattern})")
    return name


class HeadRegistry:
    """Named heads under one root directory (see the module docstring)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _dir(self, name: str) -> Path:
        return self.root / require_head_name(name)

    def names(self) -> tuple[str, ...]:
        """Names of the registered heads, sorted."""
        if not self.root.is_dir():
            return ()
        return tuple(
            sorted(
                child.name
                for child in self.root.iterdir()
                if child.is_dir()
                and HEAD_NAME.fullmatch(child.name) is not None
                and (child / HEAD_JSON).is_file()
            )
        )

    def entry(self, name: str) -> HeadEntry:
        """The record of head ``name`` (weights are not read)."""
        directory = self._dir(name)
        if not (directory / HEAD_JSON).is_file():
            raise HeadLoadError(f"no head named {name!r} in {self.root}")
        record = read_head_record(directory)
        return HeadEntry(name, directory, record, sha256_file(directory / HEAD_JSON))

    def entries(self) -> tuple[HeadEntry, ...]:
        """Every registered head, by name."""
        return tuple(self.entry(name) for name in self.names())

    def load(
        self,
        name: str,
        encoder: EncoderIdentity,
        *,
        expected_record_sha256: str | None = None,
    ) -> LoadedHead:
        """Load head ``name`` for ``encoder``, verified as the module docstring lists."""
        entry = self.entry(name)
        if expected_record_sha256 is not None:
            require_sha256(expected_record_sha256)
            if entry.record_sha256 != expected_record_sha256:
                raise HeadLoadError(
                    f"head {name!r} record has SHA-256 {entry.record_sha256}, "
                    f"expected {expected_record_sha256}"
                )
        if entry.record.encoder_weights_sha256 is None:
            raise HeadLoadError(f"head {name!r} records no encoder; it cannot be served")
        loaded = load_head(
            entry.path,
            encoder_hidden_size=encoder.hidden_size,
            encoder_weights_sha256=encoder.weights_sha256,
        )
        if sha256_file(entry.path / HEAD_JSON) != entry.record_sha256:
            raise HeadLoadError(f"head {name!r} record changed while it was loaded")
        return loaded

    def load_all(self, encoder: EncoderIdentity) -> dict[str, LoadedHead]:
        """Every registered head, loaded for ``encoder``; any failure raises."""
        return {name: self.load(name, encoder) for name in self.names()}

    def publish(self, source: Path, name: str, *, replace: bool = False) -> HeadEntry:
        """Verify the saved head in ``source`` and copy it in as ``name``."""
        final = self._dir(name)
        if final.exists() and not replace:
            raise FileExistsError(f"a head named {name!r} is already registered in {self.root}")
        # Verifies the record, the weight hash and that the weights fit the kind.
        load_head(source)
        with staged_directory(final) as staging:
            for filename in (HEAD_JSON, HEAD_WEIGHTS):
                shutil.copyfile(source / filename, staging / filename)
        return self.entry(name)


__all__ = [
    "HEAD_NAME",
    "EncoderIdentity",
    "HeadEntry",
    "HeadRegistry",
    "require_head_name",
]
