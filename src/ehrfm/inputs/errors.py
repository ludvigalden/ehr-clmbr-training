"""Typed errors for consumed artifacts. Every check fails closed by raising one."""

from __future__ import annotations

from pathlib import Path

from ehrfm.inputs.contract import PRODUCER_REPO, VERIFY_COMMAND


class InputArtifactError(Exception):
    """An input artifact is missing, malformed, tampered with or incompatible.

    The message names the artifact directory and points at the producing
    repository and its verification command.
    """

    def __init__(self, root: Path, reason: str) -> None:
        self.root = root
        self.reason = reason
        super().__init__(
            f"{root}: {reason}. The artifact is produced by {PRODUCER_REPO}; "
            f"re-create it there, or run `{VERIFY_COMMAND} {root}` in that repository "
            "for a full diagnosis."
        )


class MissingArtifactError(InputArtifactError):
    """The artifact directory, its manifest, or a listed file is absent."""


class ManifestFormatError(InputArtifactError):
    """``artifact_manifest.json`` (or another contract JSON file) does not parse or validate."""


class ContractVersionError(InputArtifactError):
    """The artifact's contract version is not readable by this package."""


class ArtifactTypeError(InputArtifactError):
    """The manifest describes a different kind of artifact than the one requested."""


class UnexpectedFileError(InputArtifactError):
    """A file is present that the manifest does not list, or is not a regular file."""


class IntegrityError(InputArtifactError):
    """A file's size, SHA-256 or row count differs from the manifest record."""


class SchemaError(InputArtifactError):
    """A Parquet file's columns, types or nullability differ from the contract."""


class LinkageError(InputArtifactError):
    """Two artifacts that must refer to each other do not (for example a label set
    recorded against a different extract)."""


class DatabaseBuildError(InputArtifactError):
    """``meds_reader_convert`` failed or is unavailable."""


__all__ = [
    "ArtifactTypeError",
    "ContractVersionError",
    "DatabaseBuildError",
    "InputArtifactError",
    "IntegrityError",
    "LinkageError",
    "ManifestFormatError",
    "MissingArtifactError",
    "SchemaError",
    "UnexpectedFileError",
]
