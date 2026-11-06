"""Manifest reading and fail-closed integrity verification.

Hashes follow the upstream contract: SHA-256, lowercase hex, over the exact
file bytes; Parquet row counts come from the file footer. The file set under an
artifact root must equal the manifest's list exactly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq
from pydantic import ValidationError

from ehrfm.inputs.contract import CONTRACT_VERSION, MANIFEST_NAME, is_compatible
from ehrfm.inputs.errors import (
    ArtifactTypeError,
    ContractVersionError,
    IntegrityError,
    ManifestFormatError,
    MissingArtifactError,
    UnexpectedFileError,
)
from ehrfm.inputs.models import ArtifactManifest, ArtifactType

_CHUNK: Final = 1 << 20


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, lowercase hex."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def concept_list_sha256(codes: list[str]) -> str:
    """Contract concept-list hash: SHA-256 of the distinct codes, sorted by code
    point and joined with ``\\n`` (no trailing newline), as UTF-8."""
    joined = "\n".join(sorted(set(codes)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def artifact_files(root: Path) -> list[str]:
    """Relative POSIX paths of every file under ``root`` except the manifest, sorted.

    Symlinks and anything that is neither a file nor a directory are rejected.
    """
    paths: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise UnexpectedFileError(
                root, f"{rel} is a symlink; artifacts hold regular files only"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise UnexpectedFileError(root, f"{rel} is not a regular file")
        if rel != MANIFEST_NAME:
            paths.append(rel)
    return paths


def read_manifest(root: Path, expected_type: ArtifactType) -> ArtifactManifest:
    """Parse ``artifact_manifest.json`` under ``root`` and check version and type."""
    if not root.is_dir():
        raise MissingArtifactError(root, "artifact directory does not exist")
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise MissingArtifactError(
            root, f"{MANIFEST_NAME} is missing, so the artifact is incomplete or not one"
        )
    text = path.read_text(encoding="utf-8")
    try:
        payload: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestFormatError(root, f"{MANIFEST_NAME} is not JSON: {exc}") from exc
    version = payload.get("contract_version") if isinstance(payload, dict) else None
    if not isinstance(version, str):
        raise ManifestFormatError(root, f"{MANIFEST_NAME} has no contract_version string")
    if not is_compatible(version):
        raise ContractVersionError(
            root,
            f"contract version {version} is not readable by this package "
            f"(built against {CONTRACT_VERSION}: same MAJOR, MINOR no newer)",
        )
    try:
        manifest = ArtifactManifest.model_validate_json(text)
    except ValidationError as exc:
        raise ManifestFormatError(root, f"{MANIFEST_NAME} is invalid: {exc}") from exc
    if manifest.artifact_type != expected_type:
        raise ArtifactTypeError(
            root, f"manifest describes a {manifest.artifact_type}, expected a {expected_type}"
        )
    return manifest


def verify_files(root: Path, manifest: ArtifactManifest) -> None:
    """Check the files under ``root`` against ``manifest``; raise on any difference."""
    listed = {record.path: record for record in manifest.files}
    present = set(artifact_files(root))
    missing = sorted(set(listed) - present)
    if missing:
        raise MissingArtifactError(root, f"files listed in the manifest are missing: {missing}")
    unlisted = sorted(present - set(listed))
    if unlisted:
        raise UnexpectedFileError(root, f"files not listed in the manifest: {unlisted}")
    for rel, record in sorted(listed.items()):
        path = root / rel
        size = path.stat().st_size
        digest = sha256_file(path)
        if digest != record.sha256 or size != record.bytes:
            raise IntegrityError(
                root,
                f"{rel}: sha256 {digest} ({size} bytes) differs from the manifest's "
                f"{record.sha256} ({record.bytes} bytes)",
            )
        rows = pq.ParquetFile(path).metadata.num_rows if rel.endswith(".parquet") else None
        if rows != record.rows:
            raise IntegrityError(root, f"{rel}: {rows} rows, the manifest records {record.rows}")


def verified_manifest(root: Path, expected_type: ArtifactType) -> tuple[ArtifactManifest, str]:
    """Read and verify the manifest under ``root``; return it with its own SHA-256."""
    manifest = read_manifest(root, expected_type)
    verify_files(root, manifest)
    return manifest, sha256_file(root / MANIFEST_NAME)


__all__ = [
    "artifact_files",
    "concept_list_sha256",
    "read_manifest",
    "sha256_file",
    "verified_manifest",
    "verify_files",
]
