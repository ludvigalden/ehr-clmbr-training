"""Fetch and verify the pinned CLMBR-T-Base model release.

The release is gated: Hugging Face grants access only after the requester
accepts the EHRSHOT Credentialed Health Data License (a CITI certificate
plus a non-commercial use agreement), which is why the fetch needs the
user's own ``HF_TOKEN`` from the environment. The token is never written
to disk, cached or printed; it is passed straight to the download call.

The pinned revision's ``dictionary.msgpack`` is verified against
``ehrfm.inputs.pins.CLMBR_T_BASE_PIN``; a checksum mismatch, an
unreachable source or a missing token fails closed. The licence situation
is printed on every run (see :func:`licence_notice`).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ehrfm.inputs.pins import CLMBR_T_BASE_PIN, DictionaryPin

TOKEN_VARIABLE: Final = "HF_TOKEN"
"""The environment variable for the user's gated-access token."""

HUGGING_FACE_URL: Final = "https://huggingface.co/{repo}/tree/{revision}"
"""A template for the pinned release's web address."""

_TOKEN_MESSAGE: Final = (
    f"{TOKEN_VARIABLE} is not set. The release is gated: accept its access terms on "
    "Hugging Face, then export HF_TOKEN; the token is read from the environment only, "
    "never written to disk, cached or printed."
)

_LICENCE_BODY: Final = (
    "Licence notice: access to {repo} is gated behind the EHRSHOT Credentialed Health "
    "Data License (a CITI certificate plus a non-commercial use agreement accepted on "
    "Hugging Face). The model card states CC-BY-NC-4.0; the bundled LICENSE file states "
    "CC BY-NC-SA 4.0. The two differ; read both before any redistribution."
)


class FetchError(Exception):
    """The pinned release could not be fetched and verified; fail closed."""


class TokenMissingError(FetchError):
    """The gated release needs HF_TOKEN and it is not set."""


class UnreachableSourceError(FetchError):
    """Hugging Face could not be reached, or refused the request."""


class ChecksumMismatchError(FetchError):
    """The downloaded dictionary does not match the pinned SHA-256."""


def licence_notice(pin: DictionaryPin = CLMBR_T_BASE_PIN) -> str:
    """The licence notice printed on every fetch (dry-run or real)."""
    return _LICENCE_BODY.format(repo=pin.repo)


def token_missing() -> bool:
    """Whether a real fetch would fail for lack of a token."""
    return not os.environ.get(TOKEN_VARIABLE, "").strip()


def destination_default() -> Path:
    """Where the release lands when --dest is omitted (the user cache)."""
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return cache_home / "ehrfm" / "clmbr-t-base"


@dataclass(frozen=True, slots=True)
class FetchPlan:
    """What a run would do, printed by --dry-run and before every real fetch."""

    pin: DictionaryPin
    destination: Path

    def render(self) -> str:
        return "\n".join(
            (
                f"repository:      {self.pin.repo}",
                f"revision:       {self.pin.revision}",
                f"file:           {self.pin.filename}",
                f"expected sha256: {self.pin.file_sha256}",
                f"destination:    {self.destination}",
            )
        )


@dataclass(frozen=True, slots=True)
class FetchResult:
    """The outcome of a verified download."""

    pin: DictionaryPin
    destination: Path
    snapshot_dir: Path
    file_sha256: str


Downloader = Callable[[str, str, Path, str], Path]
"""``(repo_id, revision, local_dir, token) -> snapshot directory``."""


def _hf_snapshot(repo_id: str, revision: str, destination: Path, token: str) -> Path:
    import huggingface_hub

    return Path(
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=destination,
            token=token,
        )
    )


def fetch(
    destination: Path,
    *,
    dry_run: bool,
    pin: DictionaryPin = CLMBR_T_BASE_PIN,
    downloader: Downloader | None = None,
) -> FetchResult | None:
    """Download and verify the pinned release; return None for a dry run.

    ``downloader(repo_id, revision, local_dir, token) -> snapshot_dir``
    replaces the Hugging Face call in tests. Raises TokenMissingError,
    UnreachableSourceError or ChecksumMismatchError; the caller maps these
    to exit codes.
    """
    print(licence_notice(pin))
    print(FetchPlan(pin=pin, destination=destination).render())
    if dry_run:
        print("dry run: no network call was made")
        return None
    token = os.environ.get(TOKEN_VARIABLE, "").strip()
    if not token:
        raise TokenMissingError(_TOKEN_MESSAGE)
    if downloader is None:
        downloader = _hf_snapshot
    try:
        snapshot_dir = downloader(pin.repo, pin.revision, destination, token)
    except FetchError:
        raise
    except Exception as exc:
        raise UnreachableSourceError(
            f"the pinned release could not be downloaded from Hugging Face: {exc}"
        ) from exc
    dictionary = snapshot_dir / pin.filename
    if not dictionary.is_file():
        raise ChecksumMismatchError(f"{pin.filename} is missing from the snapshot")
    sha = hashlib.sha256(dictionary.read_bytes()).hexdigest()
    if sha != pin.file_sha256:
        raise ChecksumMismatchError(
            f"{pin.filename} sha256 {sha} does not match the pin "
            f"{pin.file_sha256}; the upstream release changed or the pin is wrong; "
            "refusing to use it"
        )
    print(f"verified: {pin.filename} sha256 matches the pin")
    return FetchResult(
        pin=pin,
        destination=destination,
        snapshot_dir=snapshot_dir,
        file_sha256=sha,
    )


__all__ = [
    "ChecksumMismatchError",
    "Downloader",
    "FetchError",
    "FetchPlan",
    "FetchResult",
    "HUGGING_FACE_URL",
    "TOKEN_VARIABLE",
    "TokenMissingError",
    "UnreachableSourceError",
    "destination_default",
    "fetch",
    "licence_notice",
    "token_missing",
]
