#!/usr/bin/env python
"""Fetch the pinned CLMBR-T-Base release; the same behaviour as `ehrfm fetch-model`.

Run it with the project environment (``uv run scripts/fetch_clmbr_t_base.py
[--dest DIR] [--dry-run]``). The release is gated behind the EHRSHOT
Credentialed Health Data License; see the licence notice the script prints.
A checksum mismatch, an unreachable source or a missing token exits 2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ehrfm.inputs.fetch import (
    FetchError,
    destination_default,
    fetch,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and verify the pinned CLMBR-T-Base release.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Destination directory (default: a user cache dir).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the licence notice and the plan; make no network call.",
    )
    args = parser.parse_args(argv)
    target = Path(args.dest) if args.dest else destination_default()
    try:
        result = fetch(target, dry_run=args.dry_run)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if result is None:
        return 0
    print(f"snapshot: {result.snapshot_dir}")
    weights = result.snapshot_dir / "model.safetensors"
    if weights.is_file():
        import hashlib

        sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        print(
            f"weights:  {weights} (sha256 {sha}; "
            "set EHRFM_SERVER_WEIGHTS_SHA256 to this value to serve it)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
