"""Shared helpers of the server tests (pure functions; fixtures live in conftest)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import torch

from ehrfm.inference.head_loader import save_head
from ehrfm.models.head import HeadSpec, LinearHead
from ehrfm.models.representations import ClmbrEncoder
from ehrfm.server.models import SubjectPayload
from ehrfm.server.subject import build_subject
from ehrfm.toy.checkpoint import ToyCheckpoint


def make_head(spec: HeadSpec, seed: int, out_dir: Path, encoder_sha: str) -> LinearHead:
    """A deterministic linear head with seeded weights, saved into ``out_dir/head``."""
    torch.manual_seed(seed)
    head = LinearHead(spec)
    save_head(head, out_dir / "head", encoder_weights_sha256=encoder_sha)
    return head


BIRTH = "2019-06-01T00:00:00"
"""A birth before every request event (requests must carry one)."""


def events(codes: list[str], *, start: dt.datetime | None = None) -> list[dict[str, object]]:
    """Naive-time event payloads for a request."""
    base = start or dt.datetime(2020, 1, 1)
    return [
        {"code": code, "time": (base + dt.timedelta(days=offset)).isoformat()}
        for offset, code in enumerate(codes)
    ]


def direct_embedding(
    encoder: ClmbrEncoder, toy_checkpoint: ToyCheckpoint, payload: dict[str, object]
) -> torch.Tensor:
    """The last-position embedding the server must reproduce."""
    from ehrfm.data.batches import BatchBuilder
    from ehrfm.data.tokenization import CodeOntology
    from ehrfm.models.checkpoint import load_tokenizer

    tokenizer = load_tokenizer(toy_checkpoint.path, CodeOntology({}))
    builder = BatchBuilder(tokenizer)
    subject = build_subject(SubjectPayload.model_validate(payload))
    batch = builder.build([subject])
    tokens = encoder.encode(batch)
    return tokens.hidden[tokens.hidden.shape[0] - 1]


def parse_sse(text: str) -> list[tuple[str, str, dict[str, object]]]:
    """Parse an SSE body into (event, id, data) triples."""
    frames: list[tuple[str, str, dict[str, object]]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event, identifier, data = "", "", ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("id: "):
                identifier = line.removeprefix("id: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        frames.append((event, identifier, json.loads(data)))
    return frames
