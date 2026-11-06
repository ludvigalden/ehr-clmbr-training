"""Serving computations: encode a subject, run a head, postprocess, format.

Everything here is synchronous; async handlers run it in a worker thread (see
:mod:`ehrfm.server.app`). The subject's last token position — the one that
has attended to the whole record — is the representation heads run on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ehrfm.inference.head_loader import LoadedHead
from ehrfm.models import _compat
from ehrfm.server.models import (
    EmbedProvenance,
    EmbedRequest,
    EmbedResponse,
    PredictRequest,
    PredictResponse,
    Provenance,
    SubjectPayload,
)
from ehrfm.server.postprocess import apply_postprocess
from ehrfm.server.state import AppState
from ehrfm.server.subject import build_subject, count_unknown_codes

_compat.install()


@dataclass(frozen=True, slots=True)
class EncodedSubject:
    """The representation of one subject's last token position."""

    subject_id: int
    embedding: torch.Tensor
    """Float32 row of shape ``(hidden_size,)``."""
    n_positions: int


class UnknownHeadError(LookupError):
    """A request named a head the server did not load."""


class EmptySubjectError(ValueError):
    """A subject produced no token positions to represent."""


def resolve_head(state: AppState, name: str | None) -> LoadedHead:
    """The requested head; ``None`` selects the only loaded head.

    An unknown or ambiguous name raises :class:`UnknownHeadError`, which the
    routes map to HTTP 404.
    """
    if name is None:
        if len(state.heads) != 1:
            raise UnknownHeadError(
                "name the head to use: the server has "
                f"{len(state.heads)} loaded heads ({', '.join(sorted(state.heads)) or 'none'})"
            )
        return next(iter(state.heads.values()))
    head = state.heads.get(name)
    if head is None:
        raise UnknownHeadError(
            f"no head named {name!r} is loaded; loaded: {', '.join(sorted(state.heads)) or 'none'}"
        )
    return head


def encode_subject(state: AppState, payload: SubjectPayload) -> EncodedSubject:
    """Encode ``payload``'s subject and select its last token position."""
    unknown = count_unknown_codes(payload, state.known_codes, limit=state.config.unknown_code_limit)
    if unknown.events == len(payload.events):
        raise EmptySubjectError(
            f"every code of subject {payload.subject_id} is unknown to the "
            "dictionary; there is nothing to represent"
        )
    subject = build_subject(payload)
    batch = state.builder.build([subject])
    tokens = state.encoder.encode(batch)
    n_positions = int(tokens.hidden.shape[0])
    if n_positions == 0:
        raise EmptySubjectError(f"subject {payload.subject_id} produced no token positions")
    return EncodedSubject(
        subject_id=payload.subject_id,
        embedding=tokens.hidden[n_positions - 1],
        n_positions=n_positions,
    )


def head_values(head: LoadedHead, embedding: torch.Tensor) -> list[float]:
    """The head's predictions for one subject, as plain values."""
    scores = head.head.predict(embedding.unsqueeze(0))
    if head.record.spec.label_type == "categorical":
        return [float(value) for value in scores[0]]
    return [float(scores[0])]


def format_output(head: LoadedHead, values: list[float]) -> float | dict[str, float]:
    """Values in response form: one number, or a mapping per category."""
    if head.record.spec.label_type == "categorical":
        categories = head.record.spec.categories or ()
        return dict(zip(categories, values, strict=True))
    return values[0]


def run_prediction(state: AppState, request: PredictRequest) -> PredictResponse:
    """Encode the request's subject, run one head, postprocess, build the response."""
    started = time.perf_counter()
    head = resolve_head(state, request.head)
    unknown = count_unknown_codes(request, state.known_codes, limit=state.config.unknown_code_limit)
    encoded = encode_subject(state, request)
    raw = head_values(head, encoded.embedding)
    values = apply_postprocess(state.postprocess, raw, head.record)
    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    record = head.record
    return PredictResponse(
        subject_id=request.subject_id,
        head=head_name(state, head),
        task=record.spec.task_name,
        label_type=str(record.spec.label_type),
        output=format_output(head, values),
        postprocess=state.postprocess,
        unknown_codes=unknown,
        provenance=Provenance(
            encoder_weights_sha256=state.encoder_weights_sha256,
            head_weights_sha256=record.weights_sha256,
            head_record_sha256=state.record_sha256[head_name(state, head)],
        ),
        duration_ms=duration_ms,
    )


def head_name(state: AppState, head: LoadedHead) -> str:
    """The registry name ``head`` was loaded under."""
    for name, loaded in state.heads.items():
        if loaded is head:
            return name
    raise UnknownHeadError("the head is not loaded on this server")


def run_embedding(state: AppState, request: EmbedRequest) -> EmbedResponse:
    """Encode the request's subject and report its last-position representation."""
    started = time.perf_counter()
    unknown = count_unknown_codes(request, state.known_codes, limit=state.config.unknown_code_limit)
    encoded = encode_subject(state, request)
    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    return EmbedResponse(
        subject_id=request.subject_id,
        embedding=[float(value) for value in encoded.embedding],
        hidden_size=int(encoded.embedding.shape[0]),
        n_positions=encoded.n_positions,
        unknown_codes=unknown,
        provenance=EmbedProvenance(encoder_weights_sha256=state.encoder_weights_sha256),
        duration_ms=duration_ms,
    )
