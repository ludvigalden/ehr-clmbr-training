"""Request and response schemas of the serving API.

Requests carry a subject inline: its identifier, a birth time (required; femr
derives ages from a birth event) and its events as ``<vocabulary>/<concept>``
codes with naive datetimes (read as UTC). A code without its own entry in the
checkpoint dictionary is unknown; it is never silently dropped, it is counted
and reported on every response.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from ehrfm.inputs.contract import is_vocabulary_code

MAX_EVENTS: Final = 100
"""Upper bound on events per request (one request stays one batch)."""

Code = Annotated[str, Field(min_length=1, max_length=256)]


class SubjectEvent(BaseModel):
    """One event of a subject, with its code and its naive UTC time."""

    model_config = ConfigDict(extra="forbid")

    code: Code
    time: dt.datetime
    numeric_value: Annotated[float, Field(allow_inf_nan=False)] | None = None
    text_value: str | None = None

    @field_validator("code")
    @classmethod
    def _code_format(cls, value: str) -> str:
        if not is_vocabulary_code(value):
            raise ValueError(
                f"code must follow the <vocabulary_id>/<concept_code> format, got {value!r}"
            )
        return value

    @field_validator("time")
    @classmethod
    def _naive_time(cls, value: dt.datetime) -> dt.datetime:
        if value.tzinfo is not None:
            raise ValueError("time must be naive (read as UTC)")
        return value


class SubjectPayload(BaseModel):
    """The inline subject shared by every prediction and embedding request."""

    model_config = ConfigDict(extra="forbid")

    subject_id: int = Field(ge=0)
    events: Annotated[list[SubjectEvent], Field(min_length=1, max_length=MAX_EVENTS)]
    birth_datetime: dt.datetime
    """Required: femr derives ages from a birth event, and a birth that equals
    the earliest event would collapse that day into one token."""

    @field_validator("birth_datetime")
    @classmethod
    def _birth_rules(cls, value: dt.datetime, info: ValidationInfo) -> dt.datetime:
        if value.tzinfo is not None:
            raise ValueError("birth_datetime must be naive (read as UTC)")
        events = info.data.get("events")
        if events and value > min(event.time for event in events):
            raise ValueError("birth_datetime must not be after the earliest event")
        return value


class EmbedRequest(SubjectPayload):
    """A subject to represent."""


class PredictRequest(SubjectPayload):
    """A subject to predict for; ``head`` names a loaded head.

    With ``head`` unset, the single loaded head is used; when more than one
    head is loaded, naming the head is required.
    """

    head: str | None = None


class StreamRequest(SubjectPayload):
    """A subject and the heads to predict with, streamed as SSE events."""

    heads: Annotated[list[str], Field(min_length=1, max_length=MAX_EVENTS)]


class UnknownCodes(BaseModel):
    """How many request codes the dictionary does not know (see the module docstring)."""

    model_config = ConfigDict(frozen=True)

    events: int
    """Events whose code has no dictionary entry."""
    distinct: int
    """Distinct unknown codes."""
    codes: list[str]
    """The distinct unknown codes, capped at ``unknown_code_limit``."""


class Provenance(BaseModel):
    """Weight provenance of a served prediction."""

    model_config = ConfigDict(frozen=True)

    encoder_weights_sha256: str
    head_weights_sha256: str
    head_record_sha256: str


class EmbedProvenance(BaseModel):
    """Weight provenance of a served representation."""

    model_config = ConfigDict(frozen=True)

    encoder_weights_sha256: str


class PredictResponse(BaseModel):
    """One head's output for one subject."""

    model_config = ConfigDict(frozen=True)

    subject_id: int
    head: str
    task: str
    label_type: str
    output: float | dict[str, float]
    """The postprocessed output: the probability or score, or a per-category mapping."""
    postprocess: str
    unknown_codes: UnknownCodes
    provenance: Provenance
    duration_ms: float


class EmbedResponse(BaseModel):
    """The representation of a subject's last token position."""

    model_config = ConfigDict(frozen=True)

    subject_id: int
    embedding: list[float]
    hidden_size: int
    n_positions: int
    unknown_codes: UnknownCodes
    provenance: EmbedProvenance
    duration_ms: float


class StreamStarted(BaseModel):
    """First SSE event: the heads the stream will report."""

    model_config = ConfigDict(frozen=True)

    subject_id: int
    heads: list[str]


class StreamSubject(BaseModel):
    """SSE event emitted after the subject is encoded."""

    model_config = ConfigDict(frozen=True)

    subject_id: int
    n_positions: int
    unknown_codes: UnknownCodes


class StreamComplete(BaseModel):
    """Final SSE event of a successful stream."""

    model_config = ConfigDict(frozen=True)

    subject_id: int
    heads: list[str]
    duration_ms: float


class StreamError(BaseModel):
    """Terminal SSE event of a failed stream."""

    model_config = ConfigDict(frozen=True)

    kind: str
    """``invalid_input``, ``unknown_head``, ``encoding`` or ``internal``."""
    message: str


__all__ = [
    "MAX_EVENTS",
    "Code",
    "EmbedProvenance",
    "EmbedRequest",
    "EmbedResponse",
    "PredictRequest",
    "PredictResponse",
    "Provenance",
    "StreamComplete",
    "StreamError",
    "StreamRequest",
    "StreamStarted",
    "StreamSubject",
    "SubjectEvent",
    "SubjectPayload",
    "UnknownCodes",
]
