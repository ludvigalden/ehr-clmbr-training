"""Building a femr subject from a request, and counting unknown request codes.

The subject is a ``meds_reader.transform.MutableSubject``: femr's batch
builder consumes it directly, without a database. Events are sorted
chronologically, and ties on a timestamp keep the extract's canonical order
(time, then code, then values). The birth event is required and must not be
after the earliest event; both rules keep what the model sees aligned with
how training subjects were built.

An event's code is *recognised* when the checkpoint dictionary has a ``code``
entry for it — the same definition the :mod:`ehrfm.data.coverage` report
uses. Unrecognised codes are counted per event and per distinct code and
reported on every response; they are never silently dropped. The response's
code list is capped (the counts always cover every event); numeric values are
narrowed to float32, the database's storage type, so a value tokenizes into
the same bin as it would from an extract.
"""

from __future__ import annotations

import numpy as np
from meds_reader.transform import MutableEvent, MutableSubject

from ehrfm.server.models import SubjectEvent, SubjectPayload, UnknownCodes


def build_subject(payload: SubjectPayload) -> MutableSubject:
    """Convert a request payload into a ``MutableSubject`` ready for the batch builder."""
    events: list[MutableEvent] = [
        MutableEvent(time=payload.birth_datetime, code="MEDS_BIRTH", properties={})
    ]
    events.extend(
        MutableEvent(
            time=event.time,
            code=event.code,
            properties=_properties(event),
        )
        for event in payload.events
    )
    events.sort(key=_order)
    return MutableSubject(subject_id=payload.subject_id, events=events)


def _properties(event: SubjectEvent) -> dict[str, object]:
    """The event's value properties, numeric values narrowed to float32."""
    properties: dict[str, object] = {}
    if event.numeric_value is not None:
        # The database stores float32; narrowing here keeps a value on the same
        # side of a tokenizer bin boundary as it would be from an extract.
        properties["numeric_value"] = float(np.float32(event.numeric_value))
    if event.text_value is not None:
        properties["text_value"] = event.text_value
    return properties


def _order(event: MutableEvent) -> tuple[object, ...]:
    """The extract's canonical within-time order: time, code, then values."""
    numeric = getattr(event, "numeric_value", None)
    text = getattr(event, "text_value", None)
    return (event.time, event.code, 0.0 if numeric is None else numeric, text or "")


def count_unknown_codes(
    payload: SubjectPayload, known: frozenset[str], *, limit: int
) -> UnknownCodes:
    """The unknown codes of ``payload``; the counts are complete, the list is capped."""
    unknown_events = 0
    distinct: set[str] = set()
    for event in payload.events:
        if event.code not in known:
            unknown_events += 1
            distinct.add(event.code)
    return UnknownCodes(
        events=unknown_events,
        distinct=len(distinct),
        codes=sorted(distinct)[:limit],
    )


__all__ = ["build_subject", "count_unknown_codes"]
