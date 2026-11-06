"""Parity between the served request path and the database encoding path.

These tests guard against the reference path silently calling the server's
own construction code: the expected embedding is produced by the database
route (extract subject -> meds_reader database -> ``subject_representations``)
and the served one by the request route (JSON events -> ``MutableSubject``).
"""

from __future__ import annotations

import datetime as dt
from typing import cast

import meds_reader
import numpy as np
import pytest
from fastapi.testclient import TestClient
from meds_reader.transform import MutableEvent, MutableSubject

from ehrfm.data.batches import BatchBuilder
from ehrfm.data.tokenization import CodeOntology
from ehrfm.models.checkpoint import load_tokenizer
from ehrfm.models.representations import ClmbrEncoder
from ehrfm.toy.checkpoint import ToyCheckpoint
from tests.server import helpers

TOLERANCE = 1e-5


def _subject_request(
    database: meds_reader.SubjectDatabase, subject_id: int
) -> tuple[dict[str, object], MutableSubject]:
    """A /embed request and a hand-built reference subject from one database subject.

    The extract's static rows carry no time; a request cannot express them, so
    both the request and the reference keep only the timed events.
    """
    subject = database[subject_id]
    birth = None
    events: list[dict[str, object]] = []
    timed: list[MutableEvent] = []
    for event in subject.events:
        # The stub types time as non-None, but the extract's static rows carry None.
        time = cast("dt.datetime | None", event.time)
        if time is None:
            continue
        if event.code == "MEDS_BIRTH":
            birth = time
            timed.append(MutableEvent(time=time, code="MEDS_BIRTH", properties={}))
            continue
        entry: dict[str, object] = {"code": event.code, "time": time.isoformat()}
        properties: dict[str, object] = {}
        numeric = getattr(event, "numeric_value", None)
        text = getattr(event, "text_value", None)
        if numeric is not None:
            properties["numeric_value"] = float(np.float32(numeric))
            entry["numeric_value"] = float(numeric)
        if text is not None:
            properties["text_value"] = str(text)
            entry["text_value"] = str(text)
        events.append(entry)
        timed.append(MutableEvent(time=event.time, code=event.code, properties=properties))
    assert birth is not None, f"subject {subject_id} has no MEDS_BIRTH event"
    assert events
    request: dict[str, object] = {
        "subject_id": subject_id,
        "events": events,
        "birth_datetime": birth.isoformat(),
    }
    reference = MutableSubject(subject_id=subject_id, events=sorted(timed, key=lambda e: e.time))
    return request, reference


def test_embed_parity_with_database_path(
    client: TestClient,
    toy_checkpoint: ToyCheckpoint,
    encoder: ClmbrEncoder,
    database: meds_reader.SubjectDatabase,
) -> None:
    """The request path reproduces the database path subject for subject."""
    tokenizer = load_tokenizer(toy_checkpoint.path, CodeOntology({}))
    builder = BatchBuilder(tokenizer)
    for subject_id in (1, 7, 11):
        request, reference = _subject_request(database, subject_id)
        response = client.post("/embed", json=request)
        assert response.status_code == 200, response.text
        served = response.json()["embedding"]
        tokens = encoder.encode(builder.build([reference]))
        expected = tokens.hidden[tokens.hidden.shape[0] - 1]
        assert served == pytest.approx([float(value) for value in expected], abs=TOLERANCE), (
            f"subject {subject_id} diverged"
        )


def test_events_at_the_same_time_keep_canonical_order(client: TestClient) -> None:
    """Reordering same-timestamp events in the request does not change the output."""
    when = dt.datetime(2021, 3, 4, 8, 30)
    events = [
        {"code": "DEMOCOND/C01", "time": when.isoformat()},
        {"code": "DEMODRUG/D01", "time": when.isoformat()},
        {"code": "DEMOLAB/L01", "time": when.isoformat()},
    ]
    first = client.post(
        "/embed",
        json={"subject_id": 20, "events": events, "birth_datetime": helpers.BIRTH},
    ).json()
    reversed_events = list(reversed(events))
    second = client.post(
        "/embed",
        json={"subject_id": 20, "events": reversed_events, "birth_datetime": helpers.BIRTH},
    ).json()
    assert first["embedding"] == pytest.approx(second["embedding"], abs=TOLERANCE)
