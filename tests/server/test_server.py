"""Endpoint behaviour over HTTP: correctness, selection, streaming, provenance."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from ehrfm.inference.head_loader import LoadedHead
from ehrfm.models.representations import ClmbrEncoder
from ehrfm.toy.checkpoint import ToyCheckpoint
from tests.server import helpers

TOLERANCE = 1e-5

CODES = ["DEMOCOND/C01", "DEMODRUG/D01", "DEMOLAB/L01"]
UNKNOWN = "NOTREAL/ZZ"
REQUEST: dict[str, Any] = {
    "subject_id": 1,
    "events": helpers.events(CODES),
    "birth_datetime": helpers.BIRTH,
    "head": "alpha",
}
BARE_REQUEST: dict[str, Any] = {key: value for key, value in REQUEST.items() if key != "head"}


@pytest.fixture(scope="module")
def expected(toy_checkpoint: ToyCheckpoint, encoder: ClmbrEncoder) -> torch.Tensor:
    """The embedding every endpoint output is checked against."""
    return helpers.direct_embedding(encoder, toy_checkpoint, BARE_REQUEST)


def test_health_and_ready(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["loaded"] is True
    assert body["heads"] == ["alpha", "beta"]
    assert "raw" in body["postprocess"]
    ready = client.get("/ready").json()
    assert ready["ready"] is True
    assert ready["heads"] == ["alpha", "beta"]


def test_heads_lists_loaded_heads(client: TestClient) -> None:
    body = client.get("/heads").json()
    assert len(body["encoder_weights_sha256"]) == 64
    listed = [head["name"] for head in body["heads"]]
    assert listed == ["alpha", "beta"]
    for head in body["heads"]:
        assert head["spec"]["kind"] == "linear"
        assert head["spec"]["input_dim"] == 32
        assert len(head["record_sha256"]) == 64


def test_predict_matches_direct_head(
    client: TestClient,
    toy_loaded: dict[str, LoadedHead],
    expected: torch.Tensor,
) -> None:
    response = client.post("/predict", json=REQUEST)
    assert response.status_code == 200
    body = response.json()
    direct = float(toy_loaded["alpha"].head.predict(expected.unsqueeze(0))[0])
    assert body["output"] == pytest.approx(direct, abs=TOLERANCE)
    assert body["head"] == "alpha"
    assert body["task"] == "demo_task"
    assert body["label_type"] == "boolean"
    assert body["postprocess"] == "raw"
    assert body["unknown_codes"]["events"] == 0


def test_head_selection_gives_distinct_outputs(
    client: TestClient, toy_loaded: dict[str, LoadedHead], expected: torch.Tensor
) -> None:
    """Regression test for the upstream multi-head assignment bug."""
    alpha = client.post("/predict", json={**REQUEST, "head": "alpha"}).json()
    beta = client.post("/predict", json={**REQUEST, "head": "beta"}).json()
    assert alpha["output"] != beta["output"]
    direct_alpha = float(toy_loaded["alpha"].head.predict(expected.unsqueeze(0))[0])
    direct_beta = float(toy_loaded["beta"].head.predict(expected.unsqueeze(0))[0])
    assert alpha["output"] == pytest.approx(direct_alpha, abs=TOLERANCE)
    assert beta["output"] == pytest.approx(direct_beta, abs=TOLERANCE)


def test_unknown_head_is_not_found(client: TestClient) -> None:
    response = client.post("/predict", json={**REQUEST, "head": "missing"})
    assert response.status_code == 404
    assert "missing" in response.json()["detail"]


def test_omitted_head_with_two_loaded_is_rejected(client: TestClient) -> None:
    """Without a name and two loaded heads, the request is ambiguous."""
    response = client.post("/predict", json=BARE_REQUEST)
    assert response.status_code == 404
    assert "2 loaded heads" in response.json()["detail"]


def test_embed_matches_direct_encode(client: TestClient, expected: torch.Tensor) -> None:
    response = client.post("/embed", json=BARE_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["hidden_size"] == 32
    assert body["n_positions"] > 0
    assert body["embedding"] == pytest.approx([float(value) for value in expected], abs=TOLERANCE)
    assert len(body["provenance"]["encoder_weights_sha256"]) == 64


def test_unknown_codes_are_reported(client: TestClient) -> None:
    request = {
        "subject_id": 2,
        "events": helpers.events(CODES + [UNKNOWN]),
        "birth_datetime": helpers.BIRTH,
        "head": "alpha",
    }
    body = client.post("/predict", json=request).json()
    assert body["unknown_codes"]["events"] == 1
    assert body["unknown_codes"]["distinct"] == 1
    assert body["unknown_codes"]["codes"] == [UNKNOWN]


def test_stream_frames_and_content(
    client: TestClient, toy_loaded: dict[str, LoadedHead], expected: torch.Tensor
) -> None:
    request = {
        "subject_id": 3,
        "events": helpers.events(CODES),
        "birth_datetime": helpers.BIRTH,
        "heads": ["alpha", "beta"],
    }
    response = client.post("/predict/stream", json=request)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = helpers.parse_sse(response.text)
    names = [name for name, _, _ in frames]
    assert names == ["started", "subject", "prediction", "prediction", "complete"]
    identifiers = [identifier for _, identifier, _ in frames]
    assert identifiers == ["0", "1", "2", "3", "4"]
    started = frames[0][2]
    assert started["heads"] == ["alpha", "beta"]
    subject_event = frames[1][2]
    positions = subject_event["n_positions"]
    assert isinstance(positions, int) and positions > 0
    unknown = subject_event["unknown_codes"]
    assert isinstance(unknown, dict) and isinstance(unknown["events"], (int, float))
    assert unknown["events"] == 0
    direct_alpha = float(toy_loaded["alpha"].head.predict(expected.unsqueeze(0))[0])
    direct_beta = float(toy_loaded["beta"].head.predict(expected.unsqueeze(0))[0])
    assert frames[2][2]["head"] == "alpha"
    assert frames[2][2]["output"] == pytest.approx(direct_alpha, abs=TOLERANCE)
    assert frames[3][2]["head"] == "beta"
    assert frames[3][2]["output"] == pytest.approx(direct_beta, abs=TOLERANCE)
    assert frames[4][2]["heads"] == ["alpha", "beta"]


def test_stream_unknown_head_is_not_found(client: TestClient) -> None:
    response = client.post("/predict/stream", json={**BARE_REQUEST, "heads": ["alpha", "missing"]})
    assert response.status_code == 404


def test_provenance_headers_and_request_id(client: TestClient) -> None:
    response = client.post("/predict", json={**REQUEST, "head": "beta"})
    assert response.status_code == 200
    assert len(response.headers["x-ehrfm-encoder-weights-sha256"]) == 64
    assert len(response.headers["x-ehrfm-head-weights-sha256"]) == 64
    assert len(response.headers["x-ehrfm-head-record-sha256"]) == 64
    echoed = client.get("/health", headers={"X-Request-Id": "test-123"})
    assert echoed.headers["x-request-id"] == "test-123"
    generated = client.get("/health")
    assert len(generated.headers["x-request-id"]) >= 8
