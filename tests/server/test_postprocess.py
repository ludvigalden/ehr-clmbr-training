"""Postprocess hooks: the neutral default, a custom hook, fail-closed names."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ehrfm.inference.head_loader import HeadRecord, LoadedHead
from ehrfm.models.head import HeadSpec
from ehrfm.models.representations import ClmbrEncoder
from ehrfm.server.config import ServerConfig
from ehrfm.server.postprocess import (
    apply_postprocess,
    postprocess_names,
    register_postprocess,
    registered_postprocess,
)
from ehrfm.toy.checkpoint import ToyCheckpoint
from tests.server import helpers

CODES = ["DEMOCOND/C01", "DEMODRUG/D01", "DEMOLAB/L01"]

_RECORD = HeadRecord(
    format_version=1,
    spec=HeadSpec(kind="linear", task_name="demo_task", label_type="boolean", input_dim=32),
    weights_sha256="0" * 64,
    encoder_weights_sha256=None,
    metadata={},
)


def test_default_raw_is_identity() -> None:
    """The neutral default returns the head's raw probabilities unchanged."""
    raw = registered_postprocess("raw")
    assert raw([0.25, 0.5], _RECORD) == [0.25, 0.5]
    assert apply_postprocess("raw", [0.75], _RECORD) == [0.75]
    assert "raw" in postprocess_names()


def test_unknown_hook_raises() -> None:
    with pytest.raises(ValueError, match="unknown postprocess"):
        registered_postprocess("no-such-hook")


def test_registered_hook_is_applied() -> None:
    def complement(values, record):  # type: ignore[no-untyped-def]
        return [1.0 - value for value in values]

    register_postprocess("complement", complement)
    assert apply_postprocess("complement", [0.1], _RECORD) == pytest.approx([0.9])
    with pytest.raises(ValueError, match="already registered"):
        register_postprocess("complement", complement)


def test_hook_output_shape_is_validated() -> None:
    def two(values, record):  # type: ignore[no-untyped-def]
        return [float(value) for value in values]

    register_postprocess("two", two)
    with pytest.raises(ValueError, match="1 outputs"):
        apply_postprocess("two", [0.1, 0.4], _RECORD)
    with pytest.raises(ValueError, match="non-finite"):
        apply_postprocess("raw", [float("inf")], _RECORD)


def test_custom_hook_is_served(
    toy_checkpoint: ToyCheckpoint,
    toy_registry: Path,
    toy_loaded: dict[str, LoadedHead],
    encoder: ClmbrEncoder,
) -> None:
    """A configured postprocess hook transforms the served output."""
    from ehrfm.server.app import create_app

    def complement(values, record):  # type: ignore[no-untyped-def]
        return [1.0 - value for value in values]

    register_postprocess("served-complement", complement)
    config = ServerConfig(
        model_dir=toy_checkpoint.path,
        heads_dir=toy_registry,
        device="cpu",
        postprocess="served-complement",
        ontology="none",
    )
    request: dict[str, object] = {
        "subject_id": 9,
        "events": helpers.events(CODES),
        "birth_datetime": helpers.BIRTH,
        "head": "alpha",
    }
    with TestClient(create_app(config)) as client:
        served = client.post("/predict", json=request)
        assert served.status_code == 200
        body = served.json()
    payload = {key: value for key, value in request.items() if key != "head"}
    expected = helpers.direct_embedding(encoder, toy_checkpoint, payload)
    direct = float(toy_loaded["alpha"].head.predict(expected.unsqueeze(0))[0])
    assert body["postprocess"] == "served-complement"
    assert isinstance(body["output"], float)
    assert body["output"] == pytest.approx(1.0 - direct, abs=1e-5)
