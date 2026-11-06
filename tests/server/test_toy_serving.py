"""``--toy`` serving: the fitted toy head is bound, published and served."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from ehrfm.models.representations import ClmbrEncoder
from ehrfm.server.config import ServerConfig
from ehrfm.server.models import SubjectPayload
from ehrfm.server.state import AppState
from tests.server import helpers

TOY_CODES = ["DEMOCOND/C01", "DEMODRUG/D01", "DEMOLAB/L01"]
TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def toy_state(tmp_path_factory: pytest.TempPathFactory) -> AppState:
    """The state of a toy server: the toy checkpoint plus the on-the-fly head."""
    from ehrfm.server.state import load_state

    config = ServerConfig(
        toy=True,
        toy_subjects=20,
        toy_work_dir=tmp_path_factory.mktemp("toy-server"),
        device="cpu",
    )
    return load_state(config)


def test_toy_head_is_fitted_and_bound(toy_state: AppState) -> None:
    names = sorted(toy_state.heads)
    assert names == ["demo_outcome_30d"]
    record = toy_state.heads[names[0]].record
    assert record.spec.label_type == "boolean"
    assert record.encoder_weights_sha256 == toy_state.encoder_weights_sha256


def test_toy_registry_is_reused(tmp_path_factory: pytest.TempPathFactory) -> None:
    """A second prepare_toy over the same work directory skips fitting."""
    from ehrfm.inference.registry import HeadRegistry
    from ehrfm.server.toy import prepare_toy

    work = tmp_path_factory.mktemp("toy-reuse")
    config = ServerConfig(toy=True, toy_subjects=20, toy_work_dir=work, device="cpu")
    first = prepare_toy(config)
    before = {
        path.name: path.read_bytes()
        for path in sorted((first.heads_dir / "demo_outcome_30d").iterdir())
    }
    second = prepare_toy(config)
    assert second.model_dir == first.model_dir
    after = {
        path.name: path.read_bytes()
        for path in sorted((second.heads_dir / "demo_outcome_30d").iterdir())
    }
    assert before == after
    assert HeadRegistry(second.heads_dir).names() == ("demo_outcome_30d",)


def test_toy_serving_uses_the_training_ontology(
    toy_state: AppState,
    encoder: ClmbrEncoder,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Served toy embeddings match a direct encode through the toy extract's ontology.

    The toy head is fitted through the extract ontology (its parent-group
    tokens included); if serving dropped ancestors, the head would see inputs
    unlike its training inputs. The none-ontology encode of the same request
    differs, so this test discriminates.
    """
    from ehrfm.data.batches import BatchBuilder
    from ehrfm.data.tokenization import CodeOntology, extract_tokenizer
    from ehrfm.inputs.artifacts import open_extract
    from ehrfm.models.checkpoint import load_tokenizer
    from ehrfm.server.app import create_app
    from ehrfm.server.subject import build_subject

    work = toy_state.config.toy_work_dir
    assert work is not None
    checkpoint = _checkpoint_dir(toy_state)
    extract = work / "inputs" / "artifacts" / "meds" / "synthetic_demo"
    request: dict[str, Any] = {
        "subject_id": 21,
        "events": helpers.events(TOY_CODES),
        "birth_datetime": helpers.BIRTH,
    }
    payload = SubjectPayload.model_validate(request)
    subject = build_subject(payload)
    with_extract = encoder.encode(
        BatchBuilder(extract_tokenizer(checkpoint, open_extract(extract))).build([subject])
    )
    without_ancestors = encoder.encode(
        BatchBuilder(load_tokenizer(checkpoint, CodeOntology({}))).build([subject])
    )
    assert not torch.allclose(with_extract.hidden[-1], without_ancestors.hidden[-1])

    served_app = create_app(
        ServerConfig(toy=True, toy_subjects=20, toy_work_dir=work, device="cpu")
    )
    with TestClient(served_app) as client:
        served = client.post("/embed", json=request)
        assert served.status_code == 200, served.text
        assert served.json()["embedding"] == pytest.approx(
            [float(value) for value in with_extract.hidden[-1]], abs=TOLERANCE
        )


def test_toy_server_serves(toy_state: AppState) -> None:
    """A toy app answers health, heads and predict over HTTP."""
    from ehrfm.server.app import create_app

    config = ServerConfig(
        toy=True,
        toy_subjects=20,
        toy_work_dir=toy_state.config.toy_work_dir,
        device="cpu",
    )
    request: dict[str, Any] = {
        "subject_id": 11,
        "events": helpers.events(TOY_CODES),
        "birth_datetime": helpers.BIRTH,
    }
    with TestClient(create_app(config)) as client:
        health = client.get("/health").json()
        assert health["loaded"] is True
        heads = client.get("/heads").json()["heads"]
        assert [head["name"] for head in heads] == ["demo_outcome_30d"]
        response = client.post("/predict", json=request)
        assert response.status_code == 200
        body = response.json()
        assert body["head"] == "demo_outcome_30d"
        assert body["task"] == "demo_outcome_30d"
        assert 0.0 <= body["output"] <= 1.0
        assert len(body["provenance"]["head_record_sha256"]) == 64


def _checkpoint_dir(state: AppState) -> Path:
    """The checkpoint directory of the state's encoder (from its weights path)."""
    return state.encoder.weights.path.parent
