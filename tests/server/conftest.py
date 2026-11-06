"""Fixtures of the server tests: a registry of two crafted heads and a client.

The heads are deterministic ``LinearHead``s with distinct, hand-seeded
weights, both bound to the toy checkpoint's weights, so head selection by
name has two genuinely different outputs to tell apart — the regression test
for the multi-head assignment bug the upstream server had.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from fastapi.testclient import TestClient

from ehrfm.inference.head_loader import LoadedHead
from ehrfm.inference.registry import EncoderIdentity, HeadRegistry
from ehrfm.models.head import HeadSpec
from ehrfm.models.representations import ClmbrEncoder
from ehrfm.models.weights import weights_sha256
from ehrfm.server.config import ServerConfig
from ehrfm.toy.checkpoint import ToyCheckpoint
from tests.server.helpers import make_head

REGISTRY_HEADS: Final = ("alpha", "beta")


@pytest.fixture(scope="session")
def toy_registry(tmp_path_factory: pytest.TempPathFactory, toy_checkpoint: ToyCheckpoint) -> Path:
    """A head registry with two distinct heads bound to the toy checkpoint."""
    registry_dir = tmp_path_factory.mktemp("registry")
    encoder_sha = weights_sha256(toy_checkpoint.path / "model.safetensors")
    spec = HeadSpec(kind="linear", task_name="demo_task", label_type="boolean", input_dim=32)
    for name, seed in zip(REGISTRY_HEADS, (1, 2), strict=True):
        stage = tmp_path_factory.mktemp(f"stage-{name}")
        make_head(spec, seed, stage, encoder_sha)
        HeadRegistry(registry_dir).publish(stage / "head", name)
    return registry_dir


@pytest.fixture(scope="session")
def toy_sha(toy_checkpoint: ToyCheckpoint) -> str:
    """The toy checkpoint's weight digest (the heads' encoder binding)."""
    return weights_sha256(toy_checkpoint.path / "model.safetensors")


@pytest.fixture(scope="session")
def toy_loaded(
    toy_registry: Path, toy_checkpoint: ToyCheckpoint, toy_sha: str
) -> dict[str, LoadedHead]:
    """The two registry heads, loaded the way the server loads them."""
    identity = EncoderIdentity(hidden_size=32, weights_sha256=toy_sha)
    return dict(HeadRegistry(toy_registry).load_all(identity))


@pytest.fixture(scope="session")
def encoder(toy_checkpoint: ToyCheckpoint) -> ClmbrEncoder:
    """A direct encoder for computing expected outputs without the server."""
    return ClmbrEncoder.load(toy_checkpoint.path, device="cpu")


@pytest.fixture(scope="session")
def client(toy_checkpoint: ToyCheckpoint, toy_registry: Path) -> Iterator[TestClient]:
    """The serving app over the toy checkpoint and the two crafted heads."""
    config = ServerConfig(
        model_dir=toy_checkpoint.path, heads_dir=toy_registry, device="cpu", ontology="none"
    )
    from ehrfm.server.app import create_app

    with TestClient(create_app(config)) as test_client:
        yield test_client
