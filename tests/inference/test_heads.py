"""Task heads: typed spec, predictions per label type, save/load with integrity checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import meds_reader
import pytest
import torch
from pydantic import ValidationError
from torch import Tensor, nn

from ehrfm.data.batches import BatchBuilder
from ehrfm.data.tokenization import extract_tokenizer
from ehrfm.inference.head_loader import (
    HEAD_JSON,
    HEAD_WEIGHTS,
    HeadLoadError,
    load_head,
    save_head,
)
from ehrfm.inputs import MedsExtract
from ehrfm.models.head import (
    HeadSpec,
    LinearHead,
    TaskHead,
    build_head,
    head_kinds,
    register_head_kind,
)
from ehrfm.models.representations import ClmbrEncoder
from ehrfm.toy.checkpoint import HIDDEN_SIZE, ToyCheckpoint

SHA_A = "a" * 64
SHA_B = "b" * 64


def _spec(**changes: object) -> HeadSpec:
    base: dict[str, object] = {
        "kind": "linear",
        "task_name": "demo_outcome_30d",
        "label_type": "boolean",
        "input_dim": HIDDEN_SIZE,
    }
    base.update(changes)
    return HeadSpec.model_validate(base)


def _seeded(spec: HeadSpec, seed: int = 0) -> TaskHead:
    torch.manual_seed(seed)
    return build_head(spec)


def test_spec_output_dims() -> None:
    assert _spec().output_dim == 1
    assert _spec(label_type="float").output_dim == 1
    assert _spec(label_type="categorical", categories=("a", "b", "c")).output_dim == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"label_type": "categorical"},
        {"label_type": "categorical", "categories": ("a",)},
        {"label_type": "categorical", "categories": ("a", "a")},
        {"categories": ("a", "b")},
        {"input_dim": 0},
        {"task_name": "Bad Name"},
        {"unknown": 1},
    ],
)
def test_invalid_specs_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _spec(**changes)


def test_predictions_per_label_type() -> None:
    x = torch.randn(5, HIDDEN_SIZE)
    boolean = _seeded(_spec()).predict(x)
    assert boolean.shape == (5,) and ((boolean > 0) & (boolean < 1)).all()
    cat = _seeded(_spec(label_type="categorical", categories=("a", "b", "c"))).predict(x)
    assert cat.shape == (5, 3) and torch.allclose(cat.sum(dim=1), torch.ones(5))
    reg = _seeded(_spec(label_type="float")).predict(x)
    assert reg.shape == (5,)


def test_predict_checks_width() -> None:
    with pytest.raises(ValueError, match="shape"):
        _seeded(_spec()).predict(torch.randn(2, HIDDEN_SIZE + 1))


def test_spec_kind_must_match_class() -> None:
    with pytest.raises(ValueError, match="kind"):
        LinearHead(_spec(kind="other"))


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown head kind"):
        build_head(_spec(kind="nonexistent"))


def test_new_kinds_extend_the_registry(tmp_path: Path) -> None:
    """A head kind added later is saved and loaded through the same functions."""

    class TwoLayerHead(TaskHead):
        kind: ClassVar[str] = "two_layer_test"

        def __init__(self, spec: HeadSpec) -> None:
            super().__init__(spec)
            self.body = nn.Sequential(
                nn.Linear(spec.input_dim, 4), nn.ReLU(), nn.Linear(4, spec.output_dim)
            )

        def forward(self, representations: Tensor) -> Tensor:
            return self.body(representations)

    register_head_kind(TwoLayerHead)
    assert "two_layer_test" in head_kinds()
    head = _seeded(_spec(kind="two_layer_test"))
    save_head(head, tmp_path / "h")
    loaded = load_head(tmp_path / "h")
    assert isinstance(loaded.head, TwoLayerHead)
    x = torch.randn(3, HIDDEN_SIZE)
    assert torch.equal(loaded.head.predict(x), head.predict(x))


def test_registry_rejects_a_conflicting_kind() -> None:
    class Clash(LinearHead):
        kind: ClassVar[str] = "linear"

    with pytest.raises(ValueError, match="already registered"):
        register_head_kind(Clash)


# -- serialization -------------------------------------------------------------------


def test_round_trip(tmp_path: Path) -> None:
    head = _seeded(_spec())
    record = save_head(
        head, tmp_path / "h", encoder_weights_sha256=SHA_A, metadata={"l2": 0.1, "note": "x"}
    )
    assert sorted(p.name for p in (tmp_path / "h").iterdir()) == [HEAD_JSON, HEAD_WEIGHTS]
    loaded = load_head(
        tmp_path / "h", encoder_hidden_size=HIDDEN_SIZE, encoder_weights_sha256=SHA_A
    )
    assert loaded.record == record
    assert not loaded.head.training
    x = torch.randn(4, HIDDEN_SIZE)
    assert torch.equal(loaded.head.predict(x), head.predict(x))


def test_saving_is_deterministic(tmp_path: Path) -> None:
    head = _seeded(_spec())
    save_head(head, tmp_path / "a", metadata={"l2": 0.1})
    save_head(head, tmp_path / "b", metadata={"l2": 0.1})
    for name in (HEAD_JSON, HEAD_WEIGHTS):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_tampered_head_fails_closed(tmp_path: Path) -> None:
    save_head(_seeded(_spec()), tmp_path / "h")
    weights = tmp_path / "h" / HEAD_WEIGHTS
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0x01
    weights.write_bytes(bytes(data))
    with pytest.raises(HeadLoadError, match="SHA-256"):
        load_head(tmp_path / "h")


def test_width_mismatch_fails(tmp_path: Path) -> None:
    save_head(_seeded(_spec()), tmp_path / "h")
    with pytest.raises(HeadLoadError, match="wide"):
        load_head(tmp_path / "h", encoder_hidden_size=HIDDEN_SIZE * 2)


def test_encoder_mismatch_fails(tmp_path: Path) -> None:
    save_head(_seeded(_spec()), tmp_path / "h", encoder_weights_sha256=SHA_A)
    with pytest.raises(HeadLoadError, match="fitted on"):
        load_head(tmp_path / "h", encoder_weights_sha256=SHA_B)


def test_missing_or_malformed_record_fails(tmp_path: Path) -> None:
    with pytest.raises(HeadLoadError, match=HEAD_JSON):
        load_head(tmp_path)
    save_head(_seeded(_spec()), tmp_path / "h")
    record = json.loads((tmp_path / "h" / HEAD_JSON).read_text())
    record["format_version"] = 99
    (tmp_path / "h" / HEAD_JSON).write_text(json.dumps(record))
    with pytest.raises(HeadLoadError, match="format version"):
        load_head(tmp_path / "h")
    record["format_version"] = 1
    record["surprise"] = True
    (tmp_path / "h" / HEAD_JSON).write_text(json.dumps(record))
    with pytest.raises(HeadLoadError, match="not a valid head record"):
        load_head(tmp_path / "h")


def test_existing_directory_is_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / "h").mkdir()
    with pytest.raises(FileExistsError):
        save_head(_seeded(_spec()), tmp_path / "h")


def test_head_on_toy_encoder(
    toy_checkpoint: ToyCheckpoint,
    extract: MedsExtract,
    database: meds_reader.SubjectDatabase,
    tmp_path: Path,
) -> None:
    """Representations from the toy checkpoint feed a saved and reloaded head."""
    encoder = ClmbrEncoder.load(toy_checkpoint.path, device="cpu")
    builder = BatchBuilder(extract_tokenizer(toy_checkpoint.path, extract))
    reps = encoder.subject_representations(database, list(range(1, 21)), builder)
    head = _seeded(_spec(input_dim=encoder.hidden_size))
    save_head(head, tmp_path / "h", encoder_weights_sha256=encoder.weights.sha256)
    loaded = load_head(
        tmp_path / "h",
        encoder_hidden_size=encoder.hidden_size,
        encoder_weights_sha256=encoder.weights.sha256,
    )
    probabilities = loaded.head.predict(reps.embeddings)
    assert probabilities.shape == (20,)
    assert torch.isfinite(probabilities).all()
