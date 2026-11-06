"""Toy CLMBR checkpoint: layout, surgery load, forward pass over the synthetic extract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import femr.ontology
import meds_reader
import msgpack
import pytest
import torch
from femr.models.processor import FEMRBatchProcessor
from femr.models.tasks import LabeledSubjectTask
from femr.models.transformer import CLMBRTaskHead

from ehrfm.inputs import ensure_database, open_extract
from ehrfm.models import _compat
from ehrfm.models.checkpoint import (
    CheckpointError,
    load_femr_model,
    load_tokenizer,
    nested_age_stats,
)
from ehrfm.toy.artifacts import SyntheticArtifacts
from ehrfm.toy.checkpoint import (
    CLMBR_VOCAB_SIZE,
    HIDDEN_SIZE,
    N_LAYERS,
    ToyCheckpoint,
)


def subject_representations(
    model_dir: Path,
    database: meds_reader.SubjectDatabase,
    ontology: femr.ontology.Ontology,
) -> dict[int, torch.Tensor]:
    """Encode every subject and return its final-token representation."""
    model, _ = load_femr_model(model_dir, task_config=LabeledSubjectTask([]).get_task_config())
    processor = FEMRBatchProcessor(load_tokenizer(model_dir, ontology))
    result: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for subject_id in sorted(database):
            batch: Any = processor.convert_subject(database[subject_id], tensor_type="pt")
            _, output = model(**processor.collate([batch]))
            result[subject_id] = output["representations"][-1].clone()
    return result


def test_layout_mirrors_release(toy_checkpoint: ToyCheckpoint) -> None:
    files = sorted(p.name for p in toy_checkpoint.path.iterdir())
    assert files == ["config.json", "dictionary.msgpack", "model.safetensors"]
    config = json.loads((toy_checkpoint.path / "config.json").read_text())
    assert config["architectures"] == ["FEMRModel"]
    assert config["task_config"]["task_type"] == "clmbr"
    assert config["task_config"]["task_kwargs"] == {"clmbr_vocab_size": CLMBR_VOCAB_SIZE}
    assert "is_hierarchical" not in config["transformer_config"]
    assert config["transformer_config"]["vocab_size"] == toy_checkpoint.vocab_size


def test_dictionary_follows_clmbr_v2_schema(toy_checkpoint: ToyCheckpoint) -> None:
    with (toy_checkpoint.path / "dictionary.msgpack").open("rb") as handle:
        dictionary = cast(dict[str, Any], msgpack.unpack(handle))
    assert sorted(dictionary) == ["age_stats", "is_hierarchical", "vocab"]
    assert dictionary["is_hierarchical"] is True
    assert sorted(dictionary["age_stats"]) == ["mean", "std"]
    types = {entry["type"] for entry in dictionary["vocab"]}
    assert types == {"code", "numeric", "text", "unused"}
    base = {"code_string", "text_string", "type", "val_end", "val_start", "weight"}
    for entry in dictionary["vocab"]:
        expected = base | {"property"} if entry["type"] in ("numeric", "text") else base
        assert set(entry) == expected
    codes = [e["code_string"] for e in dictionary["vocab"] if e["type"] == "code"]
    assert codes == list(toy_checkpoint.dictionary_codes)


def test_dictionary_covers_part_of_the_extract(
    artifacts: SyntheticArtifacts, toy_checkpoint: ToyCheckpoint
) -> None:
    known = set(toy_checkpoint.dictionary_codes)
    vocabulary_codes = {c for c in artifacts.codes if "/" in c}
    recognised = vocabulary_codes & known
    assert 0 < len(recognised) < len(vocabulary_codes)


def test_surgery_on_load(toy_checkpoint: ToyCheckpoint) -> None:
    model, report = load_femr_model(toy_checkpoint.path)
    assert report.hierarchical_patched
    assert report.embedding_remapped
    assert report.hawk_layers_replaced == N_LAYERS // 2
    assert all(not layer.use_hawk for layer in model.transformer.layers)
    assert model.config.transformer_config.is_hierarchical
    assert not model.training


def test_incomplete_checkpoint_fails(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(CheckpointError, match="missing"):
        load_femr_model(tmp_path)


def test_age_stats_are_expanded(
    toy_checkpoint: ToyCheckpoint, ontology: femr.ontology.Ontology
) -> None:
    tokenizer = load_tokenizer(toy_checkpoint.path, ontology)
    assert sorted(tokenizer.dictionary["age_stats"]) == ["age", "delta", "log_age", "log_delta"]
    nested = tokenizer.dictionary["age_stats"]
    assert nested_age_stats(nested) is nested


def test_forward_produces_per_subject_representations(
    artifacts: SyntheticArtifacts,
    toy_checkpoint: ToyCheckpoint,
    ontology: femr.ontology.Ontology,
    database_cache: Path,
) -> None:
    extract = open_extract(artifacts.extract)
    with meds_reader.SubjectDatabase(str(ensure_database(extract, database_cache))) as database:
        reps = subject_representations(toy_checkpoint.path, database, ontology)
    assert sorted(reps) == list(range(1, 21))
    stacked = torch.stack(list(reps.values()))
    assert stacked.shape == (20, HIDDEN_SIZE)
    assert torch.isfinite(stacked).all()
    # Different histories give different representations.
    assert torch.unique(stacked, dim=0).shape[0] > 1


def test_clmbr_head_kept_when_no_task_given(toy_checkpoint: ToyCheckpoint) -> None:
    model, _ = load_femr_model(toy_checkpoint.path)
    head = model.task_model
    assert isinstance(head, CLMBRTaskHead)
    assert head.final_layer.out_features == CLMBR_VOCAB_SIZE
    assert head.final_layer.in_features == HIDDEN_SIZE


def test_shim_never_called_on_loaded_model(toy_checkpoint: ToyCheckpoint) -> None:
    model, _ = load_femr_model(toy_checkpoint.path)
    if _compat.is_active():
        hawk_blocks = [m for m in model.modules() if isinstance(m, _compat.RecurrentBlock)]
        assert hawk_blocks == []
