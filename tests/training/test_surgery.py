"""Every continued-pretraining surgery invariant, with the failure each one catches."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import safetensors.torch
from femr.models.transformer import FEMRModel
from torch import nn

from ehrfm.data.coverage import dictionary_codes
from ehrfm.models.checkpoint import (
    DICTIONARY_FILE,
    WEIGHTS_FILE,
    CheckpointError,
    load_checkpoint,
    read_config,
    uniform_attention_model,
)
from ehrfm.models.weights import WeightIntegrityError
from ehrfm.toy.artifacts import write_concept_set
from ehrfm.toy.checkpoint import CLMBR_VOCAB_SIZE, ToyCheckpoint
from ehrfm.training.surgery import (
    SurgeryError,
    check_clmbr_head,
    check_embedding,
    check_hierarchical,
    check_surgery,
    check_uniform_attention,
    prepare_base_model,
    read_dictionary,
)


def _entries(checkpoint: ToyCheckpoint) -> int:
    return len(read_dictionary(checkpoint.path)["vocab"])


def test_prepared_base_model_passes_every_check(toy_checkpoint: ToyCheckpoint) -> None:
    prepared = prepare_base_model(toy_checkpoint.path, clmbr_vocab_size=CLMBR_VOCAB_SIZE)
    assert prepared.surgery.hierarchical_patched
    assert prepared.surgery.hawk_layers_replaced == 1
    assert prepared.surgery.embedding_remapped
    assert prepared.clmbr_vocab_size == CLMBR_VOCAB_SIZE
    assert prepared.model.training
    assert check_surgery(prepared.model, _entries(toy_checkpoint)) == CLMBR_VOCAB_SIZE


# --- hierarchical flag -----------------------------------------------------------------


def test_unpatched_hierarchical_flag_is_caught(toy_checkpoint: ToyCheckpoint) -> None:
    config = read_config(toy_checkpoint.path)
    assert not config.transformer_config.is_hierarchical  # as released
    model, _ = uniform_attention_model(config)
    with pytest.raises(SurgeryError, match="is_hierarchical is false"):
        check_hierarchical(model)


def test_flag_without_embedding_bag_is_caught(toy_checkpoint: ToyCheckpoint) -> None:
    model = load_checkpoint(toy_checkpoint.path).model
    del model.transformer.embed_bag
    with pytest.raises(SurgeryError, match="no embed_bag"):
        check_hierarchical(model)


def test_leftover_flat_embedding_is_caught(toy_checkpoint: ToyCheckpoint) -> None:
    model = load_checkpoint(toy_checkpoint.path).model
    model.transformer.embed = nn.Embedding(4, 4)
    with pytest.raises(SurgeryError, match="flat embed"):
        check_hierarchical(model)


# --- hawk to uniform attention ---------------------------------------------------------


def test_hawk_layer_is_caught(toy_checkpoint: ToyCheckpoint) -> None:
    config = read_config(toy_checkpoint.path)
    config.transformer_config.is_hierarchical = True
    model = FEMRModel(config)  # femr's default: hawk in every even layer
    with pytest.raises(SurgeryError, match=r"layers \[0\] still use hawk"):
        check_uniform_attention(model)


def test_recurrent_module_left_behind_is_caught(toy_checkpoint: ToyCheckpoint) -> None:
    config = read_config(toy_checkpoint.path)
    config.transformer_config.is_hierarchical = True
    model = FEMRModel(config)
    object.__setattr__(model.transformer.layers[0], "use_hawk", False)  # flag only, block kept
    with pytest.raises(SurgeryError, match="recurrent modules remain"):
        check_uniform_attention(model)


# --- embedding remap --------------------------------------------------------------------


def test_embedding_size_must_match_dictionary(toy_checkpoint: ToyCheckpoint) -> None:
    model = load_checkpoint(toy_checkpoint.path).model
    check_embedding(model, _entries(toy_checkpoint))
    with pytest.raises(SurgeryError, match="dictionary has"):
        check_embedding(model, _entries(toy_checkpoint) + 1)


def test_missing_embedding_weights_fail_the_load(
    toy_checkpoint: ToyCheckpoint, tmp_path: Path
) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(toy_checkpoint.path, broken)
    state = safetensors.torch.load_file(str(broken / WEIGHTS_FILE))
    del state["transformer.embed.weight"]
    safetensors.torch.save_file(state, str(broken / WEIGHTS_FILE))
    with pytest.raises(CheckpointError, match="embed_bag.weight"):
        prepare_base_model(broken)


def test_unexpected_weights_fail_the_load(toy_checkpoint: ToyCheckpoint, tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(toy_checkpoint.path, broken)
    state = safetensors.torch.load_file(str(broken / WEIGHTS_FILE))
    state["transformer.layers.0.hawk_module.weight"] = state["transformer.in_norm.weight"].clone()
    safetensors.torch.save_file(state, str(broken / WEIGHTS_FILE))
    with pytest.raises(CheckpointError, match="unexpected"):
        prepare_base_model(broken)


# --- CLMBR head -------------------------------------------------------------------------


def test_missing_task_config_is_caught(toy_checkpoint: ToyCheckpoint) -> None:
    model = load_checkpoint(toy_checkpoint.path).model
    model.config.task_config = None
    with pytest.raises(SurgeryError, match="would compute no loss"):
        check_clmbr_head(model, _entries(toy_checkpoint))


def test_non_clmbr_head_module_is_caught(toy_checkpoint: ToyCheckpoint) -> None:
    model = load_checkpoint(toy_checkpoint.path).model
    model.task_model = nn.Linear(4, 4)
    with pytest.raises(SurgeryError, match="no CLMBR task head"):
        check_clmbr_head(model, _entries(toy_checkpoint))


def test_head_width_pin_mismatch_is_caught(toy_checkpoint: ToyCheckpoint) -> None:
    with pytest.raises(SurgeryError, match="8192 were pinned"):
        prepare_base_model(toy_checkpoint.path, clmbr_vocab_size=8192)


def test_head_width_disagreeing_with_config_is_caught(toy_checkpoint: ToyCheckpoint) -> None:
    model = load_checkpoint(toy_checkpoint.path).model
    assert model.config.task_config is not None
    model.config.task_config.task_kwargs = {"clmbr_vocab_size": CLMBR_VOCAB_SIZE + 1}
    with pytest.raises(SurgeryError, match="config records"):
        check_clmbr_head(model, _entries(toy_checkpoint))


def test_head_wider_than_dictionary_is_caught(toy_checkpoint: ToyCheckpoint) -> None:
    model = load_checkpoint(toy_checkpoint.path).model
    with pytest.raises(SurgeryError, match="dictionary only"):
        check_clmbr_head(model, CLMBR_VOCAB_SIZE - 1)


# --- provenance pins ---------------------------------------------------------------------


def test_weight_pin_mismatch_fails_before_training(toy_checkpoint: ToyCheckpoint) -> None:
    with pytest.raises(WeightIntegrityError):
        prepare_base_model(toy_checkpoint.path, expected_weights_sha256="0" * 64)


def test_concept_set_must_come_from_the_base_dictionary(
    toy_checkpoint: ToyCheckpoint, tmp_path: Path
) -> None:
    codes, sha = dictionary_codes(toy_checkpoint.path)
    matching = tmp_path / "matching"
    write_concept_set(matching, sorted(codes), sha)
    prepare_base_model(toy_checkpoint.path, concept_set=matching)

    other = tmp_path / "other"
    write_concept_set(other, sorted(codes), "f" * 64)
    with pytest.raises(SurgeryError, match="is not the dictionary"):
        prepare_base_model(toy_checkpoint.path, concept_set=other)
    assert (toy_checkpoint.path / DICTIONARY_FILE).is_file()
