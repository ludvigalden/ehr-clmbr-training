"""Batches and representations end to end: extract -> meds_reader -> batches -> embeddings.

Also covers device fallback, weight provenance (a tampered weight fails
closed) and the xformers attention-mask fallback.
"""

from __future__ import annotations

import datetime as dt
import shutil
import types
from pathlib import Path

import femr.ontology
import meds_reader
import numpy as np
import pytest
import torch
from femr.models.processor import BatchCreator, FEMRBatchProcessor
from meds_reader.transform import MutableEvent, MutableSubject
from numpy.typing import NDArray

from ehrfm.data.batches import BatchBuilder
from ehrfm.data.prediction_times import PredictionTime, epoch_seconds, prediction_times
from ehrfm.data.tokenization import extract_tokenizer
from ehrfm.inputs import MedsExtract, open_label_set
from ehrfm.models import _compat
from ehrfm.models.checkpoint import WEIGHTS_FILE, load_checkpoint
from ehrfm.models.device import DeviceUnavailableError, resolve_device
from ehrfm.models.representations import ClmbrEncoder, RepresentationError, position_at
from ehrfm.models.weights import WeightIntegrityError, read_weights, weights_sha256
from ehrfm.toy.artifacts import SyntheticArtifacts
from ehrfm.toy.checkpoint import HIDDEN_SIZE, ToyCheckpoint
from tests.models.test_toy_checkpoint import subject_representations

ALL_SUBJECTS = list(range(1, 21))


@pytest.fixture(scope="module")
def builder(toy_checkpoint: ToyCheckpoint, extract: MedsExtract) -> BatchBuilder:
    return BatchBuilder(extract_tokenizer(toy_checkpoint.path, extract), tokens_per_batch=64)


@pytest.fixture(scope="module")
def encoder(toy_checkpoint: ToyCheckpoint) -> ClmbrEncoder:
    return ClmbrEncoder.load(toy_checkpoint.path, device="cpu")


# -- batches -----------------------------------------------------------------


def test_batches_cover_every_subject_once(
    builder: BatchBuilder, database: meds_reader.SubjectDatabase
) -> None:
    batches = list(builder.batches(database, ALL_SUBJECTS + [3, 1]))
    packed = [sid for batch in batches for sid in batch.subject_ids]
    assert packed == ALL_SUBJECTS
    assert len(batches) > 1
    for batch in batches:
        assert batch.n_tokens <= builder.tokens_per_batch or len(batch.subject_ids) == 1


def test_batch_matches_femr_processor(
    builder: BatchBuilder, database: meds_reader.SubjectDatabase
) -> None:
    """A one-subject batch is exactly what femr's own convert_subject + collate gives."""
    ours = builder.build([database[4]]).model_input
    processor = FEMRBatchProcessor(builder.tokenizer)
    theirs = processor.collate([processor.convert_subject(database[4], tensor_type="pt")])
    assert ours.keys() == theirs.keys()
    for key in ("hierarchical_tokens", "token_indices", "ages", "timestamps", "subject_lengths"):
        assert np.array_equal(
            np.asarray(ours["batch"]["transformer"][key]),
            np.asarray(theirs["batch"]["transformer"][key]),
        )


def test_packed_batch_shapes(builder: BatchBuilder, database: meds_reader.SubjectDatabase) -> None:
    batch = builder.build([database[1], database[2]])
    transformer = batch.model_input["batch"]["transformer"]
    lengths = transformer["subject_lengths"]
    assert lengths.shape[0] == 1 and int(np.asarray(lengths).sum()) == batch.n_tokens
    assert transformer["ages"].shape == (1, batch.n_tokens)
    assert "task" not in batch.model_input["batch"]


def test_empty_batch_is_rejected(builder: BatchBuilder) -> None:
    with pytest.raises(ValueError):
        builder.build([])


# -- representations ----------------------------------------------------------


def test_subject_representations_shape_and_order(
    encoder: ClmbrEncoder, builder: BatchBuilder, database: meds_reader.SubjectDatabase
) -> None:
    reps = encoder.subject_representations(database, ALL_SUBJECTS, builder)
    assert reps.embeddings.shape == (20, HIDDEN_SIZE)
    assert reps.embeddings.dtype == torch.float32
    assert reps.subject_ids.tolist() == ALL_SUBJECTS
    assert torch.isfinite(reps.embeddings).all()
    assert torch.unique(reps.embeddings, dim=0).shape[0] == 20


def test_packing_does_not_change_representations(
    encoder: ClmbrEncoder,
    database: meds_reader.SubjectDatabase,
    toy_checkpoint: ToyCheckpoint,
    extract: MedsExtract,
) -> None:
    """Packed batches give the same rows as one subject per batch (attention is per subject)."""
    tokenizer = extract_tokenizer(toy_checkpoint.path, extract)
    packed = encoder.subject_representations(database, ALL_SUBJECTS, BatchBuilder(tokenizer, 4096))
    single = encoder.subject_representations(database, ALL_SUBJECTS, BatchBuilder(tokenizer, 1))
    assert torch.allclose(packed.embeddings, single.embeddings, atol=1e-5)


def test_matches_the_reference_forward(
    encoder: ClmbrEncoder,
    builder: BatchBuilder,
    database: meds_reader.SubjectDatabase,
    toy_checkpoint: ToyCheckpoint,
    ontology: femr.ontology.Ontology,
) -> None:
    reps = encoder.subject_representations(database, ALL_SUBJECTS, builder)
    reference = subject_representations(toy_checkpoint.path, database, ontology)
    for row, subject_id in enumerate(ALL_SUBJECTS):
        assert torch.allclose(reps.embeddings[row], reference[subject_id], atol=1e-5)


def test_deterministic_across_loads(
    toy_checkpoint: ToyCheckpoint, builder: BatchBuilder, database: meds_reader.SubjectDatabase
) -> None:
    first = ClmbrEncoder.load(toy_checkpoint.path, device="cpu")
    second = ClmbrEncoder.load(toy_checkpoint.path, device="cpu")
    a = first.subject_representations(database, ALL_SUBJECTS, builder)
    b = second.subject_representations(database, ALL_SUBJECTS, builder)
    assert torch.equal(a.embeddings, b.embeddings)
    assert first.weights.sha256 == second.weights.sha256


def test_end_to_end_from_label_set(
    artifacts: SyntheticArtifacts,
    extract: MedsExtract,
    encoder: ClmbrEncoder,
    builder: BatchBuilder,
    database: meds_reader.SubjectDatabase,
) -> None:
    """Extract -> meds_reader -> batches -> one embedding per upstream prediction time."""
    times = prediction_times(open_label_set(artifacts.labels, extract), extract)
    reps = encoder.representations_at(database, times, builder)
    assert len(reps) == len(times)
    assert reps.embeddings.shape == (len(times), HIDDEN_SIZE)
    assert reps.subject_ids.tolist() == [t.subject_id for t in times]
    assert reps.times.tolist() == [t.epoch_seconds for t in times]
    assert torch.isfinite(reps.embeddings).all()


def test_prediction_time_sees_no_later_events(
    encoder: ClmbrEncoder, builder: BatchBuilder, database: meds_reader.SubjectDatabase
) -> None:
    """At a time between two events the row equals the one at the earlier event."""
    # Static events carry no time at runtime although the stub types it as datetime.
    event_times: list[dt.datetime | None] = [e.time for e in database[3].events]
    distinct = sorted({t for t in event_times if t is not None})
    first_visit, later = distinct[1], distinct[2]
    between = first_visit + (later - first_visit) / 2
    at_event = encoder.representations_at(database, [PredictionTime(3, first_visit)], builder)
    at_between = encoder.representations_at(database, [PredictionTime(3, between)], builder)
    at_later = encoder.representations_at(database, [PredictionTime(3, later)], builder)
    assert torch.equal(at_event.embeddings, at_between.embeddings)
    assert not torch.equal(at_event.embeddings, at_later.embeddings)
    full = encoder.subject_representations(database, [3], builder)
    assert full.times[0] >= epoch_seconds(later)


def test_time_before_any_event_fails(
    encoder: ClmbrEncoder, builder: BatchBuilder, database: meds_reader.SubjectDatabase
) -> None:
    with pytest.raises(RepresentationError, match="precedes"):
        encoder.representations_at(database, [PredictionTime(3, dt.datetime(1900, 1, 1))], builder)


# -- birth position timestamps ------------------------------------------------


def _subject(code: str, birth: dt.datetime, times: list[dt.datetime]) -> MutableSubject:
    events = [MutableEvent(time=birth, code="MEDS_BIRTH", properties={})]
    events += [MutableEvent(time=time, code=code, properties={}) for time in times]
    return MutableSubject(subject_id=1, events=events)


def _stamps(builder: BatchBuilder, subject: MutableSubject) -> NDArray[np.int64]:
    creator = BatchCreator(builder.tokenizer)
    creator.start_batch()
    creator.add_subject(subject)
    return np.asarray(creator.timestamps, dtype=np.int64)


def test_femr_stamps_the_birth_position_with_the_last_event_time(builder: BatchBuilder) -> None:
    """femr 0.2.4 behaviour that :func:`position_at` relies on: the stamps are unsorted."""
    code = sorted(c for c in builder.tokenizer.code_lookup if not c.startswith("MEDS"))[0]
    later = [dt.datetime(2000, 1, 1), dt.datetime(2005, 6, 1), dt.datetime(2010, 3, 3)]
    stamps = _stamps(builder, _subject(code, dt.datetime(1980, 1, 1), later))
    assert stamps.tolist() == [epoch_seconds(later[-1])] + [epoch_seconds(t) for t in later]
    assert not bool(np.all(np.diff(stamps) >= 0))


def test_position_at_skips_the_birth_position() -> None:
    stamps = np.asarray([300, 100, 200, 300], dtype=np.int64)
    assert position_at(stamps, 50) is None
    assert position_at(stamps, 99) is None
    assert position_at(stamps, 100) == 1
    assert position_at(stamps, 250) == 2
    assert position_at(stamps, 300) == 3
    assert position_at(stamps, 10_000) == 3


def test_position_at_never_selects_a_birth_only_timeline() -> None:
    """A subject whose events all fall on the birth day has only the birth position."""
    assert position_at(np.asarray([500], dtype=np.int64), 10_000) is None


def test_position_at_rejects_unsorted_post_birth_stamps() -> None:
    with pytest.raises(RepresentationError, match="not sorted"):
        position_at(np.asarray([300, 200, 100, 300], dtype=np.int64), 150)


def test_birth_day_only_subject_has_no_representation_at_a_time(
    encoder: ClmbrEncoder, builder: BatchBuilder
) -> None:
    """Matches femr's labelled-subject task, which never labels the birth position."""
    birth_only = _subject(
        sorted(c for c in builder.tokenizer.code_lookup if not c.startswith("MEDS"))[0],
        dt.datetime(1980, 1, 1, 8),
        [dt.datetime(1980, 1, 1, 9)],
    )
    tokens = encoder.encode(builder.build([birth_only]))
    later = epoch_seconds(dt.datetime(1990, 1, 1))
    assert tokens.timestamps.shape[0] == 1
    # A plain binary search over femr's stamps selects the birth position here.
    assert int(np.searchsorted(tokens.timestamps, later, side="right")) - 1 == 0
    assert position_at(tokens.timestamps, later) is None


# -- weight provenance ----------------------------------------------------------


def test_provenance_is_recorded(toy_checkpoint: ToyCheckpoint, encoder: ClmbrEncoder) -> None:
    expected = weights_sha256(toy_checkpoint.path / WEIGHTS_FILE)
    assert encoder.weights.sha256 == expected
    assert encoder.weights.size == (toy_checkpoint.path / WEIGHTS_FILE).stat().st_size
    pinned = ClmbrEncoder.load(toy_checkpoint.path, device="cpu", expected_weights_sha256=expected)
    assert pinned.weights.sha256 == expected


def test_tampered_weights_fail_closed(toy_checkpoint: ToyCheckpoint, tmp_path: Path) -> None:
    pinned = weights_sha256(toy_checkpoint.path / WEIGHTS_FILE)
    tampered = tmp_path / "tampered"
    shutil.copytree(toy_checkpoint.path, tampered)
    data = bytearray((tampered / WEIGHTS_FILE).read_bytes())
    data[-1] ^= 0x01
    (tampered / WEIGHTS_FILE).write_bytes(bytes(data))
    with pytest.raises(WeightIntegrityError, match="pinned"):
        load_checkpoint(tampered, expected_weights_sha256=pinned)
    with pytest.raises(WeightIntegrityError):
        read_weights(tampered / WEIGHTS_FILE, pinned)


def test_malformed_pin_is_rejected(toy_checkpoint: ToyCheckpoint) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(toy_checkpoint.path, expected_weights_sha256="abc")


# -- device selection -------------------------------------------------------------


def test_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("ehrfm.models.device.mps_available", lambda: False)
    monkeypatch.delenv("EHRFM_DEVICE", raising=False)
    assert resolve_device() == torch.device("cpu")
    assert resolve_device("auto") == torch.device("cpu")


def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EHRFM_DEVICE", "cpu")
    assert resolve_device() == torch.device("cpu")


def test_unavailable_device_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("ehrfm.models.device.mps_available", lambda: False)
    for name in ("cuda", "cuda:1", "mps", "not-a-device"):
        with pytest.raises(DeviceUnavailableError):
            resolve_device(name)


def test_auto_prefers_cuda_then_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("ehrfm.models.device.mps_available", lambda: True)
    assert resolve_device("auto") == torch.device("mps")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda", 0)


# -- xformers attention-mask fallback ------------------------------------------------


def test_fallback_mask_matches_xformers() -> None:
    xformers_ops = pytest.importorskip("xformers").ops
    lengths, window = [3, 5, 1], 2
    real = xformers_ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(lengths)
    ours = _compat.BlockDiagonalMask.from_seqlens(lengths)
    shape = (1, 2, 9, 9)
    assert torch.equal(
        real.make_local_attention(window).materialize(shape),
        ours.make_local_attention(window).materialize(shape),
    )
    full = ours.materialize(shape)[0, 0]
    assert torch.equal(full[:3, :3], torch.zeros(3, 3))
    assert torch.isinf(full[:3, 3:]).all()


def test_fallback_registers_only_when_missing() -> None:
    xformers_ops = pytest.importorskip("xformers").ops
    bare = types.ModuleType("fake_ops")
    assert _compat.install_attention_bias(bare)
    mask = bare.fmha.attn_bias.BlockDiagonalMask.from_seqlens([2])
    assert isinstance(mask, bare.AttentionBias)
    assert not _compat.install_attention_bias(bare)
    assert not _compat.install_attention_bias(xformers_ops)


def test_encoder_runs_under_the_fallback_mask(
    monkeypatch: pytest.MonkeyPatch,
    encoder: ClmbrEncoder,
    builder: BatchBuilder,
    database: meds_reader.SubjectDatabase,
) -> None:
    """With an xformers lacking fmha the fallback gives the same representations."""
    xformers_ops = pytest.importorskip("xformers").ops
    expected = encoder.subject_representations(database, [1, 2, 3], builder)
    fake = types.ModuleType("xformers.ops")
    _compat.install_attention_bias(fake)
    monkeypatch.setattr(xformers_ops, "fmha", fake.fmha)
    monkeypatch.setattr(xformers_ops, "AttentionBias", fake.AttentionBias)
    got = encoder.subject_representations(database, [1, 2, 3], builder)
    assert torch.allclose(got.embeddings, expected.embeddings, atol=1e-5)


def test_resolve_cpu_explicitly() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
