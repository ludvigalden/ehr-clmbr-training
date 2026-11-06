"""Contract verification of the synthetic artifacts: pass, tamper, missing, linkage."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from ehrfm.cli import app
from ehrfm.inputs import (
    InputArtifactError,
    open_concept_set,
    open_extract,
    open_label_set,
)
from ehrfm.inputs.errors import (
    ArtifactTypeError,
    ContractVersionError,
    IntegrityError,
    LinkageError,
    ManifestFormatError,
    MissingArtifactError,
    SchemaError,
    UnexpectedFileError,
)
from ehrfm.inputs.manifest import sha256_file
from ehrfm.toy.artifacts import (
    MANIFEST_NAME,
    SyntheticArtifacts,
    build_artifacts,
    write_concept_set,
    write_manifest,
)
from ehrfm.toy.checkpoint import ToyCheckpoint


@pytest.fixture
def fresh(tmp_path: Path) -> SyntheticArtifacts:
    """A private artifact set each test may modify."""
    return build_artifacts(tmp_path)


def test_extract_verifies(artifacts: SyntheticArtifacts) -> None:
    extract = open_extract(artifacts.extract)
    assert extract.summary.subjects == 20
    assert extract.dataset.meds_version == "0.3.3"
    assert extract.shards["train"] == ("data/train/0.parquet", "data/train/1.parquet")
    splits = extract.subject_splits()
    assert len(splits) == 20
    assert set(splits.values()) == {"train", "tuning", "held_out"}


def test_label_set_verifies_against_extract(artifacts: SyntheticArtifacts) -> None:
    extract = open_extract(artifacts.extract)
    labels = open_label_set(artifacts.labels, extract)
    assert labels.task.label_type == "boolean"
    assert labels.summary.extract_manifest_sha256 == extract.manifest_sha256


def test_concept_set_verifies(tmp_path: Path, toy_checkpoint: ToyCheckpoint) -> None:
    dictionary = toy_checkpoint.path / "dictionary.msgpack"
    root = tmp_path / "clmbr"
    digest = write_concept_set(root, toy_checkpoint.dictionary_codes, sha256_file(dictionary))
    concepts = open_concept_set(root)
    assert concepts.summary.concept_list_sha256 == digest
    assert "Gender/F" in concepts.codes()
    assert "MEDS_BIRTH" not in concepts.codes()


def test_tampered_shard_fails(fresh: SyntheticArtifacts) -> None:
    shard = fresh.extract / "data" / "train" / "0.parquet"
    table = pq.read_table(shard)
    pq.write_table(table.slice(1), shard)
    with pytest.raises(IntegrityError, match="sha256"):
        open_extract(fresh.extract)


def test_single_flipped_byte_fails(fresh: SyntheticArtifacts) -> None:
    dataset_json = fresh.extract / "metadata" / "dataset.json"
    data = bytearray(dataset_json.read_bytes())
    data[-2] ^= 0x01
    dataset_json.write_bytes(bytes(data))
    with pytest.raises(IntegrityError):
        open_extract(fresh.extract)


def test_missing_manifest_fails(fresh: SyntheticArtifacts) -> None:
    (fresh.extract / MANIFEST_NAME).unlink()
    with pytest.raises(MissingArtifactError, match="artifact_manifest.json is missing"):
        open_extract(fresh.extract)


def test_missing_directory_fails(tmp_path: Path) -> None:
    with pytest.raises(MissingArtifactError):
        open_extract(tmp_path / "absent")


def test_missing_listed_file_fails(fresh: SyntheticArtifacts) -> None:
    (fresh.extract / "data" / "tuning" / "0.parquet").unlink()
    with pytest.raises(MissingArtifactError, match="missing"):
        open_extract(fresh.extract)


def test_unlisted_file_fails(fresh: SyntheticArtifacts) -> None:
    (fresh.extract / "data" / "train" / "notes.txt").write_text("extra\n")
    with pytest.raises(UnexpectedFileError, match="not listed"):
        open_extract(fresh.extract)


def test_symlink_fails(fresh: SyntheticArtifacts) -> None:
    (fresh.extract / "link").symlink_to(fresh.extract / "metadata" / "dataset.json")
    with pytest.raises(UnexpectedFileError, match="symlink"):
        open_extract(fresh.extract)


def test_incompatible_contract_version_fails(fresh: SyntheticArtifacts) -> None:
    path = fresh.extract / MANIFEST_NAME
    payload = json.loads(path.read_text())
    payload["contract_version"] = "2.0.0"
    path.write_text(json.dumps(payload))
    with pytest.raises(ContractVersionError):
        open_extract(fresh.extract)


def test_newer_minor_version_fails(fresh: SyntheticArtifacts) -> None:
    path = fresh.extract / MANIFEST_NAME
    payload = json.loads(path.read_text())
    payload["contract_version"] = "1.1.0"
    path.write_text(json.dumps(payload))
    with pytest.raises(ContractVersionError):
        open_extract(fresh.extract)


def test_unknown_manifest_key_fails(fresh: SyntheticArtifacts) -> None:
    path = fresh.extract / MANIFEST_NAME
    payload = json.loads(path.read_text())
    payload["extra"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestFormatError):
        open_extract(fresh.extract)


def test_wrong_artifact_type_fails(artifacts: SyntheticArtifacts) -> None:
    with pytest.raises(ArtifactTypeError):
        open_extract(artifacts.labels)


def test_schema_drift_fails_even_with_matching_manifest(fresh: SyntheticArtifacts) -> None:
    """A re-hashed shard with a large_string column is still rejected."""
    shard = fresh.extract / "data" / "train" / "0.parquet"
    table = pq.read_table(shard)
    pq.write_table(table.set_column(2, "code", table.column("code").cast(pa.large_string())), shard)
    summary = json.loads((fresh.extract / MANIFEST_NAME).read_text())["summary"]
    write_manifest(fresh.extract, summary)
    with pytest.raises(SchemaError, match="large_string"):
        open_extract(fresh.extract)


def test_label_set_from_other_extract_fails(artifacts: SyntheticArtifacts, tmp_path: Path) -> None:
    other = build_artifacts(tmp_path, seed=8)
    with pytest.raises(LinkageError, match="built against"):
        open_label_set(artifacts.labels, open_extract(other.extract))


def test_concept_set_against_wrong_pin_fails(tmp_path: Path, toy_checkpoint: ToyCheckpoint) -> None:
    dictionary = toy_checkpoint.path / "dictionary.msgpack"
    root = tmp_path / "clmbr"
    write_concept_set(root, toy_checkpoint.dictionary_codes, sha256_file(dictionary))
    pin = tmp_path / "manifest.json"
    pin.write_text(
        json.dumps(
            {
                "care_site_excluded": 0,
                "concept_list_sha256": "0" * 64,
                "contract_version": "1.0.0",
                "entry_type_counts": {"code": 1, "numeric": 0, "text": 0, "unused": 0},
                "file_sha256": sha256_file(dictionary),
                "filename": "dictionary.msgpack",
                "revision": "0" * 40,
                "source_repo": "example/toy",
                "total": 1,
                "vocabulary_counts": {"Gender": 1},
            }
        )
    )
    with pytest.raises(LinkageError):
        open_concept_set(root, pin)


def test_clmbr_manifest_malformed_revision_fails(
    tmp_path: Path, toy_checkpoint: ToyCheckpoint
) -> None:
    """A revision that is not a 40-digit hex commit is rejected at parse time."""
    dictionary = toy_checkpoint.path / "dictionary.msgpack"
    root = tmp_path / "clmbr"
    write_concept_set(root, toy_checkpoint.dictionary_codes, sha256_file(dictionary))
    pin = tmp_path / "manifest.json"
    pin.write_text(
        json.dumps(
            {
                "care_site_excluded": 0,
                "concept_list_sha256": "0" * 64,
                "contract_version": "1.0.0",
                "entry_type_counts": {"code": 1, "numeric": 0, "text": 0, "unused": 0},
                "file_sha256": sha256_file(dictionary),
                "filename": "dictionary.msgpack",
                "revision": "main",
                "source_repo": "example/toy",
                "total": 1,
                "vocabulary_counts": {"Gender": 1},
            }
        )
    )
    with pytest.raises(ManifestFormatError, match="revision"):
        open_concept_set(root, pin)


def test_clmbr_manifest_with_care_site_fails(tmp_path: Path, toy_checkpoint: ToyCheckpoint) -> None:
    """CARE_SITE entries are excluded upstream and must not appear in the pin."""
    dictionary = toy_checkpoint.path / "dictionary.msgpack"
    root = tmp_path / "clmbr"
    write_concept_set(root, toy_checkpoint.dictionary_codes, sha256_file(dictionary))
    pin = tmp_path / "manifest.json"
    pin.write_text(
        json.dumps(
            {
                "care_site_excluded": 0,
                "concept_list_sha256": "0" * 64,
                "contract_version": "1.0.0",
                "entry_type_counts": {"code": 1, "numeric": 0, "text": 0, "unused": 0},
                "file_sha256": sha256_file(dictionary),
                "filename": "dictionary.msgpack",
                "revision": "0" * 40,
                "source_repo": "example/toy",
                "total": 1,
                "vocabulary_counts": {"Gender": 1, "CARE_SITE": 0},
            }
        )
    )
    with pytest.raises(ManifestFormatError, match="CARE_SITE"):
        open_concept_set(root, pin)


def test_concept_set_summary_with_care_site_fails(
    tmp_path: Path, toy_checkpoint: ToyCheckpoint
) -> None:
    """A concept-set summary counting CARE_SITE entries is not a contract artifact."""
    dictionary = toy_checkpoint.path / "dictionary.msgpack"
    root = tmp_path / "clmbr"
    write_concept_set(root, toy_checkpoint.dictionary_codes, sha256_file(dictionary))
    path = root / MANIFEST_NAME
    payload = json.loads(path.read_text())
    payload["summary"]["vocabulary_counts"]["CARE_SITE"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestFormatError, match="CARE_SITE"):
        open_concept_set(root)


def test_error_points_at_producer(fresh: SyntheticArtifacts) -> None:
    (fresh.extract / MANIFEST_NAME).unlink()
    with pytest.raises(InputArtifactError) as info:
        open_extract(fresh.extract)
    message = str(info.value)
    assert "github.com/ludvigalden/ehr-omop-meds-femr" in message
    assert "omopmap artifacts verify" in message


def test_cli_verify_passes(artifacts: SyntheticArtifacts) -> None:
    result = CliRunner().invoke(app, ["inputs", "verify", str(artifacts.extract)])
    assert result.exit_code == 0, result.output
    assert "20 subjects" in result.output


def test_cli_verify_tampered_exits_nonzero(fresh: SyntheticArtifacts) -> None:
    shard = fresh.extract / "data" / "held_out" / "0.parquet"
    shard.write_bytes(shard.read_bytes() + b"\0")
    result = CliRunner().invoke(app, ["inputs", "verify", str(fresh.extract)])
    assert result.exit_code == 1
    assert "IntegrityError" in result.output
    assert "omopmap artifacts verify" in result.output


def test_cli_verify_missing_manifest_exits_nonzero(fresh: SyntheticArtifacts) -> None:
    (fresh.labels / MANIFEST_NAME).unlink()
    result = CliRunner().invoke(
        app,
        [
            "inputs",
            "verify",
            str(fresh.labels),
            "--kind",
            "labels",
            "--extract",
            str(fresh.extract),
        ],
    )
    assert result.exit_code == 1
    assert "MissingArtifactError" in result.output


def test_cli_verify_label_set_passes(artifacts: SyntheticArtifacts) -> None:
    result = CliRunner().invoke(
        app,
        [
            "inputs",
            "verify",
            str(artifacts.labels),
            "--kind",
            "labels",
            "--extract",
            str(artifacts.extract),
        ],
    )
    assert result.exit_code == 0, result.output


def test_copy_of_valid_extract_still_verifies(
    artifacts: SyntheticArtifacts, tmp_path: Path
) -> None:
    """Verification depends on bytes, not on location."""
    copy = tmp_path / "moved"
    shutil.copytree(artifacts.extract, copy)
    assert open_extract(copy).manifest_sha256 == open_extract(artifacts.extract).manifest_sha256
