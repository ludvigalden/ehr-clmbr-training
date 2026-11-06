"""The pinned-release fetch: dry run, checksum verification, token handling."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ehrfm.cli import app
from ehrfm.inputs import CLMBR_T_BASE_PIN
from ehrfm.inputs.fetch import (
    ChecksumMismatchError,
    Downloader,
    TokenMissingError,
    UnreachableSourceError,
    fetch,
)


@pytest.fixture(autouse=True)
def _no_token(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    yield


def _fake_downloader(payload: bytes) -> Downloader:
    """A downloader writing payload as the dictionary into local_dir."""

    def download(repo_id: str, revision: str, destination: Path, token: str) -> Path:
        del repo_id, revision, token
        destination.mkdir(parents=True, exist_ok=True)
        (destination / CLMBR_T_BASE_PIN.filename).write_bytes(payload)
        return destination

    return download


def test_dry_run_prints_notice_and_plan_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(*args: object, **kwargs: object) -> Path:
        raise AssertionError("a dry run must not make a network call")

    monkeypatch.setattr("huggingface_hub.snapshot_download", explode)
    result = fetch(tmp_path / "dest", dry_run=True)
    assert result is None
    out = capsys.readouterr().out
    assert "Licence notice" in out
    assert "CC-BY-NC-4.0" in out
    assert "CC BY-NC-SA 4.0" in out
    assert CLMBR_T_BASE_PIN.repo in out
    assert CLMBR_T_BASE_PIN.revision in out
    assert CLMBR_T_BASE_PIN.file_sha256 in out
    assert "no network call was made" in out


def test_missing_token_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(TokenMissingError):
        fetch(tmp_path / "dest", dry_run=False)


def test_unreachable_source_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")

    def refuse(repo_id: str, revision: str, destination: Path, token: str) -> Path:
        del repo_id, revision, destination, token
        raise ConnectionError("network down")

    with pytest.raises(UnreachableSourceError):
        fetch(tmp_path / "dest", dry_run=False, downloader=refuse)


def test_checksum_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    with pytest.raises(ChecksumMismatchError):
        fetch(tmp_path / "dest", dry_run=False, downloader=_fake_downloader(b"wrong"))


def test_verified_download_reports_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    real = hashlib.sha256(b"real dictionary bytes").hexdigest()
    pin = CLMBR_T_BASE_PIN.__class__(
        repo=CLMBR_T_BASE_PIN.repo,
        revision=CLMBR_T_BASE_PIN.revision,
        filename=CLMBR_T_BASE_PIN.filename,
        file_sha256=real,
        concept_list_sha256=CLMBR_T_BASE_PIN.concept_list_sha256,
        licence=CLMBR_T_BASE_PIN.licence,
    )
    result = fetch(
        tmp_path / "dest",
        dry_run=False,
        pin=pin,
        downloader=_fake_downloader(b"real dictionary bytes"),
    )
    assert result is not None
    assert result.file_sha256 == real
    assert (result.snapshot_dir / CLMBR_T_BASE_PIN.filename).is_file()


# -- CLI wiring ---------------------------------------------------------------------


def test_cli_dry_run_lists_the_command_and_exits_zero(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["fetch-model", "--dry-run", "--dest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Licence notice" in result.output
    assert "no network call was made" in result.output


def test_cli_missing_token_exits_two(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["fetch-model", "--dest", str(tmp_path)])
    assert result.exit_code == 2
    assert "HF_TOKEN" in result.output


def test_cli_checksum_mismatch_exits_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "token-value-never-printed")
    monkeypatch.setattr("ehrfm.inputs.fetch._hf_snapshot", _fake_downloader(b"wrong"))
    runner = CliRunner()
    result = runner.invoke(app, ["fetch-model", "--dest", str(tmp_path)])
    assert result.exit_code == 2
    assert "does not match the pin" in result.output
    assert "token-value-never-printed" not in result.output
    assert "token-value-never-printed" not in result.stderr
