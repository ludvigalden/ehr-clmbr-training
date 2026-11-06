"""Dictionary coverage of the synthetic extract: exact counts, CLI, and femr agreement.

The toy dictionary keeps every second code of each invented family plus the
parent groups, so on the seed-7 extract the expected numbers are fixed:

- 25 distinct vocabulary codes (``MEDS_BIRTH`` excluded), 318 events;
- 17 codes have their own entry (253 events);
- 23 codes yield a token: the 6 condition and drug codes left out of the
  dictionary still reach it through their ``GROUP-A``/``GROUP-B`` parent;
- 2 codes yield none: ``DEMOLAB/L02`` and ``DEMOPROC/P02``, 15 events each.
"""

from __future__ import annotations

from pathlib import Path

import meds_reader
import pytest
from typer.testing import CliRunner

from ehrfm.cli import app
from ehrfm.data.coverage import (
    CoverageReport,
    UnrecognisedCode,
    dictionary_codes,
    extract_code_events,
    extract_coverage,
    measure_coverage,
    vocabulary_id,
)
from ehrfm.data.tokenization import CodeOntology, extract_tokenizer
from ehrfm.inputs import MedsExtract
from ehrfm.inputs.contract import RESERVED_CODES
from ehrfm.inputs.manifest import sha256_file
from ehrfm.toy.artifacts import SyntheticArtifacts
from ehrfm.toy.checkpoint import ToyCheckpoint

EXPECTED_BY_VOCABULARY = {
    # vocabulary: (distinct, recognised, tokenized, events, tokenized events)
    "DEMOCOND": (8, 4, 8, 38, 38),
    "DEMODRUG": (4, 2, 4, 36, 36),
    "DEMOLAB": (3, 2, 2, 43, 28),
    "DEMOOBS": (1, 1, 1, 41, 41),
    "DEMOPROC": (3, 2, 2, 50, 35),
    "Gender": (2, 2, 2, 20, 20),
    "LOINC": (1, 1, 1, 11, 11),
    "RxNorm": (1, 1, 1, 12, 12),
    "Visit": (2, 2, 2, 67, 67),
}


@pytest.fixture(scope="module")
def report(extract: MedsExtract, toy_checkpoint: ToyCheckpoint) -> CoverageReport:
    return extract_coverage(extract, toy_checkpoint.path)


def test_exact_totals(report: CoverageReport) -> None:
    assert report.distinct_codes == 25
    assert report.recognised_codes == 17
    assert report.tokenized_codes == 23
    assert report.events == 318
    assert report.recognised_events == 253
    assert report.tokenized_events == 288
    assert report.recognised_share == pytest.approx(17 / 25)
    assert report.tokenized_share == pytest.approx(23 / 25)
    assert report.tokenized_event_share == pytest.approx(288 / 318)


def test_exact_per_vocabulary(report: CoverageReport) -> None:
    got = {
        row.vocabulary_id: (
            row.distinct_codes,
            row.recognised_codes,
            row.tokenized_codes,
            row.events,
            row.tokenized_events,
        )
        for row in report.by_vocabulary
    }
    assert got == EXPECTED_BY_VOCABULARY
    assert [row.vocabulary_id for row in report.by_vocabulary] == sorted(EXPECTED_BY_VOCABULARY)


def test_out_of_vocabulary_listing(report: CoverageReport) -> None:
    assert report.unrecognised == (
        UnrecognisedCode(code="DEMOLAB/L02", events=15),
        UnrecognisedCode(code="DEMOPROC/P02", events=15),
    )


def test_listing_threshold_and_cap(extract: MedsExtract, toy_checkpoint: ToyCheckpoint) -> None:
    assert extract_coverage(extract, toy_checkpoint.path, min_events=16).unrecognised == ()
    capped = extract_coverage(extract, toy_checkpoint.path, top_n=1)
    assert [c.code for c in capped.unrecognised] == ["DEMOLAB/L02"]
    assert capped.tokenized_codes == 23


def test_report_is_pinned_to_its_inputs(
    report: CoverageReport, extract: MedsExtract, toy_checkpoint: ToyCheckpoint
) -> None:
    assert report.extract_manifest_sha256 == extract.manifest_sha256
    assert report.dictionary_sha256 == sha256_file(toy_checkpoint.path / "dictionary.msgpack")


def test_event_counts_match_the_manifest(extract: MedsExtract) -> None:
    counts = extract_code_events(extract)
    assert sum(counts.values()) == extract.summary.events == 338
    assert len(counts) == extract.summary.distinct_codes


def test_tokenized_events_agree_with_femr_tokenizer(
    report: CoverageReport,
    extract: MedsExtract,
    toy_checkpoint: ToyCheckpoint,
    database: meds_reader.SubjectDatabase,
) -> None:
    """The report's out-of-vocabulary events are exactly those femr gives no token."""
    tokenizer = extract_tokenizer(toy_checkpoint.path, extract)
    dropped = 0
    for subject_id in database:
        for event in database[subject_id].events:
            if event.code in RESERVED_CODES:
                continue
            features, _ = tokenizer.get_feature_codes(event)
            dropped += not features
    assert dropped == report.events - report.tokenized_events == 30


def test_dictionary_codes_reads_code_entries(toy_checkpoint: ToyCheckpoint) -> None:
    codes, digest = dictionary_codes(toy_checkpoint.path)
    assert codes == frozenset(toy_checkpoint.dictionary_codes)
    assert len(digest) == 64


def test_measure_coverage_on_a_hand_table() -> None:
    ontology = CodeOntology({"V/a": ["V/root"], "V/b": ["V/mid"], "V/mid": ["V/root"]})
    report = measure_coverage(
        {"V/a": 3, "V/b": 5, "W/c": 12, "MEDS_BIRTH": 2},
        {"V/root"},
        ontology,
        extract_manifest_sha256="0" * 64,
        dictionary_sha256="1" * 64,
        min_events=10,
    )
    assert (report.distinct_codes, report.recognised_codes, report.tokenized_codes) == (3, 0, 2)
    assert (report.events, report.recognised_events, report.tokenized_events) == (20, 0, 8)
    assert report.unrecognised == (UnrecognisedCode(code="W/c", events=12),)


def test_empty_extract_has_zero_shares() -> None:
    report = measure_coverage(
        {},
        set(),
        CodeOntology({}),
        extract_manifest_sha256="0" * 64,
        dictionary_sha256="1" * 64,
    )
    assert report.recognised_share == report.tokenized_share == 0.0


def test_vocabulary_id_splits_on_first_slash() -> None:
    assert vocabulary_id("LOINC/X0001-0") == "LOINC"
    assert vocabulary_id("V/a/b") == "V"
    assert vocabulary_id("MEDS_BIRTH") == ""


def test_cli_prints_the_report(
    artifacts: SyntheticArtifacts, toy_checkpoint: ToyCheckpoint
) -> None:
    result = CliRunner().invoke(
        app, ["inputs", "coverage", str(artifacts.extract), "--model", str(toy_checkpoint.path)]
    )
    assert result.exit_code == 0, result.output
    assert "distinct codes 25: recognised 17 (68.0%), tokenized 23 (92.0%)" in result.output
    assert "out of vocabulary: DEMOLAB/L02 (15 events)" in result.output


def test_cli_json_round_trips(artifacts: SyntheticArtifacts, toy_checkpoint: ToyCheckpoint) -> None:
    result = CliRunner().invoke(
        app,
        [
            "inputs",
            "coverage",
            str(artifacts.extract),
            "--model",
            str(toy_checkpoint.path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    parsed = CoverageReport.model_validate_json(result.output)
    assert parsed.tokenized_codes == 23


def test_cli_fails_closed_on_a_bad_extract(tmp_path: Path, toy_checkpoint: ToyCheckpoint) -> None:
    result = CliRunner().invoke(
        app, ["inputs", "coverage", str(tmp_path), "--model", str(toy_checkpoint.path)]
    )
    assert result.exit_code == 1
