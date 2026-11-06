"""The small-cell suppression policy, on payloads and at the MLflow boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from ehrfm.config.experiment import SmallCellPolicy, TrackingConfig
from ehrfm.experiment.suppression import flatten, nest, suppress, suppress_flat
from ehrfm.experiment.tracking import SUPPRESSED_PARAM, SUPPRESSION_ARTIFACT, open_tracker
from tests.experiment.store import small_counts, stored_counts, stored_metric_names

POLICY = SmallCellPolicy()


def test_default_threshold_is_five() -> None:
    assert POLICY.threshold == 5
    assert POLICY.suppress_zero and POLICY.mask_group and POLICY.complementary


def test_small_scalar_count_is_masked_with_its_record() -> None:
    masked = suppress(
        {
            "site": {"n": 4, "auroc": 0.81, "name": "a"},
            "other": {"n": 40, "auroc": 0.7, "name": "b"},
        },
        SmallCellPolicy(complementary=False),
    )
    assert masked.value == {
        "site": {"n": None, "auroc": None, "name": "a"},
        "other": {"n": 40, "auroc": 0.7, "name": "b"},
    }
    reasons = {cell.path: cell.reason for cell in masked.cells}
    assert reasons == {"/site/n": "small_cell", "/site/auroc": "group"}


def test_threshold_boundary_and_zero() -> None:
    payload: JsonValue = {"a": {"count": 5}, "b": {"count": 4}, "c": {"count": 0}}
    no_comp = SmallCellPolicy(complementary=False)
    assert suppress(payload, no_comp).value == {
        "a": {"count": 5},
        "b": {"count": None},
        "c": {"count": None},
    }
    keep_zero = SmallCellPolicy(complementary=False, suppress_zero=False)
    assert suppress(payload, keep_zero).value == {
        "a": {"count": 5},
        "b": {"count": None},
        "c": {"count": 0},
    }


def test_breakdown_cells_and_complementary_suppression() -> None:
    payload: JsonValue = {
        "subjects": 60,
        "subjects_by_split": {"train": 50, "tuning": 3, "held_out": 7},
    }
    masked = suppress(payload, POLICY)
    # 3 is small; 7 is the next smallest and would give 3 back from the total.
    assert masked.value == {
        "subjects": 60,
        "subjects_by_split": {"train": 50, "tuning": None, "held_out": None},
    }
    assert {cell.path: cell.reason for cell in masked.cells} == {
        "/subjects_by_split/tuning": "small_cell",
        "/subjects_by_split/held_out": "complementary",
    }


def test_sibling_records_get_complementary_suppression() -> None:
    rows: JsonValue = {
        "by_vocabulary": [
            {"vocabulary_id": "A", "events": 2, "share": 0.5},
            {"vocabulary_id": "B", "events": 30, "share": 0.9},
            {"vocabulary_id": "C", "events": 300, "share": 0.1},
        ]
    }
    masked = suppress(rows, POLICY)
    assert masked.value == {
        "by_vocabulary": [
            {"vocabulary_id": "A", "events": None, "share": None},
            {"vocabulary_id": "B", "events": None, "share": None},
            {"vocabulary_id": "C", "events": 300, "share": 0.1},
        ]
    }


def test_two_small_cells_need_no_complement() -> None:
    masked = suppress({"counts": [1, 2, 50, 60]}, POLICY)
    assert masked.value == {"counts": [None, None, 50, 60]}


def test_non_count_fields_and_booleans_are_untouched() -> None:
    payload: JsonValue = {"auroc": 0.5, "seed": 3, "top_n": 2, "flag": True, "n": True}
    assert suppress(payload, POLICY).value == payload


def test_non_finite_counts_are_masked() -> None:
    assert suppress({"n": float("nan"), "x": 1.0}, POLICY).value == {"n": None, "x": None}


def test_configurable_count_fields() -> None:
    policy = SmallCellPolicy(count_fields=("cases",), complementary=False)
    masked = suppress({"cases": 2, "n": 2}, policy)
    assert masked.value == {"cases": None, "n": None}  # n masked as part of the record
    only_cases = suppress({"group": {"n": 2}, "cases": 9}, policy)
    assert only_cases.value == {"group": {"n": 2}, "cases": 9}


def test_input_is_not_mutated() -> None:
    payload: dict[str, JsonValue] = {"n": 1, "rate": 0.5}
    suppress(payload, POLICY)
    assert payload == {"n": 1, "rate": 0.5}


def test_flat_metrics_follow_the_same_rules() -> None:
    masked = suppress_flat(
        {"eval/a/n": 3.0, "eval/a/auroc": 0.8, "eval/b/n": 300.0, "eval/b/auroc": 0.7},
        SmallCellPolicy(complementary=False),
    )
    assert masked.value == {"eval/b/n": 300.0, "eval/b/auroc": 0.7}


def test_nest_and_flatten_round_trip() -> None:
    flat = {"a/b": 1.0, "a/c/d": 2.0, "e": 3.0}
    assert flatten(nest(flat)) == flat
    with pytest.raises(ValueError, match="both a value and a prefix"):
        nest({"a/b": 1.0, "a": 2.0} | {"a/b": 1.0})
    with pytest.raises(ValueError, match="empty segment"):
        nest({"a//b": 1.0})


def test_policy_is_strict() -> None:
    with pytest.raises(ValidationError):
        SmallCellPolicy.model_validate({"threshold": 0})
    with pytest.raises(ValidationError):
        SmallCellPolicy.model_validate({"threshold": 5, "minimum": 3})
    with pytest.raises(ValidationError):
        SmallCellPolicy.model_validate({"count_fields": []})


def _coverage_like() -> dict[str, JsonValue]:
    return {
        "events": 1032,
        "distinct_codes": 25,
        "by_vocabulary": [
            {"vocabulary_id": "DEMOLAB", "events": 3, "tokenized_events": 2, "distinct_codes": 1},
            {
                "vocabulary_id": "DEMOCOND",
                "events": 143,
                "tokenized_events": 143,
                "distinct_codes": 8,
            },
            {"vocabulary_id": "Visit", "events": 210, "tokenized_events": 210, "distinct_codes": 2},
        ],
        "unrecognised": [
            {"code": "DEMOPROC/P02", "events": 4},
            {"code": "DEMOLAB/L02", "events": 39},
        ],
        "subjects_by_split": {"train": 48, "tuning": 4, "held_out": 6},
    }


def test_nothing_below_the_threshold_reaches_the_store(tmp_path: Path) -> None:
    tracker = open_tracker(TrackingConfig(), "suppression-test", tmp_path, POLICY)
    with tracker.run("run") as run:
        run.log_params({"cohort": {"n": 3, "label": "a"}, "seed": 1, "events": 2})
        run.log_metrics(
            {
                "eval/x/n": 4.0,
                "eval/x/auroc": 0.9,
                "eval/y/n": 2000.0,
                "eval/y/auroc": 0.6,
                "eval/z/n": 900.0,
                "eval/z/auroc": 0.7,
            }
        )
        run.log_json("coverage.json", _coverage_like())
    mlruns = tmp_path / "mlruns"
    assert mlruns.is_dir()

    assert small_counts(mlruns, POLICY) == []
    counts = stored_counts(mlruns, POLICY)
    assert {count.value for count in counts} >= {1032.0, 210.0, 2000.0}

    names = stored_metric_names(mlruns)
    # x is small; z (the next smallest sibling) is masked so a total cannot give x back.
    assert not names & {"eval/x/n", "eval/x/auroc", "eval/z/n", "eval/z/auroc"}
    assert {"eval/y/n", "eval/y/auroc"} <= names
    params = {path.name: path.read_text() for path in mlruns.rglob("params/*")}
    assert params["cohort.n"] == SUPPRESSED_PARAM
    assert params["events"] == SUPPRESSED_PARAM
    assert params["cohort.label"] == "a"

    coverage = json.loads(next(mlruns.rglob("artifacts/coverage.json")).read_text())
    assert coverage["by_vocabulary"][0]["events"] is None
    assert coverage["by_vocabulary"][0]["vocabulary_id"] == "DEMOLAB"
    assert coverage["by_vocabulary"][1]["events"] is None  # complementary
    assert coverage["by_vocabulary"][2]["events"] == 210
    assert coverage["unrecognised"][0] == {"code": "DEMOPROC/P02", "events": None}
    assert coverage["subjects_by_split"] == {"train": 48, "tuning": None, "held_out": None}

    log = json.loads(next(mlruns.rglob(f"artifacts/{SUPPRESSION_ARTIFACT}")).read_text())
    assert log["threshold"] == 5
    assert {
        "source": "coverage.json",
        "path": "/unrecognised/0/events",
        "reason": "small_cell",
    } in (log["suppressed"])
    # The log names paths and reasons, never the masked values.
    assert all(set(entry) == {"source", "path", "reason"} for entry in log["suppressed"])


def test_store_scan_positive_control(tmp_path: Path) -> None:
    """The scanner finds small counts when a permissive policy lets them through."""
    permissive = SmallCellPolicy(threshold=1, suppress_zero=False)
    tracker = open_tracker(TrackingConfig(), "control", tmp_path, permissive)
    with tracker.run("run") as run:
        run.log_metrics({"eval/x/n": 2.0})
        run.log_json("coverage.json", _coverage_like())
        run.log_params({"events": 3})
    found = small_counts(tmp_path / "mlruns", POLICY)
    assert {(count.name, count.value) for count in found} >= {
        ("n", 2.0),
        ("events", 3.0),
        ("events", 4.0),
        ("tokenized_events", 2.0),
        ("subjects_by_split", 4.0),
    }


def test_failed_run_is_marked_failed_and_still_logs_the_suppression_record(tmp_path: Path) -> None:
    tracker = open_tracker(TrackingConfig(), "failing", tmp_path, POLICY)
    run_ids: list[str] = []
    with pytest.raises(RuntimeError, match="boom"), tracker.run("run") as run:
        run.log_metrics({"n": 1.0})
        run_ids.append(run.run_id)
        raise RuntimeError("boom")
    run_id = run_ids[0]
    assert tracker.client.get_run(run_id).info.status == "FAILED"
    assert list((tmp_path / "mlruns").rglob(f"artifacts/{SUPPRESSION_ARTIFACT}"))
