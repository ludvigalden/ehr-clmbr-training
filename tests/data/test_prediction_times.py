"""Prediction times come from the upstream label set, restricted by split on request."""

from __future__ import annotations

import datetime as dt

import pyarrow.parquet as pq
import pytest

from ehrfm.data.prediction_times import (
    PredictionTime,
    by_subject,
    epoch_seconds,
    labelled_times,
    prediction_times,
    subjects_of,
)
from ehrfm.inputs import MedsExtract, open_label_set
from ehrfm.inputs.schemas import column_values
from ehrfm.toy.artifacts import SyntheticArtifacts


def test_times_are_the_label_rows(artifacts: SyntheticArtifacts, extract: MedsExtract) -> None:
    label_set = open_label_set(artifacts.labels, extract)
    times = prediction_times(label_set, extract)
    table = pq.read_table(label_set.labels_path)
    expected = sorted(
        PredictionTime(sid, time)
        for sid, time in zip(
            column_values(table, "subject_id"), column_values(table, "prediction_time"), strict=True
        )
    )
    assert list(times) == expected
    assert len(times) == label_set.summary.labels


def test_values_follow_the_task_label_type(
    artifacts: SyntheticArtifacts, extract: MedsExtract
) -> None:
    rows = labelled_times(open_label_set(artifacts.labels, extract), extract)
    assert all(isinstance(row.value, bool) for row in rows)


def test_split_restriction_matches_the_summary(
    artifacts: SyntheticArtifacts, extract: MedsExtract
) -> None:
    label_set = open_label_set(artifacts.labels, extract)
    membership = extract.subject_splits()
    for split in ("train", "tuning", "held_out"):
        times = prediction_times(label_set, extract, splits={split})
        assert len(times) == label_set.summary.labels_by_split[split]
        assert all(membership[t.subject_id] == split for t in times)


def test_other_extract_is_rejected(
    artifacts: SyntheticArtifacts, extract: MedsExtract, tmp_path_factory: pytest.TempPathFactory
) -> None:
    from ehrfm.inputs import open_extract
    from ehrfm.toy.artifacts import build_artifacts

    other = build_artifacts(tmp_path_factory.mktemp("other"), seed=8)
    label_set = open_label_set(artifacts.labels, extract)
    with pytest.raises(ValueError, match="another extract"):
        prediction_times(label_set, open_extract(other.extract))


def test_grouping_and_subjects() -> None:
    a = PredictionTime(2, dt.datetime(2020, 1, 2))
    b = PredictionTime(1, dt.datetime(2020, 1, 3))
    c = PredictionTime(2, dt.datetime(2020, 1, 1))
    assert by_subject([a, b, c]) == {1: (b,), 2: (c, a)}
    assert subjects_of([a, b, c]) == (1, 2)


def test_timezone_aware_times_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-naive"):
        PredictionTime(1, dt.datetime(2020, 1, 1, tzinfo=dt.UTC))


def test_epoch_seconds_reads_naive_time_as_utc() -> None:
    assert epoch_seconds(dt.datetime(1970, 1, 2)) == 86_400
    assert epoch_seconds(dt.datetime(1969, 12, 31, 23, 59, 59, 500_000)) == 0
