"""Predictions contract, stratification, covariate tables, baselines and few-shot."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from ehrfm.config.heads import EHRSHOT_C_GRID, LinearProbeConfig
from ehrfm.eval.baselines import BaselineError, age_in_years, fit_logistic
from ehrfm.eval.fewshot import FewShotError, draw_shots, run_few_shot
from ehrfm.eval.hooks import AgeOptions, CovariateOptions, FewShotOptions, ProbeOptions
from ehrfm.eval.predictions import (
    PredictionError,
    has_predictions,
    load_predictions,
    write_predictions,
)
from ehrfm.eval.stratification import MISSING, StratificationError, bin_numeric, stratify
from ehrfm.eval.tables import TableError, file_sha256, read_subject_table
from ehrfm.experiment.representations import RepresentationMeta, RepresentationSet

SHA = "0" * 64


def _set(
    n: int, *, labels: NDArray[np.float64] | None = None, split: str = "held_out"
) -> RepresentationSet:
    rng = np.random.default_rng(n)
    return RepresentationSet(
        meta=RepresentationMeta(
            task="t",
            split=split,  # pyright: ignore[reportArgumentType]
            n=n,
            hidden_size=3,
            label_type="boolean",
            model_weights_sha256=SHA,
        ),
        subject_ids=np.arange(1, n + 1, dtype=np.int64),
        times=np.arange(n, dtype=np.int64) * 86400,
        embeddings=rng.normal(size=(n, 3)).astype(np.float32),
        labels=labels if labels is not None else (np.arange(n) % 2).astype(np.float64),
    )


# --------------------------------------------------------------------------- predictions


def test_predictions_round_trip_and_take_labels_from_the_set(tmp_path: Path) -> None:
    rows = _set(6)
    probs = np.linspace(0.1, 0.9, 6)
    write_predictions(tmp_path, "probe", rows, probs)
    assert has_predictions(tmp_path, "t", "held_out")
    assert not has_predictions(tmp_path, "t", "train")
    loaded = load_predictions(tmp_path, rows)
    assert loaded.head == "probe" and len(loaded) == 6
    np.testing.assert_array_equal(loaded.y_prob, probs)
    np.testing.assert_array_equal(loaded.y_true, rows.labels)  # pyright: ignore[reportArgumentType]


def test_predictions_for_other_rows_are_rejected(tmp_path: Path) -> None:
    write_predictions(tmp_path, "probe", _set(6), np.full(6, 0.5))
    reordered = _set(6)
    reordered = RepresentationSet(
        reordered.meta,
        reordered.subject_ids[::-1].copy(),
        reordered.times,
        reordered.embeddings,
        reordered.labels,
    )
    with pytest.raises(PredictionError, match="other rows"):
        load_predictions(tmp_path, reordered)


@pytest.mark.parametrize(
    "probs", [np.full(5, 0.5), np.array([0.1, 0.2, 1.2, 0.1, 0.1, 0.1]), np.full(6, np.nan)]
)
def test_invalid_predictions_are_refused(tmp_path: Path, probs: NDArray[np.float64]) -> None:
    with pytest.raises(PredictionError):
        write_predictions(tmp_path, "probe", _set(6), probs)


def test_missing_predictions_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(PredictionError, match="no predictions"):
        load_predictions(tmp_path, _set(4))


# --------------------------------------------------------------------------- stratification


def test_bin_numeric_labels_half_open_intervals() -> None:
    labels = bin_numeric([17.0, 18.0, 49.9, 50.0, 120.0, None], [18, 50, 120])
    assert labels.tolist() == [MISSING, "[18, 50)", "[18, 50)", "[50, 120)", MISSING, MISSING]
    with pytest.raises(StratificationError):
        bin_numeric([1.0], [5, 5])


def test_stratify_counts_every_stratum_of_every_column() -> None:
    y = np.array([1, 0, 1, 0, 1, 1], dtype=np.float64)
    p = np.array([0.9, 0.1, 0.8, 0.3, 0.2, 0.7])
    columns = {"sex": ["F", "F", "M", "M", "M", "F"], "site": ["a"] * 6}
    results = stratify(y, p, columns, lambda yt, _: {"mean": float(yt.mean())})
    by_key = {(r.column, r.value): r for r in results}
    assert set(by_key) == {("sex", "F"), ("sex", "M"), ("site", "a")}
    female = by_key[("sex", "F")]
    assert (female.n, female.n_positive, female.n_negative) == (3, 2, 1)
    assert female.metrics["mean"] == pytest.approx(2 / 3)
    record = by_key[("site", "a")].to_record()
    assert record["n"] == 6 and record["n_positive"] == 4
    with pytest.raises(StratificationError, match="has 2 values"):
        stratify(y, p, {"bad": ["x", "y"]}, lambda yt, _: {})


# --------------------------------------------------------------------------- tables


def test_subject_table_is_pinned_and_typed(tmp_path: Path) -> None:
    path = tmp_path / "cov.csv"
    path.write_text("subject_id,sex,score,birth_time\n1,F,2.5,2000-01-01\n2,,3,946684800\n")
    sha = file_sha256(path)
    table = read_subject_table(path, sha256=sha)
    ids = np.array([2, 1, 3], dtype=np.int64)
    assert table.strings(["sex"], ids)["sex"].tolist() == [MISSING, "F", MISSING]
    np.testing.assert_array_equal(
        table.epoch_seconds("birth_time", ids[:2]), [946684800.0, 946684800.0]
    )
    np.testing.assert_array_equal(table.numbers(["score"], ids[:2])[:, 0], [3.0, 2.5])
    with pytest.raises(TableError, match="no row"):
        table.numbers(["score"], ids)
    with pytest.raises(TableError, match="no column"):
        table.strings(["absent"], ids)
    with pytest.raises(TableError, match="SHA-256"):
        read_subject_table(path, sha256="f" * 64)


def test_subject_table_rejects_repeated_subjects(tmp_path: Path) -> None:
    path = tmp_path / "cov.csv"
    path.write_text("subject_id,sex\n1,F\n1,M\n")
    with pytest.raises(TableError, match="repeats"):
        read_subject_table(path, sha256=file_sha256(path))


# --------------------------------------------------------------------------- baselines


def _separable(n: int, seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(np.float64)
    x = rng.normal(size=(n, 4)) + y[:, np.newaxis] * np.array([2.0, 0.0, 0.0, 1.0])
    return x * np.array([1.0, 10.0, 0.1, 1.0]) + 5.0, y


def test_fit_logistic_selects_on_tuning_and_is_deterministic() -> None:
    x, y = _separable(80, 1)
    xt, yt = _separable(40, 2)
    first = fit_logistic(x, y, xt, yt, seed=3)
    second = fit_logistic(x, y, xt, yt, seed=3)
    assert first.model.c == second.model.c
    np.testing.assert_array_equal(first.model.coef, second.model.coef)
    assert set(first.tuning_auroc) == set(EHRSHOT_C_GRID)
    assert first.tuning_auroc[first.model.c] == max(first.tuning_auroc.values())


def test_default_c_grid_is_the_ehrshot_grid_everywhere() -> None:
    """Every logistic probe consumer defaults to the same 8-value EHRSHOT grid."""
    assert EHRSHOT_C_GRID == (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 1e2, 1e4)
    assert LinearProbeConfig().c_grid == EHRSHOT_C_GRID
    assert inspect.signature(fit_logistic).parameters["c_grid"].default == EHRSHOT_C_GRID
    assert inspect.signature(run_few_shot).parameters["c_grid"].default == EHRSHOT_C_GRID
    for options in (ProbeOptions, AgeOptions, CovariateOptions, FewShotOptions):
        assert options.model_fields["c_grid"].default == EHRSHOT_C_GRID, options.__name__


def test_fit_logistic_without_usable_tuning_uses_the_middle_strength() -> None:
    x, y = _separable(40, 1)
    fitted = fit_logistic(x, y, x[:3], np.zeros(3), c_grid=(0.1, 1.0, 10.0))
    assert fitted.model.c == 1.0 and fitted.tuning_auroc == {}
    with pytest.raises(BaselineError, match="both classes"):
        fit_logistic(x, np.zeros(40))
    with pytest.raises(BaselineError, match="align"):
        fit_logistic(x, y[:-1])


def test_linear_head_reproduces_the_probabilities() -> None:
    x, y = _separable(60, 4)
    model = fit_logistic(x, y).model
    head = model.to_linear_head("demo_task")
    with torch.no_grad():
        from_head = head.predict(torch.from_numpy(x.astype(np.float32))).numpy()
    np.testing.assert_allclose(from_head, model.predict(x), atol=1e-5)


def test_age_in_years() -> None:
    birth = np.array([0.0])
    times = np.array([int(365.25 * 24 * 3600 * 40)], dtype=np.int64)
    assert age_in_years(times, birth)[0] == pytest.approx(40.0)


# --------------------------------------------------------------------------- few-shot


def test_draws_are_balanced_disjoint_seeded_and_order_free() -> None:
    labels = (np.arange(40) % 2).astype(np.float64)
    draw = draw_shots(labels, labels, 4, 0, seed=1)
    assert draw is not None
    assert labels[draw.train_rows].sum() == 4 and draw.train_rows.shape == (8,)
    assert labels[draw.tune_rows].sum() == 4
    again = draw_shots(labels, labels, 4, 0, seed=1)
    assert again is not None
    np.testing.assert_array_equal(draw.train_rows, again.train_rows)
    other = draw_shots(labels, labels, 4, 1, seed=1)
    assert other is not None and not np.array_equal(draw.train_rows, other.train_rows)


def test_too_few_examples_skip_the_shot_count() -> None:
    labels = np.array([1, 0, 0, 0], dtype=np.float64)
    assert draw_shots(labels, labels, 2, 0, seed=0) is None
    tune_short = draw_shots(np.tile(labels, 3), labels, 2, 0, seed=0)
    assert tune_short is not None and tune_short.tune_rows.shape == (0,)
    with pytest.raises(FewShotError):
        draw_shots(labels, labels, 0, 0, seed=0)


def test_run_few_shot_curve() -> None:
    train, tune, test = _separable(200, 1), _separable(60, 2), _separable(100, 3)
    result = run_few_shot(train, tune, test, shots=(8, 2, 500), replicates=3, seed=5)
    assert [s.k for s in result.summaries] == [2, 8, 500]
    assert [s.replicates for s in result.summaries] == [3, 3, 0]
    assert len(result.runs) == 6 and result.n_held_out == 100 and result.n_held_out_positive == 50
    assert result.summaries[1].auroc_mean > 0.8
    assert np.isnan(result.summaries[2].auroc_mean)
    repeat = run_few_shot(train, tune, test, shots=(2, 8, 500), replicates=3, seed=5)
    assert repeat.runs == result.runs
    with pytest.raises(FewShotError):
        run_few_shot(train, tune, test, shots=(2, 2), seed=0)
