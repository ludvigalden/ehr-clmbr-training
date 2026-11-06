"""The evaluation toolkit's pipeline hooks.

Importing this module registers these hooks (the experiment pipeline imports
it before every run):

========================  ======  ===============================================
name                      kind    what it does
========================  ======  ===============================================
``logistic_probe``        head    L2 logistic regression on the frozen
                                  representations (strength chosen on tuning
                                  AUROC); saves it as a servable linear head
``age_baseline``          head    the same model on age at prediction time only
``covariate_baseline``    head    the same model on user-named covariate
                                  columns (numeric and one-hot categorical,
                                  optionally age)
``binary``                eval    discrimination, calibration, decision curve
                                  and optional strata for every head and split
``paired``                eval    two heads compared: DeLong, paired bootstrap,
                                  McNemar
``few_shot``              eval    the EHRSHOT few-shot protocol on the
                                  representations
``forgetting``            eval    held-out next-code loss of the base versus
                                  the continued-pretrained checkpoint
========================  ======  ===============================================

**How a head reaches the evaluator.** A head hook writes, for every split, one
positive-class probability per representation row with
:func:`ehrfm.eval.predictions.write_predictions` into its ``output_dir``. The
evaluation hooks read them with :func:`ehrfm.eval.predictions.load_predictions`,
which joins them to the labels of the representation set; a head never
supplies labels. Any head hook that follows this rule is evaluated by
``binary`` and can be compared by ``paired``.

**Covariates.** Stratum columns and baseline features come from a per-subject
table (``subject_id`` plus named columns, CSV or Parquet) given in the options
as ``covariates: {path, sha256}``. In toy mode, when the option is omitted,
the generated toy table (``subject_id,sex,birth_time``) is used.

**Options** are validated strictly (unknown keys fail) and are part of each
hook's cache key; the experiment seed seeds every resampling and draw.

**Counts.** Every record carries its own ``n``, ``n_positive`` and
``n_negative`` (and ``n_true_positive``, ``n_subjects`` ... where relevant), so
the small-cell policy at the tracking boundary masks a record's metrics
whenever one of its counts is small. Full, unmasked results stay in the hook's
``output_dir``.

The ``forgetting`` hook is the one hook that loads models: it needs the base
and the continued checkpoint by construction. It reads them only through
:func:`ehrfm.eval.next_token.compare_next_token_loss`, and its cache key covers
both through the continued model's training fingerprint.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ehrfm.config.experiment import ExperimentConfig
from ehrfm.config.heads import EHRSHOT_C_GRID
from ehrfm.eval.baselines import age_in_years, fit_logistic
from ehrfm.eval.evaluator import EvaluationSettings, compare_binary, evaluate_binary
from ehrfm.eval.fewshot import (
    DEFAULT_REPLICATES,
    Scorer,
    ShotFitter,
    logistic_fitter,
    run_few_shot,
)
from ehrfm.eval.predictions import has_predictions, load_predictions, write_predictions
from ehrfm.eval.stratification import bin_numeric
from ehrfm.eval.tables import SubjectTable, file_sha256, read_subject_table
from ehrfm.experiment.hooks import (
    HookContext,
    HookError,
    HookResult,
    RepresentationIndex,
    register_hook,
)
from ehrfm.experiment.representations import RepresentationSet
from ehrfm.inputs.contract import Split

HOOK_VERSION: Final = "1"
FEW_SHOT_VERSION: Final = "2"
TOY_DIR: Final = "toy"
DATABASE_DIR: Final = "database"
HEAD_DIR: Final = "head"


class HookOptions(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


def parse_options[T: HookOptions](model: type[T], ctx: HookContext) -> T:
    try:
        return model.model_validate_json(json.dumps(dict(ctx.options)))
    except ValueError as exc:
        raise HookError(f"invalid options for {ctx.kind} hook {ctx.name!r}: {exc}") from None


class CovariateSource(HookOptions):
    """A per-subject covariate table pinned by its SHA-256."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _covariates(source: CovariateSource | None, config: ExperimentConfig) -> SubjectTable:
    if source is not None:
        return read_subject_table(Path(source.path), sha256=source.sha256)
    if config.data.toy is None:
        raise HookError("this hook needs options.covariates {path, sha256} outside toy mode")
    from ehrfm.toy.pretraining import toy_input_paths

    path = toy_input_paths(config.experiment.work_dir.resolve() / TOY_DIR).covariates
    return read_subject_table(path, sha256=file_sha256(path))


def boolean_task(index: RepresentationIndex, task: str | None) -> str:
    if task is not None:
        if task not in index.tasks:
            raise HookError(f"no representations for task {task!r} (have {list(index.tasks)})")
        return task
    if len(index.tasks) != 1:
        raise HookError(f"name the task in options.task (have {list(index.tasks)})")
    return index.tasks[0]


def load_boolean(index: RepresentationIndex, task: str, split: Split) -> RepresentationSet:
    loaded = index.load(task, split)
    if loaded.meta.label_type != "boolean" or loaded.labels is None:
        raise HookError(f"task {task!r} is not boolean-labelled")
    return loaded


def _splits(index: RepresentationIndex, wanted: Sequence[Split]) -> tuple[Split, ...]:
    missing = [split for split in wanted if split not in index.splits]
    if missing:
        raise HookError(f"no representations for split(s) {missing} (have {list(index.splits)})")
    return tuple(wanted)


def split_counts(labels: NDArray[np.float64]) -> dict[str, float]:
    positives = float(labels.sum())
    return {"n": float(labels.shape[0]), "n_positive": positives}


# --------------------------------------------------------------------------- heads


class ProbeOptions(HookOptions):
    """Options of the ``logistic_probe`` head."""

    task: str | None = None
    c_grid: tuple[float, ...] = EHRSHOT_C_GRID
    max_iter: int = Field(default=1000, ge=1)


def _fit_and_publish(
    ctx: HookContext,
    task: str,
    features: dict[Split, NDArray[np.float64]],
    sets: dict[Split, RepresentationSet],
    c_grid: Sequence[float],
    max_iter: int,
) -> tuple[HookResult, Any]:
    train, tuning = sets["train"], sets.get("tuning")
    assert train.labels is not None
    fitted = fit_logistic(
        features["train"],
        train.labels,
        features.get("tuning"),
        None if tuning is None else tuning.labels,
        c_grid=c_grid,
        seed=ctx.seed,
        max_iter=max_iter,
    )
    metrics: dict[str, float] = {}
    for split, loaded in sets.items():
        write_predictions(ctx.output_dir, ctx.name, loaded, fitted.model.predict(features[split]))
        assert loaded.labels is not None
        metrics.update(
            {f"{split}/{name}": value for name, value in split_counts(loaded.labels).items()}
        )
    tune_n = 0 if tuning is None else len(tuning)
    tune_positive = 0 if tuning is None or tuning.labels is None else int(tuning.labels.sum())
    # One record per strength, each carrying the tuning counts it was scored on.
    selection: list[JsonValue] = [
        {
            "c": c,
            "n": tune_n,
            "n_positive": tune_positive,
            "auroc": value if np.isfinite(value) else None,
        }
        for c, value in fitted.tuning_auroc.items()
    ]
    summary: dict[str, JsonValue] = {
        "task": task,
        "c": fitted.model.c,
        "features": int(fitted.model.coef.shape[0]),
        "selection": selection,
    }
    return HookResult(metrics=metrics, summary=summary), fitted


def boolean_sets(ctx: HookContext, task: str) -> dict[Split, RepresentationSet]:
    if "train" not in ctx.representations.splits:
        raise HookError("head hooks need representations of the train split")
    return {
        split: load_boolean(ctx.representations, task, split)
        for split in ctx.representations.splits
    }


def logistic_probe(ctx: HookContext) -> HookResult:
    """Logistic-regression probe on the frozen representations; also saved as a linear head."""
    options = parse_options(ProbeOptions, ctx)
    task = boolean_task(ctx.representations, options.task)
    sets = boolean_sets(ctx, task)
    features: dict[Split, NDArray[np.float64]] = {
        split: loaded.embeddings.astype(np.float64) for split, loaded in sets.items()
    }
    result, fitted = _fit_and_publish(ctx, task, features, sets, options.c_grid, options.max_iter)
    from ehrfm.inference.head_loader import save_head

    save_head(
        fitted.model.to_linear_head(task),
        ctx.output_dir / HEAD_DIR,
        encoder_weights_sha256=ctx.model.weights_sha256,
        metadata={"c": float(fitted.model.c), "fitted_by": ctx.name},
    )
    return result


class AgeOptions(HookOptions):
    """Options of the ``age_baseline`` head."""

    task: str | None = None
    covariates: CovariateSource | None = None
    birth_column: str = "birth_time"
    c_grid: tuple[float, ...] = EHRSHOT_C_GRID
    max_iter: int = Field(default=1000, ge=1)


def _age(table: SubjectTable, column: str, loaded: RepresentationSet) -> NDArray[np.float64]:
    birth = table.epoch_seconds(column, loaded.subject_ids)
    return age_in_years(loaded.times, birth)[:, np.newaxis]


def age_baseline(ctx: HookContext) -> HookResult:
    """Logistic regression on age at the prediction time only."""
    options = parse_options(AgeOptions, ctx)
    task = boolean_task(ctx.representations, options.task)
    sets = boolean_sets(ctx, task)
    table = _covariates(options.covariates, ctx.config)
    features: dict[Split, NDArray[np.float64]] = {
        split: _age(table, options.birth_column, loaded) for split, loaded in sets.items()
    }
    result, _ = _fit_and_publish(ctx, task, features, sets, options.c_grid, options.max_iter)
    return result


class CovariateOptions(HookOptions):
    """Options of the ``covariate_baseline`` head."""

    task: str | None = None
    covariates: CovariateSource | None = None
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    """One-hot encoded over the levels seen in the train split (the first level is dropped)."""
    age_from: str | None = None
    """A birth-time column to add age at the prediction time as a feature."""
    c_grid: tuple[float, ...] = EHRSHOT_C_GRID
    max_iter: int = Field(default=1000, ge=1)


def covariate_baseline(ctx: HookContext) -> HookResult:
    """Logistic regression on hand-picked covariates (numeric, one-hot categorical, age)."""
    options = parse_options(CovariateOptions, ctx)
    if not (options.numeric or options.categorical or options.age_from):
        raise HookError("covariate_baseline needs numeric, categorical or age_from features")
    task = boolean_task(ctx.representations, options.task)
    sets = boolean_sets(ctx, task)
    table = _covariates(options.covariates, ctx.config)
    levels: dict[str, list[str]] = {}
    for name in options.categorical:
        seen = sorted(set(table.strings([name], sets["train"].subject_ids)[name].tolist()))
        levels[name] = seen[1:]
    features: dict[Split, NDArray[np.float64]] = {}
    for split, loaded in sets.items():
        blocks: list[NDArray[np.float64]] = []
        if options.numeric:
            blocks.append(table.numbers(options.numeric, loaded.subject_ids))
        for name in options.categorical:
            values = table.strings([name], loaded.subject_ids)[name]
            blocks.append(
                np.stack([(values == level).astype(np.float64) for level in levels[name]], axis=1)
                if levels[name]
                else np.empty((len(loaded), 0), dtype=np.float64)
            )
        if options.age_from is not None:
            blocks.append(_age(table, options.age_from, loaded))
        features[split] = np.concatenate(blocks, axis=1)
    if features["train"].shape[1] == 0:
        raise HookError("the chosen covariates give no features (one level per categorical)")
    result, _ = _fit_and_publish(ctx, task, features, sets, options.c_grid, options.max_iter)
    return result


# --------------------------------------------------------------------------- evaluation


class StrataOptions(HookOptions):
    """Stratum columns from a covariate table."""

    covariates: CovariateSource | None = None
    columns: tuple[str, ...] = ()
    """Categorical columns used as they are."""
    bins: dict[str, tuple[float, ...]] = Field(default_factory=dict[str, tuple[float, ...]])
    """Numeric columns cut into ``[lo, hi)`` intervals over the given edges."""
    age_bins: tuple[float, ...] | None = None
    """Age at the prediction time (years), cut over these edges."""
    birth_column: str = "birth_time"


def _strata(
    options: StrataOptions, config: ExperimentConfig, loaded: RepresentationSet
) -> dict[str, NDArray[np.str_]]:
    table = _covariates(options.covariates, config)
    columns: dict[str, NDArray[np.str_]] = dict(table.strings(options.columns, loaded.subject_ids))
    for name, edges in options.bins.items():
        numbers = table.numbers([name], loaded.subject_ids)[:, 0]
        columns[name] = bin_numeric(numbers, edges)
    if options.age_bins is not None:
        ages = _age(table, options.birth_column, loaded)[:, 0]
        columns["age"] = bin_numeric(ages, options.age_bins)
    return columns


class BinaryOptions(HookOptions):
    """Options of the ``binary`` evaluation hook."""

    task: str | None = None
    heads: tuple[str, ...] | None = None
    """Heads to evaluate; all heads that published predictions when omitted."""
    splits: tuple[Split, ...] = ("held_out",)
    settings: EvaluationSettings = EvaluationSettings()
    strata: StrataOptions | None = None


def _write_json(path: Path, payload: JsonValue) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def binary(ctx: HookContext) -> HookResult:
    """Evaluate every head's probabilities on each requested split."""
    options = parse_options(BinaryOptions, ctx)
    task = boolean_task(ctx.representations, options.task)
    splits = _splits(ctx.representations, options.splits)
    names = options.heads if options.heads is not None else tuple(sorted(ctx.heads))
    metrics: dict[str, float] = {}
    results: dict[str, JsonValue] = {}
    for head in names:
        record = ctx.heads.get(head)
        if record is None:
            raise HookError(f"no head named {head!r} ran in this experiment")
        if options.heads is None and not has_predictions(record.output_dir, task, splits[0]):
            continue
        per_split: dict[str, JsonValue] = {}
        for split in splits:
            loaded = load_boolean(ctx.representations, task, split)
            predictions = load_predictions(record.output_dir, loaded)
            strata = None if options.strata is None else _strata(options.strata, ctx.config, loaded)
            evaluation = evaluate_binary(
                predictions.y_true,
                predictions.y_prob,
                seed=ctx.seed,
                settings=options.settings,
                strata=strata,
            )
            per_split[split] = evaluation.record()
            metrics.update(evaluation.flat_metrics(f"{head}/{split}"))
            _write_json(ctx.output_dir / f"{head}.{split}.json", evaluation.record())
            if evaluation.bootstrap:
                np.savez(ctx.output_dir / f"{head}.{split}.bootstrap.npz", **evaluation.bootstrap)
        results[head] = per_split
    if not results:
        raise HookError(f"no head published predictions for task {task!r}")
    summary: dict[str, JsonValue] = {
        "task": task,
        "settings": options.settings.model_dump(mode="json"),
        "results": results,
    }
    return HookResult(metrics=metrics, summary=summary)


class PairedOptions(HookOptions):
    """Options of the ``paired`` evaluation hook."""

    a: str
    b: str
    task: str | None = None
    splits: tuple[Split, ...] = ("held_out",)
    threshold: float = Field(default=0.5, gt=0, lt=1)
    """Decision threshold for McNemar's test."""
    settings: EvaluationSettings = EvaluationSettings()


def paired(ctx: HookContext) -> HookResult:
    """Compare head ``a`` with head ``b`` on the same rows (differences are ``a - b``)."""
    options = parse_options(PairedOptions, ctx)
    if options.a == options.b:
        raise HookError("paired needs two different heads")
    task = boolean_task(ctx.representations, options.task)
    splits = _splits(ctx.representations, options.splits)
    for head in (options.a, options.b):
        if head not in ctx.heads:
            raise HookError(f"no head named {head!r} ran in this experiment")
    metrics: dict[str, float] = {}
    results: dict[str, JsonValue] = {}
    for split in splits:
        loaded = load_boolean(ctx.representations, task, split)
        a = load_predictions(ctx.heads[options.a].output_dir, loaded)
        b = load_predictions(ctx.heads[options.b].output_dir, loaded)
        comparison = compare_binary(
            a.y_true,
            a.y_prob,
            b.y_prob,
            seed=ctx.seed,
            threshold=options.threshold,
            settings=options.settings,
        )
        results[split] = comparison.record()
        metrics.update(comparison.flat_metrics(split))
        _write_json(ctx.output_dir / f"{split}.json", comparison.record())
        np.save(ctx.output_dir / f"{split}.difference.npy", comparison.difference_distribution)
    return HookResult(
        metrics=metrics,
        summary={"task": task, "a": options.a, "b": options.b, "results": results},
    )


class FewShotOptions(HookOptions):
    """Options of the ``few_shot`` evaluation hook."""

    task: str | None = None
    shots: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    replicates: int = Field(default=DEFAULT_REPLICATES, ge=1)
    c_grid: tuple[float, ...] = EHRSHOT_C_GRID
    probe: Literal["logistic_regression", "linear_head"] = "logistic_regression"
    """``logistic_regression`` fits scikit-learn; ``linear_head`` fits the servable
    linear head of :func:`ehrfm.heads.probe.fit_linear_probe` for every draw."""
    class_weight: Literal["none", "balanced"] = "none"
    """Class weighting of ``linear_head`` probes (draws are balanced already)."""


def _few_shot_fitter(options: FewShotOptions, task: str, seed: int) -> ShotFitter:
    if options.probe == "logistic_regression":
        return logistic_fitter(options.c_grid, seed)
    from ehrfm.config.heads import LinearProbeConfig
    from ehrfm.heads.probe import fit_linear_probe

    config = LinearProbeConfig(c_grid=options.c_grid, class_weight=options.class_weight)

    def fit(
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_tune: NDArray[np.float64],
        y_tune: NDArray[np.float64],
    ) -> tuple[Scorer, float]:
        fitted = fit_linear_probe(task, x_train, y_train, x_tune, y_tune, config=config)
        return fitted.predict, fitted.c

    return fit


def few_shot(ctx: HookContext) -> HookResult:
    """The EHRSHOT few-shot protocol (Wornow et al., 2023) on the frozen representations."""
    options = parse_options(FewShotOptions, ctx)
    task = boolean_task(ctx.representations, options.task)
    _splits(ctx.representations, ("train", "tuning", "held_out"))
    sets = {
        split: load_boolean(ctx.representations, task, split)
        for split in ("train", "tuning", "held_out")
    }

    def xy(split: Split) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        loaded = sets[split]
        assert loaded.labels is not None
        return loaded.embeddings.astype(np.float64), loaded.labels

    result = run_few_shot(
        xy("train"),
        xy("tuning"),
        xy("held_out"),
        shots=options.shots,
        replicates=options.replicates,
        seed=ctx.seed,
        fitter=_few_shot_fitter(options, task, ctx.seed),
    )
    n_negative = result.n_held_out - result.n_held_out_positive
    # Each row repeats the held-out counts its scores were computed on, so a
    # small held-out cell masks the row. Shot counts are protocol settings.
    rows: list[JsonValue] = []
    metrics: dict[str, float] = {}
    for summary in result.summaries:
        row: dict[str, JsonValue] = {
            "shots": summary.k,
            "replicates": summary.replicates,
            "n_held_out": result.n_held_out,
            "n_held_out_positive": result.n_held_out_positive,
            "n_held_out_negative": n_negative,
        }
        for name in ("auroc_mean", "auroc_std", "auprc_mean", "auprc_std"):
            value = float(getattr(summary, name))
            row[name] = value if np.isfinite(value) else None
            if np.isfinite(value):
                metrics[f"k{summary.k}/{name}"] = value
        metrics[f"k{summary.k}/n_held_out"] = float(result.n_held_out)
        metrics[f"k{summary.k}/n_held_out_positive"] = float(result.n_held_out_positive)
        metrics[f"k{summary.k}/n_held_out_negative"] = float(n_negative)
        rows.append(row)
    runs: list[JsonValue] = [
        {
            "shots": run.k,
            "replicate": run.replicate,
            "c": run.c,
            "auroc": run.auroc if np.isfinite(run.auroc) else None,
            "auprc": run.auprc if np.isfinite(run.auprc) else None,
        }
        for run in result.runs
    ]
    _write_json(ctx.output_dir / "runs.json", runs)
    return HookResult(
        metrics=metrics,
        summary={
            "task": task,
            "protocol": "EHRSHOT (Wornow et al., 2023)",
            "probe": options.probe,
            "curve": rows,
        },
    )


class ForgettingOptions(HookOptions):
    """Options of the ``forgetting`` evaluation hook."""

    splits: tuple[Split, ...] = ("held_out",)
    tokens_per_batch: int | None = Field(default=None, ge=1)
    """Packing budget; the representation stage's when omitted."""


def forgetting(ctx: HookContext) -> HookResult:
    """Held-out next-code loss of the base checkpoint versus the continued-pretrained one."""
    options = parse_options(ForgettingOptions, ctx)
    config = ctx.config
    if ctx.model.training_fingerprint is None:
        return HookResult(
            summary={"applicable": False, "reason": "no continued pretraining in this experiment"}
        )
    from ehrfm.data.splits import extract_split
    from ehrfm.eval.next_token import compare_next_token_loss
    from ehrfm.inputs import open_extract

    work_dir = config.experiment.work_dir.resolve()
    if config.data.toy is not None:
        from ehrfm.toy.pretraining import toy_input_paths

        paths = toy_input_paths(work_dir / TOY_DIR)
        base, extract_root = paths.checkpoint, paths.extract
    else:
        assert config.model.base_checkpoint is not None and config.data.extract is not None
        base, extract_root = config.model.base_checkpoint, config.data.extract
    extract = open_extract(extract_root)
    tokens = options.tokens_per_batch or config.experiment.representations.tokens_per_batch
    if not options.splits:
        raise HookError("forgetting needs at least one split")
    metrics: dict[str, float] = {}
    results: dict[str, JsonValue] = {}
    weights: dict[str, JsonValue] = {}
    for split in options.splits:
        subjects = extract_split(extract, split)
        comparison = compare_next_token_loss(
            base,
            ctx.model.model_dir,
            extract,
            subjects,
            database_cache=work_dir / DATABASE_DIR,
            tokens_per_batch=tokens,
            device=config.model.device,
        )
        record: dict[str, float] = {
            "n_subjects": float(comparison.base.n_subjects),
            "base_loss": comparison.base.loss,
            "continued_loss": comparison.continued.loss,
            "loss_increase": comparison.loss_increase,
        }
        results[split] = dict(record)
        metrics.update({f"{split}/{name}": value for name, value in record.items()})
        weights = {
            "base_weights_sha256": comparison.base.model_weights_sha256,
            "continued_weights_sha256": comparison.continued.model_weights_sha256,
        }
    return HookResult(metrics=metrics, summary={"applicable": True, **weights, "results": results})


HEAD_HOOKS: Final = {
    "logistic_probe": logistic_probe,
    "age_baseline": age_baseline,
    "covariate_baseline": covariate_baseline,
}
EVAL_HOOKS: Final = {
    "binary": binary,
    "paired": paired,
    "few_shot": few_shot,
    "forgetting": forgetting,
}

for _name, _hook in HEAD_HOOKS.items():
    register_hook("head", _name, _hook, version=HOOK_VERSION)
for _name, _hook in EVAL_HOOKS.items():
    version = FEW_SHOT_VERSION if _name == "few_shot" else HOOK_VERSION
    register_hook("eval", _name, _hook, version=version)


__all__ = [
    "EVAL_HOOKS",
    "HEAD_DIR",
    "HEAD_HOOKS",
    "HookOptions",
    "AgeOptions",
    "BinaryOptions",
    "CovariateOptions",
    "CovariateSource",
    "FewShotOptions",
    "ForgettingOptions",
    "PairedOptions",
    "ProbeOptions",
    "StrataOptions",
    "age_baseline",
    "binary",
    "boolean_sets",
    "boolean_task",
    "covariate_baseline",
    "few_shot",
    "forgetting",
    "load_boolean",
    "logistic_probe",
    "paired",
    "parse_options",
    "split_counts",
]
