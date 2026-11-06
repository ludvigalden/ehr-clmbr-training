"""The experiment pipeline: verified inputs to representations, heads and evaluation.

:func:`run_experiment` runs these stages in order:

``inputs``
    Generate the toy inputs (toy mode), then verify the extract, the label
    sets and the base checkpoint and hash them. Always runs: it produces the
    hashes every other stage key is built from.
``coverage``
    Dictionary coverage of the extract by the base checkpoint
    (:func:`ehrfm.data.coverage.extract_coverage`).
``pretraining`` (optional)
    Continued pretraining through :func:`ehrfm.training.runner.run_pretraining`.
    The stage key is the training run fingerprint, so the stage and the
    training cache agree on what counts as the same run.
``representations``
    One embedding per prediction time of each label set (or per subject when
    there are none), per split, from the pretrained model or the base
    checkpoint.
``head`` / ``eval`` hooks
    The registered hooks named in ``tasks.heads`` and ``eval.hooks`` (see
    :mod:`ehrfm.experiment.hooks`).

Every stage after ``inputs`` is cached (:mod:`ehrfm.experiment.cache`); its key
chains the keys of what it consumes, so a changed input recomputes exactly the
stages downstream of it. Tracking runs last and is cached too: an MLflow run
tagged with the same pipeline key is reused, so rerunning an unchanged
experiment computes nothing and logs nothing.
"""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from pydantic import JsonValue

from ehrfm.config.experiment import ExperimentConfig, HookSpec
from ehrfm.config.training import RuntimeConfig, TrainingConfig, TrainingInputs
from ehrfm.data.coverage import extract_coverage
from ehrfm.experiment import cache
from ehrfm.experiment.cache import StageRecord
from ehrfm.experiment.hooks import (
    HookContext,
    HookKind,
    HookRecord,
    HookResult,
    ModelIdentity,
    RepresentationIndex,
    get_hook,
)
from ehrfm.experiment.representations import (
    SUBJECTS_TASK,
    RepresentationMeta,
    encode_labels,
    save_set,
)
from ehrfm.experiment.tracking import TrackedRun, open_tracker
from ehrfm.inputs import LabelSet, MedsExtract, open_extract, open_label_set
from ehrfm.inputs.manifest import sha256_file
from ehrfm.models.checkpoint import CONFIG_FILE, DICTIONARY_FILE, WEIGHTS_FILE, require_layout
from ehrfm.training.surgery import require_concept_set_match

COVERAGE_LOGIC: Final = 1
REPRESENTATION_LOGIC: Final = 2
PIPELINE_FORMAT: Final = 1
DATABASE_DIR: Final = "database"
PRETRAINING_DIR: Final = "pretraining"
TOY_DIR: Final = "toy"
COVERAGE_FILE: Final = "coverage.json"
HOOK_RESULT_FILE: Final = "result.json"
HOOK_OUTPUT_DIR: Final = "output"

StageStatus = Literal["verified", "computed", "cached", "skipped"]
StopRequest = Callable[[int], bool]


class PipelineInterrupted(RuntimeError):
    """Pretraining stopped on request; rerunning resumes it."""


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """How one stage ended in this invocation."""

    name: str
    status: StageStatus
    key: str | None = None
    path: Path | None = None
    duration_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class VerifiedInputs:
    """The verified, hashed inputs of an experiment."""

    extract: MedsExtract
    label_sets: tuple[tuple[str, LabelSet], ...]
    base_checkpoint: Path
    base_weights_sha256: str
    base_dictionary_sha256: str
    base_config_sha256: str

    def key_inputs(self) -> dict[str, JsonValue]:
        """Content hashes that identify the inputs."""
        return {
            "extract_manifest_sha256": self.extract.manifest_sha256,
            "label_sets": {name: labels.manifest_sha256 for name, labels in self.label_sets},
            "base_weights_sha256": self.base_weights_sha256,
            "base_dictionary_sha256": self.base_dictionary_sha256,
            "base_config_sha256": self.base_config_sha256,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Everything one :func:`run_experiment` call produced or reused."""

    config: ExperimentConfig
    work_dir: Path
    pipeline_key: str
    stages: tuple[StageOutcome, ...]
    model: ModelIdentity
    representations: RepresentationIndex
    heads: dict[str, HookRecord]
    evaluations: dict[str, HookRecord]
    tracking_uri: str | None
    tracking_run_id: str | None
    tracking_status: Literal["logged", "cached", "disabled"]
    duration_seconds: float
    records: dict[str, StageRecord] = field(default_factory=dict[str, StageRecord])

    @property
    def cache_hit(self) -> bool:
        """Whether nothing was computed or logged."""
        return self.tracking_status != "logged" and all(
            stage.status in ("verified", "cached", "skipped") for stage in self.stages
        )


def _timed[T](function: Callable[[], T]) -> tuple[T, float]:
    started = time.monotonic()
    value = function()
    return value, time.monotonic() - started


# --------------------------------------------------------------------------- inputs


def verify_inputs(config: ExperimentConfig, work_dir: Path) -> VerifiedInputs:
    """Generate (toy mode) and verify the extract, label sets and base checkpoint."""
    data = config.data
    if data.toy is not None:
        from ehrfm.toy.pretraining import write_toy_experiment_inputs

        toy = write_toy_experiment_inputs(
            work_dir / TOY_DIR, subjects=data.toy.subjects, seed=data.toy.seed
        )
        extract = open_extract(toy.extract)
        label_sets = ((toy.labels.name, open_label_set(toy.labels, extract)),)
        base = toy.checkpoint
    else:
        assert data.extract is not None and config.model.base_checkpoint is not None
        extract = open_extract(data.extract)
        label_sets = tuple(
            (task.name, open_label_set(task.labels, extract)) for task in config.tasks.label_sets
        )
        base = config.model.base_checkpoint
    require_layout(base)
    weights = sha256_file(base / WEIGHTS_FILE)
    pin = config.model.expected_weights_sha256
    if pin is not None and weights != pin:
        raise ValueError(f"{base / WEIGHTS_FILE} has SHA-256 {weights}, configured pin is {pin}")
    if data.concept_set is not None:
        require_concept_set_match(data.concept_set, base)
    return VerifiedInputs(
        extract=extract,
        label_sets=label_sets,
        base_checkpoint=base,
        base_weights_sha256=weights,
        base_dictionary_sha256=sha256_file(base / DICTIONARY_FILE),
        base_config_sha256=sha256_file(base / CONFIG_FILE),
    )


def inputs_summary(inputs: VerifiedInputs) -> dict[str, JsonValue]:
    """Recorded counts of the inputs (logged through suppression)."""
    return {
        "extract": inputs.extract.summary.model_dump(mode="json"),
        "label_sets": {
            name: labels.summary.model_dump(mode="json") for name, labels in inputs.label_sets
        },
    }


# --------------------------------------------------------------------------- coverage


def coverage_stage(
    config: ExperimentConfig, inputs: VerifiedInputs, work_dir: Path
) -> tuple[StageRecord, StageOutcome]:
    """Measure dictionary coverage of the extract (cached on the extract and dictionary)."""
    settings = config.experiment.coverage
    key_inputs: dict[str, JsonValue] = {
        "logic": COVERAGE_LOGIC,
        "extract_manifest_sha256": inputs.extract.manifest_sha256,
        "base_dictionary_sha256": inputs.base_dictionary_sha256,
        "top_n": settings.top_n,
        "min_events": settings.min_events,
    }

    def compute(directory: Path) -> dict[str, JsonValue]:
        report = extract_coverage(
            inputs.extract,
            inputs.base_checkpoint,
            top_n=settings.top_n,
            min_events=settings.min_events,
        )
        (directory / COVERAGE_FILE).write_text(report.model_dump_json(indent=2) + "\n")
        return {"report": COVERAGE_FILE}

    (record, status, path), took = _timed(
        lambda: cache.run_stage(work_dir, "coverage", key_inputs, compute)
    )
    return record, StageOutcome("coverage", status, record.key, path, took)


# --------------------------------------------------------------------------- pretraining


def training_config(
    config: ExperimentConfig, inputs: VerifiedInputs, work_dir: Path
) -> TrainingConfig:
    """The continued-pretraining run of the experiment."""
    stage = config.experiment.pretraining
    return TrainingConfig(
        inputs=TrainingInputs(
            extract=inputs.extract.root,
            base_checkpoint=inputs.base_checkpoint,
            expected_weights_sha256=inputs.base_weights_sha256,
        ),
        optimization=stage.optimization,
        runtime=RuntimeConfig(
            output_dir=work_dir / PRETRAINING_DIR,
            device=config.model.device,
            database_cache=work_dir / DATABASE_DIR,
            checkpoint_every=stage.checkpoint_every,
            keep_checkpoints=stage.keep_checkpoints,
            log_every=stage.log_every,
            tokenize_workers=stage.tokenize_workers,
            deterministic=stage.deterministic,
        ),
    )


def _published_model_matches(record: StageRecord) -> bool:
    model_dir = record.outputs.get("model_dir")
    sha = record.outputs.get("model_weights_sha256")
    if not isinstance(model_dir, str) or not isinstance(sha, str):
        return False
    weights = Path(model_dir) / WEIGHTS_FILE
    return weights.is_file() and sha256_file(weights) == sha


def pretraining_stage(
    config: ExperimentConfig,
    inputs: VerifiedInputs,
    work_dir: Path,
    *,
    should_stop: StopRequest | None = None,
) -> tuple[StageRecord, StageOutcome]:
    """Continue pretraining (cached on the training run fingerprint)."""
    from ehrfm.training.cache import fingerprint_inputs
    from ehrfm.training.runner import run_pretraining

    training = training_config(config, inputs, work_dir)
    fingerprint = fingerprint_inputs(
        training,
        extract_manifest_sha256=inputs.extract.manifest_sha256,
        base_weights_sha256=inputs.base_weights_sha256,
    ).digest()
    key_inputs: dict[str, JsonValue] = {"training_fingerprint": fingerprint}

    def compute(directory: Path) -> dict[str, JsonValue]:
        result = run_pretraining(training, should_stop=should_stop)
        if result.status == "interrupted":
            raise PipelineInterrupted(
                f"pretraining stopped at step {result.step} of {result.total_steps}; rerun to resume"
            )
        if result.fingerprint != fingerprint:
            raise RuntimeError("the training run fingerprint changed while the stage ran")
        return {
            "model_dir": str(result.model_dir.resolve()),
            "model_weights_sha256": result.model_weights_sha256,
            "training_fingerprint": result.fingerprint,
            "total_steps": result.total_steps,
            "train_losses": list(result.train_losses),
            "validation_loss": result.validation_loss,
        }

    (record, status, path), took = _timed(
        lambda: cache.run_stage(
            work_dir, "pretraining", key_inputs, compute, valid=_published_model_matches
        )
    )
    return record, StageOutcome("pretraining", status, record.key, path, took)


def model_identity(inputs: VerifiedInputs, pretrained: StageRecord | None) -> ModelIdentity:
    """The model downstream stages use: the pretrained one, else the base checkpoint."""
    if pretrained is None:
        return ModelIdentity(inputs.base_checkpoint, inputs.base_weights_sha256, None)
    model_dir = pretrained.outputs["model_dir"]
    sha = pretrained.outputs["model_weights_sha256"]
    fingerprint = pretrained.outputs["training_fingerprint"]
    assert isinstance(model_dir, str) and isinstance(sha, str) and isinstance(fingerprint, str)
    return ModelIdentity(Path(model_dir), sha, fingerprint)


# --------------------------------------------------------------------------- representations


def representation_stage(
    config: ExperimentConfig,
    inputs: VerifiedInputs,
    model: ModelIdentity,
    work_dir: Path,
) -> tuple[StageRecord, StageOutcome, RepresentationIndex]:
    """Extract representations per task and split (cached on model and inputs)."""
    settings = config.experiment.representations
    tasks = tuple(name for name, _ in inputs.label_sets) or (SUBJECTS_TASK,)
    key_inputs: dict[str, JsonValue] = {
        "logic": REPRESENTATION_LOGIC,
        "model_weights_sha256": model.weights_sha256,
        "training_fingerprint": model.training_fingerprint,
        "extract_manifest_sha256": inputs.extract.manifest_sha256,
        "label_sets": {name: labels.manifest_sha256 for name, labels in inputs.label_sets},
        "splits": list(settings.splits),
        "tokens_per_batch": settings.tokens_per_batch,
    }

    def compute(directory: Path) -> dict[str, JsonValue]:
        _write_representations(config, inputs, model, work_dir, directory)
        return {"tasks": list(tasks), "splits": list(settings.splits)}

    (record, status, path), took = _timed(
        lambda: cache.run_stage(work_dir, "representations", key_inputs, compute)
    )
    index = RepresentationIndex(path, tasks, settings.splits)
    return record, StageOutcome("representations", status, record.key, path, took), index


def _write_representations(
    config: ExperimentConfig,
    inputs: VerifiedInputs,
    model: ModelIdentity,
    work_dir: Path,
    directory: Path,
) -> None:
    from ehrfm.data.batches import BatchBuilder
    from ehrfm.data.prediction_times import labelled_times
    from ehrfm.data.splits import extract_split
    from ehrfm.data.tokenization import extract_tokenizer
    from ehrfm.inputs import open_database
    from ehrfm.models.representations import ClmbrEncoder

    settings = config.experiment.representations
    extract = inputs.extract
    encoder = ClmbrEncoder.load(
        model.model_dir, device=config.model.device, expected_weights_sha256=model.weights_sha256
    )
    builder = BatchBuilder(extract_tokenizer(model.model_dir, extract), settings.tokens_per_batch)
    with open_database(extract, work_dir / DATABASE_DIR) as database:
        for split in settings.splits:
            if not inputs.label_sets:
                reps = encoder.subject_representations(
                    database, extract_split(extract, split), builder
                )
                meta = RepresentationMeta(
                    task=SUBJECTS_TASK,
                    split=split,
                    n=len(reps),
                    hidden_size=encoder.hidden_size,
                    model_weights_sha256=model.weights_sha256,
                )
                save_set(directory, meta, reps, None)
                continue
            for name, label_set in inputs.label_sets:
                rows = labelled_times(label_set, extract, [split])
                reps = encoder.representations_at(database, [row.at for row in rows], builder)
                task = label_set.task
                labels = encode_labels(
                    [row.value for row in rows], task.label_type, task.categories
                )
                meta = RepresentationMeta(
                    task=name,
                    split=split,
                    n=len(reps),
                    hidden_size=encoder.hidden_size,
                    label_type=task.label_type,
                    categories=task.categories,
                    model_weights_sha256=model.weights_sha256,
                )
                save_set(directory, meta, reps, labels)


def representation_metrics(index: RepresentationIndex) -> dict[str, float]:
    """Rows and (binary tasks) positives and prevalence per task and split."""
    metrics: dict[str, float] = {}
    for task in index.tasks:
        for split in index.splits:
            loaded = index.load(task, split)
            prefix = f"representations/{task}/{split}"
            metrics[f"{prefix}/n"] = float(len(loaded))
            if loaded.labels is not None and loaded.meta.label_type == "boolean":
                positives = float(loaded.labels.sum())
                metrics[f"{prefix}/n_positive"] = positives
                if len(loaded):
                    metrics[f"{prefix}/prevalence"] = positives / len(loaded)
    return metrics


# --------------------------------------------------------------------------- hooks


def hook_stage(
    kind: HookKind,
    spec: HookSpec,
    config: ExperimentConfig,
    model: ModelIdentity,
    index: RepresentationIndex,
    representations_key: str,
    heads: dict[str, HookRecord],
    work_dir: Path,
) -> tuple[HookRecord, StageOutcome]:
    """Run one hook (cached on its representations, visible heads, version and options)."""
    hook = get_hook(kind, spec.name)
    key_inputs: dict[str, JsonValue] = {
        "kind": kind,
        "name": spec.name,
        "version": hook.version,
        "options": spec.options,
        "seed": config.experiment.seed,
        "representations": representations_key,
        "heads": {name: record.key for name, record in sorted(heads.items())},
    }
    stage = f"{kind}-{spec.name}"

    def compute(directory: Path) -> dict[str, JsonValue]:
        output = directory / HOOK_OUTPUT_DIR
        output.mkdir()
        context = HookContext(
            name=spec.name,
            kind=kind,
            options=spec.options,
            seed=config.experiment.seed,
            output_dir=output,
            config=config,
            model=model,
            representations=index,
            heads=dict(heads),
        )
        result = hook.function(context)
        (directory / HOOK_RESULT_FILE).write_text(result.model_dump_json(indent=2) + "\n")
        return {"result": HOOK_RESULT_FILE}

    (record, status, path), took = _timed(
        lambda: cache.run_stage(work_dir, stage, key_inputs, compute)
    )
    result = HookResult.model_validate_json((path / HOOK_RESULT_FILE).read_bytes())
    hook_record = HookRecord(spec.name, kind, record.key, path / HOOK_OUTPUT_DIR, result)
    return hook_record, StageOutcome(stage, status, record.key, path, took)


BUILTIN_HOOK_MODULES: Final = ("ehrfm.eval.hooks", "ehrfm.heads.hooks")
"""Modules whose hooks are always registered (the evaluation toolkit and the task heads)."""


def import_hook_modules(modules: tuple[str, ...]) -> None:
    """Import the built-in hook modules and the configured ones so their hooks register."""
    for module in (*BUILTIN_HOOK_MODULES, *modules):
        importlib.import_module(module)


# --------------------------------------------------------------------------- tracking


def pipeline_key(config: ExperimentConfig, stages: tuple[StageOutcome, ...]) -> str:
    """Identity of a pipeline result for tracking: its stage keys and the logging policy."""
    return cache.stage_key(
        {
            "format": PIPELINE_FORMAT,
            "experiment": config.tracking_experiment,
            "stages": {stage.name: stage.key for stage in stages},
            "policy": config.experiment.suppression.model_dump(mode="json"),
        }
    )


def _log(
    run: TrackedRun,
    config: ExperimentConfig,
    inputs: VerifiedInputs,
    stages: tuple[StageOutcome, ...],
    records: dict[str, StageRecord],
    index: RepresentationIndex,
    hooks: list[HookRecord],
) -> None:
    run.log_params(config.model_dump(mode="json"))
    run.log_json("config.json", config.model_dump(mode="json"))
    run.log_json("inputs.json", inputs_summary(inputs))
    run.log_json(
        "stages.json",
        [{"stage": s.name, "status": s.status, "key": s.key} for s in stages],
    )
    coverage = next((s.path for s in stages if s.name == "coverage" and s.path is not None), None)
    if coverage is not None:
        report = json.loads((coverage / COVERAGE_FILE).read_text(encoding="utf-8"))
        run.log_json(COVERAGE_FILE, report)
        run.log_metrics(_coverage_metrics(report))
    pretraining = records.get("pretraining")
    if pretraining is not None:
        losses = pretraining.outputs.get("train_losses")
        if isinstance(losses, list):
            run.log_series(
                "pretraining/train_loss",
                tuple(float(v) for v in losses if isinstance(v, int | float)),
            )
        validation = pretraining.outputs.get("validation_loss")
        if isinstance(validation, int | float):
            run.log_metrics({"pretraining/validation_loss": float(validation)})
    run.log_metrics(representation_metrics(index))
    for hook in hooks:
        prefix = f"{hook.kind}/{hook.name}"
        run.log_metrics({f"{prefix}/{name}": value for name, value in hook.result.metrics.items()})
        if hook.result.summary:
            run.log_json(f"hooks/{hook.kind}/{hook.name}.json", hook.result.summary)


def _coverage_metrics(report: dict[str, JsonValue]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in ("events", "recognised_events", "tokenized_events"):
        value = report.get(name)
        if isinstance(value, int):
            metrics[f"coverage/{name}"] = float(value)
    events = report.get("events")
    tokenized = report.get("tokenized_events")
    if isinstance(events, int) and isinstance(tokenized, int) and events:
        metrics["coverage/tokenized_event_share"] = tokenized / events
    distinct = report.get("distinct_codes")
    recognised = report.get("recognised_codes")
    if isinstance(distinct, int) and isinstance(recognised, int) and distinct:
        metrics["coverage/recognised_code_share"] = recognised / distinct
    return metrics


# --------------------------------------------------------------------------- entrypoint


def run_experiment(
    config: ExperimentConfig, *, should_stop: StopRequest | None = None
) -> PipelineResult:
    """Run (or reuse) every stage of ``config`` and log the result."""
    started = time.monotonic()
    work_dir = config.experiment.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    import_hook_modules(config.experiment.hook_modules)
    for spec in config.tasks.heads:
        get_hook("head", spec.name)
    for spec in config.eval.hooks:
        get_hook("eval", spec.name)

    outcomes: list[StageOutcome] = []
    records: dict[str, StageRecord] = {}
    inputs, took = _timed(lambda: verify_inputs(config, work_dir))
    outcomes.append(StageOutcome("inputs", "verified", None, None, took))

    if config.experiment.coverage.enabled:
        coverage, outcome = coverage_stage(config, inputs, work_dir)
        records["coverage"] = coverage
        outcomes.append(outcome)
    else:
        outcomes.append(StageOutcome("coverage", "skipped"))

    pretrained: StageRecord | None = None
    if config.experiment.pretraining.enabled:
        pretrained, outcome = pretraining_stage(config, inputs, work_dir, should_stop=should_stop)
        records["pretraining"] = pretrained
        outcomes.append(outcome)
    else:
        outcomes.append(StageOutcome("pretraining", "skipped"))
    model = model_identity(inputs, pretrained)

    record, outcome, index = representation_stage(config, inputs, model, work_dir)
    records["representations"] = record
    outcomes.append(outcome)

    heads: dict[str, HookRecord] = {}
    for spec in config.tasks.heads:
        head, outcome = hook_stage("head", spec, config, model, index, record.key, {}, work_dir)
        heads[spec.name] = head
        outcomes.append(outcome)
    evaluations: dict[str, HookRecord] = {}
    for spec in config.eval.hooks:
        evaluation, outcome = hook_stage(
            "eval", spec, config, model, index, record.key, heads, work_dir
        )
        evaluations[spec.name] = evaluation
        outcomes.append(outcome)

    stages = tuple(outcomes)
    key = pipeline_key(config, stages)
    tracking = config.experiment.tracking
    run_id: str | None = None
    uri: str | None = None
    tracking_status: Literal["logged", "cached", "disabled"] = "disabled"
    if tracking.enabled:
        tracker = open_tracker(
            tracking, config.tracking_experiment, work_dir, config.experiment.suppression
        )
        uri = tracker.uri
        previous = [
            run for run in tracker.find_runs("pipeline_key", key) if run.info.status == "FINISHED"
        ]
        if previous:
            run_id, tracking_status = str(previous[0].info.run_id), "cached"
        else:
            tags = {
                "pipeline_key": key,
                "model_weights_sha256": model.weights_sha256,
                "extract_manifest_sha256": inputs.extract.manifest_sha256,
            }
            if model.training_fingerprint is not None:
                tags["training_fingerprint"] = model.training_fingerprint
            with tracker.run(config.experiment.name, tags) as run:
                _log(
                    run,
                    config,
                    inputs,
                    stages,
                    records,
                    index,
                    [*heads.values(), *evaluations.values()],
                )
                run_id = run.run_id
            tracking_status = "logged"

    return PipelineResult(
        config=config,
        work_dir=work_dir,
        pipeline_key=key,
        stages=stages,
        model=model,
        representations=index,
        heads=heads,
        evaluations=evaluations,
        tracking_uri=uri,
        tracking_run_id=run_id,
        tracking_status=tracking_status,
        duration_seconds=time.monotonic() - started,
        records=records,
    )


__all__ = [
    "PipelineInterrupted",
    "PipelineResult",
    "StageOutcome",
    "VerifiedInputs",
    "coverage_stage",
    "hook_stage",
    "inputs_summary",
    "model_identity",
    "pipeline_key",
    "pretraining_stage",
    "representation_metrics",
    "representation_stage",
    "run_experiment",
    "training_config",
    "verify_inputs",
]
