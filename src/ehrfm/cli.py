"""Command-line interface for ehrfm."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from ehrfm.inputs import InputArtifactError, open_concept_set, open_extract, open_label_set
from ehrfm.models.checkpoint import CheckpointError

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
inputs_app = typer.Typer(no_args_is_help=True, help="Check consumed input artifacts.")
app.add_typer(inputs_app, name="inputs")
heads_app = typer.Typer(
    no_args_is_help=True,
    help="List trained task heads and publish them to a head registry. Heads are "
    "trained by `ehrfm run` (they are cached pipeline stages over the representations).",
)
app.add_typer(heads_app, name="heads")


class ArtifactKind(StrEnum):
    """Kinds of input artifact."""

    extract = "extract"
    labels = "labels"
    concepts = "concepts"


@inputs_app.command("verify")
def verify_inputs(
    artifact: Annotated[Path, typer.Argument(help="Artifact directory to verify.")],
    kind: Annotated[ArtifactKind, typer.Option(help="Kind of artifact.")] = ArtifactKind.extract,
    extract: Annotated[
        Path | None,
        typer.Option(help="MEDS extract the label set must refer to (labels only)."),
    ] = None,
    clmbr_manifest: Annotated[
        Path | None,
        typer.Option(help="Committed CLMBR dictionary manifest to match (concepts only)."),
    ] = None,
) -> None:
    """Verify an artifact against its manifest; exit 1 on any violation."""
    try:
        match kind:
            case ArtifactKind.extract:
                opened = open_extract(artifact)
                detail = f"{opened.summary.subjects} subjects, {opened.summary.events} events"
            case ArtifactKind.labels:
                linked = open_extract(extract) if extract is not None else None
                opened_labels = open_label_set(artifact, linked)
                detail = f"task {opened_labels.task.name}, {opened_labels.summary.labels} labels"
            case ArtifactKind.concepts:
                opened_concepts = open_concept_set(artifact, clmbr_manifest)
                detail = f"{opened_concepts.summary.concepts} concepts"
    except InputArtifactError as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"ok: {artifact} ({detail})")


@inputs_app.command("coverage")
def inputs_coverage(
    extract: Annotated[Path, typer.Argument(help="MEDS extract directory.")],
    model: Annotated[Path, typer.Option(help="Checkpoint directory holding dictionary.msgpack.")],
    top_n: Annotated[
        int, typer.Option(min=0, help="Most frequent out-of-vocabulary codes to list.")
    ] = 20,
    min_events: Annotated[
        int, typer.Option(min=1, help="Never list a code seen in fewer events than this.")
    ] = 10,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Measure how much of an extract the checkpoint dictionary covers; exit 1 on bad input."""
    from ehrfm.data.coverage import extract_coverage

    try:
        report = extract_coverage(open_extract(extract), model, top_n=top_n, min_events=min_events)
    except (InputArtifactError, CheckpointError, ValueError) as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo(
        f"distinct codes {report.distinct_codes}: recognised {report.recognised_codes} "
        f"({report.recognised_share:.1%}), tokenized {report.tokenized_codes} "
        f"({report.tokenized_share:.1%})"
    )
    typer.echo(
        f"events {report.events}: tokenized {report.tokenized_events} "
        f"({report.tokenized_event_share:.1%})"
    )
    for row in report.by_vocabulary:
        name = row.vocabulary_id or "(none)"
        typer.echo(
            f"  {name}: {row.recognised_codes}/{row.distinct_codes} recognised, "
            f"{row.tokenized_codes}/{row.distinct_codes} tokenized"
        )
    for code in report.unrecognised:
        typer.echo(f"  out of vocabulary: {code.code} ({code.events} events)")


@app.command("pretrain")
def pretrain(
    config: Annotated[
        Path | None, typer.Option(help="Training configuration (YAML or JSON).")
    ] = None,
    toy: Annotated[
        bool,
        typer.Option(
            "--toy",
            help="Train on a generated synthetic extract and toy checkpoint (CPU, seconds).",
        ),
    ] = False,
    work_dir: Annotated[
        Path | None,
        typer.Option(help="Directory for --toy inputs and outputs (default: a user cache dir)."),
    ] = None,
    print_torchrun: Annotated[
        int | None,
        typer.Option(
            min=1, help="Print the torchrun command for this many processes per node and exit."
        ),
    ] = None,
) -> None:
    """Continue pretraining a CLMBR checkpoint; a completed identical run is a cache hit."""
    import logging
    import os

    from ehrfm.config.training import load_training_config
    from ehrfm.training.runner import SignalStop, run_pretraining, torchrun_command

    if (config is None) == (not toy):
        typer.echo("error: pass exactly one of --config or --toy", err=True)
        raise typer.Exit(code=2)
    if print_torchrun is not None:
        if config is None:
            typer.echo("error: --print-torchrun needs --config", err=True)
            raise typer.Exit(code=2)
        typer.echo(" ".join(torchrun_command(config, nproc_per_node=print_torchrun)))
        return
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if toy:
            from ehrfm.toy.pretraining import toy_training_config

            cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            training = toy_training_config(work_dir or cache_home / "ehrfm" / "toy-pretrain")
        else:
            assert config is not None
            training = load_training_config(config)
        with SignalStop() as stop:
            result = run_pretraining(training, should_stop=stop)
    except (InputArtifactError, CheckpointError, ValueError, OSError) as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    summary = {
        "status": result.status,
        "fingerprint": result.fingerprint,
        "model_dir": str(result.model_dir),
        "step": result.step,
        "total_steps": result.total_steps,
        "steps_run": result.steps_run,
        "first_train_loss": result.train_losses[0] if result.train_losses else None,
        "last_train_loss": result.train_losses[-1] if result.train_losses else None,
        "validation_loss": result.validation_loss,
        "model_weights_sha256": result.model_weights_sha256,
        "duration_seconds": round(result.duration_seconds, 3),
    }
    if os.environ.get("RANK", "0") == "0":
        typer.echo(json.dumps(summary, indent=2))
    if result.status == "interrupted":
        raise typer.Exit(code=3)


class _StoreError(Exception):
    pass


def _store_location(
    config: Path | None, work_dir: Path | None, tracking_uri: str | None
) -> tuple[Path | None, str]:
    """The work directory and tracking URI named by --config, --work-dir or --tracking-uri."""
    from ehrfm.config.loader import load_experiment_config
    from ehrfm.experiment.tracking import default_uri

    if config is not None:
        loaded = load_experiment_config(config)
        root = loaded.experiment.work_dir
        return root, tracking_uri or loaded.experiment.tracking.uri or default_uri(root)
    if work_dir is not None:
        return work_dir, tracking_uri or default_uri(work_dir)
    if tracking_uri is not None:
        return None, tracking_uri
    raise _StoreError("pass one of --config, --work-dir or --tracking-uri")


_ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="Experiment configuration naming the work directory."),
]
_WorkDirOption = Annotated[
    Path | None, typer.Option(help="Experiment work directory (its mlruns/ is the default store).")
]
_UriOption = Annotated[str | None, typer.Option(help="MLflow tracking URI.")]


@app.command(
    "run",
    context_settings={"allow_extra_args": False, "ignore_unknown_options": False},
)
def run(
    args: Annotated[
        list[str] | None,
        typer.Argument(
            help="Configuration file, then Hydra overrides (key=value); with --config, overrides only."
        ),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Experiment configuration (YAML or JSON).")
    ] = None,
) -> None:
    """Run an experiment end to end; unchanged stages and logged runs are reused from the cache."""
    import contextlib
    import logging
    import sys

    from ehrfm.config.loader import ConfigError, load_experiment_config
    from ehrfm.experiment.hooks import HookError
    from ehrfm.experiment.pipeline import PipelineInterrupted, run_experiment
    from ehrfm.training.runner import SignalStop

    rest = list(args or [])
    if config is None:
        if not rest:
            typer.echo("error: pass a configuration file (positionally or with --config)", err=True)
            raise typer.Exit(code=2)
        config = Path(rest.pop(0))
    bad = [item for item in rest if "=" not in item]
    if bad:
        typer.echo(f"error: overrides must be key=value, got {bad}", err=True)
        raise typer.Exit(code=2)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        experiment = load_experiment_config(config, rest)
        # Library progress output goes to stderr so stdout carries only the summary.
        with SignalStop() as stop, contextlib.redirect_stdout(sys.stderr):
            result = run_experiment(experiment, should_stop=stop)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except PipelineInterrupted as exc:
        typer.echo(f"interrupted: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    except (InputArtifactError, CheckpointError, HookError, ValueError, OSError) as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    summary = {
        "experiment": experiment.experiment.name,
        "cache_hit": result.cache_hit,
        "pipeline_key": result.pipeline_key,
        "work_dir": str(result.work_dir),
        "stages": [
            {"stage": stage.name, "status": stage.status, "key": stage.key and stage.key[:16]}
            for stage in result.stages
        ],
        "model_weights_sha256": result.model.weights_sha256,
        "tracking": {
            "status": result.tracking_status,
            "uri": result.tracking_uri,
            "run_id": result.tracking_run_id,
        },
        "duration_seconds": round(result.duration_seconds, 3),
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command("ls")
def list_runs(
    config: _ConfigOption = None,
    work_dir: _WorkDirOption = None,
    tracking_uri: _UriOption = None,
) -> None:
    """List tracked runs and cached stages."""
    from ehrfm.config.loader import ConfigError
    from ehrfm.experiment.cache import KEY_PREFIX, list_stages
    from ehrfm.experiment.tracking import TAG_PREFIX, store_client

    try:
        root, uri = _store_location(config, work_dir, tracking_uri)
    except _StoreError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    client = store_client(uri)
    typer.echo(f"runs in {uri}:")
    for experiment in client.search_experiments():
        runs = client.search_runs(
            [experiment.experiment_id], order_by=["attributes.start_time DESC"]
        )
        for tracked in runs:
            key = tracked.data.tags.get(f"{TAG_PREFIX}pipeline_key", "")
            typer.echo(
                f"  {tracked.info.run_id}  {experiment.name}/{tracked.info.run_name}  "
                f"{tracked.info.status}  pipeline {key[:KEY_PREFIX]}"
            )
    if root is not None:
        typer.echo(f"stages in {root}:")
        for path, record in list_stages(root):
            typer.echo(
                f"  {record.stage:<20} {record.key[:KEY_PREFIX]}  {record.created_at}  "
                f"{record.duration_seconds:.2f}s  {path}"
            )


@app.command("show")
def show_run(
    ref: Annotated[str, typer.Argument(help="MLflow run id, or a stage key prefix.")],
    config: _ConfigOption = None,
    work_dir: _WorkDirOption = None,
    tracking_uri: _UriOption = None,
) -> None:
    """Show a tracked run (parameters, metrics, artifacts) or a cached stage record."""

    from ehrfm.config.loader import ConfigError
    from ehrfm.experiment.cache import list_stages
    from ehrfm.experiment.tracking import store_client

    try:
        root, uri = _store_location(config, work_dir, tracking_uri)
    except _StoreError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if root is not None:
        matches = [(p, r) for p, r in list_stages(root) if r.key.startswith(ref)]
        if len(matches) == 1:
            typer.echo(json.dumps(matches[0][1].model_dump(mode="json"), indent=2))
            return
        if len(matches) > 1:
            typer.echo(f"error: stage key prefix {ref!r} is ambiguous", err=True)
            raise typer.Exit(code=1)
    client = store_client(uri)
    from mlflow.exceptions import MlflowException

    try:
        tracked = client.get_run(ref)
    except MlflowException as exc:
        typer.echo(f"error: no run or stage {ref!r} in {uri}", err=True)
        raise typer.Exit(code=1) from exc
    payload = {
        "run_id": tracked.info.run_id,
        "name": tracked.info.run_name,
        "status": tracked.info.status,
        "tags": dict(sorted(tracked.data.tags.items())),
        "params": dict(sorted(tracked.data.params.items())),
        "metrics": dict(sorted(tracked.data.metrics.items())),
        "artifacts": sorted(_artifact_paths(client, tracked.info.run_id)),
    }
    typer.echo(json.dumps(payload, indent=2))


def _artifact_paths(client: object, run_id: str, path: str | None = None) -> list[str]:
    from mlflow.tracking import MlflowClient

    assert isinstance(client, MlflowClient)
    found: list[str] = []
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            found.extend(_artifact_paths(client, run_id, item.path))
        else:
            found.append(item.path)
    return found


def _heads_work_dir(config: Path | None, work_dir: Path | None) -> Path | None:
    if config is not None:
        from ehrfm.config.loader import load_experiment_config

        return load_experiment_config(config).experiment.work_dir
    return work_dir


@heads_app.command("ls")
def heads_list(
    config: _ConfigOption = None,
    work_dir: _WorkDirOption = None,
    registry: Annotated[
        Path | None, typer.Option(help="Head registry directory to list instead.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """List the heads an experiment trained, or the heads in a registry."""
    from ehrfm.config.loader import ConfigError
    from ehrfm.experiment.cache import KEY_PREFIX
    from ehrfm.heads.store import trained_heads
    from ehrfm.inference.head_loader import HeadLoadError
    from ehrfm.inference.registry import HeadRegistry

    if sum(option is not None for option in (config, work_dir, registry)) != 1:
        typer.echo("error: pass exactly one of --config, --work-dir or --registry", err=True)
        raise typer.Exit(code=2)
    try:
        if registry is not None:
            rows = [entry.summary() for entry in HeadRegistry(registry).entries()]
        else:
            root = _heads_work_dir(config, work_dir)
            assert root is not None
            rows = [
                {
                    "hook": head.hook,
                    "stage_key": head.stage_key[:KEY_PREFIX],
                    "created_at": head.created_at,
                    "kind": head.record.spec.kind,
                    "task": head.record.spec.task_name,
                    "input_dim": head.record.spec.input_dim,
                    "encoder_weights_sha256": head.record.encoder_weights_sha256,
                    "path": str(head.path),
                }
                for head in trained_heads(root)
            ]
    except (ConfigError, HeadLoadError, OSError) as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    for row in rows:
        label = row.get("name") or f"{row['hook']} {row['stage_key']}"
        encoder = str(row["encoder_weights_sha256"] or "-")[:KEY_PREFIX]
        typer.echo(
            f"  {label}  {row['kind']}  task {row['task']}  in {row['input_dim']}  "
            f"encoder {encoder}"
        )


@heads_app.command("publish")
def heads_publish(
    ref: Annotated[str, typer.Argument(help="Head hook name, or a head stage key prefix.")],
    registry: Annotated[Path, typer.Option(help="Head registry directory.")],
    name: Annotated[str | None, typer.Option(help="Registry name (default: the task).")] = None,
    config: _ConfigOption = None,
    work_dir: _WorkDirOption = None,
    replace: Annotated[bool, typer.Option(help="Replace a head of the same name.")] = False,
) -> None:
    """Copy a trained head into a registry after verifying it."""
    from ehrfm.config.loader import ConfigError
    from ehrfm.heads.store import find_trained_head
    from ehrfm.inference.head_loader import HeadLoadError
    from ehrfm.inference.registry import HeadRegistry

    if (config is None) == (work_dir is None):
        typer.echo("error: pass exactly one of --config or --work-dir", err=True)
        raise typer.Exit(code=2)
    try:
        root = _heads_work_dir(config, work_dir)
        assert root is not None
        head = find_trained_head(root, ref)
        entry = HeadRegistry(registry).publish(
            head.path, name or head.record.spec.task_name, replace=replace
        )
    except (ConfigError, HeadLoadError, LookupError, ValueError, OSError) as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(entry.summary(), indent=2))


@app.command("serve")
def serve(
    toy: Annotated[
        bool,
        typer.Option(
            "--toy",
            help="Serve the generated toy checkpoint with a head fitted on the fly (CPU, seconds).",
        ),
    ] = False,
    model_dir: Annotated[
        Path | None,
        typer.Option(help="Checkpoint directory to serve (required unless --toy)."),
    ] = None,
    heads_dir: Annotated[
        Path | None,
        typer.Option(help="Head registry root to serve (required unless --toy)."),
    ] = None,
    extract: Annotated[
        Path | None,
        typer.Option(
            help="Extract whose codes.parquet parents expand request codes into ancestor tokens."
        ),
    ] = None,
    weights_sha256: Annotated[
        str | None, typer.Option(help="Expected SHA-256 of the encoder weights.")
    ] = None,
    postprocess: Annotated[
        str | None, typer.Option(help="Registered postprocess hook (default: raw).")
    ] = None,
    device: Annotated[str | None, typer.Option(help="Encoder device (default: cpu).")] = None,
    host: Annotated[str | None, typer.Option(help="Bind address.")] = None,
    port: Annotated[int | None, typer.Option(min=1, help="Bind port.")] = None,
    toy_work_dir: Annotated[
        Path | None, typer.Option(help="Directory for --toy inputs, database and registry.")
    ] = None,
    toy_subjects: Annotated[
        int | None, typer.Option(min=10, help="Subjects of the --toy extract.")
    ] = None,
) -> None:
    """Serve CLMBR representations and task heads; every option has an
    EHRFM_SERVER_ environment override (see `python -m ehrfm.server --help`)."""
    import uvicorn

    from ehrfm.server.app import create_app
    from ehrfm.server.config import ServerConfig

    settings: dict[str, object] = {}
    if toy:
        settings["toy"] = True
    if model_dir is not None:
        settings["model_dir"] = model_dir
    if heads_dir is not None:
        settings["heads_dir"] = heads_dir
    if extract is not None:
        settings["extract_dir"] = extract
    if weights_sha256 is not None:
        settings["expected_weights_sha256"] = weights_sha256
    if postprocess is not None:
        settings["postprocess"] = postprocess
    if device is not None:
        settings["device"] = device
    if host is not None:
        settings["host"] = host
    if port is not None:
        settings["port"] = port
    if toy_work_dir is not None:
        settings["toy_work_dir"] = toy_work_dir
    if toy_subjects is not None:
        settings["toy_subjects"] = toy_subjects
    try:
        config = ServerConfig.model_validate({**ServerConfig.env_kwargs(), **settings})
    except ValueError as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        json.dumps(
            {
                "host": config.host,
                "port": config.port,
                "toy": config.toy,
                "model_dir": str(config.model_dir) if config.model_dir else None,
                "heads_dir": str(config.heads_dir) if config.heads_dir else None,
                "postprocess": config.postprocess,
            },
            indent=2,
        ),
        err=True,
    )
    uvicorn.run(create_app(config), host=config.host, port=config.port)


@app.command("fetch-model")
def fetch_model(
    dest: Annotated[
        Path | None,
        typer.Option(help="Destination directory (default: a user cache dir)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the licence notice and the plan; make no network call.",
        ),
    ] = False,
) -> None:
    """Download the pinned CLMBR-T-Base release and verify it against the pin.

    The release is gated; pass --dry-run without a token to read the licence
    notice first. A checksum mismatch, an unreachable source or a missing
    token exits 2.
    """
    from ehrfm.inputs.fetch import FetchError, destination_default, fetch

    target = dest or destination_default()
    try:
        result = fetch(target, dry_run=dry_run)
    except FetchError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if result is not None:
        typer.echo(f"snapshot: {result.snapshot_dir}")
        weights = result.snapshot_dir / "model.safetensors"
        if weights.is_file():
            import hashlib

            sha = hashlib.sha256(weights.read_bytes()).hexdigest()
            typer.echo(
                f"weights:  {weights} (sha256 {sha}; "
                "set EHRFM_SERVER_WEIGHTS_SHA256 to this value to serve it)"
            )


@app.command("doctor")
def doctor() -> None:
    """Check the environment: torch, femr, meds_reader, the torch_hawk shim, device."""
    import sys

    typer.echo(f"python:   {sys.version.split()[0]}")
    try:
        import torch

        typer.echo(f"torch:    {torch.__version__}")
        if torch.cuda.is_available():
            typer.echo(f"device:   cuda ({torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            typer.echo("device:   mps")
        else:
            typer.echo("device:   cpu")
    except Exception as exc:
        typer.echo(f"torch:    error ({exc})")
        raise typer.Exit(code=1) from exc
    from ehrfm.models import _compat

    hawk_shim = _compat.install()
    if hawk_shim:
        typer.echo(
            "torch_hawk: compatibility shim (the real package is absent; the "
            "uniform-attention path never calls it)"
        )
    else:
        typer.echo("torch_hawk: real package")
    for label, module in (
        ("femr", "femr.models.transformer"),
        ("meds_reader", "meds_reader"),
        ("xformers", "xformers.ops"),
    ):
        try:
            __import__(module)
        except Exception as exc:
            typer.echo(f"{label}:  error ({exc})")
            raise typer.Exit(code=1) from exc
        typer.echo(f"{label}:  ok")
    try:
        import huggingface_hub

        typer.echo(f"huggingface_hub: {huggingface_hub.__version__}")
    except Exception as exc:
        typer.echo(f"huggingface_hub: error ({exc})")
        raise typer.Exit(code=1) from exc
    typer.echo("environment ok")


def main() -> None:
    """Run the ehrfm command-line interface."""
    app()
