"""Run continued pretraining: cache decision, resume, train, publish.

:func:`run_pretraining` is the one entrypoint. It fingerprints the run
(:mod:`ehrfm.training.cache`), returns the published model unchanged when a
completed run with the same fingerprint exists, resumes an interrupted run
from its newest checkpoint, and otherwise starts fresh. Training uses the
patched and checked base model (:mod:`ehrfm.training.surgery`) and femr's
CLMBR objective (:mod:`ehrfm.training.pretrain`).

The same code runs on the CPU, on one GPU and under ``torchrun``. Under
``torchrun`` (``WORLD_SIZE`` > 1) every process trains on its own share of
each step's batches through ``DistributedDataParallel`` (NCCL on CUDA, gloo
on the CPU); rank 0 alone writes the run directory. :func:`torchrun_command`
builds the launcher command for a configuration file, for any scheduler or a
plain shell.

A stop request (``should_stop`` returning True, or SIGTERM/SIGINT under the
CLI) is honoured after the current step: a checkpoint is written, the record
is marked ``interrupted`` and the run returns; the next invocation resumes.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import shutil
import signal
import sys
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Literal, cast

import datasets
import torch
import torch.distributed as dist
from femr.models.processor import FEMRBatchProcessor
from femr.models.transformer import FEMRModel
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from ehrfm.config.training import TrainingConfig
from ehrfm.data.coverage import extract_coverage
from ehrfm.data.splits import SplitSpec, extract_split, split_subjects
from ehrfm.data.tokenization import ontology_from_extract
from ehrfm.inputs import ensure_database, open_extract
from ehrfm.inputs.manifest import sha256_file
from ehrfm.models.checkpoint import DICTIONARY_FILE, WEIGHTS_FILE
from ehrfm.models.device import resolve_device
from ehrfm.training import cache
from ehrfm.training.cache import CheckpointMeta, RunPaths, RunRecord
from ehrfm.training.pretrain import (
    ClmbrPretrainingTask,
    EmptyTrainingSetError,
    autocast,
    batch_index,
    collated,
    dictionary_bytes,
    export_checkpoint,
    make_optimizer,
    make_scheduler,
    packed_batches,
    seed_everything,
    steps_per_epoch,
    to_device,
    tokenization_key,
    total_steps,
    training_tokenizer,
    validation_loss,
)
from ehrfm.training.surgery import prepare_base_model
from ehrfm.training.vocab_expansion import expand_vocabulary, zero_replaced_embeddings

_log = logging.getLogger("ehrfm.training")

StopRequest = Callable[[int], bool]
"""Called with the number of completed steps after each step; True stops the run."""


class NonFiniteLossError(FloatingPointError):
    """A training step produced a NaN or infinite loss."""


class TrainingResult(BaseModel):
    """Outcome of one :func:`run_pretraining` call."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    status: Literal["cached", "completed", "interrupted"]
    fingerprint: str
    model_dir: Path
    step: int
    total_steps: int
    steps_run: int
    """Optimizer steps executed by this call (0 on a cache hit)."""
    train_losses: tuple[float, ...]
    validation_loss: float | None
    model_weights_sha256: str | None
    resumed_from_step: int | None
    duration_seconds: float

    @property
    def cache_hit(self) -> bool:
        """Whether the result was served from a completed run."""
        return self.status == "cached"


@dataclass(frozen=True, slots=True)
class World:
    """This process's place in a (possibly single-process) run."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def world_from_env() -> World:
    """The process's rank as set by ``torchrun`` (a single process when unset)."""
    return World(
        rank=int(os.environ.get("RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
    )


def _device_for(world: World, requested: str) -> torch.device:
    if not world.distributed:
        return resolve_device(requested)
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda", world.local_rank)
    return resolve_device(requested)


def _barrier(world: World) -> None:
    if world.distributed:
        dist.barrier()


@contextlib.contextmanager
def _process_group(world: World, device: torch.device) -> Generator[None]:
    if not world.distributed or dist.is_initialized():
        yield
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend=backend)
    try:
        yield
    finally:
        dist.destroy_process_group()


@contextlib.contextmanager
def _main_only_lock(world: World, paths: RunPaths) -> Generator[None]:
    if world.is_main:
        with cache.run_lock(paths):
            yield
    else:
        yield


def _mean_across(world: World, value: float, device: torch.device) -> float:
    if not world.distributed:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item()) / world.world_size


def _any_across(world: World, flag: bool, device: torch.device) -> bool:
    if not world.distributed:
        return flag
    tensor = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())


def _cached_result(record: RunRecord, paths: RunPaths, started: float) -> TrainingResult:
    return TrainingResult(
        status="cached",
        fingerprint=record.fingerprint,
        model_dir=paths.model,
        step=record.step,
        total_steps=record.total_steps,
        steps_run=0,
        train_losses=record.train_losses,
        validation_loss=record.validation_loss,
        model_weights_sha256=record.model_weights_sha256,
        resumed_from_step=record.resumed_from_step,
        duration_seconds=time.monotonic() - started,
    )


def _split_train_subjects(
    subjects: Sequence[int], fraction: float, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if fraction == 0:
        return tuple(sorted(subjects)), ()
    spec = SplitSpec.of({"fit": 1.0 - fraction, "validation": fraction}, f"pretrain:{seed}")
    parts = split_subjects(subjects, spec)
    if not parts["fit"]:
        raise EmptyTrainingSetError(
            f"validation_fraction {fraction} leaves no training subjects of {len(subjects)}"
        )
    return parts["fit"], parts["validation"]


def _unwrap(model: nn.Module) -> FEMRModel:
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    return cast(FEMRModel, inner)


def run_pretraining(
    config: TrainingConfig,
    *,
    should_stop: StopRequest | None = None,
    world: World | None = None,
) -> TrainingResult:
    """Run (or skip, or resume) the continued-pretraining run ``config`` describes."""
    started = time.monotonic()
    world = world_from_env() if world is None else world
    device = _device_for(world, config.runtime.device)
    paths = RunPaths(config.runtime.output_dir)
    optimization = config.optimization

    extract = open_extract(config.inputs.extract)
    base = config.inputs.base_checkpoint
    base_weights_sha256 = sha256_file(base / WEIGHTS_FILE)
    inputs = cache.fingerprint_inputs(
        config,
        extract_manifest_sha256=extract.manifest_sha256,
        base_weights_sha256=base_weights_sha256,
        world_size=world.world_size,
    )
    fingerprint = inputs.digest()
    decision = cache.decide(paths, fingerprint)
    if decision.action == "cached" and decision.record is not None:
        _log.info("cache hit: %s (%s)", paths.model, fingerprint[:12])
        return _cached_result(decision.record, paths, started)

    with _process_group(world, device), _main_only_lock(world, paths):
        _barrier(world)
        if world.is_main and decision.action == "fresh" and decision.record is not None:
            _log.info("starting over: %s", decision.reason)
            cache.reset_run(paths)
        seed_everything(optimization.seed, deterministic=config.runtime.deterministic)

        prepared = prepare_base_model(
            base,
            expected_weights_sha256=config.inputs.expected_weights_sha256,
            clmbr_vocab_size=optimization.clmbr_vocab_size,
            concept_set=config.inputs.concept_set,
        )
        ontology = ontology_from_extract(extract)
        expansion = expand_vocabulary(
            optimization.vocab_expansion.hook,
            optimization.vocab_expansion.n_replace,
            prepared.dictionary,
            extract=extract,
            ontology=ontology,
            clmbr_vocab_size=prepared.clmbr_vocab_size,
        )
        zero_replaced_embeddings(prepared.model, expansion.replaced)
        dictionary_sha256 = (
            sha256_file(base / DICTIONARY_FILE)
            if not expansion.replaced
            else cache.sha256_bytes(dictionary_bytes(expansion.dictionary))
        )
        tokenizer = training_tokenizer(expansion.dictionary, ontology)
        processor = FEMRBatchProcessor(
            tokenizer, ClmbrPretrainingTask(clmbr_vocab_size=prepared.clmbr_vocab_size)
        )

        fit_subjects, validation_subjects = _split_train_subjects(
            extract_split(extract, "train"), optimization.validation_fraction, optimization.seed
        )

        def packed(database: Path, subjects: Sequence[int]) -> datasets.Dataset:
            key = tokenization_key(
                extract_manifest_sha256=extract.manifest_sha256,
                dictionary_sha256=dictionary_sha256,
                clmbr_vocab_size=prepared.clmbr_vocab_size,
                tokens_per_batch=optimization.tokens_per_batch,
                subject_ids=subjects,
                femr_version=inputs.versions["femr"],
            )
            return packed_batches(
                database,
                subjects,
                processor,
                tokens_per_batch=optimization.tokens_per_batch,
                store=paths.tokenized,
                key=key,
                workers=config.runtime.tokenize_workers,
            )

        if world.is_main:
            # Rank 0 builds the database and the packed batches; the others then read them.
            packed(ensure_database(extract, config.database_cache), fit_subjects)
            if validation_subjects:
                packed(ensure_database(extract, config.database_cache), validation_subjects)
            coverage = extract_coverage(extract, base)
            cache.write_json_atomic(paths.root / "coverage.json", coverage.model_dump(mode="json"))
        _barrier(world)
        database = ensure_database(extract, config.database_cache)
        fit_data = packed(database, fit_subjects)
        validation_data = packed(database, validation_subjects) if validation_subjects else None
        fit_data.set_format("pt")
        if validation_data is not None:
            validation_data.set_format("pt")

        steps = total_steps(len(fit_data), optimization, world.world_size)
        model: nn.Module = prepared.model
        cast(Any, model).to(device)
        optimizer = make_optimizer(model, optimization)
        scheduler = make_scheduler(optimizer, steps, optimization)

        start_step = 0
        losses: list[float] = []
        resumed_from: int | None = None
        if decision.action == "resume" and decision.checkpoint is not None:
            meta, state = cache.load_checkpoint_state(decision.checkpoint)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_step = meta.step
            losses = list(meta.train_losses)
            resumed_from = meta.step
            _log.info("resuming at step %d of %d", start_step, steps)
        if world.distributed:
            model = DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None
            )

        record = RunRecord(
            fingerprint=fingerprint,
            inputs=inputs,
            status="in_progress",
            step=start_step,
            total_steps=steps,
            started_at=cache.now_iso(),
            train_losses=tuple(losses),
            dictionary_sha256=dictionary_sha256,
            replaced_dictionary_entries=expansion.replaced,
            resumed_from_step=resumed_from,
        )
        if world.is_main:
            cache.write_record(paths, record)

        def checkpoint(step: int) -> None:
            if not world.is_main:
                return
            epoch = step // steps_per_epoch(len(fit_data), world.world_size)
            cache.save_checkpoint(
                paths,
                CheckpointMeta(
                    fingerprint=fingerprint, step=step, epoch=epoch, train_losses=tuple(losses)
                ),
                {
                    "model": _unwrap(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                keep=config.runtime.keep_checkpoints,
            )
            cache.write_record(
                paths, record.model_copy(update={"step": step, "train_losses": tuple(losses)})
            )

        step = start_step
        try:
            model.train()
            while step < steps:
                _, row = batch_index(
                    step, len(fit_data), optimization.seed, world.rank, world.world_size
                )
                batch = to_device(collated(processor, fit_data[row]), device)
                with autocast(device, optimization.precision):
                    loss, _ = model(**batch)
                loss_tensor = cast(torch.Tensor, loss)
                if not torch.isfinite(loss_tensor):
                    raise NonFiniteLossError(f"loss is {float(loss_tensor)} at step {step}")
                loss_tensor.backward()
                if optimization.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), optimization.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(_mean_across(world, float(loss_tensor.detach()), device))
                step += 1
                if step % config.runtime.log_every == 0 or step == steps:
                    _log.info("step %d/%d loss %.4f", step, steps, losses[-1])
                stop = _any_across(world, should_stop is not None and should_stop(step), device)
                if stop and step < steps:
                    checkpoint(step)
                    if world.is_main:
                        cache.write_record(
                            paths,
                            record.model_copy(
                                update={
                                    "status": "interrupted",
                                    "step": step,
                                    "train_losses": tuple(losses),
                                    "finished_at": cache.now_iso(),
                                }
                            ),
                        )
                    _barrier(world)
                    _log.info("stopped at step %d of %d; rerun to resume", step, steps)
                    return TrainingResult(
                        status="interrupted",
                        fingerprint=fingerprint,
                        model_dir=paths.model,
                        step=step,
                        total_steps=steps,
                        steps_run=step - start_step,
                        train_losses=tuple(losses),
                        validation_loss=None,
                        model_weights_sha256=None,
                        resumed_from_step=resumed_from,
                        duration_seconds=time.monotonic() - started,
                    )
                if step % config.runtime.checkpoint_every == 0 and step < steps:
                    checkpoint(step)

            final_validation: float | None = None
            weights_sha256: str | None = None
            if world.is_main:
                trained = _unwrap(model)
                if validation_data is not None and len(validation_data) > 0:
                    final_validation = validation_loss(
                        trained, validation_data, processor, device, optimization.precision
                    )
                    if not math.isfinite(final_validation):
                        raise NonFiniteLossError(f"validation loss is {final_validation}")
                provenance = {
                    "fingerprint": fingerprint,
                    "base_weights_sha256": base_weights_sha256,
                    "extract_manifest_sha256": extract.manifest_sha256,
                    "dictionary_sha256": dictionary_sha256,
                    "replaced_dictionary_entries": list(expansion.replaced),
                    "steps": steps,
                    "final_train_loss": losses[-1] if losses else None,
                    "validation_loss": final_validation,
                    "versions": inputs.versions,
                }
                weights_sha256 = export_checkpoint(
                    trained,
                    base,
                    paths.model,
                    dictionary=expansion.dictionary if expansion.replaced else None,
                    provenance=provenance,
                )
                cache.write_record(
                    paths,
                    record.model_copy(
                        update={
                            "status": "completed",
                            "step": steps,
                            "train_losses": tuple(losses),
                            "validation_loss": final_validation,
                            "model_weights_sha256": weights_sha256,
                            "finished_at": cache.now_iso(),
                        }
                    ),
                )
                shutil.rmtree(paths.checkpoints, ignore_errors=True)
            _barrier(world)
        except BaseException as exc:
            if world.is_main:
                status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                cache.write_record(
                    paths,
                    record.model_copy(
                        update={
                            "status": status,
                            "step": step,
                            "train_losses": tuple(losses),
                            "finished_at": cache.now_iso(),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    ),
                )
            raise

    return TrainingResult(
        status="completed",
        fingerprint=fingerprint,
        model_dir=paths.model,
        step=steps,
        total_steps=steps,
        steps_run=steps - start_step,
        train_losses=tuple(losses),
        validation_loss=final_validation,
        model_weights_sha256=weights_sha256,
        resumed_from_step=resumed_from,
        duration_seconds=time.monotonic() - started,
    )


class SignalStop:
    """A stop request raised by SIGTERM or SIGINT, honoured after the current step."""

    def __init__(self) -> None:
        self.requested = False
        self._previous: dict[int, Any] = {}

    def __call__(self, step: int) -> bool:
        del step
        return self.requested

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        del frame
        _log.warning("signal %d received; stopping after the current step", signum)
        self.requested = True

    def __enter__(self) -> SignalStop:
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous[signum] = signal.signal(signum, self._handle)
        return self

    def __exit__(self, *exc: object) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)


def torchrun_command(
    config_path: Path,
    *,
    nproc_per_node: int,
    nnodes: int = 1,
    rdzv_endpoint: str | None = None,
    rdzv_id: str | None = None,
    python: str | None = None,
) -> list[str]:
    """The ``torchrun`` command that runs ``ehrfm pretrain --config config_path`` distributed.

    One node uses ``--standalone``; several nodes need a rendezvous endpoint
    (``host:port``) reachable from every node, and the same ``rdzv_id``.
    """
    if nproc_per_node < 1 or nnodes < 1:
        raise ValueError("nproc_per_node and nnodes must be positive")
    command = [python or sys.executable, "-m", "torch.distributed.run"]
    if nnodes == 1:
        command += ["--standalone", "--nnodes", "1"]
    else:
        if rdzv_endpoint is None:
            raise ValueError("a multi-node run needs a rendezvous endpoint (host:port)")
        command += [
            "--nnodes",
            str(nnodes),
            "--rdzv-backend",
            "c10d",
            "--rdzv-endpoint",
            rdzv_endpoint,
            "--rdzv-id",
            rdzv_id or "ehrfm-pretrain",
        ]
    command += [
        "--nproc-per-node",
        str(nproc_per_node),
        "-m",
        "ehrfm",
        "pretrain",
        "--config",
        str(config_path),
    ]
    return command


__all__ = [
    "NonFiniteLossError",
    "SignalStop",
    "StopRequest",
    "TrainingResult",
    "World",
    "run_pretraining",
    "torchrun_command",
    "world_from_env",
]
