"""Toy serving inputs: the toy checkpoint and a head fitted on the fly.

``--toy`` serves the generated toy checkpoint (see
:mod:`ehrfm.toy.checkpoint`) plus a boolean ``linear_probe`` head fitted on
the fly from the toy label set: the extract is verified, its meds_reader
database is built under the toy work directory, the encoder embeds the label
set's ``train`` and ``tuning`` prediction times, and
:func:`ehrfm.heads.probe.fit_linear_probe` fits the head, bound to the toy
encoder's weights, and publishes it into ``<work>/registry`` through
:meth:`~ehrfm.inference.registry.HeadRegistry.publish`.

Every input regenerates byte-identically from the seed, so a restart reuses
the fitted head: as long as the registry holds a head, fitting is skipped.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ehrfm.config.heads import LinearProbeConfig
from ehrfm.data.batches import BatchBuilder
from ehrfm.data.prediction_times import labelled_times
from ehrfm.data.tokenization import extract_tokenizer
from ehrfm.heads.probe import fit_linear_probe
from ehrfm.inference.head_loader import save_head
from ehrfm.inference.registry import HeadRegistry
from ehrfm.inputs.artifacts import open_extract, open_label_set
from ehrfm.inputs.contract import Split
from ehrfm.inputs.database import open_database
from ehrfm.models import _compat
from ehrfm.models.representations import ClmbrEncoder
from ehrfm.server.config import ServerConfig, default_toy_work_dir
from ehrfm.toy.pretraining import ToyInputs, write_toy_experiment_inputs

_compat.install()

REGISTRY_SUBDIR: Final = "registry"
DATABASE_SUBDIR: Final = "database"

TOY_C_GRID: Final = (1e-2, 1e-1, 1.0, 10.0)


@dataclass(frozen=True, slots=True)
class ToyServing:
    """The paths ``--toy`` serves from."""

    model_dir: Path
    heads_dir: Path
    extract: Path
    """The toy extract; the tokenizer's ancestor source (matching head training)."""


def prepare_toy(config: ServerConfig) -> ToyServing:
    """The toy checkpoint and head registry under ``config.toy_work_dir``."""
    work = config.toy_work_dir or default_toy_work_dir()
    inputs = write_toy_experiment_inputs(work, subjects=config.toy_subjects)
    heads_dir = work / REGISTRY_SUBDIR
    if not HeadRegistry(heads_dir).names():
        _fit_toy_head(inputs, heads_dir, work / DATABASE_SUBDIR)
    return ToyServing(
        model_dir=inputs.checkpoint,
        heads_dir=heads_dir,
        extract=inputs.extract,
    )


def _fit_toy_head(inputs: ToyInputs, heads_dir: Path, database_cache: Path) -> None:
    """Fit one boolean linear head on the toy label set and publish it."""
    extract = open_extract(inputs.extract)
    labels = open_label_set(inputs.labels, extract)
    encoder = ClmbrEncoder.load(inputs.checkpoint, device="cpu")
    tokenizer = extract_tokenizer(inputs.checkpoint, extract)
    builder = BatchBuilder(tokenizer)
    database = open_database(extract, database_cache)
    task = labels.task.name
    split_features: dict[Split, NDArray[np.float64]] = {}
    split_labels: dict[Split, NDArray[np.float64]] = {}
    for split in ("train", "tuning"):
        rows = labelled_times(labels, extract, splits={split})
        times = [row.at for row in rows]
        representations = encoder.representations_at(database, times, builder)
        split_features[split] = representations.embeddings.numpy().astype(np.float64)
        split_labels[split] = np.asarray([row.value for row in rows], dtype=np.float64)
    fit = fit_linear_probe(
        task,
        split_features["train"],
        split_labels["train"],
        split_features["tuning"],
        split_labels["tuning"],
        config=LinearProbeConfig(c_grid=TOY_C_GRID, max_iter=200),
    )
    registry = HeadRegistry(heads_dir)
    staging = heads_dir.parent / "head-staging"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        save_head(
            fit.head,
            staging,
            encoder_weights_sha256=encoder.weights.sha256,
            metadata={"fitted_by": "toy", "c": fit.c},
        )
        registry.publish(staging, task)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


__all__ = ["ToyServing", "prepare_toy"]
