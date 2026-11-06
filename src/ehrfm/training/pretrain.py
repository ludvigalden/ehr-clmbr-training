"""CLMBR next-code pretraining on femr 0.2.4: task, packed batches, steps, export.

**Task.** femr's ``CLMBRTask`` predicts the next event's code, restricted to
the first ``clmbr_vocab_size`` dictionary entries. In femr 0.2.4 it asserts a
single feature per event and does not accept the ``actually_add`` argument
femr's own batch creator passes; with the hierarchical tokenizer an event has
one feature per ancestor in the dictionary (plus value tokens), so the stock
task cannot run. :class:`ClmbrPretrainingTask` keeps femr's semantics and
picks, deterministically, the feature with the lowest dictionary position
among those inside the head (dictionary entries are ordered by information
weight, so it is the most informative one); an event with no such feature
gives no label, as in femr.

**Batches.** Training subjects are packed with femr's
``FEMRBatchProcessor.convert_dataset`` into batches of at most
``tokens_per_batch`` tokens (one subject per batch at minimum; a subject
longer than the budget contributes its last window). Each packed batch is one
optimizer step. The packed dataset is stored under the run's
``tokenized/<key>/`` keyed by everything that produces it, so reruns and
resumed runs do not tokenize again.

**Steps.** Each epoch visits the packed batches in a permutation seeded by
``(seed, epoch)``. With ``world_size`` processes, step ``s`` of an epoch
gives process ``r`` the batch at permutation position ``s * world_size + r``;
leftover batches that do not fill a step are skipped. The order is a pure
function of the seed, so resuming at a step reproduces the uninterrupted
sequence.

**Export.** The trained model is written in the layout the loader reads:
``config.json`` (the base config with ``is_hierarchical`` set),
``dictionary.msgpack`` (the base bytes, or the expanded dictionary) and
``model.safetensors`` (with the ``embed_bag`` key), so
:func:`ehrfm.models.checkpoint.load_checkpoint` and
:class:`ehrfm.models.representations.ClmbrEncoder` read it unchanged.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import hashlib
import json
import math
import random
import tempfile
from collections.abc import Callable, Generator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import datasets
import femr.ontology
import meds_reader
import msgpack
import numpy as np
import safetensors.torch
import torch
from femr.models.processor import FEMRBatchProcessor
from femr.models.tasks import CLMBRTask
from femr.models.tokenizer import HierarchicalTokenizer
from femr.models.transformer import FEMRModel
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR

from ehrfm.config.training import OptimizationConfig
from ehrfm.inputs.manifest import sha256_file
from ehrfm.models.checkpoint import (
    CONFIG_FILE,
    DICTIONARY_FILE,
    WEIGHTS_FILE,
    nested_age_stats,
)
from ehrfm.training.cache import staged_directory, write_bytes_atomic, write_json_atomic

TOKENIZATION_FORMAT: Final = 1
_DATASET_COMPLETE: Final = "tokenized.json"


class EmptyTrainingSetError(ValueError):
    """The training subjects yield no labelled positions to learn from."""


class ClmbrPretrainingTask(CLMBRTask):
    """femr's CLMBR task, usable with the hierarchical tokenizer (see the module docstring)."""

    def add_event(
        self,
        current_date: datetime.datetime,
        next_date: datetime.datetime | None,
        next_features: Sequence[int] | None = None,
        actually_add: bool = True,
    ) -> int:
        """Record the next event's label; return how many labels were added (0 or 1)."""
        del current_date, next_date
        if next_features is None:
            return 0
        # femr's dry run (actually_add=False) only counts labels; recording one is harmless
        # because every subject starts with an empty label list.
        del actually_add
        in_head = [feature for feature in next_features if feature < self.clmbr_vocab_size]
        if not in_head:
            return 0
        self.per_subject_batch_labels.append(min(in_head))
        return 1


def training_tokenizer(
    dictionary: Mapping[str, Any], ontology: femr.ontology.Ontology
) -> HierarchicalTokenizer:
    """A hierarchical tokenizer over ``dictionary`` (age statistics expanded for femr 0.2.4)."""
    payload = copy.deepcopy(dict(dictionary))
    if not payload.get("is_hierarchical", False):
        raise ValueError("continued pretraining needs a hierarchical dictionary")
    payload["age_stats"] = nested_age_stats(payload["age_stats"])
    return HierarchicalTokenizer(payload, ontology=ontology)


def dictionary_bytes(dictionary: Mapping[str, Any]) -> bytes:
    """The msgpack encoding of a dictionary payload."""
    return cast(bytes, msgpack.packb(dict(dictionary)))


@contextlib.contextmanager
def _generator_in_process(cache_dir: Path) -> Generator[None]:
    """Make femr's ``Dataset.from_generator`` run in-process and cache under ``cache_dir``.

    femr passes ``num_proc // 4``, which is 0 for fewer than four workers and
    makes ``datasets`` fail; it also leaves the cache at the user-wide
    default. Both are pinned here for the duration of the conversion.
    """
    original = datasets.Dataset.from_generator

    def from_generator(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("num_proc") is not None and int(kwargs["num_proc"]) < 2:
            kwargs["num_proc"] = None
        kwargs["cache_dir"] = str(cache_dir)
        return original(*args, **kwargs)

    setattr(datasets.Dataset, "from_generator", from_generator)  # noqa: B010
    try:
        yield
    finally:
        setattr(datasets.Dataset, "from_generator", original)  # noqa: B010


def tokenization_key(
    *,
    extract_manifest_sha256: str,
    dictionary_sha256: str,
    clmbr_vocab_size: int,
    tokens_per_batch: int,
    subject_ids: Sequence[int],
    femr_version: str,
) -> str:
    """Key of a packed dataset: everything that determines its batches."""
    payload = {
        "format": TOKENIZATION_FORMAT,
        "extract": extract_manifest_sha256,
        "dictionary": dictionary_sha256,
        "clmbr_vocab_size": clmbr_vocab_size,
        "tokens_per_batch": tokens_per_batch,
        "subjects": hashlib.sha256(
            ",".join(str(s) for s in sorted(subject_ids)).encode()
        ).hexdigest(),
        "femr": femr_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def packed_batches(
    database_path: Path,
    subject_ids: Sequence[int],
    processor: FEMRBatchProcessor,
    *,
    tokens_per_batch: int,
    store: Path,
    key: str,
    workers: int = 1,
) -> datasets.Dataset:
    """Packed CLMBR batches of ``subject_ids``, read from ``store/key`` or built and stored there.

    The dataset directory is assembled in a temporary sibling and renamed into
    place with its completion marker, so a partial build is never reused.
    """
    stored = load_packed_batches(store, key)
    if stored is not None:
        return stored
    if not subject_ids:
        raise EmptyTrainingSetError("no subjects to tokenize")
    store.mkdir(parents=True, exist_ok=True)
    target = store / key
    with (
        staged_directory(target) as staging,
        tempfile.TemporaryDirectory(prefix=".generator-", dir=store) as scratch,
    ):
        with (
            meds_reader.SubjectDatabase(str(database_path), num_threads=workers) as database,
            _generator_in_process(Path(scratch)),
        ):
            selected = database.filter(list(subject_ids))
            try:
                dataset = processor.convert_dataset(
                    selected,
                    tokens_per_batch=tokens_per_batch,
                    min_subjects_per_batch=1,
                    num_proc=workers,
                )
            except AssertionError as exc:
                raise EmptyTrainingSetError(
                    f"{len(subject_ids)} subjects yield no next-code labels inside the head"
                ) from exc
        if len(dataset) == 0:
            raise EmptyTrainingSetError("tokenization produced no batches")
        dataset.save_to_disk(str(staging / "data"))
        write_json_atomic(staging / _DATASET_COMPLETE, {"batches": len(dataset), "key": key})
    stored = load_packed_batches(store, key)
    if stored is None:
        raise RuntimeError(f"packed batches were not stored under {target}")
    return stored


def load_packed_batches(store: Path, key: str) -> datasets.Dataset | None:
    """A previously stored packed dataset, or None when absent or incomplete."""
    target = store / key
    if not (target / _DATASET_COMPLETE).is_file():
        return None
    loaded = datasets.load_from_disk(str(target / "data"))
    if not isinstance(loaded, datasets.Dataset):
        raise RuntimeError(f"{target} does not hold a single packed dataset")
    return loaded


def steps_per_epoch(n_batches: int, world_size: int) -> int:
    """Optimizer steps in one epoch: every process gets one batch per step."""
    return n_batches // world_size


def total_steps(n_batches: int, optimization: OptimizationConfig, world_size: int) -> int:
    """All optimizer steps of the run (``max_steps`` caps the epochs)."""
    per_epoch = steps_per_epoch(n_batches, world_size)
    if per_epoch < 1:
        raise EmptyTrainingSetError(
            f"{n_batches} packed batches cannot give each of {world_size} processes one per step"
        )
    steps = per_epoch * optimization.epochs
    if optimization.max_steps is not None:
        steps = min(steps, optimization.max_steps)
    return steps


def epoch_order(n_batches: int, seed: int, epoch: int) -> list[int]:
    """Batch visiting order of one epoch: a permutation seeded by ``(seed, epoch)``."""
    generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return [int(i) for i in torch.randperm(n_batches, generator=generator)]


def batch_index(
    step: int, n_batches: int, seed: int, rank: int, world_size: int
) -> tuple[int, int]:
    """``(epoch, dataset row)`` of process ``rank`` at global optimizer ``step``."""
    per_epoch = steps_per_epoch(n_batches, world_size)
    epoch, within = divmod(step, per_epoch)
    return epoch, epoch_order(n_batches, seed, epoch)[within * world_size + rank]


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch; optionally request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def make_optimizer(model: nn.Module, optimization: OptimizationConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only (not on biases, norms or the embedding bag)."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or "norm" in name or "embed_bag" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": optimization.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=optimization.learning_rate,
    )


def lr_factor(step: int, steps: int, optimization: OptimizationConfig) -> float:
    """Learning-rate multiplier at ``step``: linear warmup, then constant or cosine to zero."""
    warmup = optimization.warmup_steps
    if warmup and step < warmup:
        return (step + 1) / warmup
    if optimization.schedule == "constant":
        return 1.0
    span = max(1, steps - warmup)
    progress = min(1.0, (step - warmup) / span)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_scheduler(
    optimizer: torch.optim.Optimizer, steps: int, optimization: OptimizationConfig
) -> LambdaLR:
    """The run's learning-rate schedule."""
    schedule: Callable[[int], float] = lambda step: lr_factor(step, steps, optimization)  # noqa: E731
    return LambdaLR(optimizer, schedule)


def to_device(data: Any, device: torch.device) -> Any:
    """Move every tensor in a nested batch to ``device``."""
    if isinstance(data, Mapping):
        mapping = cast(Mapping[str, Any], data)
        return {key: to_device(value, device) for key, value in mapping.items()}
    if isinstance(data, Tensor):
        return data.to(device)
    return data


def collated(processor: FEMRBatchProcessor, row: Mapping[str, Any]) -> Mapping[str, Any]:
    """One stored packed batch, collated for ``FEMRModel(**batch)``."""
    return processor.collate([dict(row)])


def autocast(device: torch.device, precision: str) -> contextlib.AbstractContextManager[Any]:
    """bf16 autocast on CUDA when requested; full precision otherwise."""
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def validation_loss(
    model: FEMRModel,
    dataset: datasets.Dataset,
    processor: FEMRBatchProcessor,
    device: torch.device,
    precision: str,
) -> float:
    """Mean CLMBR loss over a packed dataset, weighted by labels per batch."""
    was_training = model.training
    model.eval()
    total = 0.0
    labels = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = to_device(collated(processor, dataset[index]), device)
            count = int(batch["batch"]["transformer"]["label_indices"].shape[-1])
            with autocast(device, precision):
                loss, _ = model(**batch)
            total += float(loss) * count
            labels += count
    model.train(was_training)
    if labels == 0:
        raise EmptyTrainingSetError("validation batches carry no labels")
    return total / labels


def export_checkpoint(
    model: FEMRModel,
    base_checkpoint: Path,
    out_dir: Path,
    *,
    dictionary: Mapping[str, Any] | None,
    provenance: Mapping[str, Any],
) -> str:
    """Write ``model`` into ``out_dir`` in the loader's layout; return the weights' SHA-256.

    ``dictionary`` is the expanded dictionary, or None to copy the base
    dictionary bytes unchanged. ``out_dir`` is replaced as a whole.
    """
    config = json.loads((base_checkpoint / CONFIG_FILE).read_text(encoding="utf-8"))
    config.setdefault("transformer_config", {})["is_hierarchical"] = True
    state = {
        name: tensor.detach().to("cpu").contiguous() for name, tensor in model.state_dict().items()
    }
    with staged_directory(out_dir) as staging:
        safetensors.torch.save_file(state, str(staging / WEIGHTS_FILE), metadata={"format": "pt"})
        if dictionary is None:
            write_bytes_atomic(
                staging / DICTIONARY_FILE, (base_checkpoint / DICTIONARY_FILE).read_bytes()
            )
        else:
            write_bytes_atomic(staging / DICTIONARY_FILE, dictionary_bytes(dictionary))
        write_json_atomic(staging / CONFIG_FILE, config)
        write_json_atomic(staging / "training.json", dict(provenance))
        digest = sha256_file(staging / WEIGHTS_FILE)
    return digest


__all__ = [
    "ClmbrPretrainingTask",
    "EmptyTrainingSetError",
    "autocast",
    "batch_index",
    "collated",
    "dictionary_bytes",
    "epoch_order",
    "export_checkpoint",
    "load_packed_batches",
    "lr_factor",
    "make_optimizer",
    "make_scheduler",
    "packed_batches",
    "seed_everything",
    "steps_per_epoch",
    "to_device",
    "tokenization_key",
    "total_steps",
    "training_tokenizer",
    "validation_loss",
]
