"""Load a FEMR 0.2.4 CLMBR checkpoint directory with the uniform-attention surgery.

A CLMBR-T-Base checkpoint directory holds ``config.json`` (a ``FEMRModelConfig``),
``dictionary.msgpack`` (the tokenizer dictionary) and ``model.safetensors``.
Loading it with femr 0.2.4 needs three corrections, applied here in order:

1. The released ``config.json`` omits ``transformer_config.is_hierarchical``
   (so it defaults to ``False``) although the dictionary is hierarchical; the
   flag is set to ``True`` so the model embeds through ``nn.EmbeddingBag``.
2. femr 0.2.4 builds a hawk (recurrent) block in every even-indexed layer,
   while the checkpoint was trained with uniform attention in every layer;
   each hawk layer is replaced by a uniform-attention layer.
3. The weights store the flat ``transformer.embed.weight``; it is renamed to
   ``transformer.embed_bag.weight`` to match step 1.

The state dict is then loaded strictly: any missing or unexpected key raises.
The weight bytes are hashed as they are read (:mod:`ehrfm.models.weights`), so
every load records the SHA-256 of exactly the tensors it used; a pinned hash
that does not match fails before any tensor is built.

The released dictionary records ``age_stats`` as one flat ``{mean, std}`` of
age in seconds, while femr 0.2.4's tokenizer reads per-feature statistics
(``age``, ``log_age``, ``delta``, ``log_delta``); :func:`load_tokenizer`
expands the flat form. Those statistics only feed femr's normalised-age input,
which CLMBR-T-Base leaves disabled (``use_normed_ages`` is false).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import femr.models.transformer
import femr.ontology
from femr.models.config import FEMRModelConfig, FEMRTaskConfig
from femr.models.tokenizer import HierarchicalTokenizer

from ehrfm.models.weights import WeightProvenance, read_weights

CONFIG_FILE: Final = "config.json"
DICTIONARY_FILE: Final = "dictionary.msgpack"
WEIGHTS_FILE: Final = "model.safetensors"
_FLAT_EMBED: Final = "transformer.embed.weight"
_BAG_EMBED: Final = "transformer.embed_bag.weight"


class CheckpointError(RuntimeError):
    """A checkpoint directory is incomplete or its weights do not fit the model."""


@dataclass(frozen=True, slots=True)
class SurgeryReport:
    """What the loader changed to fit the checkpoint to femr 0.2.4."""

    hierarchical_patched: bool
    hawk_layers_replaced: int
    embedding_remapped: bool


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """An evaluation-mode model with what was changed and which weights were used."""

    model: femr.models.transformer.FEMRModel
    surgery: SurgeryReport
    weights: WeightProvenance


def require_layout(model_dir: Path) -> None:
    """Fail closed unless ``model_dir`` holds the three checkpoint files."""
    missing = [
        name
        for name in (CONFIG_FILE, DICTIONARY_FILE, WEIGHTS_FILE)
        if not (model_dir / name).is_file()
    ]
    if missing:
        raise CheckpointError(f"checkpoint directory {model_dir} is missing {missing}")


def read_config(model_dir: Path) -> FEMRModelConfig:
    """Read ``config.json`` of a checkpoint directory as a ``FEMRModelConfig``."""
    config = FEMRModelConfig.from_pretrained(str(model_dir))
    if not isinstance(config, FEMRModelConfig):
        raise CheckpointError(f"{model_dir / CONFIG_FILE} is not a FEMR model config")
    return config


def uniform_attention_model(
    config: FEMRModelConfig,
) -> tuple[femr.models.transformer.FEMRModel, int]:
    """Build a ``FEMRModel`` from ``config`` with every hawk layer replaced.

    Returns the model and the number of layers replaced.
    """
    model = femr.models.transformer.FEMRModel(config)
    layers = model.transformer.layers
    replaced = 0
    for index, layer in enumerate(layers):
        if layer.use_hawk:
            layers[index] = femr.models.transformer.FEMREncoderLayer(
                config.transformer_config, use_hawk=False
            )
            replaced += 1
    return model, replaced


def load_checkpoint(
    model_dir: Path,
    task_config: FEMRTaskConfig | None = None,
    expected_weights_sha256: str | None = None,
) -> LoadedCheckpoint:
    """Load the checkpoint in ``model_dir`` as an evaluation-mode ``FEMRModel``.

    ``task_config`` replaces the checkpoint's task head; ``None`` keeps the head
    recorded in ``config.json``. Weights of a replaced head are not loaded.
    ``expected_weights_sha256`` pins the weight file; a different file raises
    :class:`ehrfm.models.weights.WeightIntegrityError`.
    """
    require_layout(model_dir)
    config = read_config(model_dir)
    transformer_config = config.transformer_config
    hierarchical_patched = not transformer_config.is_hierarchical
    transformer_config.is_hierarchical = True
    keep_head = task_config is None
    if not keep_head:
        config.task_config = task_config

    state, provenance = read_weights(model_dir / WEIGHTS_FILE, expected_weights_sha256)
    model, replaced = uniform_attention_model(config)

    remapped = _FLAT_EMBED in state
    if remapped:
        state[_BAG_EMBED] = state.pop(_FLAT_EMBED)
    if not keep_head:
        state = {key: value for key, value in state.items() if not key.startswith("task_model.")}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if not keep_head:
        missing = [key for key in missing if not key.startswith("task_model.")]
    if missing or unexpected:
        raise CheckpointError(
            f"weights in {model_dir} do not fit the model: missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}"
        )
    model.eval()
    report = SurgeryReport(
        hierarchical_patched=hierarchical_patched,
        hawk_layers_replaced=replaced,
        embedding_remapped=remapped,
    )
    return LoadedCheckpoint(model=model, surgery=report, weights=provenance)


def load_femr_model(
    model_dir: Path,
    task_config: FEMRTaskConfig | None = None,
) -> tuple[femr.models.transformer.FEMRModel, SurgeryReport]:
    """Load the checkpoint as in :func:`load_checkpoint`; return the model and its surgery."""
    loaded = load_checkpoint(model_dir, task_config)
    return loaded.model, loaded.surgery


def nested_age_stats(age_stats: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Expand a flat ``{mean, std}`` age statistic into femr 0.2.4's per-feature form.

    The release records no inter-event statistics, so ``delta`` reuses the age
    scale and ``log_delta`` is left unscaled; both keep every normalised value
    finite in femr's float16 ``time_data``. Already-nested statistics are
    returned unchanged.
    """
    if "age" in age_stats:
        return age_stats
    mean = float(age_stats["mean"])
    std = float(age_stats["std"])
    return {
        "age": {"mean": mean, "std": std},
        "log_age": {"mean": math.log(mean + 1), "std": std / (mean + 1)},
        "delta": {"mean": 0.0, "std": std},
        "log_delta": {"mean": 0.0, "std": 1.0},
    }


def load_tokenizer(model_dir: Path, ontology: femr.ontology.Ontology) -> HierarchicalTokenizer:
    """Load the checkpoint's ``dictionary.msgpack`` as a femr ``HierarchicalTokenizer``."""
    require_layout(model_dir)
    tokenizer = HierarchicalTokenizer.from_pretrained(str(model_dir), ontology=ontology)
    dictionary = cast(dict[str, Any], tokenizer.dictionary)
    if not dictionary.get("is_hierarchical", False):
        raise CheckpointError(f"{model_dir / DICTIONARY_FILE} is not a hierarchical dictionary")
    dictionary["age_stats"] = nested_age_stats(dictionary["age_stats"])
    return tokenizer


__all__ = [
    "CONFIG_FILE",
    "DICTIONARY_FILE",
    "WEIGHTS_FILE",
    "CheckpointError",
    "LoadedCheckpoint",
    "SurgeryReport",
    "load_checkpoint",
    "load_femr_model",
    "load_tokenizer",
    "nested_age_stats",
    "read_config",
    "require_layout",
    "uniform_attention_model",
]
