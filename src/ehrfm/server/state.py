"""Startup loading: encoder, tokenizer, batch builder and heads, failing closed.

:func:`load_state` turns a :class:`~ehrfm.server.config.ServerConfig` into the
objects a request touches. Any problem raises, and the server never starts:
a checkpoint that does not load, a weight-pin mismatch, a head that refuses
the encoder, a tampered head, a missing or empty head registry, or an unknown
postprocess hook. There is no fallback head and no fallback transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from femr.models.tokenizer import HierarchicalTokenizer

from ehrfm.data.batches import BatchBuilder
from ehrfm.data.tokenization import CodeOntology, ontology_from_extract
from ehrfm.inference.head_loader import HeadLoadError, LoadedHead
from ehrfm.inference.registry import EncoderIdentity, HeadRegistry
from ehrfm.inputs.artifacts import open_extract
from ehrfm.models import _compat
from ehrfm.models.checkpoint import load_tokenizer
from ehrfm.models.representations import ClmbrEncoder
from ehrfm.server.config import ServerConfig
from ehrfm.server.postprocess import registered_postprocess

_compat.install()


@dataclass(frozen=True, slots=True)
class AppState:
    """Everything a request touches, built once at startup."""

    config: ServerConfig
    encoder: ClmbrEncoder
    tokenizer: HierarchicalTokenizer
    builder: BatchBuilder
    heads: dict[str, LoadedHead]
    record_sha256: dict[str, str]
    """SHA-256 of each loaded head's ``head.json``, by registry name."""
    known_codes: frozenset[str]
    """The dictionary's code entries, for unknown-code counting."""
    postprocess: str

    @property
    def encoder_weights_sha256(self) -> str:
        """SHA-256 of the encoder weights every head is bound to."""
        return self.encoder.weights.sha256


def _ontology(config: ServerConfig, extract: Path | None) -> CodeOntology:
    """The ontology for ancestor expansion (see ``ServerConfig.ontology``)."""
    if config.ontology == "none":
        return CodeOntology({})
    if extract is None:
        raise ValueError('ontology="extract" needs an extract path')
    return ontology_from_extract(open_extract(extract))


def load_state(config: ServerConfig) -> AppState:
    """Load everything the server needs; any failure raises before serving starts."""
    model_dir: Path
    heads_dir: Path
    extract: Path | None = config.extract_dir
    if config.toy:
        from ehrfm.server.toy import prepare_toy

        serving = prepare_toy(config)
        model_dir = serving.model_dir
        heads_dir = serving.heads_dir
        # The toy head is fitted through the toy extract's ontology; serving it
        # without that ontology would feed it inputs unlike its training inputs.
        extract = serving.extract
    else:
        if config.model_dir is None or config.heads_dir is None:
            raise ValueError("pass model_dir and heads_dir, or set toy=true")
        model_dir = config.model_dir
        heads_dir = config.heads_dir
    encoder = ClmbrEncoder.load(
        model_dir,
        device=config.device,
        expected_weights_sha256=config.expected_weights_sha256,
    )
    if not heads_dir.is_dir():
        raise HeadLoadError(f"no head registry directory at {heads_dir}")
    identity = EncoderIdentity(
        hidden_size=encoder.hidden_size,
        weights_sha256=encoder.weights.sha256,
    )
    # One pass over the registry: the digest reported with each prediction is
    # pinned while the record is loaded, so a head swapped in between the two
    # steps cannot serve under a stale digest.
    registry = HeadRegistry(heads_dir)
    entries = registry.entries()
    if not entries:
        raise HeadLoadError(f"the head registry at {heads_dir} holds no heads")
    heads = {
        entry.name: registry.load(entry.name, identity, expected_record_sha256=entry.record_sha256)
        for entry in entries
    }
    record_sha256 = {entry.name: entry.record_sha256 for entry in entries}
    tokenizer = load_tokenizer(model_dir, _ontology(config, extract))
    builder = BatchBuilder(tokenizer, config.tokens_per_batch)
    registered_postprocess(config.postprocess)
    return AppState(
        config=config,
        encoder=encoder,
        tokenizer=tokenizer,
        builder=builder,
        heads=heads,
        record_sha256=record_sha256,
        known_codes=frozenset(tokenizer.code_lookup),
        postprocess=config.postprocess,
    )
