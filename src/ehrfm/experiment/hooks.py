"""Hooks: the extension points for head training and evaluation.

A hook is a function ``(HookContext) -> HookResult`` registered under a kind
(``head`` or ``eval``) and a name. An experiment names the hooks it runs in
``tasks.heads`` and ``eval.hooks`` (each with free-form ``options``); modules
listed in ``experiment.hook_modules`` are imported first so they can register
theirs. Head hooks run first, in configuration order; evaluation hooks run
after all heads and see every head's published result.

Contract:

- A hook reads representations through ``ctx.representations`` and, for an
  evaluation hook, head outputs through ``ctx.heads``. It must not read the
  extract or the model directly; ``ctx.model`` identifies the model for
  provenance only.
- It writes any files into ``ctx.output_dir``, which is empty and is
  published atomically when the hook returns (and discarded if it raises).
  Files stay in the work directory; they never reach the tracking store.
- It returns a :class:`HookResult`. ``metrics`` (flat ``a/b`` names to
  floats) and ``summary`` (JSON) are logged to the tracking store, both
  through the small-cell suppression policy; name count fields so the
  policy recognises them (for example ``n_positive``).
- It is deterministic given its context and ``ctx.seed``: its result is
  cached under a key over the representations, the head results it can see
  (for evaluation hooks), its name, ``version`` and options. Bump ``version``
  when the hook's logic changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ehrfm.config.experiment import ExperimentConfig
from ehrfm.experiment.representations import RepresentationSet, load_set
from ehrfm.inputs.contract import Split

HookKind = Literal["head", "eval"]


class HookError(RuntimeError):
    """A hook is unknown, registered twice or returned an invalid result."""


class HookResult(BaseModel):
    """What a hook reports (logged through suppression)."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    metrics: dict[str, float] = Field(default_factory=dict[str, float])
    summary: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """The model the representations came from."""

    model_dir: Path
    weights_sha256: str
    training_fingerprint: str | None
    """The continued-pretraining run fingerprint, or None for the base checkpoint."""


@dataclass(frozen=True, slots=True)
class RepresentationIndex:
    """The representation sets of one experiment."""

    directory: Path
    tasks: tuple[str, ...]
    splits: tuple[Split, ...]

    def load(self, task: str, split: Split) -> RepresentationSet:
        """The set for ``task`` and ``split``."""
        if task not in self.tasks or split not in self.splits:
            raise KeyError(f"no representations for task {task!r}, split {split!r}")
        return load_set(self.directory, task, split)


@dataclass(frozen=True, slots=True)
class HookRecord:
    """A completed hook: its published directory and result."""

    name: str
    kind: HookKind
    key: str
    output_dir: Path
    result: HookResult


@dataclass(frozen=True, slots=True)
class HookContext:
    """Everything a hook may use."""

    name: str
    kind: HookKind
    options: Mapping[str, JsonValue]
    seed: int
    output_dir: Path
    config: ExperimentConfig
    model: ModelIdentity
    representations: RepresentationIndex
    heads: Mapping[str, HookRecord]
    """Completed head hooks by name (empty for head hooks)."""


Hook = Callable[[HookContext], HookResult]


@dataclass(frozen=True, slots=True)
class RegisteredHook:
    """A hook function with its kind, name and logic version."""

    kind: HookKind
    name: str
    version: str
    function: Hook


_REGISTRY: dict[tuple[HookKind, str], RegisteredHook] = {}


def register_hook(kind: HookKind, name: str, function: Hook, *, version: str = "1") -> None:
    """Register ``function`` as the ``kind`` hook ``name``; a name registers once."""
    if (kind, name) in _REGISTRY:
        raise HookError(f"a {kind} hook named {name!r} is already registered")
    _REGISTRY[(kind, name)] = RegisteredHook(kind, name, version, function)


def unregister_hook(kind: HookKind, name: str) -> None:
    """Remove a registration (missing names are ignored)."""
    _REGISTRY.pop((kind, name), None)


def registered_hooks(kind: HookKind) -> tuple[str, ...]:
    """Names registered for ``kind``, sorted."""
    return tuple(sorted(name for registered, name in _REGISTRY if registered == kind))


def get_hook(kind: HookKind, name: str) -> RegisteredHook:
    """The registration of ``kind`` hook ``name``."""
    try:
        return _REGISTRY[(kind, name)]
    except KeyError:
        known = ", ".join(registered_hooks(kind)) or "none"
        raise HookError(f"no {kind} hook named {name!r} is registered (known: {known})") from None


__all__ = [
    "Hook",
    "HookContext",
    "HookError",
    "HookKind",
    "HookRecord",
    "HookResult",
    "ModelIdentity",
    "RegisteredHook",
    "RepresentationIndex",
    "get_hook",
    "register_hook",
    "registered_hooks",
    "unregister_hook",
]
