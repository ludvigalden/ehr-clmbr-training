"""Postprocess hooks: the configurable transform applied to every head output.

A postprocess hook takes the head's per-head output values and the head's
record, and returns the values a response reports. The registry is filled by
importing modules that call :func:`register_postprocess`; the shipped hook is
the neutral default ``raw``, which returns the head's output unchanged — the
raw probabilities for a boolean head, the raw scores for integer and float
heads, and the raw category probabilities for a categorical head.

Selecting an unregistered name fails startup closed; there is no fallback
transform.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Final

from ehrfm.inference.head_loader import HeadRecord

type Postprocess = Callable[[Sequence[float], HeadRecord], list[float]]

_POSTPROCESS: dict[str, Postprocess] = {}


def register_postprocess(name: str, fn: Postprocess) -> None:
    """Register ``fn`` under ``name``; a name may be registered only once."""
    if name in _POSTPROCESS:
        raise ValueError(f"postprocess hook {name!r} is already registered")
    _POSTPROCESS[name] = fn


def registered_postprocess(name: str) -> Postprocess:
    """The hook registered under ``name``; an unknown name raises ``ValueError``."""
    fn = _POSTPROCESS.get(name)
    if fn is None:
        raise ValueError(f"unknown postprocess hook {name!r}; registered: {sorted(_POSTPROCESS)}")
    return fn


def postprocess_names() -> tuple[str, ...]:
    """The registered hook names, sorted."""
    return tuple(sorted(_POSTPROCESS))


def apply_postprocess(name: str, values: Sequence[float], record: HeadRecord) -> list[float]:
    """Apply the hook ``name`` to ``values``, checking the result's shape.

    A hook must return exactly one finite value per output dimension; anything
    else is a loud error, never a degraded response.
    """
    transformed = registered_postprocess(name)(values, record)
    if len(transformed) != record.spec.output_dim:
        raise ValueError(
            f"postprocess hook {name!r} returned {len(transformed)} values for a head "
            f"with {record.spec.output_dim} outputs"
        )
    for value in transformed:
        if not math.isfinite(value):
            raise ValueError(f"postprocess hook {name!r} returned a non-finite value")
    return transformed


def _raw(values: Sequence[float], record: HeadRecord) -> list[float]:
    """The neutral default: the head's output, unchanged."""
    return list(values)


register_postprocess("raw", _raw)


__all__: Final = [
    "Postprocess",
    "apply_postprocess",
    "postprocess_names",
    "register_postprocess",
    "registered_postprocess",
]
