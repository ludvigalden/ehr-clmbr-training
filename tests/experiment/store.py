"""Read back what an MLflow file store holds, for asserting on suppression."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ehrfm.config.experiment import SmallCellPolicy
from ehrfm.experiment.suppression import is_count_field
from ehrfm.experiment.tracking import SUPPRESSED_PARAM, SUPPRESSION_ARTIFACT


@dataclass(frozen=True, slots=True)
class StoredCount:
    """A count-field value found in the store."""

    where: str
    name: str
    value: float


def _numbers(node: Any) -> Iterator[float]:
    if isinstance(node, dict):
        for child in node.values():
            yield from _numbers(child)
    elif isinstance(node, list):
        for child in node:
            yield from _numbers(child)
    elif isinstance(node, int | float) and not isinstance(node, bool):
        yield float(node)


def _json_counts(node: Any, where: str, policy: SmallCellPolicy) -> Iterator[StoredCount]:
    if isinstance(node, dict):
        for key, child in node.items():
            if is_count_field(str(key), policy):
                for value in _numbers(child):
                    yield StoredCount(where, str(key), value)
            else:
                yield from _json_counts(child, where, policy)
    elif isinstance(node, list):
        for child in node:
            yield from _json_counts(child, where, policy)


def stored_counts(mlruns: Path, policy: SmallCellPolicy) -> list[StoredCount]:
    """Every count-field number in params, metrics and JSON artifacts of the store."""
    found: list[StoredCount] = []
    for path in sorted(mlruns.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(mlruns).parts
        if "metrics" in parts:
            name = path.name
            if is_count_field(name, policy):
                for line in path.read_text().splitlines():
                    found.append(StoredCount(str(path), name, float(line.split()[1])))
        elif "params" in parts:
            name = path.name.rsplit(".", 1)[-1]
            text = path.read_text()
            if is_count_field(name, policy) and text != SUPPRESSED_PARAM:
                try:
                    value: Any = json.loads(text)
                except json.JSONDecodeError:
                    continue
                for number in _numbers(value):
                    found.append(StoredCount(str(path), name, number))
        elif "artifacts" in parts and path.suffix == ".json" and path.name != SUPPRESSION_ARTIFACT:
            found.extend(_json_counts(json.loads(path.read_text()), str(path), policy))
    return found


def small_counts(mlruns: Path, policy: SmallCellPolicy) -> list[StoredCount]:
    """Stored count values the policy should have masked."""
    return [
        count
        for count in stored_counts(mlruns, policy)
        if not math.isfinite(count.value)
        or count.value < policy.threshold
        and (count.value != 0 or policy.suppress_zero)
    ]


def stored_metric_names(mlruns: Path) -> set[str]:
    """``a/b/c`` names of every stored metric."""
    names: set[str] = set()
    for path in mlruns.rglob("*"):
        parts = path.relative_to(mlruns).parts
        if path.is_file() and "metrics" in parts:
            names.add("/".join(parts[parts.index("metrics") + 1 :]))
    return names
