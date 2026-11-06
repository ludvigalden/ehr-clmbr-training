"""Small-cell suppression of counts before they leave the work directory.

:func:`suppress` applies a :class:`~ehrfm.config.experiment.SmallCellPolicy`
to a JSON-like payload and returns a masked copy plus a log of what was masked
(paths and reasons, never values). :func:`suppress_flat` does the same for
flat ``a/b/c`` named values such as metrics, by nesting them on ``/`` first,
so metrics and JSON artifacts follow one rule set.

Rules:

1. **Cells.** A key whose name matches a ``count_fields`` pattern holds a
   count. A number under it is a cell; when it holds a list or mapping (a
   breakdown such as ``subjects_by_split``), every number inside is a cell.
   Booleans are never cells. A cell is *small* when it is below the threshold
   (0 included unless ``suppress_zero`` is off) or is not finite.
2. **Group masking** (``mask_group``). When a record (a mapping) has a small
   scalar count, every other number directly in that record is masked too,
   because rates and scores over a small cell disclose it. Nested records and
   breakdowns keep their own rules.
3. **Complementary suppression** (``complementary``). Within a breakdown, and
   down a count column of sibling records (a list of mappings, or a mapping
   of mappings with one shared key set), when exactly one cell is small the
   smallest other cell is suppressed as well, so a published total or the
   other siblings cannot recover it.

Rules 2 and 3 repeat until nothing changes. Masked values become ``None``.
The policy errs towards hiding: it may mask numbers that are not counts,
never publish a small count.
"""

from __future__ import annotations

import copy
import fnmatch
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from ehrfm.config.experiment import SmallCellPolicy

Reason = Literal["small_cell", "complementary", "group"]
_Path = tuple[str | int, ...]


class SuppressedCell(BaseModel):
    """One masked value: where it was and why (its value is never recorded)."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    path: str
    reason: Reason


@dataclass(frozen=True, slots=True)
class Suppressed:
    """A masked payload and the log of what was masked."""

    value: JsonValue
    cells: tuple[SuppressedCell, ...]

    @property
    def paths(self) -> frozenset[str]:
        """Paths of the masked values."""
        return frozenset(cell.path for cell in self.cells)


@dataclass(slots=True)
class _Cell:
    value: float
    small: bool = False
    reason: Reason = "small_cell"


@dataclass(slots=True)
class _Scan:
    cells: dict[_Path, _Cell] = field(default_factory=dict[_Path, _Cell])
    groups: list[list[_Path]] = field(default_factory=list[list[_Path]])
    """Cell sets under complementary suppression."""
    own_counts: dict[_Path, list[_Path]] = field(default_factory=dict[_Path, list[_Path]])
    """Record path -> its scalar count cells."""
    own_numbers: dict[_Path, list[_Path]] = field(default_factory=dict[_Path, list[_Path]])
    """Record path -> every number directly in it."""


def format_path(path: _Path) -> str:
    """``/a/0/b`` form of a path."""
    return "/" + "/".join(str(part) for part in path)


def is_count_field(name: str, policy: SmallCellPolicy) -> bool:
    """Whether a key (the last segment of a name) holds a count under ``policy``."""
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in policy.count_fields)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _numbers(node: Any, path: _Path) -> Iterator[tuple[_Path, float]]:
    if isinstance(node, dict):
        for key, child in node.items():
            yield from _numbers(child, (*path, str(key)))
    elif isinstance(node, list):
        for index, child in enumerate(node):
            yield from _numbers(child, (*path, index))
    else:
        number = _number(node)
        if number is not None:
            yield path, number


def _is_small(value: float, policy: SmallCellPolicy) -> bool:
    if not math.isfinite(value):
        return True
    if value == 0 and not policy.suppress_zero:
        return False
    return value < policy.threshold


def _siblings(node: Any) -> list[tuple[str | int, dict[str, Any]]] | None:
    if isinstance(node, list):
        items: list[Any] = list(node)
        if len(items) >= 2 and all(isinstance(item, dict) for item in items):
            return list(enumerate(items))
        return None
    if isinstance(node, dict) and len(node) >= 2:
        values: list[Any] = list(node.values())
        if (
            all(isinstance(item, dict) for item in values)
            and len({frozenset(item) for item in values}) == 1
        ):
            return [(str(key), item) for key, item in node.items()]
    return None


def _scan(node: Any, path: _Path, policy: SmallCellPolicy, scan: _Scan) -> None:
    if isinstance(node, dict):
        counts: list[_Path] = []
        numbers: list[_Path] = []
        for raw_key, child in node.items():
            key = str(raw_key)
            child_path = (*path, key)
            number = _number(child)
            if number is not None:
                numbers.append(child_path)
                if is_count_field(key, policy):
                    scan.cells[child_path] = _Cell(number)
                    counts.append(child_path)
                continue
            if is_count_field(key, policy) and isinstance(child, dict | list):
                breakdown = list(_numbers(child, child_path))
                for cell_path, value in breakdown:
                    scan.cells[cell_path] = _Cell(value)
                if len(breakdown) >= 2:
                    scan.groups.append([cell_path for cell_path, _ in breakdown])
                continue
            _scan(child, child_path, policy, scan)
        scan.own_counts[path] = counts
        scan.own_numbers[path] = numbers
    elif isinstance(node, list):
        for index, child in enumerate(node):
            _scan(child, (*path, index), policy, scan)
    siblings = _siblings(node)
    if siblings is not None:
        keys = sorted({key for _, record in siblings for key in record})
        for key in keys:
            if not is_count_field(key, policy):
                continue
            column = [
                (*path, name, key)
                for name, record in siblings
                if _number(record.get(key)) is not None
            ]
            if len(column) >= 2:
                scan.groups.append(column)


def _propagate(scan: _Scan, policy: SmallCellPolicy, masked: dict[_Path, Reason]) -> None:
    changed = True
    while changed:
        changed = False
        if policy.mask_group:
            for record, counts in scan.own_counts.items():
                if not any(scan.cells[path].small for path in counts):
                    continue
                for path in scan.own_numbers[record]:
                    cell = scan.cells.get(path)
                    if cell is not None:
                        if not cell.small:
                            cell.small, cell.reason, changed = True, "group", True
                    elif path not in masked:
                        masked[path] = "group"
        if policy.complementary:
            for group in scan.groups:
                cells = [scan.cells[path] for path in group]
                if sum(cell.small for cell in cells) != 1:
                    continue
                remaining = sorted(
                    (cell for cell in cells if not cell.small), key=lambda c: c.value
                )
                if remaining:
                    remaining[0].small, remaining[0].reason, changed = True, "complementary", True


def _assign(root: Any, path: _Path) -> None:
    node = root
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = None


def suppress(payload: JsonValue, policy: SmallCellPolicy) -> Suppressed:
    """Return ``payload`` with small cells (and what discloses them) replaced by ``None``."""
    root: Any = copy.deepcopy(payload)
    scan = _Scan()
    _scan(root, (), policy, scan)
    for cell in scan.cells.values():
        cell.small = _is_small(cell.value, policy)
    masked: dict[_Path, Reason] = {}
    _propagate(scan, policy, masked)
    for path, cell in scan.cells.items():
        if cell.small:
            masked[path] = cell.reason
    for path in masked:
        _assign(root, path)
    cells = tuple(
        sorted(
            (
                SuppressedCell(path=format_path(path), reason=reason)
                for path, reason in masked.items()
            ),
            key=lambda cell: cell.path,
        )
    )
    return Suppressed(value=root, cells=cells)


def nest(flat: Mapping[str, float], separator: str = "/") -> dict[str, JsonValue]:
    """Nest ``a/b/c`` names into mappings; a name that is also a prefix of another fails."""
    tree: dict[str, Any] = {}
    for name, value in flat.items():
        parts = name.split(separator)
        if not all(parts):
            raise ValueError(f"name {name!r} has an empty segment")
        node = tree
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"name {name!r} extends the value {part!r}")
            node = child
        if parts[-1] in node:
            raise ValueError(f"name {name!r} is both a value and a prefix")
        node[parts[-1]] = value
    return tree


def flatten(tree: JsonValue, separator: str = "/") -> dict[str, JsonValue]:
    """Leaf values of nested mappings by ``a/b/c`` name (the inverse of :func:`nest`)."""
    flat: dict[str, JsonValue] = {}

    def walk(node: JsonValue, prefix: str) -> None:
        if isinstance(node, dict) and node:
            for key, child in node.items():
                walk(child, f"{prefix}{separator}{key}" if prefix else key)
        else:
            flat[prefix] = node

    walk(tree, "")
    return flat


def suppress_flat(values: Mapping[str, float], policy: SmallCellPolicy) -> Suppressed:
    """Suppress flat ``a/b/c`` named numbers; masked names are dropped from the result."""
    masked = suppress(nest(values), policy)
    kept: dict[str, JsonValue] = {
        name: value for name, value in flatten(masked.value).items() if value is not None
    }
    return Suppressed(value=kept, cells=masked.cells)


__all__ = [
    "Reason",
    "Suppressed",
    "SuppressedCell",
    "flatten",
    "format_path",
    "is_count_field",
    "nest",
    "suppress",
    "suppress_flat",
]
