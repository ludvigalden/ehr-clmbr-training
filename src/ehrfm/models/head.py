"""Task heads over encoder representations: the typed interface and a linear head.

A head maps ``(n, input_dim)`` representations to ``(n, output_dim)`` raw
scores (logits for classification, values for regression). Its
:class:`HeadSpec` says what it predicts and how scores become predictions:

=============  ==========  ===============================
label type     output_dim  :meth:`TaskHead.predict`
=============  ==========  ===============================
boolean        1           sigmoid, the positive-class probability
categorical    categories  softmax over the categories
integer/float  1           the score itself
=============  ==========  ===============================

Head kinds are registered by name (:func:`register_head_kind`) so that a
serialized head can be rebuilt from its spec; new kinds subclass
:class:`TaskHead` and register themselves without changing the loader.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, ClassVar, Final, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from ehrfm.inputs.models import LabelType

HEAD_FORMAT_VERSION: Final = 1

_SPEC_NAME = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]


class HeadSpec(BaseModel):
    """What a head predicts, for which encoder width, and which kind it is."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: _SPEC_NAME
    task_name: _SPEC_NAME
    label_type: LabelType
    input_dim: int = Field(ge=1)
    categories: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _categories_match_label_type(self) -> Self:
        if self.label_type == "categorical":
            if not self.categories or len(self.categories) < 2:
                raise ValueError("a categorical head needs at least two categories")
            if len(set(self.categories)) != len(self.categories):
                raise ValueError("categories must be unique")
        elif self.categories is not None:
            raise ValueError(f"a {self.label_type} head has no categories")
        return self

    @property
    def output_dim(self) -> int:
        """Number of raw scores per row."""
        return len(self.categories) if self.categories is not None else 1


class TaskHead(nn.Module, ABC):
    """Base of every task head: a module from representations to raw scores."""

    kind: ClassVar[str]

    def __init__(self, spec: HeadSpec) -> None:
        super().__init__()
        if spec.kind != self.kind:
            raise ValueError(f"spec of kind {spec.kind!r} given to a {self.kind!r} head")
        self.spec = spec

    @abstractmethod
    def forward(self, representations: Tensor) -> Tensor:
        """Raw scores of shape ``(n, spec.output_dim)``."""

    def predict(self, representations: Tensor) -> Tensor:
        """Predictions in label space (see the module table); no gradients are kept."""
        if representations.ndim != 2 or representations.shape[1] != self.spec.input_dim:
            raise ValueError(
                f"expected representations of shape (n, {self.spec.input_dim}), "
                f"got {tuple(representations.shape)}"
            )
        with torch.no_grad():
            scores = self(representations)
        match self.spec.label_type:
            case "boolean":
                return torch.sigmoid(scores[:, 0])
            case "categorical":
                return torch.softmax(scores, dim=-1)
            case "integer" | "float":
                return scores[:, 0]


class LinearHead(TaskHead):
    """A single affine layer over the representation."""

    kind: ClassVar[str] = "linear"

    def __init__(self, spec: HeadSpec) -> None:
        super().__init__(spec)
        self.linear = nn.Linear(spec.input_dim, spec.output_dim)

    def forward(self, representations: Tensor) -> Tensor:
        return self.linear(representations)


_HEAD_KINDS: dict[str, type[TaskHead]] = {}


def register_head_kind(head_type: type[TaskHead]) -> type[TaskHead]:
    """Register a head class under its ``kind`` so serialized heads of it can be rebuilt."""
    existing = _HEAD_KINDS.get(head_type.kind)
    if existing is not None and existing is not head_type:
        raise ValueError(f"head kind {head_type.kind!r} is already registered to {existing}")
    _HEAD_KINDS[head_type.kind] = head_type
    return head_type


def head_kinds() -> tuple[str, ...]:
    """Registered head kinds, sorted."""
    return tuple(sorted(_HEAD_KINDS))


def build_head(spec: HeadSpec) -> TaskHead:
    """Construct an untrained head of the kind ``spec`` names."""
    head_type = _HEAD_KINDS.get(spec.kind)
    if head_type is None:
        raise ValueError(f"unknown head kind {spec.kind!r}; registered: {list(head_kinds())}")
    return head_type(spec)


register_head_kind(LinearHead)


__all__ = [
    "HEAD_FORMAT_VERSION",
    "HeadSpec",
    "LinearHead",
    "TaskHead",
    "build_head",
    "head_kinds",
    "register_head_kind",
]
