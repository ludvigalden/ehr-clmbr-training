"""Configuration of task-head training.

A head is trained by a head hook named in ``tasks.heads`` (see
``configs/tasks/``); its ``options`` are validated against the models here,
strictly and frozen, so a misspelled key or a wrong type fails before any
fitting starts. Options are part of the hook's cache key.
"""

from __future__ import annotations

import math
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EHRSHOT_C_GRID: Final = (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 1e2, 1e4)
"""Inverse L2 strengths of the EHRSHOT benchmark's default regularisation
search (Wornow et al., NeurIPS D&B 2023, ``LR_PARAMS`` in
``ehrshot/utils.py``). This is the single default grid for every logistic
probe in the package: the ``linear_probe`` head hook, the ``logistic_probe``,
``age_baseline`` and ``covariate_baseline`` reference heads, the ``few_shot``
evaluation hook, and :func:`ehrfm.eval.baselines.fit_logistic`."""

ClassWeight = Literal["none", "balanced"]


class LinearProbeConfig(BaseModel):
    """A logistic linear head fitted on frozen encoder representations.

    The objective is the scikit-learn one: ``sum_i s_i * log_loss_i +
    ||w||^2 / (2 C)``, where ``s_i`` is 1 (``class_weight: none``) or
    ``n / (2 n_class)`` (``balanced``). Features are standardised on the
    training split when ``standardize`` is set; the scaling is folded into the
    saved head, so the head takes raw representations.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    task: str | None = None
    """Label set to fit; may be omitted when the experiment has exactly one."""
    c_grid: tuple[float, ...] = EHRSHOT_C_GRID
    """Strengths tried; the one with the best tuning AUROC is kept (the middle
    one when the tuning split lacks a class)."""
    class_weight: ClassWeight = "balanced"
    standardize: bool = True
    max_iter: int = Field(default=500, ge=1)
    """L-BFGS iterations per strength."""

    @field_validator("c_grid")
    @classmethod
    def _positive_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(not math.isfinite(c) or c <= 0 for c in value):
            raise ValueError("c_grid must be a non-empty list of positive finite numbers")
        if len(set(value)) != len(value):
            raise ValueError("c_grid values must be distinct")
        return value


__all__ = ["EHRSHOT_C_GRID", "ClassWeight", "LinearProbeConfig"]
