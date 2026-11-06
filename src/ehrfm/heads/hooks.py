"""The ``linear_probe`` head hook: a logistic linear head on frozen representations.

The hook fits :func:`ehrfm.heads.probe.fit_linear_probe` on the ``train``
representations of one boolean task, chooses the L2 strength on ``tuning``,
and then, for every split the experiment extracted:

- publishes the positive-class probability of every row with
  :func:`ehrfm.eval.predictions.write_predictions`, so the ``binary``,
  ``paired`` evaluation hooks score it like any other head;
- reports ``<split>/n`` and ``<split>/n_positive`` (count-named, so the
  small-cell policy masks a split's metrics when a count is small).

It saves the head with :func:`ehrfm.inference.head_loader.save_head` to
``output_dir/head``, bound to the encoder weights the representations came
from. The saved record carries the training settings and selected strength,
never row counts. Options are :class:`ehrfm.config.heads.LinearProbeConfig`.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from ehrfm.config.heads import LinearProbeConfig
from ehrfm.eval.hooks import (
    HEAD_DIR,
    HookOptions,
    boolean_sets,
    boolean_task,
    parse_options,
    split_counts,
)
from ehrfm.eval.predictions import write_predictions
from ehrfm.experiment.hooks import HookContext, HookResult, register_hook
from ehrfm.heads.probe import fit_linear_probe
from ehrfm.inference.head_loader import save_head
from ehrfm.inputs.contract import Split

LINEAR_PROBE: Final = "linear_probe"
HOOK_VERSION: Final = "1"


class LinearProbeOptions(LinearProbeConfig, HookOptions):
    """The ``linear_probe`` options: exactly :class:`LinearProbeConfig`."""


def linear_probe(ctx: HookContext) -> HookResult:
    """Fit, publish predictions for and save a logistic linear head."""
    options = parse_options(LinearProbeOptions, ctx)
    task = boolean_task(ctx.representations, options.task)
    sets = boolean_sets(ctx, task)
    features: dict[Split, NDArray[np.float32]] = {
        split: loaded.embeddings for split, loaded in sets.items()
    }
    train, tuning = sets["train"], sets.get("tuning")
    assert train.labels is not None
    fit = fit_linear_probe(
        task,
        features["train"],
        train.labels,
        features.get("tuning"),
        None if tuning is None else tuning.labels,
        config=options,
    )
    metrics: dict[str, float] = {}
    for split, loaded in sets.items():
        assert loaded.labels is not None
        write_predictions(ctx.output_dir, ctx.name, loaded, fit.predict(features[split]))
        metrics.update({f"{split}/{k}": v for k, v in split_counts(loaded.labels).items()})
    tune_n = 0 if tuning is None else len(tuning)
    tune_positive = 0 if tuning is None or tuning.labels is None else int(tuning.labels.sum())
    selection: list[JsonValue] = [
        {
            "c": c,
            "n": tune_n,
            "n_positive": tune_positive,
            "auroc": value if np.isfinite(value) else None,
        }
        for c, value in fit.tuning_auroc.items()
    ]
    record = save_head(
        fit.head,
        ctx.output_dir / HEAD_DIR,
        encoder_weights_sha256=ctx.model.weights_sha256,
        metadata={
            "fitted_by": ctx.name,
            "c": fit.c,
            "class_weight": options.class_weight,
            "standardize": options.standardize,
            "iterations": fit.iterations,
            "converged": fit.converged,
            "seed": ctx.seed,
            "training_fingerprint": ctx.model.training_fingerprint,
        },
    )
    summary: dict[str, JsonValue] = {
        "task": task,
        "kind": record.spec.kind,
        "c": fit.c,
        "converged": fit.converged,
        "features": record.spec.input_dim,
        "head_weights_sha256": record.weights_sha256,
        "encoder_weights_sha256": ctx.model.weights_sha256,
        "selection": selection,
    }
    return HookResult(metrics=metrics, summary=summary)


register_hook("head", LINEAR_PROBE, linear_probe, version=HOOK_VERSION)


__all__ = ["LINEAR_PROBE", "LinearProbeOptions", "linear_probe"]
