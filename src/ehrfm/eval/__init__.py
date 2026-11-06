"""Evaluation toolkit: metrics, statistical tests, baselines and evaluation hooks.

- :mod:`ehrfm.eval.metrics`: discrimination, calibration, decision curves,
  paired tests, DeLong inference and forgetting summaries;
- :mod:`ehrfm.eval._bootstrap`: seeded, stratified percentile bootstrap;
- :mod:`ehrfm.eval.evaluator`: one scored split in, one record out;
- :mod:`ehrfm.eval.predictions`: the contract by which a head's probabilities
  reach the evaluator;
- :mod:`ehrfm.eval.stratification` and :mod:`ehrfm.eval.tables`: strata from
  user-supplied per-subject columns;
- :mod:`ehrfm.eval.baselines`, :mod:`ehrfm.eval.fewshot` and
  :mod:`ehrfm.eval.next_token`: probes, covariate baselines, the EHRSHOT
  few-shot protocol and the held-out next-code loss;
- :mod:`ehrfm.eval.hooks`: all of the above as experiment pipeline hooks.

Importing this package does not register the hooks; the pipeline imports
:mod:`ehrfm.eval.hooks` itself.
"""

from ehrfm.eval.evaluator import (
    BinaryEvaluation,
    EvaluationSettings,
    PairedEvaluation,
    compare_binary,
    evaluate_binary,
)
from ehrfm.eval.predictions import (
    BinaryPredictions,
    PredictionError,
    load_predictions,
    write_predictions,
)

__all__ = [
    "BinaryEvaluation",
    "BinaryPredictions",
    "EvaluationSettings",
    "PairedEvaluation",
    "PredictionError",
    "compare_binary",
    "evaluate_binary",
    "load_predictions",
    "write_predictions",
]
