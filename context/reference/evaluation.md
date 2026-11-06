# Evaluation

`ehrfm.eval` is a statistical evaluation toolkit exposed as pipeline
eval hooks (`binary`, `paired`, `few_shot`, `forgetting`), with every
metric unit-tested against hand tables and independent reference
implementations.

## Components

- **Discrimination**: AUROC/AUPRC with **stratified percentile
  bootstrap** CIs (default 10,000 resamples, chunked index draws so
  memory stays bounded; alpha 0.05). Seeded, so results are
  deterministic and cache-sound.
- **DeLong**: `eval/_delong.py` is vendored from
  `yandexdataschool/roc_comparison` (`compare_auc_delong_xu.py`, MIT,
  Copyright (c) 2021 Nikita Kazeev; see NOTICE and
  `LICENSES/MIT-roc_comparison.txt`, which carries the full permission
  notice) implementing Xu Sun & Weichao Xu, "Fast Implementation of
  DeLong's Algorithm…", IEEE Signal Processing Letters 21(11),
  1389–1393, 2014. The module docstring records the upstream source, the
  copyright and licence pointer, the
  citation and the adjustments made (type annotations, removal of an
  unused import); the body is a near-verbatim transcription. Tests pin
  it against the
  Hanley & McNeil 1982 rating-scale table (AUROC ≈ 0.893) and an
  independent O(n²) structural-components implementation (DeLong,
  DeLong & Clarke-Pearson 1988) agreeing to 1e-12, single and paired.
- **Calibration**: binned ECE, Brier score, and a logistic
  calibration fit (slope/intercept via Newton; the intercept is
  calibration-in-the-large).
- **Decision-curve analysis**: `net_benefit` / treat-all; agrees with
  the `dcurves` package to 1e-12 (default threshold range 0.01–0.20).
- **Paired tests**: DeLong paired z, paired bootstrap of the AUROC
  difference (same resample indices for both models), McNemar
  (continuity-corrected and exact), sign test, Clopper-Pearson exact
  binomial CIs, Bonferroni correction.
- **Stratification**: only user-supplied covariate columns; strata are
  generic records carrying `n` / `n_positive` / `n_negative`.
- **Covariate tables**: `read_subject_table` reads a CSV/Parquet file
  pinned by SHA-256 (`options.covariates {path, sha256}`); demographics
  are never auto-extracted from MEDS. Toy mode ships a generated table.
- **Next-token loss**: `compare_next_token_loss` measures the continued
  vs base encoder on a common extract and refuses checkpoints whose
  dictionaries differ.
- **Predictions contract**: heads publish row-hash-pinned
  probabilities (`write_predictions`); `load_predictions` rejects other
  rows or orders, and labels always come from the representation set,
  never from the head. Only boolean tasks are evaluated; non-boolean
  tasks fail loudly.

## The `forgetting` hook — documented exception

Every other hook is forbidden from reading the model directly; the
`forgetting` hook is the one documented exception, because forgetting
by definition needs **both** checkpoints: it loads the base and the
continued encoder and compares their held-out next-code loss
(`base_loss`, `continued_loss`, `loss_increase`, plus `n_subjects` per
split). Its cache key covers both models through the representations
key: the continued weights hash plus the training fingerprint, whose
`base_weights_sha256` pins the base checkpoint. It is skipped with
`applicable: false` when there was no continued pretraining. It also
reads the extract directly (to rebuild tokenized batches), a second
documented departure from the hook contract.

## Small-cell suppression interplay

Every evaluation record carries count-named fields (`n`, `n_positive`,
`n_negative`; calibration bins, DCA rows, McNemar cells, few-shot rows
likewise), so the tracking policy masks any stratum or split below the
threshold — including its AUROC/CI/calibration numbers, with
complementary suppression on top. Unmasked full results (bootstrap
distributions, per-row predictions) stay in each hook's `output_dir`
and are never logged. Details:
[tracking-and-small-cell-suppression.md](tracking-and-small-cell-suppression.md).

No real-world performance is cited anywhere in this repository; all
validated numbers live in tests (hand tables, known-value sources) and
in synthetic-fixture runs.
