# Tracking and small-cell suppression

All experiment tracking goes through one MLflow wrapper
(`ehrfm.experiment.tracking`), and every value that reaches the store
passes a small-cell suppression policy first. The policy lives at the
boundary because that is the only place nothing can bypass it: there is
no raw-logging method on the tracked-run object.

## Why suppression

Count disclosures about small cohorts can re-identify individuals; a
count below a threshold is a small cell in the GDPR sense. Any stratum
with a tiny `n` would otherwise leak both the count and — by masking
failure — the metrics computed over it. The policy therefore treats
small counts as disclosures and masks with them.

## The policy

`SmallCellPolicy` (default threshold **5**, configurable) fields:
`threshold`, `count_fields` (fnmatch patterns on the last key segment:
`n`, `n_*`, `count(s)`, `*_count(s)`, `events`, `subjects`, `labels`,
`positives`, `negatives`, `support`, …), `suppress_zero`,
`mask_group`, `complementary`.

Rules:

- A count-field number is a **cell** (as is every number inside a
  count-field list or mapping, e.g. `subjects_by_split`). A cell is
  small if it is below the threshold or non-finite.
- **Group masking**: a record with a small scalar count has every other
  number directly in it masked — the rate/AUROC over a 3-row stratum is
  not publishable either.
- **Complementary suppression**: if exactly one cell in a breakdown (or
  across sibling records) is small, the next smallest is masked too —
  otherwise the complement would reveal the small value by subtraction.
  Rules repeat until stable.

Masking produces `null` in JSON, drops the metric name, and writes the
literal `suppressed` for params. Masked values are never recorded.
Every run logs `suppression.json` — a list of `{source, path, reason}`
entries with the threshold — so what was hidden is auditable.

Known conservatism (documented in the module): the policy masks
conservatively (e.g. a zero `n_replace` is masked as a count field) and
complementary suppression is one level (siblings and breakdowns), not a
full audit of every published total. Cohort-level aggregates remain
published; runs are keyed by extract hash and the store is local by
default, which keeps differencing risk contained.

## The MLflow boundary

The tracked-run object is the only path to the MLflow client;
`log_params` / `log_metrics` / `log_series` / `log_json` all suppress
first. Tags carry identifiers and hashes only. What is logged per run:
every config parameter, `config.json`, input summaries, stage records,
the coverage report, per-step train losses, per-split representation
counts/prevalence, and hook metrics/summaries — all after suppression.

Store note: the pinned MLflow puts the file store in maintenance mode
behind `MLFLOW_ALLOW_FILE_STORE`; the wrapper sets it (via `setdefault`,
never overriding an explicit setting) for `file:` URIs only, and the
default tracking URI is a local `file:<work_dir>/mlruns`. Any explicit
`tracking.uri` (sqlite, http) works unchanged.

Tests: unit rules and the 5/4/0 boundary, plus store-level scans that
walk a real `mlruns/` tree asserting no count-field value below the
threshold survives — with a positive control that finds the small values
when a permissive policy is used.
