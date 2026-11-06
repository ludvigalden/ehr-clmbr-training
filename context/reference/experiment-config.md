# Experiment configuration

`ehrfm run configs/experiment/<name>.yaml` composes a whole experiment
(coverage → optional continued pretraining → representations → heads →
evaluation → tracking) and reuses every unchanged stage from the cache.

## Strict config schema

Every config model is strict, frozen pydantic with `extra="forbid"`;
validation runs in pydantic JSON mode, so YAML lists become tuples and
strings become paths, and nothing else is coerced (`"20"` for an int
fails). `ExperimentConfig` has exactly one field per Hydra group
(`model`, `data`, `tasks`, `experiment`, `eval`), and a test asserts
this field set equals the `configs/` group directories. Every failure
becomes a single `ConfigError` naming the key.

Group highlights:

- `model`: `base_checkpoint` (required outside toy mode),
  `expected_weights_sha256` (lowercase hex), `device`.
- `data`: exactly one of `extract` and `toy{subjects≥10, seed}`;
  `concept_set` is not allowed with toy.
- `tasks`: `label_sets` and `heads: [{name, options}]`.
- `eval`: `hooks: [{name, options}]`.
- `experiment`: `name`, `seed`, `work_dir`, `hook_modules`, plus
  `coverage`, `pretraining` (nesting the training `OptimizationConfig`),
  `representations`, `tracking`, `suppression` (the small-cell policy).

## Environment interpolation

`configs/*/default.yaml` uses `${oc.env:EHRFM_BASE_CHECKPOINT}`,
`${oc.env:EHRFM_EXTRACT}` and `${oc.env:EHRFM_WORK_DIR}/<name>` with no
site defaults and no fallbacks; a missing variable fails with a message
naming the variable and the config file that needs it. The toy configs
use `${oc.env:EHRFM_WORK_DIR,outputs}` — the only fallback is the
git-ignored relative `outputs/`.

Adding an experiment: `configs/experiment/<name>.yaml` with
`# @package _global_` and defaults
`[default@experiment, override /<group>: <option>, _self_]`
(`default@experiment` must precede the overrides — Hydra requires
overrides at the end). Any plain YAML/JSON holding the whole tree also
works, with dot-list overrides on the command line.

## Stage cache

Each stage publishes `<work_dir>/stages/<stage>/<key[:16]>/stage.json`,
keyed by SHA-256 of canonical JSON over its inputs:

| Stage | Key inputs |
|---|---|
| `inputs` (always runs) | verifies manifests, pins, weights; hashes everything |
| `coverage` | logic version, extract hash, dictionary hash, top_n, min_events |
| `pretraining` (optional) | the training fingerprint (see [training-cache.md](training-cache.md)); a validity re-hash of the published weights makes a truncated model retrain |
| `representations` | logic version, weights hash, training fingerprint, extract hash, label-set hashes, splits, tokens per batch |
| `head-<name>` | kind, name, hook version, options, seed, representations key |
| `eval-<name>` | same, plus every head key |

Device, worker counts and output locations never enter a key. Tracking
is keyed on the pipeline key (experiment name + all stage keys +
suppression policy); a FINISHED MLflow run carrying the matching
`ehrfm.pipeline_key` tag is reused, so a rerun logs nothing. Changing an
eval option recomputes only `eval-*`; changing a training setting
recomputes from pretraining onward while the coverage key stays shared.

## Hook contract

```python
register_hook(kind: Literal["head", "eval"], name, fn, *, version="1")
```

Hooks receive a `HookContext` (options, seed, `output_dir` published
atomically, config, model identity for provenance, a
`RepresentationIndex`, and for eval hooks the earlier head records) and
return `HookResult(metrics, summary)`. Rules: hooks are deterministic
for a given context and seed, must not read the extract or the model
directly, and must bump `version` when their logic changes or stale
cached results are reused. Count-valued metric names should be shaped so
the suppression policy recognises them (`n`, `n_positive`, `*_count`,
`support`) — see
[tracking-and-small-cell-suppression.md](tracking-and-small-cell-suppression.md).
Unknown hook names fail before any stage runs.
