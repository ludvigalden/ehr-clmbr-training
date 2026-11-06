# Task heads

Task heads here are **linear probes fitted on frozen encoder
representations**. The encoder is never updated by head training. The
pipeline ships the trained `linear_probe` head hook plus three
reference hooks — the sklearn `logistic_probe`, `age_baseline` and
`covariate_baseline` — which the toy experiment also runs.

## Linear probe

`ehrfm.heads.probe.fit_linear_probe` fits an L2 logistic head with
full-batch `torch.optim.LBFGS` in float64 from zero weights
(strong-Wolfe line search) — deterministic to the byte across runs (no
random init, no sampling). The objective is scikit-learn's L2 logistic
regression objective divided by n, with balanced class weights by
default; a test matches sklearn's `LogisticRegression` fit to 1e-5.
Standardisation is folded into the saved weights, so the served head
takes raw representations.

Regularisation strength C comes from a grid selected on tuning-split
AUROC; without a usable tuning split the grid's middle C is used. Every
logistic probe shares one default: the **EHRSHOT C grid**
(`ehrfm.config.heads.EHRSHOT_C_GRID`) — the 8 strengths 1e-8, 1e-6,
1e-4, 1e-2, 1e-1, 1, 1e2, 1e4 from the benchmark's `LR_PARAMS` search
(`ehrshot/utils.py`, Wornow et al. 2023). It is the default of the
`linear_probe` head hook, the sklearn `logistic_probe`, `age_baseline`
and `covariate_baseline` reference heads, the `few_shot` hook, and
`ehrfm.eval.baselines.fit_logistic`; pass `c_grid` explicitly to
override. Selection
scores the float32 `LinearHead` that will actually be served, so
published predictions equal `head.predict` exactly. A test matches the
fitted probabilities against sklearn's `LogisticRegression` (the
unweighted, unstandardised configuration) to 1e-5, and a test pins the
grid values and every default to the constant.

## Few-shot protocol

`ehrfm.eval.fewshot.run_few_shot` implements the EHRSHOT few-shot
protocol (Wornow et al., "EHRSHOT: An EHR Benchmark for Few-Shot
Evaluation of Foundation Models", NeurIPS Datasets and Benchmarks
2023): for each k and replicate it draws k positives plus k negatives
from the train split and a disjoint k+k from the tuning split (seeded
per `(seed, k, replicate)` via `SeedSequence`), fits a k-shot probe,
and scores the held-out split. A k larger than a class can supply is
skipped, and with no usable tuning draw the default strength C is used.
With `probe: linear_head` each draw is a `linear_probe` head; the
shipped default fits the sklearn-based `logistic_regression` probe. No
EHRSHOT data ships.

## Encoder binding

Heads record the encoder they were fitted on (`encoder_weights_sha256`)
in `head.json`. The serving registry refuses a head whose recorded
encoder does not match the served encoder, a head with no encoder
binding, a tampered weight hash, a width mismatch, or a pinned record
digest mismatch — there is no fallback head. A head fitted on a
continued encoder cannot be served against the base checkpoint, and
vice versa.

## The registry

`ehrfm.inference.registry.HeadRegistry` maps names to published heads
(`<root>/<name>/{head.json, weights.safetensors}`):
`names()` / `entry(name)` / `entries()` for listing, `load(name,
encoder_identity)` / `load_all(...)` for fail-closed loading, and
`publish(dir, name, replace)` to verify-then-atomically-copy a trained
head in. CLI: `ehrfm heads ls` and `ehrfm heads publish`; head training
itself runs inside `ehrfm run` as a cached stage (`tasks.heads`), so it
shares the experiment's cache identity.

## What is not provided

Encoder fine-tuning is not provided: heads do not unfreeze the encoder,
there is no LoRA/partial-unfreezing path, no supervised-label
re-encoding of timelines, and no multi-class or regression head
trainer. This is deliberate — every head is fitted on frozen
representations pinned to a specific encoder weights hash, and the
continued-pretraining loop is a next-code pretraining objective, not a
labelled-task path. Trained heads are boolean logistic probes; the
`HeadSpec` carries the other label types, but no trainer/evaluator
handles them and the prediction loader rejects them loudly.
