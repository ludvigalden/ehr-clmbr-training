# Architecture

`ehrfm` is a single Python package (`src/ehrfm/`, console script
`ehrfm`) covering the whole adaptation lifecycle: read verified inputs →
prepare the base checkpoint → continue pretraining → embed → fit heads →
evaluate → serve.

## End-to-end flow

```
MEDS extract + label sets + concept set   (sibling repo, see inputs-contract.md)
        │  ehrfm.inputs   fail-closed verification, meds_reader database, pins/fetch
        ▼
CLMBR-T-Base checkpoint   (ehrfm.models.checkpoint: surgery, weight pins)
        │  ehrfm.training  continued pretraining (fingerprinted, idempotent, resumable)
        ▼
continued checkpoint
        │  ehrfm.data      tokenization, batches, splits, prediction times, coverage
        │  ehrfm.models.representations  ClmbrEncoder embeddings at prediction times
        ▼
representation sets (safetensors, cached)
        │  ehrfm.heads     linear probes as head hooks
        │  ehrfm.eval      predictions, metrics, evaluator, few-shot, forgetting
        ▼
experiment pipeline (ehrfm.experiment)  →  MLflow tracking (small-cell suppressed)
        │
        ▼
head registry (ehrfm.inference.registry)  →  FastAPI server (ehrfm.server)
```

`ehrfm run configs/experiment/<name>.yaml` drives the middle of this as
one cached pipeline; `ehrfm pretrain` / `ehrfm serve` expose the two
ends standalone.

## Package map

| Package | Role |
|---|---|
| `ehrfm.inputs` | The sibling artifact contract: manifest + schema verification (fail closed), `meds_reader` database build, the dictionary pin and the gated model fetch |
| `ehrfm.models` | Checkpoint loading with surgery, weight provenance, device resolution, `ClmbrEncoder`, `TaskHead`/`LinearHead`, the `torch_hawk`/xformers compat shims (importing `ehrfm.models` installs them) |
| `ehrfm.data` | `CodeOntology` from `codes.parquet`, tokenizer, batch packing, seeded splits, prediction times, dictionary-coverage measurement |
| `ehrfm.config` | Strict frozen pydantic schemas for training, experiments, head probes; Hydra/plain-YAML loader |
| `ehrfm.training` | Surgery checks, optional vocab-expansion hook, CLMBR pretraining loop, content-hash cache, runner (CPU/GPU/torchrun) |
| `ehrfm.experiment` | Stage cache, hook contract, representation I/O, small-cell suppression, MLflow tracking |
| `ehrfm.heads` | The L-BFGS linear probe, its head hook, trained-head store |
| `ehrfm.eval` | DeLong/bootstrap statistics, calibration, decision curves, paired tests, stratification, covariate tables, baselines, few-shot, next-token loss, eval hooks |
| `ehrfm.inference` | Head serialization, `HeadRegistry` (fail-closed loading with encoder binding) |
| `ehrfm.server` | FastAPI app: `/health`, `/ready`, `/heads`, `/embed`, `/predict`, `/predict/stream` |
| `ehrfm.toy` | Synthetic fixtures shipped in the wheel (byte-deterministic; invented codes only) — power `--toy` everywhere |
| `ehrfm.cli` | The typer app: `pretrain`, `run`, `ls`, `show`, `serve`, `fetch-model`, `doctor`, plus `inputs` and `heads` sub-apps |

## Extension points

- Head kinds: subclass `TaskHead`, `register_head_kind(cls)`; heads are
  saved/loaded by `ehrfm.inference.head_loader` unchanged.
- Hook kinds: `register_hook("head"|"eval", name, fn, version)`; third
  party modules register via `experiment.hook_modules`.
- Postprocess hooks on the server: `register_postprocess` (default `raw`).
- Vocabulary expansion during continued pretraining:
  `register_vocab_expansion(name, hook)`; no strategy ships, the hook is
  a no-op by default.

Design invariants across all of the above: strict frozen pydantic at
every config boundary, fail-closed verification (no degraded paths, no
fallback heads), SHA-256 provenance on weights/heads/dictionaries, and
content-hash caching so unchanged work is never repeated.
