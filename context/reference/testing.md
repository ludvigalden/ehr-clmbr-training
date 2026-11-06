# Testing

The suite is hermetic: CPU-only, no network, no gated artifacts, no real
patient data. Everything runs on CPU.

## Toy fixtures (shipped, not test-only)

The synthetic fixtures live in the package itself (`src/ehrfm/toy/`) so
the installed console script can use them:

- `ehrfm.toy.artifacts` — a byte-deterministic synthetic MEDS 0.3.3
  extract (20 subjects, seed 7, per-subject seeded generation, fixed
  `created_at`), a linked boolean label set (`demo_outcome_30d`), a
  covariate table, and a CLMBR concept-set export. Codes are invented
  (`DEMOCOND/`, `DEMODRUG/`, `DEMOLAB/`, `DEMOPROC/`, `DEMOOBS/`,
  `Gender/F|M`, `Visit/ER|OP`, and `X`-prefixed LOINC/RxNorm shapes);
  every second code of each invented family is left out of the toy
  dictionary so coverage is strictly between 0 and 1. Manifests are
  written last; the same arguments reproduce identical bytes.
- `ehrfm.toy.checkpoint` — a toy FEMR checkpoint in the real
  CLMBR-T-Base layout (`config.json`, hierarchical `dictionary.msgpack`
  in the CLMBR V2 schema, seeded safetensors weights), deliberately
  omitting `is_hierarchical` so the checkpoint surgery is exercised.
  Toy hidden size 32, CLMBR head 16.
- `ehrfm.toy.pretraining` — a CPU training config (18 steps) whose
  fingerprint equals the toy experiment config's, so `pretrain --toy`
  and the toy pipeline share one cache identity.

Nothing is committed as a blob: everything is generated into work
directories at run time.

## The torch_hawk shim

femr imports `torch_hawk` unconditionally at module level, but the
real package (x86_64 wheel / Bazel+CUDA sdist) is not installable on all
hosts. `ehrfm.models._compat` installs a parameter-free shim whose
forward raises; importing `ehrfm.models` installs it (idempotently, and
only when the real package is absent). Because the checkpoint surgery
replaces hawk layers with uniform attention before any forward pass, the
shim is never invoked — asserted by tests. An xformers `fmha` fallback
(registered only when missing) is bit-equal to xformers' mask.

## Markers

Registered strict markers (see `pyproject.toml`):

- `real_model` — loads the real CLMBR checkpoint; skipped unless
  `EHRFM_MODEL_DIR` is set; deselected in CI.
- `gpu` — requires a GPU device; deselected in CI.

CI runs `uv run pytest -m "not real_model and not gpu"`. The default
local run (no marker filter) also passes on CPU.

## How CI stays hermetic

`.github/workflows/ci.yml`, four jobs, all actions pinned by full
commit SHA, workflow-level `permissions: contents: read`:

- **lint** — `uvx ruff` only; no ML dependencies installed.
- **type-check** — `uv sync --extra dev` + strict `basedpyright`. Torch
  comes from the locked CPU index on Linux (`[tool.uv.sources]` with a
  `sys_platform == 'linux'` marker), so no GPU libraries are pulled.
- **docker-build** — builds the image (CPU default, no push) so the
  container path is exercised on every change.
- **test** — same locked CPU sync, then the marker-filtered suite.

No job downloads model weights, contacts external services, or mounts
data. Test determinism is pinned by `tests/test_determinism.py` (same
seed → byte-identical artifacts/checkpoints and `torch.equal`
representations). Evaluation metrics are validated against hand tables
and independent reference implementations — see
[evaluation.md](evaluation.md).

Known environment caveats (documented in
[compute-environment.md](compute-environment.md)): aarch64 hosts build
`meds-reader` from sdist via Bazel; do not relax the `nptyping`/`numpy`/
`pyarrow` pins without re-proving the import chain.
