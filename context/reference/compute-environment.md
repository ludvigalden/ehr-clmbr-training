# Compute environment

The research behind this repository ran on **Bianca (UPPMAX)**.
Description (verbatim):

> Bianca is a research system dedicated to analysing sensitive personal
> data, or other types of sensitive data. Bianca provides 4480 cores in
> the form of 204 dual CPU (Intel Xeon E5-2630 v3) Huawei XH620 V3 nodes
> with 128GB memory, 75 fat nodes with 256 GB of memory, 15 nodes with
> 512 GB of memory, ten nodes with two NVIDIA A100 40GB GPUs each.

GPU experiment runs used those A100 40GB nodes. This repository ships no
cluster-specific submit code: the distributed launcher is generic
`torchrun` (`ehrfm pretrain --config FILE --print-torchrun N`), so it
composes with whatever job scheduler the enclave provides. See
[training-cache.md](training-cache.md) for the distributed-run contract.

## How the design reflects a sensitive-data enclave

Several design decisions exist because the environment assumes no
outbound network, no shared external services, and privacy constraints
on anything leaving the machine:

- **Pinned, offline-resolvable dependencies.** `uv.lock` is committed
  and both the Dockerfile syncs use `uv sync --frozen`; torch is pinned
  through an explicit CPU index. Nothing at runtime needs an external
  service except the explicitly requested gated model download (see
  [model-licensing.md](model-licensing.md)), which verifies the pinned
  SHA-256 and fails closed.
- **Artifacts move as hash-verified files, not via services.** The
  whole producer/consumer interface is directories with manifests: the
  consumer verifies size, SHA-256 and schema of every file and fails
  closed, so an artifact can be carried between environments (for
  example in and out of an enclave) and checked byte for byte on
  arrival. No database or queue connects the two repos.
- **Tracking stays local.** The default MLflow tracking URI is a local
  file store under the work directory; no external tracking service is
  contacted, so logged data does not leave the machine.
- **Small-cell suppression at the tracking boundary.** Counts below the
  threshold (default 5) and their dependent metrics never reach the
  store — the GDPR small-cell discipline applied to experiment logging
  (see [tracking-and-small-cell-suppression.md](tracking-and-small-cell-suppression.md)).
- **Privacy-shaped coverage reporting.** Dictionary-coverage reports
  list codes only (never descriptions), cap the listed unrecognised
  codes, and never list a code seen in fewer than a minimum number of
  events.
- **`ehrfm doctor`** prints the environment facts (python, torch,
  device, the `torch_hawk` shim state, femr, meds_reader, xformers,
  huggingface_hub) so an offline host can be diagnosed without network
  probes.

## aarch64 / meds-reader build notes

The stack resolves with prebuilt wheels on x86_64. On aarch64 (for
example arm development machines or arm64 container builds) two
packages need source builds, handled in the committed tree:

- **`meds-reader`** 0.1.x ships linux-arm64 as sdist only and builds via
  Bazel. Its sdist's `.bazelrc` hard-codes x86-only `-mavx` flags; the
  sdist self-recovers — `setup.py` probes with a tiny Bazel build and on
  failure rebuilds with a fallback rc file that carries no arch flags,
  so no source patching is needed. A fresh aarch64 host needs only
  `bazel` on PATH (bazelisk works); the first cold build takes roughly
  half an hour, warm caches minutes. The Dockerfile installs bazelisk
  (checksum-pinned) inside a `TARGETARCH=arm64` block only — amd64
  builds use the manylinux wheel and skip the block entirely.
- **`torch-hawk`** has no arm64 wheel and its sdist needs Bazel plus
  CUDA, so it is dropped from resolution everywhere via a uv
  `override-dependencies` entry, and a parameter-free shim
  (`ehrfm.models._compat`) satisfies femr's module-level import. The
  shim's forward raises `HawkUnavailableError`; the checkpoint surgery
  replaces hawk layers with uniform attention before any forward pass,
  so the shim is never invoked on the uniform-attention path (tested).
  Deleting the override and re-locking re-enables the real package.
- **NumPy/pyarrow ABI pin.** femr pins `nptyping==2.4.1`, which holds
  NumPy below 2.0, and pyarrow wheels built against the NumPy 2 ABI
  fail at import with no resolver-visible conflict; `pyproject.toml`
  caps `pyarrow<26` with a reason comment. Do not relax the three
  (`nptyping`, `numpy`, `pyarrow`) pins without re-proving the import
  chain.
