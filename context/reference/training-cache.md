# Training cache

Continued pretraining (`ehrfm.training`) is idempotent by construction:
a run is identified by a content hash of everything that can change the
result, and a rerun of an identical configuration is a cache hit that
loads no model at all.

## The fingerprint

`FingerprintInputs` is serialised as canonical JSON (sorted keys,
compact) and hashed with SHA-256. Fields:

- `format` and `logic_version` — the latter is bumped whenever the
  training procedure itself changes, invalidating all old entries.
- `extract_manifest_sha256` — the verified artifact manifest hash,
  covering every extract file byte.
- `base_weights_sha256`, `base_dictionary_sha256`, `base_config_sha256`
  — the full base checkpoint identity (the weights hash is the same
  value recorded as weight provenance at load).
- `optimization` — the whole `OptimizationConfig` (seed, tokens per
  batch, epochs, learning rate, schedule, precision, the CLMBR head
  width pin, the vocab-expansion hook name and `n_replace`, …).
- `world_size` — the same config on 1 or 4 processes is a different run
  (intended: the step order differs).
- `versions` — installed package versions of this repo, `femr`, `torch`,
  `transformers`, `meds-reader`.

Runtime settings (output dir, device, worker counts, checkpoint
frequency) are deliberately excluded — a test asserts they do not change
the digest.

## Cache decision

`decide(paths, fingerprint)` returns one of:

- `cached` — a completed record with the same fingerprint exists **and**
  `model/model.safetensors` still hashes to the recorded weights hash.
  An altered or truncated published model retrains.
- `resume` — an interrupted record with a usable latest checkpoint
  (partial checkpoint dirs and foreign fingerprints are skipped).
- `fresh` — otherwise.

A cache hit returns before any model load (asserted by a test that
monkeypatches the loader to raise). Cache hits cost milliseconds plus
imports.

## Crash safety

All record/checkpoint writes are atomic: temp sibling, fsync,
`os.replace`, directory fsync; checkpoint directories are staged and
published all-or-nothing; a corrupt record counts as no record. A
non-blocking `flock` run lock guards the run directory — a second
trainer on the same run raises instead of racing. Interrupted runs are
marked, the model is never published unless the run finished, and
`SignalStop` (SIGTERM/SIGINT) stops after the current step with a
checkpoint, so it composes with any scheduler's pre-emption. Resume
reproduces the uninterrupted run's final weights exactly (tested, both
for a clean stop and a hard interrupt).

## Distributed runs

There is no scheduler-specific submit branch. The generic launcher is
`torchrun`: `ehrfm pretrain --config FILE --print-torchrun N` prints the
command (`--standalone` for one node, c10d rendezvous for several), and
`python -m ehrfm` is provided so `torchrun -m ehrfm` works. Under
`WORLD_SIZE > 1` the process group is NCCL on CUDA or gloo on CPU, the
model is wrapped in `DistributedDataParallel`, rank 0 alone writes the
run directory/cache/lock, and losses and the stop flag are all-reduced.
The packed tokenized-batch cache has its own key (extract hash,
dictionary hash, head size, tokens per batch, subject-list hash, femr
version, format number) and is staged then renamed like everything else.
