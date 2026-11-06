# Model licensing

## The pinned release

The base model is the gated Hugging Face release
[`StanfordShahLab/clmbr-t-base`](https://huggingface.co/StanfordShahLab/clmbr-t-base).
`ehrfm.inputs.pins.CLMBR_T_BASE_PIN` pins exactly one dictionary
acquisition:

| Field | Value |
|---|---|
| repo | `StanfordShahLab/clmbr-t-base` |
| revision | `a0f2b36ef3976faf032bfbce23030cb640b5e97f` |
| filename | `dictionary.msgpack` |
| file sha256 | `481ddac70a37e79bbfadde1411ad674544a161e3c1948149989cd159a34203ee` |
| concept-list sha256 | `71f396167f40096c9ff11dc1b762746e946a7ddaec195f11a813a341bffedbc1` |
| licence | Gated release under the EHRSHOT Credentialed Health Data License; model card: CC-BY-NC-4.0; bundled LICENSE file: CC BY-NC-SA 4.0 |

Both acquisition routes — the pinned Hugging Face revision (needs a
token accepted for the repo's access terms) or a local copy of the same
file — must check `file_sha256`, and a derived concept set must check
`concept_list_sha256`. A hash mismatch or an unreachable source stops
with exit 2 before anything is written; nothing is downgraded to a
warning.

Note the licence discrepancy inherited from the release itself: the
model card declares CC-BY-NC-4.0 while the bundled LICENSE file says
CC BY-NC-SA 4.0. Treat the more restrictive reading as governing and
consult the release's own terms before any redistribution.

## Fetch flow

`ehrfm fetch-model [--dry-run] [--dest DIR]` (or
`scripts/fetch_clmbr_t_base.py`): the dry run prints the licence notice
and the plan and makes no network call; the real run downloads from the
pinned revision with `HF_TOKEN` from the environment (never persisted),
verifies the checksum, and on success prints the snapshot directory and
the weights SHA-256 — set that value as `EHRFM_SERVER_WEIGHTS_SHA256` to
serve it. Every failure mode (missing token, unreachable source,
checksum mismatch) exits 2.

## What this repository ships and does not ship

Ships: code, configs, tests, and the `ehrfm.toy` synthetic fixtures
(invented codes only). Committed pin facts (repo id, revision, hashes)
are metadata, not model content.

Does not ship: the model weights, the `dictionary.msgpack`, the concept
membership list, or any concept designations. The dictionary and weights
are components of a gated, credentialed release; the repo's own fixtures
use invented `DEMO*/`, `Gender/F|M`, `Visit/*` and `X`-prefixed codes so
the full pipeline runs hermetically without touching gated content.
Concept-set verification is done by hash (`concept_list_sha256`), which
identifies the concept set without publishing it.
