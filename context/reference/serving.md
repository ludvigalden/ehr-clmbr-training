# Serving

`ehrfm serve` (or `python -m ehrfm.server`, env-configured) runs a
FastAPI server over a CLMBR encoder and a head registry.

## Startup

The lifespan loads state off the event loop (`asyncio.to_thread`) and
fails closed: the encoder via `ClmbrEncoder.load` with an optional
weights pin, then every head in the registry in a single pass whose
pinned record digests are captured during load — any failure
(tampered weights, encoder mismatch, missing binding, width mismatch,
missing/empty registry) aborts startup. There is no fallback head.
A readiness gate serves 503 until loading completes.

Toy mode (`--toy`) regenerates the synthetic inputs, fits one boolean
`linear_probe` head on the fly through the toy extract's ontology (the
same ontology serving uses — ancestor expansion matches training),
publishes it into the work directory's registry, and skips refitting on
restart when the registry is non-empty.

## Endpoints

| Endpoint | Behaviour |
|---|---|
| `GET /health` | always 200: loaded flag, head names, postprocess names |
| `GET /ready` | 200 after startup: ready flag, encoder weights SHA, heads |
| `GET /heads` | per-head spec, weights SHA, record SHA, metadata |
| `POST /embed` | subject → last-position embedding; returns hidden size, position count, unknown codes, encoder provenance, duration |
| `POST /predict` | subject + optional `head` (unset = the only head when exactly one is loaded; ambiguous/unknown → 404) → label-type-shaped output with full provenance |
| `POST /predict/stream` | SSE over 1–100 named heads: frames `started → subject → prediction×n → complete`, with a terminal catch-all `error` event so the stream always ends |

Every prediction/embedding request carries an inline subject:
`subject_id`, 1–100 events of `{code, time, numeric_value?, text_value?}`
with `<vocabulary>/<concept>` codes, and a required `birth_datetime`
that must not be after the earliest event (femr derives age from birth).
Same-timestamp events are canonically sorted (`(time, code, values)`),
numeric values are narrowed to float32 and must be finite, so request
reordering and float noise cannot change the encoding. Parity tests
assert `/embed` equals a direct hand-built `meds_reader` encode.

## Unknown-code accounting

femr silently drops codes outside the dictionary. The server never does:
unknown codes (no dictionary `code` entry — the "recognised" definition
from [inputs-contract.md](inputs-contract.md)) are counted per event and
per distinct code and reported on every prediction and in the stream's
`subject` event; the code list is capped (`unknown_code_limit`, default
20) while counts stay complete. A request whose events are all unknown
is rejected (422) instead of served from its birth token.

## Provenance and postprocess

Responses carry `x-ehrfm-encoder-weights-sha256`,
`x-ehrfm-head-weights-sha256`, `x-ehrfm-head-record-sha256` headers and
the same values in the body, plus a request id middleware (echoes
`X-Request-Id` or generates one). Postprocess hooks
(`register_postprocess`, default `raw`) transform head output; the
configured hook name is echoed in every response and output
shape/finiteness is validated.

Configuration is strict frozen pydantic (`ServerConfig`) with
`EHRFM_SERVER_*` env overrides for every CLI flag; a non-toy deployment
needs `model_dir` + `heads_dir` and, unless `ontology="none"` is chosen
explicitly (degrading to code-only tokenization, echoed in
`/health`/`/ready`), an `extract_dir` whose `codes.parquet` parents
match the checkpoint dictionary. After `ehrfm fetch-model`, set
`EHRFM_SERVER_WEIGHTS_SHA256` from the printed weights digest.
