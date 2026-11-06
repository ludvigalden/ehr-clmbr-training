# Checkpoint surgery

CLMBR-T-Base is released as a FEMR checkpoint (`config.json`,
`dictionary.msgpack`, `model.safetensors`). Loading it verbatim into
femr 0.2.4 fails or silently mis-trains, so `ehrfm.models.checkpoint`
patches the checkpoint at load time. The strict load (missing or
unexpected key raises) and the optional weights SHA-256 pin apply on
every load, training and inference alike; the full invariant checks
(`ehrfm.training.surgery.check_surgery`) gate continued pretraining.

## The three patches

1. **Hierarchical flag.** The released `config.json` omits
   `transformer_config.is_hierarchical` (so femr defaults it to `False`)
   although the shipped `dictionary.msgpack` is hierarchical. Without
   the patch femr builds a flat `nn.Embedding` and reads a missing
   `tokens` key. The loader sets the flag to `True`.
2. **Hawk layer replacement.** femr builds a hawk (recurrent) block in
   every even-indexed layer. The released checkpoint's recurrent weights
   are not usable on this path, so each hawk layer is replaced by a
   uniform-attention layer before weights are loaded. On hosts without
   the real `torch_hawk` package, `ehrfm.models._compat` installs a
   parameter-free shim whose `forward` raises — the surgery guarantees
   it is never called, and the checks below catch any leftover
   recurrent block.
3. **Embedding remap.** The checkpoint stores flat weights as
   `transformer.embed.weight`; the hierarchical model uses
   `EmbeddingBag` (`transformer.embed_bag.weight`). Keys are renamed and
   the load is strict: any missing or unexpected key raises (a tolerant
   load would leave the embedding at random init).

Additionally `load_tokenizer` expands the dictionary's flat `age_stats`
to femr 0.2.4's age/log-age/delta/log-delta form; the delta std reuses
the age std because a constant 1.0 overflows femr's float16 `time_data`.

## Surgery invariants (tested in `tests/training/test_surgery.py`)

Each patch has a negative test that fails loudly when the invariant is
violated: unpatched hierarchical flag, flag without an `EmbeddingBag`,
leftover flat embedding, hawk layer present, recurrent module left
behind a flipped flag, missing/unexpected weights on the strict
load, bag rows ≠ config vocab_size ≠ dictionary entries, missing task
config (femr would return a constant 0 loss — training would silently
do nothing), wrong head module, head width disagreeing with config or
with the configured pin (8192 for the release), head wider than the
dictionary, base weights SHA-256 pin mismatch, and base dictionary ≠
concept-set dictionary (`require_concept_set_match` — training on a
checkpoint the vocabulary mapping did not target).

`SurgeryReport` records what was done (`hierarchical_patched`,
`hawk_layers_replaced`, `embedding_remapped`). Checkpoints exported by
this repo are written already-hierarchical, so re-loading them reports
`hierarchical_patched=False` and `embedding_remapped=False` — but hawk
layers are still replaced on every load, because femr always constructs
them in even-indexed layers.

## The femr birth-timestamp finding and fix

femr 0.2.4's `BatchCreator.add_subject` re-stamps the birth position
with a leaked loop variable: the timestamp ends up being the subject's
**last** event time, so position stamps are unsorted
(`[last, e1, e2, …]`) and any `searchsorted` over all stamps is unsafe.
Impact analysis showed the naive `searchsorted` disagrees with a
post-birth-only search only for **birth-day-only subjects** (a single
position stamped with the last event's time), where the naive code
selected the birth position for any later prediction time — a
divergence from femr's own labelling semantics, and a dependency on an
undocumented femr quirk besides.

Fix: `ehrfm.models.representations.position_at(stamps, epoch_seconds)`
skips position 0 (birth), asserts post-birth stamps are sorted (raises
`RepresentationError` otherwise — fail closed), and returns `None`
before the first post-birth position. `representations_at` uses it. A
regression test pins femr's behaviour so a femr change flags itself
(see the birth-position tests in
`tests/models/test_representations.py`); the representations cache is
keyed with `REPRESENTATION_LOGIC = 2`
(`ehrfm.experiment.pipeline`), so caches built before the fix recompute.
