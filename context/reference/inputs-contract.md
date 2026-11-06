# Inputs contract

This repository is a pure consumer of artifacts. It does not import the
producer's code; it mirrors the producer's schemas as typed structures
and verifies every artifact against its manifest before reading.

Producer contract and full schema documentation:
[ehr-omop-meds-femr](https://github.com/ludvigalden/ehr-omop-meds-femr)
(`context/reference/artifacts.md`). The consumer pins contract version
`1.0.0` / MEDS `0.3.3` and accepts same-MAJOR, MINOR ≤ ours.

## Upstream artifacts

| Artifact | Used for |
|---|---|
| `artifacts/meds/<dataset>/` — MEDS 0.3.3 extract with `train`/`tuning`/`held_out` splits and `metadata/` (dataset.json, codes.parquet with `parent_codes`, subject_splits.parquet) | Continued pretraining (train), representation building, coverage measurement, evaluation |
| `artifacts/labels/<dataset>/<task>/` — label set (`labels.parquet` MEDS label schema + `task.json`) | Task heads and evaluation |
| `artifacts/clmbr/` — concept set (`concept_set.parquet`) | Checked against the base checkpoint dictionary before training |
| `vocab/clmbr/manifest.json` — the only committed producer artifact (counts + hashes; the concept list itself is never published) | Optional pin check on the concept set |

## Verification is fail closed

`ehrfm.inputs` reads nothing before `verify_files` passes: exact file
set (no missing, no unlisted, no symlinks), per-file size and SHA-256,
Parquet footer row counts, exact Arrow schemas (names, order, types,
nullability), summary recomputed from the files, and for label sets the
`extract_manifest_sha256` linkage. Any violation raises a typed error
(`MissingArtifactError`, `IntegrityError`, `SchemaError`,
`LinkageError`, …) whose message names the producer and the verify
command. Nothing is downgraded to a warning. The CLI equivalent is
`ehrfm inputs verify <dir>` (exit 0/1).

Deliberate elision: row-level content rules (sort order, one
`MEDS_BIRTH` per subject, per-row code regex, subject→shard partition)
are not re-checked here — the manifest hash proves the bytes are exactly
the ones the producer's validator accepted. This is stated in the module
docstring.

## `meds_reader_convert` ownership

The consumer owns building the `meds_reader` database:
`ehrfm.inputs.database.ensure_database(extract, cache_dir)` runs
`meds_reader_convert` into `<cache>/<extract-manifest-sha>/`, staged
then atomically renamed, with a completion marker; an incomplete cache
entry raises and is never silently reused. Reasons (per the producer
contract): the binary index format is tied to the `meds-reader` version
the consumer's `femr` loads, the producer stays free of `femr`/`torch`,
and the MEDS Parquet extract is the tool-neutral interface. The extract
schema is convertible by construction — no normalisation step is needed
before conversion.

## Dictionary coverage

`ehrfm inputs coverage <extract> --model <ckpt>` (and
`ehrfm.data.coverage.extract_coverage`) reports how much of an extract
the checkpoint's dictionary covers, following the producer contract's
field names and privacy rules (codes only, a min-events floor, top-n;
never descriptions).

Two distinct measures exist because femr's tokenizer is hierarchical:

- **recognised** — the code has its own dictionary `code` entry. This is
  the producer definition behind the thesis finding that fewer than 20%
  of distinct codes in the transformed Swedish records were recognised
  (see [project-context.md](project-context.md)).
- **tokenized** — the code or any ancestor has an entry, so femr still
  emits a token; anything else is out-of-vocabulary and femr drops it
  silently. Both shares are always reported.

### Synthetic fixture numbers (synthetic, traced to the code)

On the built-in 20-subject seed-7 synthetic fixture
(`ehrfm.toy.artifacts`, invented codes only), pinned by
`tests/data/test_coverage.py`: 25 distinct codes, 17 recognised
(68.0%), 23 tokenized (92.0%) — coverage strictly between 0 and 1
because every second code of each invented family is left out of the
toy dictionary. These are synthetic fixture numbers only; they carry no
real-world meaning.
