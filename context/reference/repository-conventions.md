# Repository conventions

The rules a change to this repository follows, beyond what the code and
its tests enforce. They apply to every contributor, human or automated.

## Scope

This repository consumes model-ready artifacts; it does not make them.
OMOP and vocabulary handling, MEDS writing or generation, and the CLMBR
dictionary import belong to
[ehr-omop-meds-femr](https://github.com/ludvigalden/ehr-omop-meds-femr),
whose artifact contract [inputs-contract.md](inputs-contract.md)
verifies, failing closed.

## Layout

| Path | Contents |
|---|---|
| `src/ehrfm/` | The package: `inputs/` (the input contract and the `meds_reader` database build), `models/` (checkpoint loading and the compatibility shim), `config/`, `data/`, `training/`, `experiment/`, `eval/`, `inference/`, `server/`, `cli.py` and `logging.py` ([architecture.md](architecture.md)) |
| `configs/` | The default YAML for each configuration slice ([experiment-config.md](experiment-config.md)) |
| `scripts/` | `fetch_clmbr_t_base.py`, the gated-checkpoint fetch ([model-licensing.md](model-licensing.md)) |
| `tests/` | Hermetic CPU tests over synthetic fixtures generated at test time ([testing.md](testing.md)) |
| `context/reference/` | These reference notes; the README names every one of them |

## What is never committed

- **Records and licensed content.** No patient data, no personal
  identity numbers, no licensed vocabulary text (SNOMED CT
  designations, CPT4, ICD-O-3) and no CLMBR dictionary membership
  content. Fixtures match the producer's contract and are synthetic.
- **Gated weights.** CI needs no gated weights and downloads no model.
- **Identifying detail.** No allocation ids, usernames, hostnames, host
  paths or personal email addresses. The research provenance is stated
  once, in [project-context.md](project-context.md) and
  [compute-environment.md](compute-environment.md).
- **Secrets.** Every input that needs a credential reads it from the
  environment.

## Documentation

The committed documentation describes the current state of the code:
no decision records, changelogs, "why we chose" prose, planned or
upcoming work, or deadlines.

## Commits

Commit messages follow Conventional Commits: a lowercase
`type(scope): subject` of at most 72 characters, with no body and no
trailers.
