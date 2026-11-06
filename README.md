# ehr-clmbr-training

Continued pretraining of the CLMBR-T-Base clinical foundation model on
MEDS data, linear-probe task heads on the frozen representations,
rigorous statistical evaluation, and HTTP/SSE serving of predictions —
the whole CLMBR adaptation lifecycle behind one command line, `ehrfm`.

## Context

The pipeline follows the research setting of
[GMLF](https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-590690)
("Global Models, Local Friction: Interoperability, Adaptation, and
Deployment of Clinical Foundation Models in Swedish Healthcare",
developed at Uppsala University and Uppsala University Hospital within
the EmergAI research project): Swedish emergency-department records are
standardised into the OMOP CDM, converted to MEDS, and fed to
CLMBR-T-Base, a foundation model trained on American records. The thesis
frames adaptation cost in three parts — terminological (mapping Swedish
code systems onto SNOMED CT, RxNorm and LOINC), distributional (a
different event mix than the model learned from), and implementation
(organisational and regulatory distance to deployment). One finding
motivates this repository: fewer than 20% of distinct codes in the
transformed Swedish records were recognised by the model's vocabulary,
so the encoder must be adapted to the target population before
downstream heads and metrics mean anything. Continued pretraining on
real Swedish records was deferred in that research (the recognition
rate made a full run unlikely to yield reliable estimates); the
adaptation pipeline itself is ready to run, validated end to end on
synthetic data, and published here for its value to future projects.
It was developed and run on Bianca (UPPMAX), a sensitive-data HPC
enclave.

The inputs are produced by the sibling repository
[ehr-omop-meds-femr](https://github.com/ludvigalden/ehr-omop-meds-femr)
(OMOP vocabulary handling, MEDS extracts, label sets, CLMBR concept
set), with the Swedish vocabulary mapping in
[vocabmod-se](https://github.com/ludvigalden/vocabmod-se).

## Quickstart (synthetic data, CPU, seconds)

```sh
uv sync --extra dev
uv run ehrfm pretrain --toy                        # 18 steps of continued pretraining
uv run ehrfm run configs/experiment/toy.yaml       # coverage, pretraining, heads, evaluation
uv run ehrfm serve --toy                           # FastAPI server with a head fitted on the fly
```

Prediction request (all codes are invented synthetic codes):

```sh
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"subject_id":1,"birth_datetime":"2019-06-01T00:00:00","events":[{"code":"Gender/M","time":"2019-06-01T00:00:00"},{"code":"DEMOCOND/C01","time":"2020-01-01T00:00:00"}],"head":"demo_outcome_30d"}'
```

## Using the real model

CLMBR-T-Base is a gated release under the EHRSHOT Credentialed Health
Data License. `ehrfm fetch-model --dry-run` prints the licence notice
without a network call; `ehrfm fetch-model` downloads the pinned release
revision and verifies its `dictionary.msgpack` against the pinned
SHA-256, exiting 2 on any mismatch. See
[model-licensing](context/reference/model-licensing.md).

## Reference index

- [project-context](context/reference/project-context.md) — thesis context and honesty statements
- [architecture](context/reference/architecture.md) — package map and end-to-end flow
- [inputs-contract](context/reference/inputs-contract.md) — the sibling artifact contract, fail-closed verification
- [checkpoint-surgery](context/reference/checkpoint-surgery.md) — preparing the checkpoint for continued pretraining
- [training-cache](context/reference/training-cache.md) — fingerprints, idempotent reruns, resume, distributed runs
- [experiment-config](context/reference/experiment-config.md) — strict config schema, Hydra composition, stage cache
- [tracking-and-small-cell-suppression](context/reference/tracking-and-small-cell-suppression.md) — the MLflow boundary
- [evaluation](context/reference/evaluation.md) — bootstrap, DeLong, calibration, decision curves, paired tests, forgetting
- [task-heads](context/reference/task-heads.md) — linear probes, the EHRSHOT few-shot protocol, serving registry
- [serving](context/reference/serving.md) — endpoints, provenance, unknown-code accounting
- [compute-environment](context/reference/compute-environment.md) — the sensitive-data enclave, aarch64 build notes
- [model-licensing](context/reference/model-licensing.md) — pins, the gated licence, what ships
- [testing](context/reference/testing.md) — toy fixtures, markers, hermetic CPU CI

## Licence

Apache-2.0 ([LICENSE](LICENSE)); the vendored DeLong MIT code and its
algorithm citation are in [NOTICE](NOTICE).
