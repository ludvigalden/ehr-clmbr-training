# Project context

This repository is the model-side half of a two-repo adaptation pipeline
for clinical foundation models. Its framing source is the thesis
[GMLF](https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-590690)
("Global Models, Local Friction: Interoperability, Adaptation, and
Deployment of Clinical Foundation Models in Swedish Healthcare"),
authored at Uppsala University and Uppsala University Hospital within
the EmergAI research project.

## Abstract (verbatim)

> What does it cost to bring an international clinical foundation model
> into Swedish emergency care? Foundation models are large AI systems
> that learn from the time-ordered records of millions of patients to
> anticipate what is likely to happen to a patient next. The clinical
> foundation models most relevant to this project have largely been
> trained on American records and expect each record in one
> international format — which Swedish hospitals, with their own code
> systems, do not use. This project approaches that gap as a feasibility
> study, using one such model (CLMBR-T-Base) as its anchor.
>
> The adaptation cost takes three forms. Terminological cost arises when
> Swedish code systems (ICD-10-SE, KVÅ, ATC, NPU) are mapped onto
> international standard vocabularies (SNOMED CT, RxNorm, LOINC).
> Distributional cost arises because Swedish care records a different
> mix and concentration of events than the American data the model
> learned from, so the model interprets even faithfully translated codes
> through expectations tuned elsewhere. Implementation cost is the
> organisational and regulatory distance to using the model in practice.
>
> To measure that cost, a data-processing chain was built to translate
> Swedish emergency department records into the format the model
> expects, recording at each step what survives the translation and what
> is lost. A decisive finding appears when the transformed Swedish
> records enter the model: fewer than 20% of distinct codes are
> recognised by the model's vocabulary. The model can still be trained
> further on Swedish data, a step that would make it more adapted to
> Swedish care. However, further training on real Swedish records was
> deferred because the severely limited vocabulary recognition made a
> full run unlikely to produce scientifically reliable performance
> estimates. Moreover, a proposal for integration with the standard
> hospital-record interface (FHIR) is specified but not tested. Our
> assessment is that, for this model, the total cost of adapting it was
> too large relative to what the model could offer. What this work
> leaves behind, however, is not tied to this particular model. Its main
> contributions are an infrastructure for standardising Swedish
> emergency department records, a vocabulary mapping with broad
> coverage, a proposal for integrating the model with the
> hospital-record interface, and a three-part framework for adaptation
> cost.

## How this repository maps onto the framework

The thesis measures adaptation cost in three parts. This repo's code
addresses them as follows:

| Framework part | Repo machinery |
|---|---|
| Distributional cost — the model reads even translated codes through expectations tuned elsewhere | `ehrfm pretrain` continues pretraining CLMBR-T-Base on MEDS extracts; `ehrfm.data.coverage` measures how much of an extract the base vocabulary recognises and tokenizes; `ehrfm.eval` compares base vs continued encoders, including the `forgetting` hook (base vs continued held-out next-code loss) |
| Terminological cost — Swedish codes mapped to SNOMED CT / RxNorm / LOINC | Owned by the sibling [ehr-omop-meds-femr](https://github.com/ludvigalden/ehr-omop-meds-femr); this repo consumes its artifacts and checks the concept set against the checkpoint dictionary (`ehrfm.training.surgery.require_concept_set_match`) |
| Implementation cost — organisational/regulatory distance to deployment | `ehrfm.server` (HTTP/SSE inference with provenance headers), the fail-closed artifact contract (`ehrfm.inputs`), SHA-256 provenance on weights and heads, and the sensitive-data-enclave design in [compute-environment.md](compute-environment.md) |

The under-20% distinct-code recognition finding is why continued
pretraining exists here at all: it addresses distributional cost on the
codes the dictionary already recognises. Changing which codes are
recognised needs a user-supplied vocabulary-expansion hook
(`register_vocab_expansion`); none ships with this repository.

## Honesty statements

- The pipeline is validated end to end on synthetic data only. No
  real-world performance claim is made anywhere in this repository.
- Continued pretraining on real Swedish records was deferred in the
  research (reasons in the abstract above); this repository provides the
  machinery, not a real-data result.
- Every number in these docs is either quoted from the abstract or is a
  synthetic fixture number traced to the code and its tests, or a
  structure/behaviour fact checked against the code.
- The only thesis figure reproduced here is the fewer-than-20%
  distinct-code recognition rate from the abstract.
