"""Input contract for the artifacts this package consumes.

The MEDS extracts, label sets and CLMBR concept-set exports read here are
produced by github.com/ludvigalden/ehr-omop-meds-femr and specified in its
``context/reference/artifacts.md``. Every artifact is verified against its
hashed ``artifact_manifest.json`` before use; any violation raises an
:class:`InputArtifactError` and the artifact is not consumed.
"""

from ehrfm.inputs.artifacts import (
    ConceptSet,
    LabelSet,
    MedsExtract,
    dictionary_matches,
    open_concept_set,
    open_extract,
    open_label_set,
)
from ehrfm.inputs.contract import CONTRACT_VERSION, MEDS_VERSION, SPLITS
from ehrfm.inputs.database import ensure_database, open_database
from ehrfm.inputs.errors import InputArtifactError
from ehrfm.inputs.pins import CLMBR_T_BASE_PIN, DictionaryPin

__all__ = [
    "CLMBR_T_BASE_PIN",
    "CONTRACT_VERSION",
    "MEDS_VERSION",
    "SPLITS",
    "ConceptSet",
    "DictionaryPin",
    "InputArtifactError",
    "LabelSet",
    "MedsExtract",
    "dictionary_matches",
    "ensure_database",
    "open_concept_set",
    "open_database",
    "open_extract",
    "open_label_set",
]
