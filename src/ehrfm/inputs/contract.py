"""Constants of the upstream artifact contract this package reads.

The artifacts are produced by github.com/ludvigalden/ehr-omop-meds-femr and
specified in its ``context/reference/artifacts.md``. The values here mirror
that document; a reader accepts an artifact whose contract version has the
same MAJOR and a MINOR no newer than :data:`CONTRACT_VERSION`.
"""

from __future__ import annotations

import re
from typing import Final, Literal

#: Contract version this reader was built against.
CONTRACT_VERSION: Final = "1.0.0"

#: MEDS release the extract follows (the one femr 0.2.4 requires).
MEDS_VERSION: Final = "0.3.3"

#: MEDS split names, in canonical order.
SPLITS: Final = ("train", "tuning", "held_out")
type Split = Literal["train", "tuning", "held_out"]

BIRTH_CODE: Final = "MEDS_BIRTH"
DEATH_CODE: Final = "MEDS_DEATH"
RESERVED_CODES: Final = frozenset({BIRTH_CODE, DEATH_CODE})

#: The vocabulary id the CLMBR dictionary uses for site-local care-site ids;
#: its entries are excluded from concept sets and the committed CLMBR manifest.
CARE_SITE_VOCABULARY: Final = "CARE_SITE"

#: ``<vocabulary_id>/<concept_code>``, split on the first ``/``.
CODE_PATTERN: Final = r"^[^/\s](?:[^/\n]*[^/\s])?/\S(?:.*\S)?$"
_CODE_RE: Final = re.compile(CODE_PATTERN)

MANIFEST_NAME: Final = "artifact_manifest.json"
DATA_DIR: Final = "data"
DATASET_JSON: Final = "metadata/dataset.json"
CODES_PARQUET: Final = "metadata/codes.parquet"
SUBJECT_SPLITS_PARQUET: Final = "metadata/subject_splits.parquet"
LABELS_PARQUET: Final = "labels.parquet"
TASK_JSON: Final = "task.json"
CONCEPT_SET_PARQUET: Final = "concept_set.parquet"
SHARD_PATTERN: Final = re.compile(r"^data/(train|tuning|held_out)/(0|[1-9][0-9]*)\.parquet$")

#: Where the artifacts come from and how to re-check them there.
PRODUCER_REPO: Final = "github.com/ludvigalden/ehr-omop-meds-femr"
VERIFY_COMMAND: Final = "omopmap artifacts verify"

_SEMVER_RE: Final = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def is_vocabulary_code(code: str) -> bool:
    """Return True for a well-formed ``<vocabulary_id>/<concept_code>`` string."""
    return _CODE_RE.match(code) is not None


def parse_version(version: str) -> tuple[int, int, int] | None:
    """Parse ``MAJOR.MINOR.PATCH``; return None when malformed."""
    match = _SEMVER_RE.match(version)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def is_compatible(version: str) -> bool:
    """Return True when an artifact of ``version`` is readable by this package."""
    theirs = parse_version(version)
    ours = parse_version(CONTRACT_VERSION)
    if theirs is None or ours is None:
        return False
    return theirs[0] == ours[0] and theirs[1] <= ours[1]
