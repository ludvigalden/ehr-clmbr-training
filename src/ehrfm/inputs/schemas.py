"""Arrow schemas of the contract's Parquet files, and a strict schema check."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from ehrfm.inputs.errors import SchemaError

MEDS_DATA_SCHEMA: Final = pa.schema(
    [
        pa.field("subject_id", pa.int64(), nullable=False),
        pa.field("time", pa.timestamp("us"), nullable=True),
        pa.field("code", pa.string(), nullable=False),
        pa.field("numeric_value", pa.float32(), nullable=True),
        pa.field("text_value", pa.string(), nullable=True),
    ]
)

MEDS_CODES_SCHEMA: Final = pa.schema(
    [
        pa.field("code", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=True),
        pa.field("parent_codes", pa.list_(pa.string()), nullable=True),
    ]
)

SUBJECT_SPLITS_SCHEMA: Final = pa.schema(
    [
        pa.field("subject_id", pa.int64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
    ]
)

LABEL_SCHEMA: Final = pa.schema(
    [
        pa.field("subject_id", pa.int64(), nullable=False),
        pa.field("prediction_time", pa.timestamp("us"), nullable=False),
        pa.field("boolean_value", pa.bool_(), nullable=True),
        pa.field("integer_value", pa.int64(), nullable=True),
        pa.field("float_value", pa.float64(), nullable=True),
        pa.field("categorical_value", pa.string(), nullable=True),
    ]
)

CONCEPT_SET_SCHEMA: Final = pa.schema(
    [
        pa.field("vocabulary_id", pa.string(), nullable=False),
        pa.field("concept_code", pa.string(), nullable=False),
        pa.field("code", pa.string(), nullable=False),
        pa.field("entry_types", pa.list_(pa.string()), nullable=False),
    ]
)


def column_values(table: pa.Table, name: str) -> list[Any]:
    """The Python values of one column of ``table``."""
    return table.column(name).to_pylist()


def _describe(schema: pa.Schema) -> list[str]:
    return [f"{f.name}: {f.type}{'' if f.nullable else ' not null'}" for f in schema]


def check_schema(root: Path, rel: str, expected: pa.Schema) -> None:
    """Fail closed unless the Parquet file's schema equals ``expected`` exactly.

    Column names, order, Arrow types and nullability must all match; schema
    metadata is ignored.
    """
    actual = pq.read_schema(root / rel)
    if not actual.remove_metadata().equals(expected):
        raise SchemaError(
            root,
            f"{rel} has schema {_describe(actual)}, the contract requires {_describe(expected)}",
        )


__all__ = [
    "CONCEPT_SET_SCHEMA",
    "LABEL_SCHEMA",
    "MEDS_CODES_SCHEMA",
    "MEDS_DATA_SCHEMA",
    "SUBJECT_SPLITS_SCHEMA",
    "check_schema",
    "column_values",
]
