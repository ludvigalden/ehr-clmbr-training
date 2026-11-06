"""MLflow tracking behind the small-cell suppression boundary.

:class:`Tracker` is the only path from the pipeline to MLflow. Every
parameter, metric and JSON artifact goes through
:mod:`ehrfm.experiment.suppression` with the experiment's
:class:`~ehrfm.config.experiment.SmallCellPolicy` before the MLflow client sees
it; there is no method that logs raw values. Each run also gets a
``suppression.json`` artifact listing what was masked (paths and reasons,
never the masked values).

The default store is a local file store at ``<work_dir>/mlruns``. MLflow
releases that discourage the file store refuse it unless
``MLFLOW_ALLOW_FILE_STORE`` is set; :func:`open_tracker` sets it (only when it
is unset) for ``file:`` URIs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import JsonValue

from ehrfm.config.experiment import SmallCellPolicy, TrackingConfig
from ehrfm.experiment.suppression import SuppressedCell, flatten, suppress, suppress_flat

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

MLRUNS_DIR: Final = "mlruns"
SUPPRESSION_ARTIFACT: Final = "suppression.json"
SUPPRESSED_PARAM: Final = "suppressed"
TAG_PREFIX: Final = "ehrfm."
_MAX_PARAM_LENGTH: Final = 6000
RunStatus = Literal["FINISHED", "FAILED", "KILLED"]


def default_uri(work_dir: Path) -> str:
    """The local file-store URI under ``work_dir``."""
    return (work_dir / MLRUNS_DIR).resolve().as_uri()


def _client(uri: str) -> MlflowClient:
    if uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri)


@dataclass(slots=True)
class TrackedRun:
    """An open MLflow run; every method suppresses before logging."""

    client: MlflowClient
    run_id: str
    policy: SmallCellPolicy
    suppressed: list[tuple[str, SuppressedCell]] = field(
        default_factory=list[tuple[str, SuppressedCell]]
    )

    def _record(self, where: str, cells: tuple[SuppressedCell, ...]) -> None:
        self.suppressed.extend((where, cell) for cell in cells)

    def log_params(self, params: Mapping[str, JsonValue]) -> None:
        """Log nested parameters as flat ``a.b.c`` names (count fields suppressed)."""
        masked = suppress(dict(params), self.policy)
        self._record("params", masked.cells)
        for name, value in flatten(masked.value, separator=".").items():
            if not name:
                continue
            text = SUPPRESSED_PARAM if value is None else _param_text(value)
            self.client.log_param(self.run_id, name, text[:_MAX_PARAM_LENGTH])

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Log ``a/b`` named metrics; suppressed names are not logged at all."""
        if not metrics:
            return
        masked = suppress_flat(metrics, self.policy)
        self._record("metrics" if step is None else f"metrics@{step}", masked.cells)
        assert isinstance(masked.value, dict)
        for name, value in masked.value.items():
            assert isinstance(value, int | float) and not isinstance(value, bool)
            self.client.log_metric(self.run_id, name, float(value), step=step or 0)

    def log_series(self, name: str, values: tuple[float, ...]) -> None:
        """Log a per-step series (losses); the name must not be a count field."""
        for step, value in enumerate(values, start=1):
            self.log_metrics({name: value}, step=step)

    def log_json(self, artifact_file: str, payload: JsonValue) -> None:
        """Log a JSON artifact with small cells replaced by ``null``."""
        masked = suppress(payload, self.policy)
        self._record(artifact_file, masked.cells)
        text = json.dumps(masked.value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        self.client.log_text(self.run_id, text, artifact_file)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Set ``ehrfm.``-prefixed tags (identifiers and hashes only, never counts)."""
        for key, value in tags.items():
            self.client.set_tag(self.run_id, f"{TAG_PREFIX}{key}", value)

    def suppression_log(self) -> list[dict[str, str]]:
        """What this run masked so far."""
        return [
            {"source": where, "path": cell.path, "reason": cell.reason}
            for where, cell in self.suppressed
        ]


def _param_text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


@dataclass(frozen=True, slots=True)
class Tracker:
    """An MLflow experiment in a store, with the suppression policy every run applies."""

    client: MlflowClient
    uri: str
    experiment_id: str
    experiment_name: str
    policy: SmallCellPolicy

    @contextmanager
    def run(self, name: str, tags: Mapping[str, str] | None = None) -> Generator[TrackedRun]:
        """Open a run; it ends FINISHED, or FAILED when the body raises."""
        created = self.client.create_run(self.experiment_id, run_name=name)
        run = TrackedRun(self.client, created.info.run_id, self.policy)
        status: RunStatus = "FAILED"
        try:
            if tags:
                run.set_tags(tags)
            yield run
            status = "FINISHED"
        finally:
            try:
                run.client.log_text(
                    run.run_id,
                    json.dumps(
                        {"threshold": self.policy.threshold, "suppressed": run.suppression_log()},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    SUPPRESSION_ARTIFACT,
                )
            finally:
                self.client.set_terminated(run.run_id, status=status)

    def find_runs(self, tag: str, value: str) -> list[Any]:
        """Runs of this experiment whose ``ehrfm.<tag>`` equals ``value``, newest first."""
        return list(
            self.client.search_runs(
                [self.experiment_id],
                filter_string=f"tags.`{TAG_PREFIX}{tag}` = '{value}'",
                order_by=["attributes.start_time DESC"],
            )
        )


def open_tracker(
    tracking: TrackingConfig, name: str, work_dir: Path, policy: SmallCellPolicy
) -> Tracker:
    """Open (creating when needed) the MLflow experiment ``name`` in the configured store."""
    uri = tracking.uri or default_uri(work_dir)
    client = _client(uri)
    experiment = client.get_experiment_by_name(name)
    experiment_id = (
        experiment.experiment_id if experiment is not None else client.create_experiment(name)
    )
    return Tracker(client, uri, str(experiment_id), name, policy)


def store_client(uri: str) -> MlflowClient:
    """A client for reading a store (``ehrfm ls`` / ``show``)."""
    return _client(uri)


__all__ = [
    "MLRUNS_DIR",
    "SUPPRESSED_PARAM",
    "SUPPRESSION_ARTIFACT",
    "TAG_PREFIX",
    "TrackedRun",
    "Tracker",
    "default_uri",
    "open_tracker",
    "store_client",
]
