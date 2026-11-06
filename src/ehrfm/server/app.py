"""The FastAPI application: health, heads, embeddings, predictions and SSE.

Endpoints:

- ``GET /health`` — the process is up; whether the models are loaded.
- ``GET /ready`` — 200 once startup loading succeeded, 503 before or after.
- ``GET /heads`` — every loaded head with its verified record.
- ``POST /embed`` — the subject's last-position representation.
- ``POST /predict`` — one head's output for one subject, selected by name.
- ``POST /predict/stream`` — the same computation as named server-sent
  events: ``started``, ``subject`` (the subject is encoded once), one
  ``prediction`` event per requested head, then ``complete`` (or a terminal
  ``error``).

Startup is fail closed: the lifespan runs :func:`ehrfm.server.state.load_state`
and any failure stops the server before it serves a request.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from ehrfm.server.config import ServerConfig
from ehrfm.server.inference import (
    EmptySubjectError,
    EncodedSubject,
    UnknownHeadError,
    encode_subject,
    format_output,
    head_values,
    resolve_head,
    run_embedding,
    run_prediction,
)
from ehrfm.server.middleware import RequestIdMiddleware
from ehrfm.server.models import (
    EmbedRequest,
    EmbedResponse,
    PredictRequest,
    PredictResponse,
    Provenance,
    StreamComplete,
    StreamError,
    StreamRequest,
    StreamStarted,
    StreamSubject,
    UnknownCodes,
)
from ehrfm.server.postprocess import apply_postprocess, postprocess_names
from ehrfm.server.state import AppState, load_state
from ehrfm.server.subject import count_unknown_codes

type Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]

ENCODER_HEADER: str = "x-ehrfm-encoder-weights-sha256"
HEAD_WEIGHTS_HEADER: str = "x-ehrfm-head-weights-sha256"
HEAD_RECORD_HEADER: str = "x-ehrfm-head-record-sha256"


def _sse(event: str, identifier: str, payload: BaseModel) -> str:
    """One server-sent event frame (the payload is serialised in JSON mode)."""
    return (
        f"event: {event}\nid: {identifier}\ndata: "
        + json.dumps(payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n\n"
    )


def _state(app: FastAPI) -> AppState:
    """The loaded application state; 503 before startup finished."""
    state: AppState | None = getattr(app.state, "ehrfm", None)
    if state is None:
        raise HTTPException(status_code=503, detail="models are not loaded")
    return state


def _encoder_header(response: Response, state: AppState) -> None:
    response.headers[ENCODER_HEADER] = state.encoder_weights_sha256


def _provenance_headers(response: Response, provenance: Provenance) -> None:
    """Weight provenance headers every prediction response carries."""
    response.headers[ENCODER_HEADER] = provenance.encoder_weights_sha256
    response.headers[HEAD_WEIGHTS_HEADER] = provenance.head_weights_sha256
    response.headers[HEAD_RECORD_HEADER] = provenance.head_record_sha256


def _lifespan(settings: ServerConfig) -> Lifespan:
    """The startup/shutdown hook: load everything or never serve."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        state = await asyncio.to_thread(load_state, settings)
        app.state.ehrfm = state
        yield
        del app.state.ehrfm

    return lifespan


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create the serving application; ``config`` defaults to the environment."""
    settings = config or ServerConfig.from_env()
    app = FastAPI(
        title="ehrfm server",
        description="CLMBR representations and task heads over HTTP and SSE.",
        version="0.1.0",
        lifespan=_lifespan(settings),
    )

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, object]:
        """Liveness: the process is up, loaded or not."""
        state: AppState | None = getattr(app.state, "ehrfm", None)
        return {
            "status": "ok",
            "loaded": state is not None,
            "heads": sorted(state.heads) if state is not None else [],
            "postprocess": sorted(postprocess_names()),
        }

    @app.get("/ready", tags=["health"])
    async def ready() -> dict[str, object]:
        """Readiness: 200 once startup loading succeeded, 503 otherwise."""
        state = _state(app)
        return {
            "ready": True,
            "encoder_weights_sha256": state.encoder_weights_sha256,
            "heads": sorted(state.heads),
        }

    @app.get("/heads", tags=["heads"])
    async def heads(response: Response) -> dict[str, object]:
        """Every loaded head and its verified record."""
        state = _state(app)
        _encoder_header(response, state)
        return {
            "encoder_weights_sha256": state.encoder_weights_sha256,
            "heads": [
                {
                    "name": name,
                    "spec": loaded.record.spec.model_dump(mode="json"),
                    "weights_sha256": loaded.record.weights_sha256,
                    "record_sha256": state.record_sha256[name],
                    "metadata": loaded.record.metadata,
                }
                for name, loaded in sorted(state.heads.items())
            ],
        }

    @app.post("/embed", tags=["representations"])
    async def embed(request: EmbedRequest, response: Response) -> EmbedResponse:
        """The subject's last-position representation."""
        state = _state(app)
        try:
            result = await asyncio.to_thread(run_embedding, state, request)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _encoder_header(response, state)
        return result

    @app.post("/predict", tags=["predictions"])
    async def predict(request: PredictRequest, response: Response) -> PredictResponse:
        """One head's output for one subject; the head is selected by name."""
        state = _state(app)
        try:
            resolve_head(state, request.head)
            result = await asyncio.to_thread(run_prediction, state, request)
        except UnknownHeadError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _provenance_headers(response, result.provenance)
        return result

    @app.post("/predict/stream", tags=["predictions"])
    async def predict_stream(request: StreamRequest) -> StreamingResponse:
        """The prediction computation as named server-sent events (see the module docstring)."""
        state = _state(app)
        try:
            for name in request.heads:
                resolve_head(state, name)
        except UnknownHeadError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def stream() -> AsyncIterator[str]:
            started = time.perf_counter()
            counter = 0

            def frame(event: str, payload: BaseModel) -> str:
                nonlocal counter
                identifier = str(counter)
                counter += 1
                return _sse(event, identifier, payload)

            yield frame(
                "started",
                StreamStarted(subject_id=request.subject_id, heads=request.heads),
            )
            try:
                encoded = await asyncio.to_thread(encode_subject, state, request)
                subject_event = StreamSubject(
                    subject_id=request.subject_id,
                    n_positions=encoded.n_positions,
                    unknown_codes=count_unknown_codes(
                        request, state.known_codes, limit=state.config.unknown_code_limit
                    ),
                )
                yield frame("subject", subject_event)
                for name in request.heads:
                    prediction = await asyncio.to_thread(
                        _predict_with, state, encoded, subject_event.unknown_codes, name
                    )
                    yield frame("prediction", prediction)
                yield frame(
                    "complete",
                    StreamComplete(
                        subject_id=request.subject_id,
                        heads=request.heads,
                        duration_ms=round((time.perf_counter() - started) * 1000, 3),
                    ),
                )
            except EmptySubjectError as exc:
                yield frame("error", StreamError(kind="invalid_input", message=str(exc)))
            except (ValueError, RuntimeError) as exc:
                yield frame("error", StreamError(kind="encoding", message=str(exc)))
            except Exception as exc:  # noqa: BLE001 - a stream must end with an error event
                yield frame("error", StreamError(kind="internal", message=str(exc)))

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                ENCODER_HEADER: state.encoder_weights_sha256,
            },
        )

    app.add_middleware(RequestIdMiddleware)
    return app


def _predict_with(
    state: AppState,
    encoded: EncodedSubject,
    unknown: UnknownCodes,
    name: str,
) -> PredictResponse:
    """One head run over an already-encoded subject (the stream's per-head event)."""
    head = resolve_head(state, name)
    started = time.perf_counter()
    raw = head_values(head, encoded.embedding)
    values = apply_postprocess(state.postprocess, raw, head.record)
    record = head.record
    return PredictResponse(
        subject_id=encoded.subject_id,
        head=name,
        task=record.spec.task_name,
        label_type=str(record.spec.label_type),
        output=format_output(head, values),
        postprocess=state.postprocess,
        unknown_codes=unknown,
        provenance=Provenance(
            encoder_weights_sha256=state.encoder_weights_sha256,
            head_weights_sha256=record.weights_sha256,
            head_record_sha256=state.record_sha256[name],
        ),
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
    )


__all__ = ["ENCODER_HEADER", "HEAD_RECORD_HEADER", "HEAD_WEIGHTS_HEADER", "create_app"]
