"""Request identification middleware.

Every request gets an identifier: the incoming ``X-Request-Id`` header when
the caller supplied one, a generated identifier otherwise. The identifier is
bound to the structlog context for the request's lifetime and echoed back on
the response.
"""

from __future__ import annotations

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER: str = "X-Request-Id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assigns, logs and echoes a request identifier on every response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id", "path")
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def request_id() -> str | None:
    """The current request identifier, or ``None`` outside a request."""
    return structlog.contextvars.get_contextvars().get("request_id")


__all__ = ["REQUEST_ID_HEADER", "RequestIdMiddleware", "request_id"]
