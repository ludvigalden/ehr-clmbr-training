"""The ``ehrfm`` serving API: a FastAPI application over CLMBR representations.

Import :func:`ehrfm.server.create_app` for the application,
:class:`ehrfm.server.config.ServerConfig` for its configuration, and
:func:`ehrfm.server.state.load_state` for the startup loading contract.
Importing this package installs the ``torch_hawk`` shim the femr model code
needs (see :mod:`ehrfm.models._compat`).
"""

from __future__ import annotations

from ehrfm.models import _compat
from ehrfm.server.app import create_app
from ehrfm.server.config import ServerConfig
from ehrfm.server.models import PredictRequest, PredictResponse
from ehrfm.server.state import AppState, load_state

_compat.install()

__all__ = [
    "AppState",
    "PredictRequest",
    "PredictResponse",
    "ServerConfig",
    "create_app",
    "load_state",
]
