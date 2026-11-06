"""Choose the torch device a model runs on.

Precedence: an explicit request, then the ``EHRFM_DEVICE`` environment
variable, then automatic selection. Automatic selection takes the first CUDA
device, then Apple MPS, then the CPU. A device that is named explicitly but is
not available raises instead of silently running somewhere else.
"""

from __future__ import annotations

import os
from typing import Final

import torch

DEVICE_ENV: Final = "EHRFM_DEVICE"
AUTO: Final = "auto"


class DeviceUnavailableError(RuntimeError):
    """A device was requested by name but this host cannot provide it."""


def mps_available() -> bool:
    """Return True when Apple's MPS backend is built and usable."""
    backend = getattr(torch.backends, "mps", None)
    return backend is not None and backend.is_built() and backend.is_available()


def auto_device() -> torch.device:
    """The best available device: CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _check_available(device: torch.device) -> torch.device:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceUnavailableError(f"{device} was requested but CUDA is not available")
        # torch types ``index`` as int, but it is None for a bare "cuda".
        index = device.index or 0
        count = torch.cuda.device_count()
        if index >= count:
            raise DeviceUnavailableError(
                f"{device} was requested but only {count} CUDA devices exist"
            )
    elif device.type == "mps" and not mps_available():
        raise DeviceUnavailableError(f"{device} was requested but MPS is not available")
    elif device.type not in ("cpu", "cuda", "mps"):
        raise DeviceUnavailableError(f"device type {device.type!r} is not supported")
    return device


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve ``requested`` (``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``).

    ``None`` defers to ``EHRFM_DEVICE`` and then to automatic selection.
    """
    spec = requested if requested is not None else os.environ.get(DEVICE_ENV, AUTO)
    spec = spec.strip() or AUTO
    if spec == AUTO:
        return auto_device()
    try:
        device = torch.device(spec)
    except RuntimeError as exc:
        raise DeviceUnavailableError(f"{spec!r} is not a device name") from exc
    return _check_available(device)


__all__ = [
    "AUTO",
    "DEVICE_ENV",
    "DeviceUnavailableError",
    "auto_device",
    "mps_available",
    "resolve_device",
]
