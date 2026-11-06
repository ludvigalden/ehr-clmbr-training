"""The package-level torch-hawk shim: one mechanism, loud on any hawk use."""

from __future__ import annotations

import subprocess
import sys

import femr.models.transformer
import pytest
import torch
from femr.models.config import FEMRTransformerConfig

from ehrfm.models import _compat


def _shim_required() -> None:
    if not _compat.is_active():
        pytest.skip("a real torch-hawk is installed; the shim is inactive")


def test_install_is_idempotent() -> None:
    assert _compat.install() == _compat.install()
    assert "torch_hawk" in sys.modules


def test_hawk_layer_forward_raises_under_shim() -> None:
    _shim_required()
    config = FEMRTransformerConfig(
        vocab_size=8, hidden_size=16, intermediate_size=32, n_heads=2, n_layers=2
    )
    layer = femr.models.transformer.FEMREncoderLayer(config, use_hawk=True)
    x = torch.zeros(3, 16)
    s = torch.zeros(3, dtype=torch.uint8)
    with pytest.raises(_compat.HawkUnavailableError, match="torch-hawk is not installed"):
        layer(x, torch.zeros(3, 1), None, None, s)


def test_shim_block_has_no_parameters() -> None:
    _shim_required()
    block = _compat.RecurrentBlock(16, num_heads=2)
    assert list(block.parameters()) == []


def test_femr_imports_outside_pytest() -> None:
    """Production entrypoints import femr through the package shim, not test config."""
    code = "import ehrfm.models._compat; import femr.models.transformer; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
