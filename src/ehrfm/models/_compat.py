"""Runtime compatibility for femr's optional recurrent (hawk) layers.

femr 0.2.4 imports ``torch_hawk`` unconditionally at module level in
``femr.models.transformer`` and builds a ``torch_hawk.RecurrentBlock`` for every
even-indexed encoder layer. torch-hawk ships only x86_64 CUDA builds, so on a
CPU-only or aarch64 host that import fails and no femr model can be loaded.

CLMBR checkpoints are trained with uniform attention in every layer, and the
checkpoint loader replaces each hawk layer with a uniform-attention layer
before any weights are loaded or any forward pass runs. The recurrent block is
therefore only needed as an import target and a constructor. When the real
package is importable it is always used; otherwise :func:`install` registers a
minimal ``torch_hawk`` module whose ``RecurrentBlock`` constructs (holding no
parameters) and raises :class:`HawkUnavailableError` on any forward call, so a
model that still contains a hawk layer fails loudly instead of computing
meaningless values.

femr's encoder also builds its attention mask with
``xformers.ops.fmha.attn_bias.BlockDiagonalMask``. Some CPU-only xformers
builds ship without the ``fmha`` submodule; :func:`install_attention_bias`
then registers a pure-torch mask with the same semantics (block-diagonal per
subject; causal and banded once ``make_local_attention`` is applied). femr's
CPU attention only ever calls ``materialize`` on it.

Importing :mod:`ehrfm.models` installs both, so every module under
``ehrfm.models`` may import ``femr.models.transformer`` at top level. Code
outside that package must import :mod:`ehrfm.models` (or call
:func:`install`) before importing femr's model modules.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from collections.abc import Sequence
from typing import Final, NoReturn, Self

import torch
from torch import Tensor, nn

MODULE_NAME: Final = "torch_hawk"
_SHIM_MARKER: Final = "__ehrfm_compat_shim__"


class HawkUnavailableError(RuntimeError):
    """A hawk (recurrent) layer was used while only the compatibility shim is installed."""


class RecurrentBlock(nn.Module):
    """Stand-in for ``torch_hawk.RecurrentBlock`` that refuses to compute.

    It accepts femr's constructor arguments and holds no parameters, so a model
    can be built and its hawk layers replaced. Calling it raises.
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def forward(self, x: Tensor, s: Tensor) -> NoReturn:
        """Raise: the recurrent computation needs a real torch-hawk install."""
        del x, s
        raise HawkUnavailableError(
            "a hawk (recurrent) layer was called, but torch-hawk is not installed and "
            "only the ehrfm compatibility shim is active. Load checkpoints through "
            "ehrfm.models.checkpoint, which replaces hawk layers with uniform "
            "attention, or install torch-hawk on a supported x86_64 CUDA host."
        )


def is_active() -> bool:
    """Return True when ``torch_hawk`` resolves to this shim."""
    module = sys.modules.get(MODULE_NAME)
    return module is not None and getattr(module, _SHIM_MARKER, False) is True


class AttentionBias:
    """Base of the fallback attention biases (stands in for ``xformers.ops.AttentionBias``)."""

    def materialize(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Return the additive mask (0 where attention is allowed, -inf elsewhere)."""
        raise NotImplementedError


class BlockDiagonalMask(AttentionBias):
    """Fallback ``BlockDiagonalMask``: each sequence attends only within its own block.

    Without a window every position sees its whole block. With a window (from
    :meth:`make_local_attention`) position ``i`` sees positions ``j`` of its
    block with ``i - window < j <= i``, as xformers'
    ``BlockDiagonalCausalLocalAttentionMask`` does.
    """

    def __init__(self, seqlens: Sequence[int], window: int | None = None) -> None:
        if any(length <= 0 for length in seqlens):
            raise ValueError(f"sequence lengths must be positive, got {list(seqlens)}")
        if window is not None and window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self.seqlens = tuple(int(length) for length in seqlens)
        self.window = window

    @classmethod
    def from_seqlens(cls, seqlens: Sequence[int]) -> Self:
        """Build a mask for sequences of the given lengths, packed back to back."""
        return cls(seqlens)

    def make_local_attention(self, window_size: int) -> BlockDiagonalMask:
        """Return the causal, banded variant of this mask."""
        return BlockDiagonalMask(self.seqlens, window_size)

    def materialize(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        total = sum(self.seqlens)
        if shape[-2:] != (total, total):
            raise ValueError(f"mask shape {shape} does not match {total} packed positions")
        block = torch.repeat_interleave(
            torch.arange(len(self.seqlens), device=device),
            torch.tensor(self.seqlens, device=device),
        )
        position = torch.arange(total, device=device)
        allowed = block[:, None] == block[None, :]
        if self.window is not None:
            distance = position[:, None] - position[None, :]
            allowed &= (distance >= 0) & (distance < self.window)
        mask = torch.zeros((total, total), dtype=dtype, device=device)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask.expand(shape).contiguous()


def install_attention_bias(ops: types.ModuleType) -> bool:
    """Give an ``xformers.ops`` module the ``fmha.attn_bias`` mask when it lacks one.

    Returns True when the fallback was registered and False when ``ops``
    already provides ``fmha``.
    """
    if hasattr(ops, "fmha"):
        return False
    attn_bias = types.ModuleType(f"{ops.__name__}.fmha.attn_bias")
    attn_bias.BlockDiagonalMask = BlockDiagonalMask  # pyright: ignore[reportAttributeAccessIssue]
    fmha = types.ModuleType(f"{ops.__name__}.fmha")
    fmha.attn_bias = attn_bias  # pyright: ignore[reportAttributeAccessIssue]
    ops.fmha = fmha  # pyright: ignore[reportAttributeAccessIssue]
    if not hasattr(ops, "AttentionBias"):
        ops.AttentionBias = AttentionBias  # pyright: ignore[reportAttributeAccessIssue]
    return True


def _install_xformers_fallback() -> None:
    if importlib.util.find_spec("xformers") is None:
        return
    install_attention_bias(importlib.import_module("xformers.ops"))


def install() -> bool:
    """Make femr's model module importable; return True when the hawk shim is active.

    The real torch-hawk wins whenever it is importable; the shim is registered
    in ``sys.modules`` only when it is not. The xformers attention-mask
    fallback is added only when xformers lacks ``fmha``. Calling this more than
    once is safe.
    """
    _install_xformers_fallback()
    if MODULE_NAME in sys.modules:
        return is_active()
    if importlib.util.find_spec(MODULE_NAME) is not None:
        return False
    module = types.ModuleType(MODULE_NAME, "ehrfm compatibility shim for torch_hawk")
    module.RecurrentBlock = RecurrentBlock  # pyright: ignore[reportAttributeAccessIssue]
    setattr(module, _SHIM_MARKER, True)
    sys.modules[MODULE_NAME] = module
    return True


__all__ = [
    "AttentionBias",
    "BlockDiagonalMask",
    "HawkUnavailableError",
    "RecurrentBlock",
    "install",
    "install_attention_bias",
    "is_active",
]
