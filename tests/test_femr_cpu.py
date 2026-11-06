"""CPU-feasibility tests for the femr model stack.

These tests pin the hermetic-CPU contract that the rest of the
repository builds on: the femr model modules import, the
uniform-attention encoder path (the only path the checkpoint surgery
keeps) constructs and runs without torch-hawk, and a tiny FEMR model
constructed from config alone runs one forward pass on CPU once its hawk
layers are replaced (as the checkpoint loader does). The toy checkpoint
fixture builds on this.
"""

from typing import Any

import femr.models.transformer
import numpy as np
import torch
from femr.models.config import FEMRModelConfig, FEMRTransformerConfig

_HIDDEN_SIZE = 32
_N_HEADS = 4
_N_LAYERS = 2
_VOCAB_SIZE = 128
_TOTAL_TOKENS = 3


def _tiny_transformer_config() -> FEMRTransformerConfig:
    return FEMRTransformerConfig(
        vocab_size=_VOCAB_SIZE,
        is_hierarchical=False,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=64,
        n_heads=_N_HEADS,
        n_layers=_N_LAYERS,
        use_normed_ages=False,
    )


def _tiny_packed_batch() -> dict[str, Any]:
    """A minimal packed batch for one subject with three tokens.

    Shapes follow ``FEMRTransformer.forward``: a single packed dimension
    across all subjects' tokens, ages as float32 (consumed by the rotary
    age embeddings), time_data present but unused when
    ``use_normed_ages`` is False, and per-subject sequence lengths for
    the block-diagonal local attention mask. A leading batch dimension
    is included because ``FEMRModel.forward`` squeezes it away.
    """
    return {
        "subject_ids": torch.zeros(1, _TOTAL_TOKENS, dtype=torch.int64),
        "transformer": {
            "tokens": (torch.arange(_TOTAL_TOKENS) % _VOCAB_SIZE).reshape(1, _TOTAL_TOKENS),
            "ages": torch.linspace(0.0, 1.0, _TOTAL_TOKENS, dtype=torch.float32).reshape(
                1, _TOTAL_TOKENS
            ),
            "time_data": torch.zeros(1, _TOTAL_TOKENS, 1),
            "timestamps": np.zeros((1, _TOTAL_TOKENS)),
            "subject_lengths": np.asarray([_TOTAL_TOKENS]).reshape(1, 1),
        },
    }


def test_femr_transformer_module_imports() -> None:
    """femr.models.transformer imports (via the package shim when torch-hawk is absent)."""
    assert hasattr(femr.models.transformer, "FEMRModel")


def test_uniform_attention_encoder_layer_forward() -> None:
    """The uniform-attention layer constructs and runs with no torch_hawk."""
    config = _tiny_transformer_config()
    layer = femr.models.transformer.FEMREncoderLayer(config, use_hawk=False)
    assert not layer.use_hawk

    x = torch.randn(_TOTAL_TOKENS, _HIDDEN_SIZE)
    time_data = torch.zeros(_TOTAL_TOKENS, 1)
    pos_embed = femr.models.transformer.fixed_pos_embedding(
        torch.linspace(0.0, 1.0, _TOTAL_TOKENS), _HIDDEN_SIZE // _N_HEADS, x.dtype
    )
    attn_bias = femr.models.transformer.xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
        [_TOTAL_TOKENS]
    ).make_local_attention(config.attention_width)
    s = torch.zeros(_TOTAL_TOKENS, dtype=torch.uint8)

    out = layer(x, time_data, pos_embed, attn_bias, s)
    assert out.shape == (_TOTAL_TOKENS, _HIDDEN_SIZE)


def test_femr_model_construct_and_forward_cpu() -> None:
    """A tiny FEMR model constructed from config alone runs on CPU without hawk layers."""
    config = FEMRModelConfig(transformer_config=_tiny_transformer_config().to_dict())
    model = femr.models.transformer.FEMRModel(config)
    layers = model.transformer.layers
    for index, layer in enumerate(layers):
        if layer.use_hawk:
            layers[index] = femr.models.transformer.FEMREncoderLayer(
                config.transformer_config, use_hawk=False
            )
    model.eval()

    with torch.no_grad():
        loss, result = model(_tiny_packed_batch())

    assert loss == 0
    # femr flattens the packed representation to (total_tokens, hidden).
    assert result["representations"].shape == (_TOTAL_TOKENS, _HIDDEN_SIZE)
    assert torch.isfinite(result["representations"]).all()


def test_tokenizer_and_meds_reader_import() -> None:
    """The tokenizer and meds_reader modules import."""
    import meds_reader
    from femr.models.tokenizer import HierarchicalTokenizer

    assert meds_reader is not None
    assert HierarchicalTokenizer is not None
