# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wan2.2 quantization coverage tests."""

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


class RecordingQuantConfig(QuantizationConfig):
    """Lightweight quant config that records vLLM linear coverage.

    INT8 online quant methods need CUDA/NPU kernels, so this test double keeps
    construction CPU-safe while exercising the same ``quant_config`` plumbing
    used by the real INT8 config: every vLLM ``LinearBase`` must request a
    quant method during module construction.
    """

    def __init__(self) -> None:
        self.linear_count = 0

    def get_name(self) -> str:
        return "recording"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    def get_min_capability(self) -> int:
        return 0

    def get_config_filenames(self) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "RecordingQuantConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            self.linear_count += 1
            return UnquantizedLinearMethod()
        return None


def _make_tiny_wan(quant_config: QuantizationConfig) -> WanTransformer3DModel:
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=8,
        ffn_dim=32,
        num_layers=1,
        image_dim=12,
        added_kv_proj_dim=16,
        rope_max_seq_len=16,
        pos_embed_seq_len=4,
        quant_config=quant_config,
    )


def test_wan2_2_quant_config_reaches_transformer_linears():
    quant_config = RecordingQuantConfig()

    model = _make_tiny_wan(quant_config)

    quantized_module_names = {
        name for name, module in model.named_modules() if getattr(module, "quant_method", None) is not None
    }
    expected_module_names = {
        # Main Wan2.2 transformer projections.
        "blocks.0.attn1.to_qkv",
        "blocks.0.attn1.to_out",
        "blocks.0.attn2.to_q",
        "blocks.0.attn2.to_k",
        "blocks.0.attn2.to_v",
        "blocks.0.attn2.add_k_proj",
        "blocks.0.attn2.add_v_proj",
        "blocks.0.attn2.to_out",
        "blocks.0.ffn.net_0.proj",
        "blocks.0.ffn.net_2",
        "proj_out",
    }

    assert expected_module_names.issubset(quantized_module_names)
    assert quant_config.linear_count >= len(expected_module_names)

    # Condition, time-embedding, and RoPE modules should stay unquantized.
    assert "condition_embedder.text_embedder.linear_1" not in quantized_module_names
    assert "condition_embedder.text_embedder.linear_2" not in quantized_module_names
    assert "condition_embedder.image_embedder.ff.net.0.proj" not in quantized_module_names
    assert "condition_embedder.image_embedder.ff.net.2" not in quantized_module_names
    assert "condition_embedder.time_embedder.linear_1" not in quantized_module_names
    assert "condition_embedder.time_embedder.linear_2" not in quantized_module_names
    assert "condition_embedder.time_proj" not in quantized_module_names
    assert all(not name.startswith("rope") for name in quantized_module_names)
