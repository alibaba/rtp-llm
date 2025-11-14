"""Fused SiLU activation dense MLP - unified implementation."""

from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import ActivationType, ParallelismConfig
from rtp_llm.utils.model_weight import W


class FusedSiluActDenseMLP(nn.Module):
    """
    Unified FusedSiluActDenseMLP implementation that works across different architectures.
    Uses architecture-specific silu_and_mul implementation from base layer.
    """

    def __init__(
        self, activation_type: ActivationType, parallelism_config: ParallelismConfig, weights: Dict[str, torch.Tensor], quant_config: object
    ):
        super().__init__()

        assert activation_type == ActivationType.Swiglu, \
            f"FusedSiluActDenseMLP only supports SiGLU activation, got {activation_type}"
        self.parallelism_config = parallelism_config

        # Handle merged or separate weights
        if W.ffn_w13 not in weights:
            self.gate_up_proj = LinearFactory.create_merged_linear(
                weights,
                weight_keys=[W.ffn_w1, W.ffn_w3],
                scale_keys=[W.ffn_s1, W.ffn_s3],
                bias_keys=[W.ffn_b1, W.ffn_b3],
                quant_config=quant_config,
                dim=-1,
            )
            self.down_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, quant_config=quant_config
            )

        else:
            self.gate_up_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w13, W.ffn_s13, W.ffn_b13, quant_config=quant_config
            )
            self.down_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, quant_config=quant_config
            )

        # Get device-specific silu_and_mul implementation
        self.silu_and_mul_impl = FusedSiluAndMul()

    def forward(self, x: torch.Tensor):
        gate_up = self.gate_up_proj(x)
        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)

        # Use architecture-specific implementation
        self.silu_and_mul_impl.silu_and_mul(output, gate_up)

        down_proj = self.down_proj(output)
        if self.parallelism_config.tp_size > 1:
            down_proj = all_reduce(down_proj, group=Group.TP)
        return down_proj
