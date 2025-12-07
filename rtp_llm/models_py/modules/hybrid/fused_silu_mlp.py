"""Fused SiLU activation dense MLP - unified implementation."""

from typing import Dict

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group, all_reduce
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.utils.model_weight import W


class FusedSiluActDenseMLP(nn.Module):
    """
    Unified FusedSiluActDenseMLP implementation that works across different architectures.
    Uses architecture-specific silu_and_mul implementation from base layer.
    """

    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()

        assert (
            config.activation_type == "SiGLU"
        ), "FusedSiluActDenseMLP only supports SiGLU activation"
        self.config = config

        # Handle merged or separate weights
        if W.ffn_w13 not in weights:
            self.gate_up_proj = LinearFactory.create_merged_linear(
                weights,
                weight_keys=[W.ffn_w1, W.ffn_w3],
                scale_keys=[W.ffn_s1, W.ffn_s3],
                bias_keys=[W.ffn_b1, W.ffn_b3],
                config=config,
                dim=-1,
            )
            self.down_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, config
            )

        else:
            self.gate_up_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w13, W.ffn_s13, W.ffn_b13, config
            )
            self.down_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, config
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
        if self.config.tp_size > 1:
            down_proj = all_reduce(down_proj, group=Group.TP)
        return down_proj
