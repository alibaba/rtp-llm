from typing import Dict
import aiter
import torch
from torch import nn

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.linear_factory import LinearFactory
from rtp_llm.utils.model_weight import W
from rtp_llm.ops import rtp_llm_ops


class DenseMLP(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()

        # Create linear layers using LinearFactory
        self.gate_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w1, W.ffn_s1, W.ffn_b1, config
        )
        self.up_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w3, W.ffn_s3, W.ffn_b3, config
        )
        self.down_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, config
        )

        if config.activation_type == "SiGLU":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type: {config.activation_type}")

    def forward(self, x: torch.Tensor):
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        activated = self.act_fn(gate_output)
        product = activated * up_output
        down_output = self.down_proj(product)
        return down_output


class FusedSiluActDenseMLP(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        assert (
            config.activation_type == "SiGLU"
        ), "FusedSiluActDenseMLP only supports SiGLU activation"

        # Handle merged or separate weights
        if W.ffn_w13 in weights:
            # Pre-merged weights
            self.gate_up_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w13, W.ffn_s13, W.ffn_b13, config
            )
            self.down_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, config
            )
        else:
            # Separate weights: concatenate w1 and w3
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

    def forward(self, x: torch.Tensor):
        gate_up = self.gate_up_proj(x)

        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        aiter.silu_and_mul(output, gate_up)
        down_proj = self.down_proj(output)
        return down_proj
