from typing import Dict

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group, all_reduce
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops.compute_ops import rtp_llm_ops
from rtp_llm.utils.model_weight import W


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
        self.config = config

        # Handle merged or separate weights
        if W.ffn_w13 not in weights:
            raise ValueError(f"Weight {W.ffn_w13} not found in weights")
        self.gate_up_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w13, W.ffn_s13, W.ffn_b13, config
        )
        self.down_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, config
        )

    def forward(self, x: torch.Tensor):
        gate_up = self.gate_up_proj(x)

        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        stream_id = torch.cuda.current_stream().cuda_stream
        rtp_llm_ops.silu_and_mul(output, gate_up, stream_id)
        down_proj = self.down_proj(output)
        if self.config.tp_size > 1:
            down_proj = all_reduce(down_proj, group=Group.TP)
        return down_proj


class BertGeluActDenseMLP(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()

        # For BERT model, use traditional FFN structure with GeLU activation
        # BERT uses: intermediate_weight3 -> GeLU -> intermediate_weight2
        self.intermediate_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w3, W.ffn_s3, W.ffn_b3, config
        )
        self.output_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, config
        )

        # Use GeLU activation
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor):
        # Traditional BERT FFN: intermediate -> GeLU -> output
        intermediate = self.intermediate_proj(x)
        activated = self.act_fn(intermediate)
        output = self.output_proj(activated)
        return output
