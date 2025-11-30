from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import ParallelismConfig, ActivationType, rtp_llm_ops
from rtp_llm.utils.model_weight import W

class DenseMLP(nn.Module):
    def __init__(
        self, activation_type: ActivationType, parallelism_config: ParallelismConfig, weights: Dict[str, torch.Tensor], quant_config: object
    ):
        super().__init__()

        # Create linear layers using LinearFactory
        self.gate_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w1, W.ffn_s1, W.ffn_b1, quant_config=quant_config
        )
        self.up_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w3, W.ffn_s3, W.ffn_b3, quant_config=quant_config
        )
        self.down_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, quant_config=quant_config
        )

        if activation_type == ActivationType.Swiglu:
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def forward(self, x: torch.Tensor):
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        activated = self.act_fn(gate_output)
        product = activated * up_output
        down_output = self.down_proj(product)
        return down_output


class FusedSiluActDenseMLP(nn.Module):
    def __init__(
        self, activation_type: ActivationType, parallelism_config: ParallelismConfig, weights: Dict[str, torch.Tensor], quant_config: object
    ):
        super().__init__()

        assert activation_type == ActivationType.Swiglu, \
            f"FusedSiluActDenseMLP only supports SiGLU activation, got {activation_type}"
        self.parallelism_config = parallelism_config

        # Handle merged or separate weights
        if W.ffn_w13 in weights:
            self.gate_up_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w13, W.ffn_s13, W.ffn_b13, quant_config=quant_config
            )
            self.down_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, quant_config=quant_config
            )
        else:
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

    def forward(self, x: torch.Tensor):
        gate_up = self.gate_up_proj(x)

        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        stream_id = torch.cuda.current_stream().cuda_stream
        rtp_llm_ops.silu_and_mul(output, gate_up, stream_id)
        down_proj = self.down_proj(output)
        if self.parallelism_config.tp_size > 1:
            down_proj = all_reduce(down_proj, group=Group.TP)
        return down_proj


class BertGeluActDenseMLP(nn.Module):
    def __init__(
        self, activation_type: ActivationType, parallelism_config: ParallelismConfig, weights: Dict[str, torch.Tensor], quant_config: object
    ):
        super().__init__()

        # For BERT model, use traditional FFN structure with GeLU activation
        # BERT uses: intermediate_weight3 -> GeLU -> intermediate_weight2
        self.intermediate_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w3, W.ffn_s3, W.ffn_b3, quant_config=quant_config
        )
        self.output_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, quant_config=quant_config
        )

        # Use GeLU activation
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor):
        # Traditional BERT FFN: intermediate -> GeLU -> output
        intermediate = self.intermediate_proj(x)
        activated = self.act_fn(intermediate)
        output = self.output_proj(activated)
        return output
