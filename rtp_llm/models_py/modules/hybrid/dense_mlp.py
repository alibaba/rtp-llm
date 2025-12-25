"""Unified dense MLP implementation supporting multiple activation types."""

from typing import Dict, Type

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import ActivationType, ParallelismConfig
from rtp_llm.utils.model_weight import W

_ACTIVATION_FUNC_MAP: Dict[ActivationType, Type[nn.Module]] = {
    ActivationType.Swiglu: FusedSiluAndMul,
    ActivationType.Gelu: nn.GELU,
}

_GATED_ACTIVATION_TYPE_LIST = [ActivationType.Swiglu]


class DenseMLP(nn.Module):
    """
    Unified DenseMLP implementation supporting both SiGLU and GELU activations.

    - For SiGLU (Swiglu): Uses gate_up_proj + fused silu_and_mul + down_proj
    - For GELU (Gelu): Uses intermediate_proj + GELU activation + output_proj
    """

    def __init__(
        self,
        activation_type: ActivationType,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        quant_config: object,
    ):
        super().__init__()

        self.activation_type = activation_type
        self.parallelism_config = parallelism_config
        if self.activation_type not in _ACTIVATION_FUNC_MAP:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        self.act_fn = _ACTIVATION_FUNC_MAP[activation_type]()
        self.is_gated = activation_type in _GATED_ACTIVATION_TYPE_LIST

        if self.is_gated:
            if W.ffn_w13 not in weights:
                self.up_proj = LinearFactory.create_merged_linear(
                    weights,
                    weight_keys=[W.ffn_w1, W.ffn_w3],
                    scale_keys=[W.ffn_s1, W.ffn_s3],
                    bias_keys=[W.ffn_b1, W.ffn_b3],
                    quant_config=quant_config,
                    dim=-1,
                )
            else:
                self.up_porj = LinearFactory.create_linear_from_weights(
                    weights, W.ffn_w13, W.ffn_s13, W.ffn_b13, quant_config=quant_config
                )

        else:
            self.up_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w3, W.ffn_s3, W.ffn_b3, quant_config=quant_config
            )

        self.down_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w2, W.ffn_s2, W.ffn_b2, quant_config=quant_config
        )

    def forward(self, x: torch.Tensor):
        up = self.up_proj(x)
        activated = self.act_fn(up)
        output = self.down_proj(activated)
        if self.parallelism_config.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output
