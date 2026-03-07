"""Unified dense MLP implementation supporting multiple activation types."""

from typing import Dict, Optional, Type

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import ActivationType, HWKernelConfig, ParallelismConfig
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
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()

        self.activation_type = activation_type
        self.parallelism_config = parallelism_config
        if self.activation_type not in _ACTIVATION_FUNC_MAP:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        self.act_fn = _ACTIVATION_FUNC_MAP[activation_type]()
        self.is_gated = activation_type in _GATED_ACTIVATION_TYPE_LIST

        if self.is_gated:
            if W.ffn_gate_up not in weights:
                self.up_proj = LinearFactory.create_merged_linear(
                    weights,
                    weight_keys=[W.ffn_gate, W.ffn_up],
                    scale_keys=[W.ffn_gate_s, W.ffn_up_s],
                    bias_keys=[W.ffn_gate_b, W.ffn_up_b],
                    quant_config=quant_config,
                    dim=-1,
                    hw_kernel_config=hw_kernel_config,
                    scale2_keys=[W.ffn_gate_s2, W.ffn_up_s2],
                    input_scale_keys=[W.ffn_gate_i_s, W.ffn_up_i_s],
                )
            else:
                self.up_proj = LinearFactory.create_linear_from_weights(
                    weights,
                    W.ffn_gate_up,
                    W.ffn_gate_up_s,
                    W.ffn_gate_up_b,
                    quant_config=quant_config,
                    hw_kernel_config=hw_kernel_config,
                    weight_scale_2_key=W.ffn_gate_up_s2,
                    input_scale_key=W.ffn_gate_up_i_s,
                )

        else:
            self.up_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.ffn_up,
                W.ffn_up_s,
                W.ffn_up_b,
                quant_config=quant_config,
                hw_kernel_config=hw_kernel_config,
                weight_scale_2_key=W.ffn_up_s2,
                input_scale_key=W.ffn_up_i_s,
            )

        self.down_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.ffn_down,
            W.ffn_down_s,
            W.ffn_down_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.ffn_down_s2,
            input_scale_key=W.ffn_down_i_s,
        )

    def forward(self, x: torch.Tensor):
        up = self.up_proj(x)
        activated = self.act_fn(up)
        output = self.down_proj(activated)
        if self.parallelism_config.get_ffn_tp_size() > 1:
            output = all_reduce(output, group=Group.TP)
        return output
