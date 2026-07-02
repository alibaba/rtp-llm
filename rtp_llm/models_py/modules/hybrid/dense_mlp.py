"""Unified dense MLP implementation supporting multiple activation types."""

import os
from typing import Dict, Optional, Tuple, Type

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.triton_kernels.common.activation import bias_gelu
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
            if W.ffn_w13 not in weights:
                self.up_proj = LinearFactory.create_merged_linear(
                    weights,
                    weight_keys=[W.ffn_w1, W.ffn_w3],
                    scale_keys=[W.ffn_s1, W.ffn_s3],
                    bias_keys=[W.ffn_b1, W.ffn_b3],
                    quant_config=quant_config,
                    dim=-1,
                    hw_kernel_config=hw_kernel_config,
                    scale2_keys=[W.ffn_w1_s2, W.ffn_w3_s2],
                    input_scale_keys=[W.ffn_w1_i_s, W.ffn_w3_i_s],
                )
            else:
                self.up_proj = LinearFactory.create_linear_from_weights(
                    weights,
                    W.ffn_w13,
                    W.ffn_s13,
                    W.ffn_b13,
                    quant_config=quant_config,
                    hw_kernel_config=hw_kernel_config,
                    weight_scale_2_key=W.ffn_w13_s2,
                    input_scale_key=W.ffn_w13_i_s,
                )

        else:
            self.up_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.ffn_w3,
                W.ffn_s3,
                W.ffn_b3,
                quant_config=quant_config,
                hw_kernel_config=hw_kernel_config,
                weight_scale_2_key=W.ffn_w3_s2,
                input_scale_key=W.ffn_w3_i_s,
            )

        self.down_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.ffn_w2,
            W.ffn_s2,
            W.ffn_b2,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.ffn_w2_s2,
            input_scale_key=W.ffn_w2_i_s,
        )

    def _forward_impl(
        self,
        x: torch.Tensor,
        skip_allreduce: bool = False,
        defer_output_bias: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        has_hipb_bias_gelu = (
            not self.is_gated
            and self.activation_type == ActivationType.Gelu
            and hasattr(self.up_proj, "forward_hipb_bias_gelu")
            and getattr(self.up_proj, "bias", None) is not None
            and os.environ.get("DISABLE_HIPB_GELU", "0") != "1"
            and not getattr(self, "_hipb_gelu_unsupported", False)
        )
        can_fuse_up_bias_gelu = (
            not self.is_gated
            and self.activation_type == ActivationType.Gelu
            and hasattr(self.up_proj, "forward_without_bias")
        )
        if has_hipb_bias_gelu:
            try:
                activated = self.up_proj.forward_hipb_bias_gelu(x)
            except TypeError:
                self._hipb_gelu_unsupported = True
                activated = None
            if activated is None:
                up = self.up_proj.forward_without_bias(x)
                up_bias = getattr(self.up_proj, "bias", None)
                activated = (
                    bias_gelu(up, up_bias) if up_bias is not None else self.act_fn(up)
                )
        elif can_fuse_up_bias_gelu:
            up = self.up_proj.forward_without_bias(x)
            up_bias = getattr(self.up_proj, "bias", None)
            activated = (
                bias_gelu(up, up_bias) if up_bias is not None else self.act_fn(up)
            )
        else:
            up = self.up_proj(x)
            activated = self.act_fn(up)

        ffn_tp_size = self.parallelism_config.get_ffn_tp_size()
        output_bias = None
        can_defer_bias = (
            defer_output_bias
            and (skip_allreduce or ffn_tp_size == 1)
            and hasattr(self.down_proj, "forward_without_bias")
        )
        if (
            can_defer_bias
            and hasattr(self.down_proj, "forward_hipb_bias")
            and getattr(self.down_proj, "bias", None) is not None
        ):
            output = self.down_proj.forward_hipb_bias(activated)
        elif can_defer_bias:
            output = self.down_proj.forward_without_bias(activated)
            output_bias = getattr(self.down_proj, "bias", None)
        else:
            output = self.down_proj(activated)

        if not skip_allreduce and ffn_tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output, output_bias

    def forward(self, x: torch.Tensor, skip_allreduce: bool = False) -> torch.Tensor:
        output, _ = self._forward_impl(
            x, skip_allreduce=skip_allreduce, defer_output_bias=False
        )
        return output

    def forward_defer_output_bias(
        self, x: torch.Tensor, skip_allreduce: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Same as forward() but returns down_proj's bias separately for the
        caller to fuse into a following residual-add+LayerNorm. Only effective
        for FP8 linear with TP size 1; otherwise behaves like forward()."""
        return self._forward_impl(
            x, skip_allreduce=skip_allreduce, defer_output_bias=True
        )
