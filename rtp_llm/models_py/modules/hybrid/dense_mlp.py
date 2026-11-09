"""Unified dense MLP implementation supporting multiple activation types."""

import os
from typing import Dict, Optional, Type

import torch
from torch import nn
from torch.nn import functional as F

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


class ClampedSiluAndMul(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit
        self._prefill_fusion = None
        if (
            torch.cuda.is_available()
            and torch.version.hip is None
            and os.environ.get("GLM53_PREFILL_FUSED_CLAMPED_SWIGLU", "0") == "1"
        ):
            from rtp_llm.models_py.triton_kernels.common.clamped_swiglu import (
                clamped_swiglu,
            )

            self._prefill_fusion = clamped_swiglu

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        if self._prefill_fusion is not None:
            from rtp_llm.models_py.modules.dsv4.forward_metadata import metadata_cache

            if metadata_cache() is not None:
                fused = self._prefill_fusion(gate_up, self.limit)
                if fused is not None:
                    return fused
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate.clamp(max=self.limit)) * up.clamp(-self.limit, self.limit)


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
        linear_factory=None,
        *,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        self.linear_factory = linear_factory or LinearFactory

        self.activation_type = activation_type
        self.parallelism_config = parallelism_config
        if self.activation_type not in _ACTIVATION_FUNC_MAP:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        self.is_gated = activation_type in _GATED_ACTIVATION_TYPE_LIST
        self.clamp_swiglu = self.is_gated and swiglu_limit > 0
        self.act_fn = (
            ClampedSiluAndMul(swiglu_limit)
            if self.clamp_swiglu
            else _ACTIVATION_FUNC_MAP[activation_type]()
        )

        if self.is_gated:
            if W.ffn_w13 not in weights:
                self.up_proj = self.linear_factory.create_merged_linear(
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
                self.up_proj = self.linear_factory.create_linear_from_weights(
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
            self.up_proj = self.linear_factory.create_linear_from_weights(
                weights,
                W.ffn_w3,
                W.ffn_s3,
                W.ffn_b3,
                quant_config=quant_config,
                hw_kernel_config=hw_kernel_config,
                weight_scale_2_key=W.ffn_w3_s2,
                input_scale_key=W.ffn_w3_i_s,
            )

        self.down_proj = self.linear_factory.create_linear_from_weights(
            weights,
            W.ffn_w2,
            W.ffn_s2,
            W.ffn_b2,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.ffn_w2_s2,
            input_scale_key=W.ffn_w2_i_s,
        )

    @property
    def accepts_fp8_input(self) -> bool:
        return self.is_gated and callable(
            getattr(self.up_proj, "forward_quantized", None)
        )

    def forward(
        self,
        x: torch.Tensor,
        skip_allreduce: bool = False,
        *,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = None
        if not self.is_gated and self.activation_type == ActivationType.Gelu:
            activated = self.up_proj.forward_with_bias_gelu(x)
        else:
            up = (
                self.up_proj.forward_quantized(x_fp8, x_scale)
                if x_fp8 is not None and x_scale is not None and self.accepts_fp8_input
                else self.up_proj(x)
            )
            if (
                self.is_gated
                and not self.clamp_swiglu
                and getattr(self.down_proj, "supports_fused_silu_quant", False)
            ):
                output = self.down_proj.forward_silu_and_mul_quantized(up)
            if output is None:
                activated = self.act_fn(up)

        ffn_tp_size = self.parallelism_config.get_ffn_tp_size()
        if output is None:
            output = self.down_proj(activated)
        if not skip_allreduce and ffn_tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output
