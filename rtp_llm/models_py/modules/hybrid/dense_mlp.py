"""Unified dense MLP implementation supporting multiple activation types."""

from typing import Dict, Optional, Type

import torch
from torch import nn

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import ActivationType, HWKernelConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

# CUDA-only fused silu_and_mul + per-token-group fp8 quant. Activated when
# down_proj is CudaFp8GEMMLinear with UE8M0 scales — saves one launch
# (silu_and_mul) and one launch (per_token_group_quant_8bit) per forward.
_DEVICE_TYPE = get_device_type()
if _DEVICE_TYPE == DeviceType.Cuda:
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
        CudaFp8GEMMLinear,
    )
    from rtp_llm.models_py.triton_kernels.common.activation import (
        silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
    )
else:
    CudaFp8GEMMLinear = None  # type: ignore

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
                    weights, W.ffn_w13, W.ffn_s13, W.ffn_b13,
                    quant_config=quant_config, hw_kernel_config=hw_kernel_config,
                    weight_scale_2_key=W.ffn_w13_s2,
                    input_scale_key=W.ffn_w13_i_s,
                )

        else:
            self.up_proj = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w3, W.ffn_s3, W.ffn_b3,
                quant_config=quant_config, hw_kernel_config=hw_kernel_config,
                weight_scale_2_key=W.ffn_w3_s2,
                input_scale_key=W.ffn_w3_i_s,
            )

        self.down_proj = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w2, W.ffn_s2, W.ffn_b2,
            quant_config=quant_config, hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.ffn_w2_s2,
            input_scale_key=W.ffn_w2_i_s
        )

        from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled

        self._fuse_silu_quant = (
            fuse_kernels_enabled(hw_kernel_config)
            and self.is_gated
            and CudaFp8GEMMLinear is not None
            and isinstance(self.down_proj, CudaFp8GEMMLinear)
            and (self.down_proj.K % 128 == 0)
        )
        if self._fuse_silu_quant and self.down_proj.scale_ue8m0:
            self._fuse_silu_quant = self.down_proj.K % 512 == 0

    @property
    def accepts_fp8_input(self) -> bool:
        return CudaFp8GEMMLinear is not None and isinstance(
            self.up_proj, CudaFp8GEMMLinear
        )

    def forward(
        self,
        x: torch.Tensor,
        x_fp8: "Optional[torch.Tensor]" = None,
        x_scale: "Optional[torch.Tensor]" = None,
        skip_allreduce: bool = False,
    ):
        if x_fp8 is not None and x_scale is not None and self.accepts_fp8_input:
            up = self.up_proj(x_fp8, input_scales=x_scale)
        else:
            up = self.up_proj(x)
        if self._fuse_silu_quant and up.dim() == 2:
            scale_ue8m0 = self.down_proj.scale_ue8m0
            fp8_out, scale_out = (
                silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
                    up.contiguous(),
                    quant_group_size=128,
                    scale_ue8m0=scale_ue8m0,
                )
            )
            output = self.down_proj(fp8_out, input_scales=scale_out)
        else:
            activated = self.act_fn(up)
            output = self.down_proj(activated)
        if not skip_allreduce and self.parallelism_config.get_ffn_tp_size() > 1:
            output = all_reduce(output, group=Group.TP)
        return output
