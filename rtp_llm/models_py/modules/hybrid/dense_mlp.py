"""Unified dense MLP implementation supporting multiple activation types."""

from typing import Any, Dict, NamedTuple, Optional, Tuple, Type

import torch
from torch import nn

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import ActivationType, HWKernelConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

# CUDA-only fused activation + per-token-group fp8 quant. Activated when
# down_proj accepts externally quantized fp8 input (FP8 per-block or MXFP8).
_DEVICE_TYPE = get_device_type()
if _DEVICE_TYPE == DeviceType.Cuda:
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
        CudaFp8GEMMLinear,
    )
    try:
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.mxfp8_linear import (
            CudaMxfp8Linear,
        )
    except ImportError:
        CudaMxfp8Linear = None  # type: ignore
    from rtp_llm.models_py.triton_kernels.common.activation import (
        silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
    )
    from rtp_llm.models_py.triton_kernels.common.swiglu_oai import (
        swiglu_oai_torch,
    )
else:
    CudaFp8GEMMLinear = None  # type: ignore
    CudaMxfp8Linear = None  # type: ignore
    swiglu_oai_torch = None  # type: ignore

_ACTIVATION_FUNC_MAP: Dict[ActivationType, Type[nn.Module]] = {
    ActivationType.Swiglu: FusedSiluAndMul,
    ActivationType.Gelu: nn.GELU,
}

_GATED_ACTIVATION_TYPE_LIST = [ActivationType.Swiglu]


class _FusedFp8QuantParams(NamedTuple):
    group_size: int
    scale_ue8m0: bool
    round_to_pow2: bool


def _get_fused_fp8_quant_params(linear: Any) -> Optional[_FusedFp8QuantParams]:
    if CudaFp8GEMMLinear is not None and isinstance(linear, CudaFp8GEMMLinear):
        return _FusedFp8QuantParams(
            group_size=getattr(linear, "input_quant_group_size", 128),
            scale_ue8m0=getattr(
                linear, "input_quant_scale_ue8m0", linear.scale_ue8m0
            ),
            round_to_pow2=getattr(linear, "input_quant_round_to_pow2", False),
        )
    if CudaMxfp8Linear is not None and isinstance(linear, CudaMxfp8Linear):
        return _FusedFp8QuantParams(
            group_size=getattr(linear, "input_quant_group_size", 32),
            scale_ue8m0=getattr(linear, "input_quant_scale_ue8m0", False),
            round_to_pow2=getattr(linear, "input_quant_round_to_pow2", True),
        )
    return None


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
        swiglu_oai_params: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()

        self.activation_type = activation_type
        self.parallelism_config = parallelism_config
        if self.activation_type not in _ACTIVATION_FUNC_MAP:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        # SwiGLU-OAI override: when alpha/limit are provided we use the GPT-OSS /
        # MiniMax-M3 variant ``gate * sigmoid(gate*alpha) * (up + 1)`` with both
        # ends clamped at ±limit (one-sided on gate). When None, fall back to
        # plain SiLU-and-mul as before. Caller (decoder layer) reads
        # ``config.swiglu_alpha`` / ``config.swiglu_limit`` and passes them in.
        self.swiglu_oai_params = swiglu_oai_params
        if self.swiglu_oai_params is not None:
            assert activation_type == ActivationType.Swiglu, (
                "swiglu_oai_params requires Swiglu activation_type"
            )
            self.act_fn = lambda x: swiglu_oai_torch(
                x, self.swiglu_oai_params[0], self.swiglu_oai_params[1],
                gate_first=True,
            )
        else:
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

        from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled

        self._down_proj_fp8_quant_params = _get_fused_fp8_quant_params(
            self.down_proj
        )
        self._fuse_silu_quant = (
            fuse_kernels_enabled(hw_kernel_config)
            and self.is_gated
            and self._down_proj_fp8_quant_params is not None
            and (self.down_proj.K % self._down_proj_fp8_quant_params.group_size == 0)
        )
        if (
            self._fuse_silu_quant
            and self._down_proj_fp8_quant_params.scale_ue8m0
        ):
            self._fuse_silu_quant = (
                self.down_proj.K
                % (self._down_proj_fp8_quant_params.group_size * 4)
                == 0
            )

    @property
    def fp8_input_quant_params(self) -> Optional[_FusedFp8QuantParams]:
        return _get_fused_fp8_quant_params(self.up_proj)

    @property
    def accepts_fp8_input(self) -> bool:
        return self.fp8_input_quant_params is not None

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
            _params = self._down_proj_fp8_quant_params
            assert _params is not None
            _alpha, _limit = self.swiglu_oai_params or (0.0, 0.0)
            fp8_out, scale_out = silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
                up.contiguous(),
                quant_group_size=_params.group_size,
                scale_ue8m0=_params.scale_ue8m0,
                round_to_pow2=_params.round_to_pow2,
                gemm1_alpha=_alpha,
                gemm1_clamp_limit=_limit,
            )
            output = self.down_proj(fp8_out, input_scales=scale_out)
        else:
            activated = self.act_fn(up)
            output = self.down_proj(activated)
        if not skip_allreduce and self.parallelism_config.get_ffn_tp_size() > 1:
            output = all_reduce(output, group=Group.TP)
        return output
