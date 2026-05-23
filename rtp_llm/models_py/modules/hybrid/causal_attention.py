from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W

# Import device-specific FusedQKRMSNorm + SigmoidMulInplace
device_type = get_device_type()
if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm.attn_output_gate import SigmoidMulInplace
    from rtp_llm.models_py.modules.base.rocm.norm import FusedQKRMSNorm

    CudaFp8GEMMLinear = None
    sigmoid_mul_fp8_quant_fwd = None
else:
    from rtp_llm.models_py.modules.base.cuda.attn_output_gate import SigmoidMulInplace
    from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm

    try:
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
            CudaFp8GEMMLinear,
        )
        from rtp_llm.models_py.triton_kernels.common.attn_output_gate import (
            sigmoid_mul_fp8_quant_fwd,
        )
    except ImportError:
        CudaFp8GEMMLinear = None
        sigmoid_mul_fp8_quant_fwd = None


class CausalAttention(nn.Module):

    def __init__(
        self,
        attn_config: AttentionConfigs,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.parallelism_config = parallelism_config
        self.tp_size = parallelism_config.get_attn_tp_size()
        self.head_num = attn_config.head_num
        self.num_key_value_groups = attn_config.head_num // attn_config.kv_head_num
        self.head_dim = attn_config.size_per_head
        self.q_size = attn_config.head_num * self.head_dim

        # Create linear layers using LinearFactory
        self.qkv_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_qkv_w,
            W.attn_qkv_s,
            W.attn_qkv_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.attn_qkv_s2,
            input_scale_key=W.attn_qkv_i_s,
        )
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_o_w,
            W.attn_o_s,
            W.attn_o_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.attn_o_s2,
            input_scale_key=W.attn_o_i_s,
        )
        self.cache_scale_len = 1024
        self.o_proj.maybe_cache_quant_scale(self.cache_scale_len)
        self.qk_fuse_norm = None
        if W.q_ln_gamma in weights and W.k_ln_gamma in weights:
            self.qk_fuse_norm = FusedQKRMSNorm(
                weights[W.q_ln_gamma],
                weights[W.k_ln_gamma],
                attn_config.head_num,
                attn_config.kv_head_num,
                attn_config.size_per_head,
                layernorm_eps,
            )
        # Fused sigmoid+mul for Qwen3.5 attn_output_gate (replaces the
        # `attn_output * torch.sigmoid(gate)` 2-kernel sequence below).
        self.sigmoid_mul = SigmoidMulInplace()
        from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled

        # Resolve once at init: HWKernelConfig.enable_fuse_kernels (or env
        # ``ENABLE_FUSE_KERNELS``). Cached so the forward path does no
        # config / env lookup per token.
        self._fuse_kernels_enabled = fuse_kernels_enabled(hw_kernel_config)
        self._fuse_sigmoid_mul_quant = (
            self._fuse_kernels_enabled
            and CudaFp8GEMMLinear is not None
            and sigmoid_mul_fp8_quant_fwd is not None
            and isinstance(self.o_proj, CudaFp8GEMMLinear)
            and self.o_proj.K % 128 == 0
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        gate: Optional[torch.Tensor] = None,  # for qwen3 next
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        if x_fp8 is not None and x_scale is not None:
            qkv = self.qkv_proj(x_fp8, input_scales=x_scale)
        else:
            qkv = self.qkv_proj(hidden_states)
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
        attn_output = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if gate is not None:
            if self._fuse_sigmoid_mul_quant and attn_output.dim() == 2:
                fp8_out, scale = sigmoid_mul_fp8_quant_fwd(
                    attn_output,
                    gate,
                    quant_group_size=128,
                    scale_ue8m0=self.o_proj.scale_ue8m0,
                )
                output = self.o_proj(fp8_out, input_scales=scale)
            elif self._fuse_kernels_enabled:
                attn_output = self.sigmoid_mul(attn_output, gate)
                output = self.o_proj(attn_output)
            else:
                # Master switch off → use the original PyTorch baseline,
                # NOT the Triton ``sigmoid_mul_inplace_triton`` kernel.
                attn_output = attn_output * torch.sigmoid(gate)
                output = self.o_proj(attn_output)
        else:
            output = self.o_proj(attn_output)
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output
