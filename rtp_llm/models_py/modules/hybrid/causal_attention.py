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

device_type = get_device_type()
if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm.norm import FusedQKRMSNorm
else:
    from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm


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
        self.fp8_key_center = None
        qkv_bias = weights.get(W.attn_qkv_b)
        if (
            attn_config.fp8_kv_cache_mode == 2
            and self.qk_fuse_norm is None
            and qkv_bias is not None
        ):
            # One checkpoint-derived shift for every token, request and cache
            # reuse. Subtract after RoPE: subtracting the projection bias before
            # rotation would change relative-position attention scores.
            kv_size = attn_config.kv_head_num * self.head_dim
            self.fp8_key_center = (
                qkv_bias.reshape(-1)[self.q_size : self.q_size + kv_size]
                .reshape(attn_config.kv_head_num, self.head_dim)
                .float()
                .contiguous()
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        gate: Optional[torch.Tensor] = None,
        qkv: Optional[torch.Tensor] = None,
        skip_allreduce: bool = False,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        if qkv is None:
            qkv = self.qkv_proj(hidden_states)
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
        if self.fp8_key_center is None:
            attn_output = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        else:
            attn_output = fmha_impl.forward(
                qkv, kv_cache, self.layer_idx, key_center=self.fp8_key_center
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output = self.o_proj(attn_output)
        if not skip_allreduce and self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output
