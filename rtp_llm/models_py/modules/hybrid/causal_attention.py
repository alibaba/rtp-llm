from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import ParallelismConfig, AttentionConfigs
from rtp_llm.ops.compute_ops import DeviceType, KVCache, get_device
from rtp_llm.utils.model_weight import W
from rtp_llm.ops import HWKernelConfig

# Import device-specific FusedQKRMSNorm
device_type = get_device().get_device_type()
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
        hw_kernel_config: Optional['HWKernelConfig'] = None,
    ):
        super().__init__()
        self.parallelism_config = parallelism_config
        self.head_num = attn_config.head_num
        self.num_key_value_groups = attn_config.head_num // attn_config.kv_head_num
        self.head_dim = attn_config.size_per_head
        self.q_size = attn_config.head_num * self.head_dim

        # Create linear layers using LinearFactory
        self.qkv_proj = LinearFactory.create_linear_from_weights(
            weights, W.attn_qkv_w, W.attn_qkv_s, W.attn_qkv_b, quant_config=quant_config, hw_kernel_config=hw_kernel_config
        )
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights, W.attn_o_w, W.attn_o_s, W.attn_o_b, quant_config=quant_config, hw_kernel_config=hw_kernel_config
        )
        # for qwen3
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
        gate: Optional[torch.Tensor] = None,  # for qwen3 next
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
        attn_output = fmha_impl.forward(qkv, kv_cache, need_rope_kv_cache)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output = self.o_proj(attn_output)
        if self.parallelism_config.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output
