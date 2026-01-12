from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import DeviceType, KVCache, get_device
from rtp_llm.utils.model_weight import W

# Import device-specific FusedQKRMSNorm
device_type = get_device().get_device_type()
if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm.norm import FusedQKRMSNorm
else:
    from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm


class CausalAttention(nn.Module):

    def __init__(
        self, config: ModelConfig, parallelism_config: ParallelismConfig, weights: Dict[str, torch.Tensor], quant_config: Optional[object] = None
    ):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.head_dim = config.hidden_size // config.attn_config.head_num
        self.head_num = config.attn_config.head_num
        self.num_key_value_groups = config.attn_config.head_num // config.attn_config.kv_head_num
        self.q_size = config.attn_config.head_num * self.head_dim

        # Create linear layers using LinearFactory
        # Get quant_config from parameter or config
        if quant_config is None:
            quant_config = config.quant_config
        self.qkv_proj = LinearFactory.create_linear_from_weights(
            weights, W.attn_qkv_w, W.attn_qkv_s, W.attn_qkv_b, quant_config=quant_config, weight_scale_2_key=W.attn_qkv_s2, input_scale_key=W.attn_qkv_i_s)
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights, W.attn_o_w, W.attn_o_s, W.attn_o_b, quant_config=quant_config,
            weight_scale_2_key=W.attn_o_s2, input_scale_key=W.attn_o_i_s)
        self.qk_fuse_norm = None
        if W.q_ln_gamma in weights and W.k_ln_gamma in weights:
            self.qk_fuse_norm = FusedQKRMSNorm(
                weights[W.q_ln_gamma],
                weights[W.k_ln_gamma],
                config.attn_config.head_num // parallelism_config.tp_size,
                config.attn_config.kv_head_num // parallelism_config.tp_size,
                config.attn_config.size_per_head,
                config.layernorm_eps,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
        attn_output = fmha_impl.forward(qkv, kv_cache, need_rope_kv_cache)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        if self.parallelism_config.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output
