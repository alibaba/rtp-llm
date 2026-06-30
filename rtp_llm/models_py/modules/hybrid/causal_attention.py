import os
from typing import Dict, Optional, Tuple

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
        self.kv_size = attn_config.kv_head_num * self.head_dim

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

    def _should_use_hipb_qkv(self, hidden_states: torch.Tensor) -> bool:
        if os.environ.get("USE_FP8_QKV_HIPB", "1") == "0":
            return False
        expected_qkv_size = self.q_size + 2 * self.kv_size
        return (
            self.tp_size == 1
            and hidden_states.dim() == 2
            and hidden_states.shape[-1] == 768
            and hidden_states.shape[0] >= 8192
            and expected_qkv_size == 3 * hidden_states.shape[-1]
            and getattr(self.qkv_proj, "hidden_size", None) == hidden_states.shape[-1]
            and getattr(self.qkv_proj, "output_size", None) == expected_qkv_size
            and getattr(self.qkv_proj, "bias", None) is not None
            and hasattr(self.qkv_proj, "forward_hipb_bias")
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        gate: Optional[torch.Tensor] = None,
        defer_output_bias: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        if self._should_use_hipb_qkv(hidden_states):
            qkv = self.qkv_proj.forward_hipb_bias(hidden_states)
        else:
            qkv = self.qkv_proj(hidden_states)
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
        attn_output = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)

        output_bias = None
        can_defer_bias = (
            defer_output_bias
            and self.tp_size == 1
            and hasattr(self.o_proj, "forward_without_bias")
        )
        if (
            can_defer_bias
            and hasattr(self.o_proj, "forward_hipb_bias")
            and getattr(self.o_proj, "bias", None) is not None
        ):
            output = self.o_proj.forward_hipb_bias(attn_output)
        elif can_defer_bias:
            output = self.o_proj.forward_without_bias(attn_output)
            output_bias = getattr(self.o_proj, "bias", None)
        else:
            output = self.o_proj(attn_output)

        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output, output_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output, _ = self._forward_impl(
            hidden_states, fmha_impl, kv_cache, gate, defer_output_bias=False
        )
        return output

    def forward_defer_output_bias(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Same as forward() but returns o_proj's bias separately so the caller
        can fuse it into a following residual-add+LayerNorm."""
        return self._forward_impl(
            hidden_states, fmha_impl, kv_cache, gate, defer_output_bias=True
        )
