from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.base.rocm.norm import (
    HAS_FUSED_NORM_QUANT,
    RMSNormFusedQuant,
    RMSResNormFusedQuant,
)
from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
    RocmFp8PTPCLinear,
)
from rtp_llm.ops.compute_ops import LayerKVCache


_printed_fused_norm_quant_created = False
_printed_fused_norm_quant_forward = False


class _FusedNormQuant(nn.Module):
    def __init__(
        self,
        self_attn: nn.Module,
        mlp: nn.Module,
        input_layernorm_weight: torch.Tensor,
        post_attention_layernorm_weight: torch.Tensor,
        layernorm_eps: float,
    ):
        super().__init__()
        object.__setattr__(self, "self_attn", self_attn)
        object.__setattr__(self, "mlp", mlp)
        self.input_layernorm_fused = RMSNormFusedQuant(
            input_layernorm_weight, eps=layernorm_eps
        )
        self.post_attention_layernorm_fused = RMSResNormFusedQuant(
            post_attention_layernorm_weight, eps=layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        global _printed_fused_norm_quant_forward
        if not _printed_fused_norm_quant_forward:
            print(
                f"[FusedNormQuant] fused forward enabled, hidden_states_shape={tuple(hidden_states.shape)}, "
                f"dtype={hidden_states.dtype}, device={hidden_states.device}",
                flush=True,
            )
            _printed_fused_norm_quant_forward = True

        residual = hidden_states.clone()
        hidden_states_fp8, hidden_states_scale = self.input_layernorm_fused(
            hidden_states
        )

        attn_module = self.self_attn
        input_shape = hidden_states.shape[:-1]
        qkv = attn_module.qkv_proj.forward_prequantized(
            hidden_states_fp8, hidden_states_scale
        )
        if attn_module.qk_fuse_norm is not None:
            qkv = attn_module.qk_fuse_norm(qkv)
        attn_output = fmha_impl.forward(qkv, kv_cache, attn_module.layer_idx)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_module.o_proj(attn_output)
        if attn_module.tp_size > 1:
            attn_output = all_reduce(attn_output, group=Group.TP)

        hidden_states_fp8, hidden_states_scale, residual = (
            self.post_attention_layernorm_fused(attn_output, residual)
        )

        mlp_module = self.mlp
        mlp_output = mlp_module.up_proj.forward_prequantized(
            hidden_states_fp8, hidden_states_scale
        )
        mlp_output = mlp_module.act_fn(mlp_output)
        mlp_output = mlp_module.down_proj(mlp_output)
        if mlp_module.parallelism_config.get_ffn_tp_size() > 1:
            mlp_output = all_reduce(mlp_output, group=Group.TP)

        return residual + mlp_output


def FusedNormQuant(
    self_attn: nn.Module,
    mlp: nn.Module,
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    layernorm_eps: float,
) -> Optional[_FusedNormQuant]:
    if not HAS_FUSED_NORM_QUANT:
        return None

    projections = (
        self_attn.qkv_proj,
        self_attn.o_proj,
        mlp.up_proj,
        mlp.down_proj,
    )
    if not all(isinstance(projection, RocmFp8PTPCLinear) for projection in projections):
        return None

    global _printed_fused_norm_quant_created
    if not _printed_fused_norm_quant_created:
        print(
            "[FusedNormQuant] created fused ROCm norm-quant path for Qwen3 decoder layer",
            flush=True,
        )
        _printed_fused_norm_quant_created = True

    return _FusedNormQuant(
        self_attn,
        mlp,
        input_layernorm_weight,
        post_attention_layernorm_weight,
        layernorm_eps,
    )
