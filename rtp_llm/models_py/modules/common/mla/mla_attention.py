from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group, all_reduce
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.norm import RMSNorm
from rtp_llm.ops.compute_ops import KVCache
from rtp_llm.utils.model_weight import W


class MlaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.num_heads = self.config.head_num // self.config.tp_size
        self.qk_nope_head_dim = self.config.nope_head_dim
        self.qk_rope_head_dim = self.config.rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_lora_rank = self.config.kv_lora_rank
        self.v_head_dim = self.config.v_head_dim
        self.q_lora_rank = self.config.q_lora_rank
        self.softmax_scale = self.q_head_dim ** (-0.5)
        self.layer_idx = layer_idx
        self.token_per_block = self.config.seq_size_per_block

        if self.q_lora_rank > 0:
            self.fused_qkv_a_proj = LinearFactory.create_linear_from_weights(
                weights, W.mla_fusedqkrope_w, W.mla_fusedqkrope_s, None, config
            )
            self.q_a_layernorm = RMSNorm(
                weights.get(W.mla_q_a_ln_gamma, None), eps=config.layernorm_eps
            )
            self.q_b_proj = LinearFactory.create_linear_from_weights(
                weights, W.mla_q_b_w, W.mla_q_b_s, None, config
            )
        else:
            self.fused_qkv_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.mla_fusedqkrope_no_lora_w,
                W.mla_fusedqkrope_no_lora_s,
                None,
                config,
            )

        self.kv_a_layernorm = RMSNorm(
            weights.get(W.mla_kv_a_ln_gamma, None), eps=config.layernorm_eps
        )

        self.o_proj = LinearFactory.create_linear_from_weights(
            weights, W.attn_o_w, W.attn_o_s, W.attn_o_b, config
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        if self.q_lora_rank > 0:
            fused_qkv = self.fused_qkv_a_proj(hidden_states)
            kv_offset = self.config.q_lora_rank
            q, compressed_kv = torch.split(
                fused_qkv,
                [
                    kv_offset,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ],
                dim=-1,
            )
            q = self.q_a_layernorm(q.contiguous())
            q = self.q_b_proj(q)
        else:
            fused_qkv = self.fused_qkv_proj(hidden_states)
            kv_offset = self.num_heads * self.config.size_per_head
            q, compressed_kv = torch.split(
                fused_qkv,
                [
                    kv_offset,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ],
                dim=-1,
            )
        q_view = q.reshape(-1, self.num_heads, self.q_head_dim)

        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_layernorm(compressed_kv.contiguous())

        attn_output = fmha_impl.forward(
            q_view, compressed_kv, k_pe, kv_cache, self.layer_idx
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if self.config.tp_size > 1:
            attn_output = all_reduce(attn_output, group=Group.TP)
        return attn_output
