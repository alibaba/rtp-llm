from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.ep.model_moe_sparse_block import ModelMoESparseBlock
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear_factory import LinearFactory
from rtp_llm.models_py.modules.mlp import FusedSiluActDenseMLP
from rtp_llm.models_py.modules.norm import RMSNorm, RMSNormTorch
from rtp_llm.ops import KVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class DeepSeekV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.num_heads = self.config.head_num
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
        fmha_impl: FMHAImplBase,
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
            q = self.q_b_proj(fused_qkv)
        else:
            fused_qkv = self.fused_qkv_proj(hidden_states)
            kv_offset = self.config.head_num * self.config.size_per_head
            q, compressed_kv = torch.split(
                fused_qkv,
                [
                    kv_offset,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ],
                dim=-1,
            )

        q_view = q.reshape(-1, self.num_heads, self.q_head_dim)

        q_nope, q_pe = torch.split(
            q_view, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_layernorm(compressed_kv.contiguous())

        attn_output = fmha_impl.forward(
            q_nope, q_pe, compressed_kv, k_pe, kv_cache, self.layer_idx
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class DeepSeekV2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.self_attn = DeepSeekV2Attention(config, weights, layer_idx)

        if len(config.moe_layer_index) > 0 and layer_idx < config.moe_layer_index[0]:
            self.is_dense_layer = True
        else:
            self.is_dense_layer = False
            self.moe_mlp = ModelMoESparseBlock(config, weights)
        self.add_shared_expert = config.moe_style == 2

        if self.add_shared_expert:
            config.activation_type = "SiGLU"
            self.shared_mlp = FusedSiluActDenseMLP(config, weights)

        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_dense_layer:
            hidden_states = self.shared_mlp(hidden_states)
        else:
            experts_output, _router_logits = self.moe_mlp(hidden_states)
            if self.add_shared_expert:
                shared_mlp_output = self.shared_mlp(hidden_states)
                hidden_states = experts_output + shared_mlp_output
            else:
                hidden_states = experts_output
        hidden_states = residual + hidden_states

        return hidden_states


class DeepSeekV2Model(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.layer_num = config.layer_num
        self.vocab_size = config.vocab_size
        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))

        self.layers = nn.ModuleList(
            [
                DeepSeekV2DecoderLayer(
                    config,
                    weights.weights[idx],
                    idx,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        attention_inputs: PyAttentionInputs = inputs.attention_inputs

        fmha_impl = self.get_mla_impl(attention_inputs, use_torch=True)

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)

        return PyModelOutputs(hidden_states)


__all__ = [
    "DeepSeekV2Model",
]
