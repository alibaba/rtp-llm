from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.attention import CausalAttention
from rtp_llm.models_py.modules.embedding import EmbeddingBert
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.mlp import BertGeluActDenseMLP
from rtp_llm.models_py.modules.norm import LayerNorm, AddBiasResLayerNorm
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class BertDecoderLayer(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.self_attn = CausalAttention(config, weights)
        self.mlp = BertGeluActDenseMLP(config, weights)
        self.input_layernorm = AddBiasResLayerNorm(
            weights[W.post_ln_gamma],
            beta=weights[W.post_ln_beta],
            eps=config.layernorm_eps,
        )
        self.post_attention_layernorm = AddBiasResLayerNorm(
            weights[W.post_ffn_ln_gamma],
            beta=weights[W.post_ffn_ln_beta],
            eps=config.layernorm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
            need_rope_kv_cache=False,
        )
        hidden_states = self.input_layernorm(hidden_states, residual, torch.empty(0))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm(
            hidden_states, residual, torch.empty(0)
        )
        return hidden_states


class BertModel(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.embed_tokens = EmbeddingBert(
            config, weights.get_global_weight(W.embedding)
        )
        self.pre_decoder_layernorm = LayerNorm(
            weight=weights.get_global_weight(W.pre_decoder_ln_gamma),
            beta=weights.get_global_weight(W.pre_decoder_ln_beta),
            eps=config.layernorm_eps,
        )
        self.layers = nn.ModuleList(
            [
                BertDecoderLayer(config, weights.weights[idx])
                for idx in range(self.layer_num)
            ]
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        bert_embedding_inputs = inputs.bert_embedding_inputs
        inputs_embeds = self.embed_tokens(
            input_ids,
            bert_embedding_inputs.combo_position_ids,
            bert_embedding_inputs.position_encoding,
            bert_embedding_inputs.combo_tokens_type_ids,
            bert_embedding_inputs.token_type_embedding,
            bert_embedding_inputs.input_embedding_scalar,
        )
        hidden_states = self.pre_decoder_layernorm(inputs_embeds)
        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        fmha_impl = self.get_fmha_impl(attention_inputs)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
