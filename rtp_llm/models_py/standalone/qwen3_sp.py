from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3DecoderLayer
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.linear_factory import LinearFactory
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class Qwen2MtpModel(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)

        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))
        self.eh_proj = LinearFactory.create_linear_from_weights(
            weights.weights[0], W.multi_tokens_predict_eh_proj
        )
        self.e_norm = RMSNorm(
            weights.weights[0][W.multi_tokens_predict_enorm], eps=config.layernorm_eps
        )
        self.h_norm = RMSNorm(
            weights.weights[0][W.multi_tokens_predict_hnorm], eps=config.layernorm_eps
        )

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, weights.weights[idx])
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.weights[0][W.multi_tokens_predict_final_ln_gamma],
            eps=config.layernorm_eps,
        )

        self.lm_head = LinearFactory.create_linear_from_weights(
            weights.weights[0],
            W.lm_head,
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        embedding_hidden_states = inputs_embeds
        last_hidden_states = inputs.input_hiddens

        e_norm = self.e_norm(embedding_hidden_states)
        h_norm = self.h_norm(last_hidden_states)
        cat_hidden_states = torch.cat([h_norm, e_norm], -1)
        hidden_states = self.eh_proj(cat_hidden_states)

        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        fmha_impl = self.get_fmha_impl(attention_inputs)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        hidden_states = self.lm_head(hidden_states)
        hidden_states = F.softmax(hidden_states, dim=-1)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "Qwen2MtpModel",
]
