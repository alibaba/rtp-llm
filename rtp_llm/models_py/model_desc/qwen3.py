import logging
from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import FusedSiluActDenseMLP, RMSNorm
from rtp_llm.models_py.modules.attention import CausalAttention
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.ops import KVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.self_attn = CausalAttention(config, weights)
        self.mlp = FusedSiluActDenseMLP(config, weights)
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
        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3Model(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)

        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, weights.weights[idx])
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        try:
            input_ids: torch.Tensor = inputs.input_ids
            inputs_embeds = self.embed_tokens(input_ids)
            hidden_states = inputs_embeds

            attention_inputs: PyAttentionInputs = inputs.attention_inputs
            fmha_impl = self.get_fmha_impl(attention_inputs)
            for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
                hidden_states = decoder_layer(
                    hidden_states,
                    fmha_impl,
                    kv_cache=(
                        self.kv_cache.get_layer_cache(i) if self.kv_cache else None
                    ),
                )
            hidden_states = self.norm(hidden_states)
            return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
        except Exception as e:
            error_msg = f"Qwen3Model forward failed: {type(e).__name__}: {e}"
            logging.error(error_msg)
            logging.info(error_msg)
            print(error_msg)
            import traceback

            logging.error(f"Traceback: {traceback.format_exc()}")
            logging.info(f"Traceback: {traceback.format_exc()}")
            print(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e


__all__ = [
    "Qwen3Model",
]
