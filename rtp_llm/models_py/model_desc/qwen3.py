from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    AttnImplFactory,
    CausalAttention,
    Embedding,
    FMHAImplBase,
    FusedSiluActDenseMLP,
    RMSNorm,
)
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.utils.model_weight import W


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self, config: ModelConfig, parallelism_config: ParallelismConfig, weights: Dict[str, torch.Tensor], quant_config: Optional[object] = None
    ):
        super().__init__()
        self.self_attn = CausalAttention(config, parallelism_config, weights, quant_config)
        self.mlp = FusedSiluActDenseMLP(config.activation_type, parallelism_config, weights, quant_config)
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
    def __init__(
        self, 
        config: ModelConfig, 
        parallelism_config: ParallelismConfig,
        weights: ModelWeights, 
         max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config, 
            parallelism_config, 
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config, 
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        self.embed_tokens = Embedding(config, parallelism_config, weights.get_global_weight(W.embedding))
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, parallelism_config, weights.weights[idx], quant_config)
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
        fmha_impl = AttnImplFactory.get_fmha_impl(
            self.config, self.parallelism_config, self.weight, attention_inputs, self.fmha_config
        )
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "Qwen3Model",
]
