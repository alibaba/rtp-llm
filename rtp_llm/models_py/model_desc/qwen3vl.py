from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3DecoderLayer
from rtp_llm.models_py.modules import (
    AttnImplFactory,
    Embedding,
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
    RMSNorm,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class Qwen3VLModel(GptModelBase):

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

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.multimodal_deepstack_injector = MultimodalDeepstackInjector()
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config, parallelism_config, weights.weights[idx], quant_config
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def need_combo_position_ids(self) -> bool:
        return True

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        if fmha_impl is None:
            fmha_impl = AttnImplFactory.get_fmha_impl(inputs)

        position_ids = attention_inputs.combo_position_ids
        token_type_ids = attention_inputs.combo_tokens_type_ids
        text_tokens_mask = attention_inputs.text_tokens_mask
        mm_features = attention_inputs.multimodal_features
        mm_feature_locs = attention_inputs.mm_features_locs
        mm_deepstack_embeds = attention_inputs.mm_deepstack_embeds

        inputs_embeds = self.embed_tokens(
            input_ids, position_ids, token_type_ids, text_tokens_mask
        )
        hidden_states = self.multimodal_embedding_injector(
            inputs_embeds, mm_features, mm_feature_locs
        )

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            hidden_states = self.multimodal_deepstack_injector(
                hidden_states, mm_deepstack_embeds, mm_feature_locs, i
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "Qwen3VLModel",
]
