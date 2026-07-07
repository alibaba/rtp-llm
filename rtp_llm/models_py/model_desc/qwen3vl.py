from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    get_fmha_params,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3DecoderLayer
from rtp_llm.models_py.modules import (
    AttnImplFactory,
    Embedding,
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
    RMSNorm,
    reshape_extra_input_to_deepstack,
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
                    config,
                    parallelism_config,
                    idx,
                    weights.weights[idx],
                    quant_config,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids

        position_ids = inputs.combo_position_ids
        token_type_ids = inputs.embedding_inputs.combo_tokens_type_ids
        text_tokens_mask = inputs.embedding_inputs.text_tokens_mask
        mm_features = inputs.multimodal_inputs.multimodal_features
        mm_feature_locs = inputs.multimodal_inputs.mm_features_locs
        mm_extra_input = inputs.multimodal_inputs.mm_extra_input
        # extra input arrives as flat 1-D tensors; reshape back to deepstack [layers, tokens, hidden]
        mm_deepstack_embeds = (
            reshape_extra_input_to_deepstack(mm_extra_input, mm_features)
            if mm_extra_input
            else []
        )

        inputs_embeds = self.embed_tokens(
            input_ids, position_ids, token_type_ids, text_tokens_mask
        )
        hidden_states = self.multimodal_embedding_injector(
            inputs_embeds, mm_features, mm_feature_locs
        )

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        if mm_deepstack_embeds and mm_feature_locs is not None:
            cpu_locs = (
                mm_feature_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
            )
        else:
            cpu_locs = []

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            layer_fmha_impl = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            hidden_states = decoder_layer(
                hidden_states,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            hidden_states = self.multimodal_deepstack_injector(
                hidden_states, mm_deepstack_embeds, cpu_locs, i
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, get_fmha_params(fmha_impl))


__all__ = [
    "Qwen3VLModel",
]
