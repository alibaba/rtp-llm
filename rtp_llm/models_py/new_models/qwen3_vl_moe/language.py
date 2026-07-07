"""Qwen3-VL-MoE language model for the new loader.

语言骨干与 Qwen3-MoE 完全一致(旧 loader 里 ``QWen3_VL_MOE.get_weight_cls()`` 是
``QWen3VLMoeWeightInfo``,继承 ``QWenV3MoeWeight``),因此复用 ``new_models/qwen3_moe`` 的
``Qwen3MoeForCausalLM``;只在 forward 上加多模态 embedding 注入 + 逐层 deepstack 注入
(与 ``model_desc/qwen3vl_moe.py`` 对齐)。结构与 ``new_models/qwen3_vl`` 完全平行,
区别只是基类从稠密换成 MoE。
"""

from typing import Any

import torch

from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.modules import (
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
    reshape_extra_input_to_deepstack,
)
from rtp_llm.models_py.new_models.qwen3_moe.language import Qwen3MoeForCausalLM
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs


class Qwen3VLMoeForCausalLM(Qwen3MoeForCausalLM):

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__(model_config, load_config)
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.multimodal_deepstack_injector = MultimodalDeepstackInjector()

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids

        mm = inputs.multimodal_inputs
        mm_features = mm.multimodal_features
        mm_feature_locs = mm.mm_features_locs
        mm_extra_input = mm.mm_extra_input
        mm_deepstack_embeds = (
            reshape_extra_input_to_deepstack(mm_extra_input, mm_features)
            if mm_extra_input
            else []
        )

        inputs_embeds = self.embed_tokens(input_ids)
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

        for i, layer in enumerate(self.layers):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states = layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            hidden_states = self.multimodal_deepstack_injector(
                hidden_states, mm_deepstack_embeds, cpu_locs, i
            )

        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = ["Qwen3VLMoeForCausalLM"]
