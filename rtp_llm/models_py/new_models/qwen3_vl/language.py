"""Qwen3-VL language model for the new loader.

语言骨干与 Qwen3 稠密模型完全一致（旧 loader 里 ``QWen3_VL.get_weight_cls()``
返回的就是 ``QWenV3Weight``），因此直接复用 ``new_models/qwen3`` 的
``Qwen3ForCausalLM``，权重加载（``load_weights`` / ``WEIGHTS_MAPPER``）原样继承。

唯一的差异在 ``forward``：Qwen3-VL 需要
  1. 把视觉特征注入到对应占位 token 的 embedding 上（``MultimodalEmbeddingInjector``）；
  2. 在每个 decoder 层之后做 deepstack 注入（``MultimodalDeepstackInjector``）。
这段逻辑直接照抄 ``model_desc/qwen3vl.py`` 的 forward（两边都吃同一套
``PyModelInputs``，注入器是无参 stateless 模块，不影响权重加载契约）。
"""

from typing import Any

import torch

from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.modules import (
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
    reshape_extra_input_to_deepstack,
)
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs


class Qwen3VLForCausalLM(Qwen3ForCausalLM):

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__(model_config, load_config)
        # 无参注入器；放在 __init__ 里只是为了和 model_desc 对齐、便于复用。
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.multimodal_deepstack_injector = MultimodalDeepstackInjector()

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids

        mm = inputs.multimodal_inputs
        mm_features = mm.multimodal_features
        mm_feature_locs = mm.mm_features_locs
        mm_extra_input = mm.mm_extra_input
        # extra input 以扁平 1-D 张量到达；还原成 deepstack [layers, tokens, hidden]
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
            # 与 model_desc/qwen3vl.py 对齐:分组 KV 时每层需选对应的 block map，
            # 否则 group 边界之后的层会读到错误的 KV block。
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


__all__ = ["Qwen3VLForCausalLM"]
