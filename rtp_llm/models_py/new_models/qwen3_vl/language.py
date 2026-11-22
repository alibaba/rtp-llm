"""Qwen3-VL text owners for main's multimodal serving interface."""

import torch

from rtp_llm.models_py.new_models.model_base import select_fmha_impl_for_layer
from rtp_llm.models_py.new_models.qwen2.language import Qwen2ForCausalLM
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.new_models.qwen3_moe.language import Qwen3MoeForCausalLM
from rtp_llm.models_py.new_models.qwen3_vl.multimodal import Qwen3VLMultimodalMixin
from rtp_llm.models_py.weight_mapper import WeightsMapper


class _LanguageRuntime(Qwen3VLMultimodalMixin):
    _is_moe = False

    def __init__(self, model_config, load_config):
        super().__init__(model_config, load_config)
        self._init_multimodal_injectors()

    def forward(self, inputs, fmha_impl=None):
        hidden, deepstack, locations = self._embed_multimodal_inputs(inputs)
        residual = torch.zeros_like(hidden) if self._is_moe else None
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        capture = self.capture_context(inputs.capture_hidden_states)
        layers = self.layers.values() if self._pipeline_partitioned else self.layers
        for local, (layer_id, layer) in enumerate(zip(self._owned_layer_ids, layers)):
            attention = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, local)
            cache = self.kv_cache.get_layer_cache(local) if self.kv_cache else None
            if self._is_moe:
                hidden, residual = layer(hidden, residual, attention, cache)
            else:
                hidden = layer(hidden, attention, kv_cache=cache)
            hidden = self._inject_deepstack_after_layer(
                hidden, deepstack, locations, layer_id
            )
            if self._is_moe:
                capture.capture_layer(layer_id, hidden, residual)
            else:
                capture.capture_layer(layer_id, hidden)
        if self.norm is None:
            intermediate = hidden + residual if self._is_moe else hidden
            from rtp_llm.ops.compute_ops import PyModelOutputs

            out = PyModelOutputs(intermediate)
            out.pp_intermediates = {"hidden_states": intermediate}
            return out
        return (
            capture.finalize(hidden, residual)
            if self._is_moe
            else capture.finalize(hidden)
        )


class Qwen2VLForCausalLM(_LanguageRuntime, Qwen2ForCausalLM):
    """Biased Qwen2 decoder; its independent vision owner consumes visual.*."""

    def checkpoint_weight_name_filter(self):
        owns_stage = super().checkpoint_weight_name_filter()
        return lambda key: not key.startswith("visual.") and (
            owns_stage is None or owns_stage(key)
        )


class _Qwen3VLRuntime(_LanguageRuntime):
    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.language_model.": ""})

    @staticmethod
    def online_checkpoint_name(name):
        text = name.removeprefix("model.")
        if text.startswith(("layers.", "embed_tokens.", "norm.")):
            return "model.language_model." + text
        return name

    def checkpoint_weight_name_filter(self):
        owns_stage = super().checkpoint_weight_name_filter()
        return lambda key: (
            key.startswith(("model.language_model.", "lm_head."))
            or self.WEIGHTS_MAPPER.map_name(key) in self.handler_heads.sources
        ) and (owns_stage is None or owns_stage(key))


class Qwen3VLForCausalLM(_Qwen3VLRuntime, Qwen3ForCausalLM):
    pass


class Qwen3VLMoeForCausalLM(_Qwen3VLRuntime, Qwen3MoeForCausalLM):
    _is_moe = True
