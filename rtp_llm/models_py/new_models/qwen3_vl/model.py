from collections.abc import Callable, Iterator
from typing import Any

import torch

from rtp_llm.models_py.modules import (
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
    reshape_extra_input_to_deepstack,
)
from rtp_llm.models_py.new_models.model_base import select_block_map_for_layer
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs


class Qwen3VLForCausalLM(Qwen3ForCausalLM):
    """Qwen3-VL text runtime with multimodal feature injection."""

    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={"model.language_model.": ""},
    )

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__(model_config, load_config)
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.multimodal_deepstack_injector = MultimodalDeepstackInjector()

    def checkpoint_weight_name_filter(self) -> Callable[[str], bool]:
        return lambda name: name.startswith("model.language_model.") or name.startswith(
            "lm_head."
        )

    def load_weights(
        self, weights: Iterator[tuple[str, torch.Tensor]] | dict[str, torch.Tensor]
    ) -> None:
        iterator = weights.items() if isinstance(weights, dict) else weights
        super().load_weights(self.WEIGHTS_MAPPER.apply(iterator))

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        text_tokens_mask = inputs.embedding_inputs.text_tokens_mask
        if text_tokens_mask is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            text_mask = text_tokens_mask.to(device=input_ids.device, dtype=torch.bool)
            safe_input_ids = torch.where(text_mask, input_ids, 0)
            hidden_states = self.embed_tokens(safe_input_ids)
            hidden_states = hidden_states * text_mask.unsqueeze(-1)

        multimodal_inputs = inputs.multimodal_inputs
        mm_features = multimodal_inputs.multimodal_features
        mm_feature_locs = multimodal_inputs.mm_features_locs
        mm_extra_input = multimodal_inputs.mm_extra_input
        mm_deepstack_embeds = (
            reshape_extra_input_to_deepstack(mm_extra_input, mm_features)
            if mm_extra_input
            else []
        )
        hidden_states = self.multimodal_embedding_injector(
            hidden_states, mm_features, mm_feature_locs
        )

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        cpu_locs = (
            mm_feature_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
            if mm_deepstack_embeds and mm_feature_locs is not None
            else []
        )
        for layer_id, layer in enumerate(self.layers):
            select_block_map_for_layer(inputs.attention_inputs, layer_id)
            hidden_states = layer(
                hidden_states,
                fmha_impl,
                kv_cache=(
                    self.kv_cache.get_layer_cache(layer_id) if self.kv_cache else None
                ),
            )
            hidden_states = self.multimodal_deepstack_injector(
                hidden_states,
                mm_deepstack_embeds,
                cpu_locs,
                layer_id,
            )

        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = ["Qwen3VLForCausalLM"]
