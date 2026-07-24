from collections.abc import Callable, Iterator
from typing import Any

import torch

from rtp_llm.models_py.new_models.model_base import select_block_map_for_layer
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.new_models.qwen3_vl.multimodal import Qwen3VLMultimodalMixin
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs


class Qwen3VLForCausalLM(Qwen3VLMultimodalMixin, Qwen3ForCausalLM):
    """Qwen3-VL text runtime with multimodal feature injection."""

    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={"model.language_model.": ""},
    )

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__(model_config, load_config)
        self._init_multimodal_injectors()

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
        hidden_states, mm_deepstack_embeds, cpu_locs = self._embed_multimodal_inputs(
            inputs
        )

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        for layer_id, layer in enumerate(self.layers):
            select_block_map_for_layer(inputs.attention_inputs, layer_id)
            hidden_states = layer(
                hidden_states,
                fmha_impl,
                kv_cache=(
                    self.kv_cache.get_layer_cache(layer_id) if self.kv_cache else None
                ),
            )
            hidden_states = self._inject_deepstack_after_layer(
                hidden_states, mm_deepstack_embeds, cpu_locs, layer_id
            )

        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = ["Qwen3VLForCausalLM"]
