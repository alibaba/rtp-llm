from typing import Any, Optional

import torch

from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel
from rtp_llm.models_py.modules import MultimodalEmbeddingInjector
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs


class MultimodalGenericModel(GenericMoeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids

        position_ids = inputs.combo_position_ids
        token_type_ids = inputs.embedding_inputs.combo_tokens_type_ids
        text_tokens_mask = inputs.embedding_inputs.text_tokens_mask
        mm_features = inputs.multimodal_inputs.multimodal_features
        mm_feature_locs = inputs.multimodal_inputs.mm_features_locs

        inputs_embeds = self.embed_tokens(
            input_ids, position_ids, token_type_ids, text_tokens_mask
        )
        hidden_states = self.multimodal_embedding_injector(
            inputs_embeds, mm_features, mm_feature_locs
        )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(
                inputs
            )  # pyright: ignore[reportUnreachable]
        residual = torch.zeros_like(hidden_states)
        mtp_target_hidden_capture = self._begin_mtp_target_hidden_capture(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attn_inputs=inputs.attention_inputs,
            )
            hidden_states = output.hidden_states
            residual = output.residual

            if mtp_target_hidden_capture is not None:
                self._capture_mtp_target_hidden(
                    mtp_target_hidden_capture,
                    i + 1,
                    hidden_states,
                    residual,
                )

        if mtp_target_hidden_capture is not None:
            self._finish_mtp_target_hidden_capture(mtp_target_hidden_capture)
        hidden_states, _ = self.norm(hidden_states, residual)

        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
