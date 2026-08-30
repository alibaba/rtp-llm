from typing import Any

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

        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        _rt_on = _rt.ENABLED
        if _rt_on:
            _rt.begin(
                seqlen=(
                    int(input_ids.size(0))
                    if input_ids.dim() == 1
                    else int(input_ids.size(-1))
                )
            )
            if _rt._get_buf() is None:
                _rt_on = False
        if _rt_on:
            _rt.record("embed_out", hidden_states)

        residual = torch.zeros_like(hidden_states)
        mtp_target_hidden_capture = self._begin_mtp_target_hidden_capture(hidden_states)
        prev_topk_indices = None
        layers = self._layers_for_forward()
        for i, decoder_layer in enumerate(layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                prev_topk_indices=prev_topk_indices,
                attn_inputs=inputs.attention_inputs,
            )
            hidden_states = output.hidden_states
            residual = output.residual
            prev_topk_indices = output.topk_indices

            if mtp_target_hidden_capture is not None:
                self._capture_mtp_target_hidden(
                    mtp_target_hidden_capture,
                    i + 1,
                    hidden_states,
                    residual,
                )
            if _rt_on:
                _rt.record(f"layer{i:02d}_hidden", hidden_states)
                _rt.record(f"layer{i:02d}_residual", residual)
                _rt.record(f"layer{i:02d}_combined", hidden_states + residual)

        if mtp_target_hidden_capture is not None:
            self._finish_mtp_target_hidden_capture(mtp_target_hidden_capture)

        hidden_states, _ = self.norm(hidden_states, residual)
        self._finish_mtp_target_hidden_capture_after_norm(residual)
        if _rt_on:
            _rt.record("final_norm", hidden_states)
            extra: dict = {
                "input_ids_shape": tuple(input_ids.shape),
                "input_ids": input_ids.detach().cpu(),
            }
            _rt.dump(step=getattr(self, "_dbg_step", 0), extra=extra)
            self._dbg_step = getattr(self, "_dbg_step", 0) + 1
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
