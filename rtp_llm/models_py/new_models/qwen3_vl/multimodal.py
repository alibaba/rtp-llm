from typing import Any

import torch

from rtp_llm.models_py.modules import (
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
    reshape_extra_input_to_deepstack,
)


class Qwen3VLMultimodalMixin:
    """Shared Qwen3-VL embedding and DeepStack injection orchestration."""

    def _init_multimodal_injectors(self) -> None:
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.multimodal_deepstack_injector = MultimodalDeepstackInjector()

    def _embed_multimodal_inputs(
        self, inputs: Any
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
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
        cpu_locs = (
            mm_feature_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
            if mm_deepstack_embeds and mm_feature_locs is not None
            else []
        )
        return hidden_states, mm_deepstack_embeds, cpu_locs

    def _inject_deepstack_after_layer(
        self,
        hidden_states: torch.Tensor,
        mm_deepstack_embeds: list[torch.Tensor],
        cpu_locs: list[int],
        layer_id: int,
    ) -> torch.Tensor:
        return self.multimodal_deepstack_injector(
            hidden_states,
            mm_deepstack_embeds,
            cpu_locs,
            layer_id,
        )


__all__ = ["Qwen3VLMultimodalMixin"]
