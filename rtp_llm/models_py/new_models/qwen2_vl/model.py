from typing import Any, Callable

import torch

from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalEmbeddingInjector,
)
from rtp_llm.models_py.new_models.qwen2.language import Qwen2ForCausalLM
from rtp_llm.ops.compute_ops import PyModelInputs


class Qwen2VLForCausalLM(Qwen2ForCausalLM):
    """Qwen2-VL language runtime consuming the separately loaded ViT features."""

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__(model_config, load_config)
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()

    def checkpoint_weight_name_filter(self) -> Callable[[str], bool]:
        return lambda name: not name.startswith("visual.")

    def word_embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        input_ids = inputs.input_ids
        features = inputs.multimodal_inputs.multimodal_features
        locations = inputs.multimodal_inputs.mm_features_locs
        text_mask = inputs.embedding_inputs.text_tokens_mask
        if features and locations is None:
            raise ValueError("Qwen2-VL multimodal features require mm_features_locs")
        if text_mask is None or text_mask.numel() == 0:
            if features:
                raise ValueError(
                    "Qwen2-VL multimodal features require text_tokens_mask"
                )
            return super().word_embedding(inputs)
        if text_mask.shape != input_ids.shape:
            raise ValueError("Qwen2-VL text_tokens_mask must match input_ids shape")

        # C++ represents visual tokens by hashes, not vocabulary indices. Mask
        # them before the vocabulary lookup, then insert ViT features at the
        # supplied token offsets. Ordinary decode remains the text-only path.
        text_mask = text_mask.to(device=input_ids.device, dtype=torch.bool)
        safe_ids = torch.where(text_mask, input_ids, 0)
        hidden_states = self.embed_tokens(safe_ids) * text_mask.unsqueeze(-1)
        return self.multimodal_embedding_injector(hidden_states, features, locations)
