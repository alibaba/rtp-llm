"""Kimi-K2.5 text owner; the MLA/MoE forward is main's GenericMoeModel."""

from collections.abc import Mapping

import torch

from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalEmbeddingInjector,
)
from rtp_llm.models_py.new_models.deepseek_v3.language import (
    DeepSeekV32ForCausalLM,
    read_config_json,
)
from rtp_llm.models_py.weight_mapper import WeightsMapper


class KimiK25ForCausalLM(DeepSeekV32ForCausalLM):
    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={
            "language_model.model.": "",
            "language_model.": "",
            "model.": "",
        }
    )

    @staticmethod
    def _read_config_json(ckpt_path):
        config = read_config_json(ckpt_path)
        text = config.get("text_config")
        if not isinstance(text, dict):
            raise ValueError("Kimi-K2.5 requires a nested text_config")
        return {**config, **text}

    def __init__(self, model_config, load_config):
        super().__init__(model_config, load_config)
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()

    def checkpoint_weight_name_filter(self):
        text_filter = super().checkpoint_weight_name_filter()
        return lambda name: name in self._handler_checkpoint_names or (
            name.startswith("language_model.") and text_filter(name.removeprefix("language_model."))
        )

    def load_weights(self, weights):
        iterator = weights.items() if isinstance(weights, Mapping) else weights
        # The parent detects the explicit output head before considering ties.
        super().load_weights(
            (key.removeprefix("language_model."), value) for key, value in iterator
        )

    def online_weight_name_filter(self):
        # The text RPC uses canonical names without the multimodal file prefix.
        return super().checkpoint_weight_name_filter()

    def embedding(self, inputs):
        mm = getattr(inputs, "multimodal_inputs", None)
        features = getattr(mm, "multimodal_features", None)
        if not features:
            return self.embed_tokens(inputs.input_ids)
        text_mask = inputs.embedding_inputs.text_tokens_mask
        if text_mask is None:
            token_ids = inputs.input_ids
        else:
            mask = text_mask.to(device=inputs.input_ids.device, dtype=torch.bool)
            token_ids = torch.where(mask, inputs.input_ids, 0)
        embedded = self.embed_tokens(token_ids)
        if text_mask is not None:
            embedded = embedded * mask.unsqueeze(-1)
        return self.multimodal_embedding_injector(
            embedded, features, mm.mm_features_locs
        )
