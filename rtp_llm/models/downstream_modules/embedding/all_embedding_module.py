from typing import Any, Dict, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.render.all_embedding_renderer import ALLEmbeddingRenderer
from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.models.downstream_modules.embedding.misc import hidden_combo_to_batch
from rtp_llm.tokenizer_factory.tokenizers import BaseTokenizer


class ALLEmbeddingModule(CustomModule):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.renderer = ALLEmbeddingRenderer(config, tokenizer)
        self.handler = NormalHandler(config)


class NormalHandler(CustomHandler):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_combo_to_batch(hidden_states, input_lengths)
