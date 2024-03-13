import torch
from typing import List
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingBatchedInput, EmbeddingOutput


class PostProcessModule(object):
    def process(self, batch_query: EmbeddingBatchedInput, hidde_states: torch.Tensor, attention_mask: torch.Tensor) -> List[EmbeddingOutput]:
        raise NotImplementedError()