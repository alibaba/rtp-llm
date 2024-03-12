import torch
from typing import List
from maga_transformer.async_decoder_engine.embedding.embedding_batch_query import EmbeddingBatchQuery, EmbeddingOutput


class PostProcessModule(object):
    def process(self, batch_query: EmbeddingBatchQuery, hidde_states: torch.Tensor, attention_mask: torch.Tensor) -> List[EmbeddingOutput]:
        raise NotImplementedError()