import os
import torch
from typing import List, Union
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.post_process.sentence_transformer_module import SentenceTransformerModule
from maga_transformer.async_decoder_engine.embedding.post_process.post_process_module import PostProcessModule
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingBatchedInput, EmbeddingOutput

class NormalModule(PostProcessModule):
    def process(self, batch_query: EmbeddingBatchedInput, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> List[EmbeddingOutput]:
        outputs: List[EmbeddingOutput] = []
        # return last hidden states as default for gpt
        bias = 0
        for input_length in batch_query.context_lengths_list:
            outputs.append(EmbeddingOutput(sentence_embedding=hidden_states[bias + input_length - 1]))
            bias += input_length
        return outputs
class PostProcessFactory(object):
    @staticmethod
    def create_post_process_module(config: GptInitModelParameters, dtype: Union[torch.dtype, str]) -> PostProcessModule:
        if os.path.exists(os.path.join(config.ckpt_path, 'modules.json')):
            return SentenceTransformerModule(config, dtype)
        else:
            return NormalModule()