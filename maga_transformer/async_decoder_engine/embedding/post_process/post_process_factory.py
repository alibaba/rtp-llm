import os
import torch
from typing import List
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.post_process.sentence_transformer_module import SentenceTransformerModule
from maga_transformer.async_decoder_engine.embedding.post_process.post_process_module import PostProcessModule
from maga_transformer.async_decoder_engine.embedding.embedding_batch_query import EmbeddingBatchQuery, EmbeddingOutput

class NormalModule(PostProcessModule):
    def process(self, batch_query: EmbeddingBatchQuery, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> List[EmbeddingOutput]:
        outputs: List[EmbeddingOutput] = []
        # 这里合并成batch算可能性能会更好，但是由于这部分占总时间1/1000，所以先不纠结
        for input_length in batch_query.context_lengths_list:
            bias = 0
            outputs.append(EmbeddingOutput(sentence_embedding=hidden_states[bias]))
            bias += input_length
        return outputs
class PostProcessFactory(object):
    @staticmethod
    def create_post_process_module(config: GptInitModelParameters) -> PostProcessModule:
        if os.path.exists(os.path.join(config.ckpt_path, 'modules.json')):
            return SentenceTransformerModule(config)
        else:
            return NormalModule()