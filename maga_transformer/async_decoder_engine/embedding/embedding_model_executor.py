import torch
import torch.nn as nn
from typing import List, Dict
from maga_transformer.utils.util import to_cuda
from maga_transformer.models.base_model import BaseModel
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.ops.gpt_ops.gpt_op import GptOp
from maga_transformer.async_decoder_engine.embedding.embedding_batch_query import EmbeddingBatchQuery
from maga_transformer.async_decoder_engine.embedding.post_process.post_process_factory import PostProcessFactory

class EmbeddingModelExecutor(object):
    def __init__(self, model: BaseModel, config: GptInitModelParameters):
        self.model_ = model
        self.config_ = config
        self.gpt_op_ = GptOp.from_config(self.config_)
        self.gpt_op_.set_weight(self.model_.weight)

        self.post_process_module_ = PostProcessFactory.create_post_process_module(self.config_)
        self.dummy_cache_ = torch.zeros([1,1,1,1]).cuda()

    def _pre_process(self, batch_query: EmbeddingBatchQuery):        
        combo_tokens_tensor = to_cuda(torch.IntTensor(batch_query.combo_tokens))
        position_ids: List[int] = []
        for len in batch_query.context_lengths_list:
            position_ids.extend(range(len))
        position_ids_tensor = to_cuda(torch.IntTensor(position_ids))
        
        input_embeds = self.model_.async_input_word_embedding(combo_tokens_tensor, [])
        if self.model_.position_encoding is not None:
            input_embeds += self.model_.position_encoding(position_ids_tensor)

        if self.model_.token_type_embeddings is not None:
            input_embeds += self.model_.token_type_embeddings(to_cuda(torch.IntTensor(batch_query.combo_token_type_ids)))

        if self.model_.pre_decoder_layernorm is not None:
            input_embeds = self.model_.pre_decoder_layernorm(input_embeds)
        
        attention_mask = self.model_.create_context_decoder_mask(batch_query.context_lengths_list)
        return input_embeds, attention_mask, position_ids_tensor

    def process(self, batch_query: EmbeddingBatchQuery) -> None:
        input_embeds, attention_mask, position_ids = self._pre_process(batch_query)
        hidden_states = self.gpt_op_.forward(
            decoder_input=input_embeds,
            key_cache=self.dummy_cache_,
            value_cache=self.dummy_cache_,
            key_cache_scale=None,
            value_cache_scale=None,
            input_lengths=torch.IntTensor(batch_query.context_lengths_list),
            sequence_lengths=torch.IntTensor([]),
            block_index_map=torch.IntTensor([1]),
            position_ids=position_ids,
            attention_mask=attention_mask,
            linear_bias_slopes=self.model_.linear_bias_slopes,
            prefix_lengths=torch.IntTensor(batch_query.reuse_lengths_list),
            count_length=torch.BoolTensor([True]),
            max_prefix_length=torch.IntTensor([0]),
            lora_ids=torch.IntTensor(batch_query.lora_ids))    
        output = self.post_process_module_.process(batch_query, hidden_states, attention_mask)        
        batch_query.update_output(output)