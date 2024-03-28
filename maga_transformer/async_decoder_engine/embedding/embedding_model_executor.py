import torch
import torch.nn as nn
from typing import List, Dict
from maga_transformer.utils.util import to_cuda
from maga_transformer.models.base_model import BaseModel
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.ops.gpt_ops.gpt_op import GptOp
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingBatchedInput, EmbeddingOutput
from maga_transformer.async_decoder_engine.embedding.post_process.post_process_module import PostProcessModule

class EmbeddingModelExecutor(object):
    def __init__(self, model: BaseModel, config: GptInitModelParameters):
        self.model_ = model
        self.config_ = config
        self.gpt_op_ = GptOp(self.config_, False)
        self.gpt_op_.set_weight(self.model_.weight)

        self.post_process_module_ = PostProcessModule(self.config_, self.model_.dtype, self.model_.tokenizer)

    def _pre_process(self, batch_input: EmbeddingBatchedInput):
        combo_tokens_tensor = to_cuda(torch.IntTensor(batch_input.combo_tokens))
        position_ids_tensor = to_cuda(self.model_.create_context_position_ids(batch_input.context_lengths_list))
        input_embeds = self.model_.async_input_word_embedding(combo_tokens_tensor, [])
        if self.model_.position_encoding is not None:
            input_embeds += self.model_.position_encoding(position_ids_tensor)

        if self.model_.token_type_embeddings is not None:
            input_embeds += self.model_.token_type_embeddings(to_cuda(torch.IntTensor(batch_input.combo_token_type_ids)))

        if self.model_.pre_decoder_layernorm is not None:
            input_embeds = self.model_.pre_decoder_layernorm(input_embeds)

        attention_mask = self.model_.create_context_decoder_mask(batch_input.context_lengths_list)
        return input_embeds, attention_mask, position_ids_tensor

    def process(self, batch_input: EmbeddingBatchedInput) -> List[EmbeddingOutput]:
        input_embeds, attention_mask, position_ids = self._pre_process(batch_input)
        hidden_states = self.gpt_op_.forward(
            decoder_input=input_embeds,
            key_cache=None,
            value_cache=None,
            key_cache_scale=None,
            value_cache_scale=None,
            input_lengths=torch.IntTensor(batch_input.context_lengths_list),
            sequence_lengths=torch.IntTensor([]),
            block_index_map=torch.IntTensor([1]),
            position_ids=position_ids,
            attention_mask=attention_mask,
            linear_bias_slopes=self.model_.linear_bias_slopes,
            prefix_lengths=torch.IntTensor([0] * batch_input.batch_size),
            count_length=torch.BoolTensor([True]),
            max_prefix_length=torch.IntTensor([0]),
            lora_ids=torch.IntTensor([-1] * batch_input.batch_size))
        output = self.post_process_module_.process(batch_input, hidden_states, attention_mask, batch_input.embedding_config)
        return output