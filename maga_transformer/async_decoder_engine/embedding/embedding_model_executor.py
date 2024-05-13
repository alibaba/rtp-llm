import torch
import torch.nn as nn
from typing import List, Any, Union

from maga_transformer.utils.util import to_cuda
from maga_transformer.models.base_model import BaseModel
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.ops.gpt_ops.gpt_op import GptOp
from maga_transformer.ops.comm.embedding_op import EmbeddingOp
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingBatchedInput

class EmbeddingModelExecutor(object):
    def __init__(self, model: BaseModel, config: GptInitModelParameters):
        self.model_ = model
        assert model.custom_module is not None, "custom model should not be None in embedding pipeline"
        self.custom_model_ = model.custom_module
        self.config_ = config
        self.gpt_op_ = GptOp(self.config_, False)
        self.gpt_op_.set_weight(self.model_.weight)
        assert self.model_.word_embedding is not None, "word embedding should not be None"
        self.embedding_op_ = EmbeddingOp(self.model_.word_embedding.weight,
                                         self.model_.position_encoding.weight if self.model_.position_encoding is not None else None,
                                         self.model_.token_type_embeddings.weight if self.model_.token_type_embeddings is not None else None,
                                         self.config_.tp_split_emb_and_lm_head and g_parallel_info.tp_size > 1)

    def _pre_process(self, batch_input: EmbeddingBatchedInput):
        combo_tokens_tensor = to_cuda(torch.IntTensor(batch_input.combo_tokens))
        position_ids_tensor = to_cuda(self.model_.create_context_position_ids(batch_input.context_lengths_list))
        token_type_ids_tensor = to_cuda(torch.IntTensor(batch_input.combo_token_type_ids))
        input_embeds = self.embedding_op_.forward(combo_tokens_tensor, position_ids_tensor, token_type_ids_tensor)        
        if self.model_.pre_decoder_layernorm is not None:
            input_embeds = self.model_.pre_decoder_layernorm(input_embeds)
        return input_embeds, position_ids_tensor

    def process(self, batch_input: EmbeddingBatchedInput) -> Union[torch.Tensor, List[Any]]:
        with torch.cuda.nvtx.range("embedding kernel"):
            input_embeds, position_ids = self._pre_process(batch_input)
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
                attention_mask=None,
                linear_bias_slopes=self.model_.linear_bias_slopes,
                prefix_lengths=torch.IntTensor([0] * batch_input.batch_size),
                count_length=torch.BoolTensor([True]),
                max_prefix_length=torch.IntTensor([0]),
                lora_ids=torch.IntTensor([-1] * batch_input.batch_size))
        return self.custom_model_.handler.forward(
            torch.IntTensor(batch_input.combo_tokens), 
            hidden_states, 
            torch.IntTensor(batch_input.context_lengths_list), 
            batch_input.config
        )
