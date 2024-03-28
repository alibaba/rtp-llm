import torch
from typing import List, Union, Optional, Dict, Tuple
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from maga_transformer.utils.util import to_cuda
from maga_transformer.embedding.embedding_config import EmbeddingGenerateConfig, EmbeddingType
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.post_process.dense_embedding_module import  init_dense_embedding_module
from maga_transformer.async_decoder_engine.embedding.post_process.sparse_emebdding_module import init_sparse_embedding_module
from maga_transformer.async_decoder_engine.embedding.post_process.colbert_embedding_module import init_colbert_embedding_module
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingBatchedInput, EmbeddingOutput

class PostProcessModule(object):
    def __init__(self, config: GptInitModelParameters, dtype: Union[torch.dtype, str], tokenizer: PreTrainedTokenizerBase):
        self.config_ = config
        self.dtype_ = dtype
        self.tokenizer_ = tokenizer
        self.pad_token_id_ = self.tokenizer_.pad_token_id if self.tokenizer_.pad_token_id is not None else 0
        self.dense_embedding_module_ = init_dense_embedding_module(config, dtype)
        self.sparse_embedding_module_ = init_sparse_embedding_module(config, tokenizer, dtype)
        self.colbert_embedding_module_ = init_colbert_embedding_module(config, dtype)


    # attention_mask from [batch, max_seq, max_seq] to [batch, max_seq]
    # hidden_states/input_ids from [combo_length, hidden_states] to [batch, max_seq, hidden_states]    
    def _reorder_input(self, batch_input: EmbeddingBatchedInput, hidde_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sliced_hidden_states: List[torch.Tensor] = []
        sliced_input_ids: List[torch.Tensor] = []
        attention_mask_indexs: List[int] = []
        hidden_bias = 0
        mask_bias = 0
        for input_length in batch_input.context_lengths_list:
            sliced_hidden_states.append(hidde_states[hidden_bias: hidden_bias + input_length])
            sliced_input_ids.append(torch.IntTensor(batch_input.combo_tokens[hidden_bias: hidden_bias + input_length]))
            attention_mask_indexs.append(mask_bias + input_length - 1)
            mask_bias += attention_mask.shape[1]
            hidden_bias += input_length
        batched_hidden_states = pad_sequence(sliced_hidden_states, batch_first=True, padding_value=self.pad_token_id_)
        batched_input_ids = pad_sequence(sliced_input_ids, batch_first=True, padding_value=self.pad_token_id_)
        batched_attention_mask = attention_mask.reshape(-1, attention_mask.shape[2])[attention_mask_indexs].contiguous()
        return batched_input_ids, batched_hidden_states, batched_attention_mask

    def _set_outputs(self, outputs: List[EmbeddingOutput],
                     dense_embedding: Optional[torch.Tensor],
                     sparse_embedding: Optional[List[Dict[str, float]]],
                     colbert_embedding: Optional[List[torch.Tensor]]):
        if dense_embedding is not None:
            for index, dense in enumerate(dense_embedding):
                outputs[index].sentence_embedding = dense
        if sparse_embedding is not None:
            for index, sparse in enumerate(sparse_embedding):
                outputs[index].sparse_embedding = sparse
        if colbert_embedding is not None:
            for index, colbert in enumerate(colbert_embedding):
                outputs[index].colbert_embedding = colbert

    def process(self, batch_input: EmbeddingBatchedInput, hidde_states: torch.Tensor, attention_mask: torch.Tensor, embedding_config: EmbeddingGenerateConfig) -> List[EmbeddingOutput]:
        outputs = [EmbeddingOutput() for _ in range(batch_input.batch_size)]
        batch_input_ids, batch_hidden_states, batch_attention_mask = self._reorder_input(batch_input, hidde_states, attention_mask)
        dense_embedding = None
        sprase_embedding = None
        colbert_embedding = None
        if embedding_config.type == EmbeddingType.DENSE:
            dense_embedding = self.dense_embedding_module_(hidden_states=batch_hidden_states, attention_mask=batch_attention_mask,
                                                           input_length=batch_input.context_lengths_list,
                                                           do_normalize=embedding_config.do_normalize)
        if embedding_config.type == EmbeddingType.SPARSE:
            if self.sparse_embedding_module_ is None:
                raise Exception("module not support sparse embedding")
            sprase_embedding = self.sparse_embedding_module_(batch_input_ids, batch_hidden_states)
        if embedding_config.type == EmbeddingType.COLBERT:
            if self.colbert_embedding_module_ == None:
                raise Exception("module not support colbert embedding")
            colbert_embedding = self.colbert_embedding_module_(batch_hidden_states, batch_attention_mask, batch_input.context_lengths_list, do_normalize=embedding_config.do_normalize)
        self._set_outputs(outputs, dense_embedding, sprase_embedding, colbert_embedding)
        return outputs