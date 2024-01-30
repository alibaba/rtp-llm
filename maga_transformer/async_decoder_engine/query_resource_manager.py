
import os
import logging
import torch
import traceback
from typing import Any, List, Optional, Union, Dict
from maga_transformer.async_decoder_engine.cache_manager import CacheManager
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.ptuning import Ptuning, PrefixParams, MultiTaskPtuning, PrefixType
from maga_transformer.structure.raw_query import SingleRawQuery
from maga_transformer.async_decoder_engine.batch_query import QueryStats, BatchQuery
from maga_transformer.utils.model_weight import LoraResource

class QueryResourceManager:
    def __init__(self, config: GptInitModelParameters, prefix_params: PrefixParams, cache_manger: CacheManager, gen_num_per_circle: int) -> None:
        self.config_ = config
        self.cache_manager_ = cache_manger
        self.gen_num_per_circle = gen_num_per_circle
        
        self.count_prefix_length = True
        if prefix_params is not None and prefix_params.prefix_type == PrefixType.PTuningV2:
            self.count_prefix_length = False
        self.construct_ptuning(prefix_params)
        logging.info(f"reuse_cache: {self.reuse_cache_}")
        
    def construct_ptuning(self, prefix_params: PrefixParams):
        if prefix_params is None:
            self.ptuning_ = None
            self.reuse_cache_ = os.environ.get('REUSE_CACHE', None) == '1'
            return
        
        self.reuse_cache_ = False
        if isinstance(prefix_params.prefix_kvcache, dict):
            assert prefix_params.prefix_tensor is not None
            self.ptuning_ = MultiTaskPtuning(self.config_, self.cache_manager_,
                                             prefix_params.prefix_kvcache, prefix_params.prefix_type, prefix_params.prefix_tensor)
        else:
            assert isinstance(prefix_params.prefix_kvcache, torch.Tensor)
            self.ptuning_ = Ptuning(self.config_, self.cache_manager_, prefix_params.prefix_kvcache, torch.zeros([0]), prefix_params.prefix_type)
    
    def get_prefix_args(self, batch_query: BatchQuery) -> Union[torch.IntTensor, torch.BoolTensor, torch.IntTensor]:
        if self.ptuning_:
            count_length = torch.BoolTensor([self.ptuning_.count_length()])
            max_length = 0 if batch_query.generate_batch_size == 0 else max(batch_query.reuse_lengths_list[:batch_query.generate_batch_size * batch_query.beam_width])
            max_prefix_length = torch.IntTensor([max_length])
        else:
            count_length = torch.BoolTensor([1])
            max_prefix_length = torch.IntTensor([0])
        prefix_lengths = torch.IntTensor(batch_query.reuse_lengths_list)
        # TODO(xinfei.sxf) 封装
        return prefix_lengths, count_length, max_prefix_length
            
    def construct_new_query(self, single_raw_query: SingleRawQuery, lora_resource: [LoraResource]) -> QueryStats:
        slice_length = 0
        inputs = single_raw_query.input_token_ids
        if self.ptuning_:
            _, prefix_tensors = self.ptuning_.get_prefix_params(single_raw_query.generate_config)
            slice_length = len(prefix_tensors)
            inputs = torch.concat([prefix_tensors, single_raw_query.input_token_ids], dim=0)
        return QueryStats(input_tokens=inputs,
                          tokenizer=single_raw_query.tokenizer,
                          images=single_raw_query.images,
                          max_seq_len=self.config_.max_seq_len,
                          reuse_length=0,
                          block_indice=[],
                          slice_length=slice_length,
                          generate_config=single_raw_query.generate_config,
                          adapter_name=single_raw_query.adapter_name,
                          lora_resource=lora_resource)
    
    def initial_allocate_cache(self, query: QueryStats):
        # reuse length represent for ptuning length or kvcache reuse length
        block_size = (query.seq_length - query.slice_length - 2 + self.gen_num_per_circle) // self.config_.seq_size_per_block + 1
        block_indice = []
        reuse_length = 0
        try:
            if self.ptuning_:
                block_indice, reuse_length = self.ptuning_.get_block_indice(block_size, query.generate_config)
            elif self.reuse_cache_:
                block_indice, reuse_length = self.cache_manager_.malloc_with_cache(
                    block_size, query.input_token_ids.numpy().tolist(), query.generate_config.chat_id)
            else:
                block_indice = self.cache_manager_.malloc(block_size)
                reuse_length = 0
            query.add_block_index([block_indice])
            query.set_reuse_length(reuse_length)
            # query.length is acutal length for decoder, query.input_length for context_decoder
        except Exception as e:
            logging.error(f"allocate kv cache error {str(e)} {traceback.format_exc()}")
            self.cache_manager_.free(block_indice)
            raise e
    
    def incremental_allocate_cache(self, query: QueryStats):
        malloc_size = self._calc_malloc_size(query)
        if malloc_size > 0:
            try:
                query.add_block_index([self.cache_manager_.malloc(malloc_size) for _ in range(query.generate_config.num_beams)])
            except Exception as e:
                # TODO(xinfei.sxf) refactor to enum str
                query.set_error('LACK_MEM')
                logging.warning(f"lack of mem, finished. err: {str(e)}")
        
    def _calc_malloc_size(self, query: QueryStats):
        next_length = query.seq_length + self.gen_num_per_circle - 1
        # 在ptuning-v2场景下,seq_len需要加上prefix_length
        if not self.count_prefix_length:
            next_length += query.reuse_length
        current_block_length = len(query.block_indice[0]) * self.config_.seq_size_per_block
        return (next_length - current_block_length - 1) // self.config_.seq_size_per_block + 1
        
    def release_query_resource(self, query: QueryStats):
        query.release()
        self._free_block_cache(query)

    def _free_block_cache(self, query: QueryStats):
        query.set_reuse_length(0)
        block_indice = query.pop_block_indice()
        if not block_indice:
            return
        if self.ptuning_:
            prefix_block_size = self.ptuning_.calc_prefix_block_size(query.generate_config)
            self.cache_manager_.free([indice[prefix_block_size:] for indice in block_indice])
        elif self.reuse_cache_ and not query.has_error():
            self.cache_manager_.free_with_cache(
                block_indice, query.output_token_ids[0].numpy().tolist(), query.chat_id
            )
        else:
            self.cache_manager_.free(block_indice)