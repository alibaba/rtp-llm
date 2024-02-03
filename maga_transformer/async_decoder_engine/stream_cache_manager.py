
import os
import logging
import torch
from typing import Any, List, Optional, Union, Dict
from maga_transformer.async_decoder_engine.cache_manager import CacheManager
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.ptuning import Ptuning, PrefixParams, MultiTaskPtuning, PrefixType
from maga_transformer.async_decoder_engine.batch_query import BatchQuery
from maga_transformer.async_decoder_engine.generate_stream import GenerateStream

class StreamCacheManager:
    def __init__(self, config: GptInitModelParameters, prefix_params: PrefixParams, cache_manger: CacheManager, gen_num_per_circle: int) -> None:
        self.config_ = config
        self.cache_manager_ = cache_manger
        self.gen_num_per_circle = gen_num_per_circle
        self.seq_size_per_block_ = config.seq_size_per_block

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
            max_length = 0 if batch_query.generate_batch_size == 0 else max(batch_query.reuse_lengths_list[:batch_query.generate_batch_size * batch_query.num_beams])
            max_prefix_length = torch.IntTensor([max_length])
        else:
            count_length = torch.BoolTensor([1])
            max_prefix_length = torch.IntTensor([0])
        prefix_lengths = torch.IntTensor(batch_query.reuse_lengths_list)
        # TODO(xinfei.sxf) 封装
        return prefix_lengths, count_length, max_prefix_length

    def update_prefix(self, stream):
        if self.ptuning_:
            _, prefix_tensors = self.ptuning_.get_prefix_params(stream.generate_config)
            stream.update_ptuning(prefix_tensors)

    def init_kvcache(self, stream: GenerateStream):
        # reuse length represent for ptuning length or kvcache reuse length
        block_size = (stream.input_length - 2 + self.gen_num_per_circle) // self.seq_size_per_block_ + 1
        block_indice = []
        reuse_length = 0
        if self.ptuning_:
            block_indice, reuse_length = self.ptuning_.get_block_indice(block_size, stream.generate_config)
        elif self.reuse_cache_:
            block_indice, reuse_length = self.cache_manager_.malloc_with_cache(
                block_size, stream.complete_token_ids[0].numpy().tolist(), stream.generate_config.chat_id)
        else:
            block_indice = self.cache_manager_.malloc(block_size)
            reuse_length = 0

        stream.set_kvcache([block_indice], reuse_length)
        stream.add_resource_dtor(lambda: self.free_block_cache(stream))

    def incr_kvcache(self, streams):
        malloc_sizes = self._collect_malloc_sizes(streams)
        for stream, malloc_size in malloc_sizes.items():
            try:
                block_index = [self.cache_manager_.malloc(malloc_size)
                               for _ in range(stream.generate_config.num_beams)]
                stream.add_block_index(block_index)
            except Exception as e:
                stream.set_stop('LACK_MEM')
                logging.warning(f"lack of mem, finished. err: {str(e)}")

    def enough_kvcache(self, streams):
        malloc_sizes = self._collect_malloc_sizes(streams)
        return len(self.cache_manager_.free_blocks_index) > sum(malloc_sizes.values())
                
    def _collect_malloc_sizes(self, streams):
        malloc_sizes = {}
        for stream in streams:
            malloc_size = self._calc_malloc_size(stream)
            if malloc_size > 0:
                malloc_sizes[stream] = malloc_size
        return malloc_sizes

    def _calc_malloc_size(self, stream: GenerateStream):
        next_length = stream.seq_length + self.gen_num_per_circle - 1
        # 在ptuning-v2场景下,seq_len需要加上prefix_length
        if not self.count_prefix_length:
            next_length += stream.reuse_length
        current_block_length = len(stream.block_indice[0]) * self.seq_size_per_block_
        return (next_length - current_block_length - 1) // self.seq_size_per_block_ + 1

    def free_block_cache(self, stream: GenerateStream):
        block_indice = stream.pop_block_indice()
        if not block_indice:
            return
        if self.ptuning_:
            prefix_block_size = self.ptuning_.calc_prefix_block_size(stream.generate_config)
            self.cache_manager_.free([indice[prefix_block_size:] for indice in block_indice])
        elif self.reuse_cache_ and not stream.stopped:
            self.cache_manager_.free_with_cache(
                block_indice, stream.complete_token_ids[0].numpy().tolist(), stream.generate_config.chat_id
            )
        else:
            self.cache_manager_.free(block_indice)

    def get_kv_cache_base(self):
        return self.cache_manager_.get_kv_cache_base()

    def block_used_ratio(self) -> float:
        return self.cache_manager_.block_used_ratio()
