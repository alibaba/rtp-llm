
import os
import logging
import torch
from typing import Any, List, Optional, Union, Dict
from maga_transformer.async_decoder_engine.cache_manager import CacheManager
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.ptuning import Ptuning, PrefixParams, MultiTaskPtuning, PrefixType
from maga_transformer.async_decoder_engine.ptuning.ptuning import PrefixInfo
from maga_transformer.async_decoder_engine.generate_stream import GenerateStream

class StreamCacheManager:
    def __init__(self, config: GptInitModelParameters, cache_manger: CacheManager, gen_num_per_circle: int) -> None:
        self.config_ = config
        self.cache_manager_ = cache_manger
        self.gen_num_per_circle = gen_num_per_circle
        self.seq_size_per_block_ = config.seq_size_per_block        
        self.ptuning_ = None        
        self.reuse_cache_ = os.environ.get('REUSE_CACHE', None) == '1'

    def set_ptuning(self, prefix_params: Optional[Union[PrefixParams, Dict[int, PrefixParams]]]):
        if isinstance(prefix_params, dict):
            # reuse cache must be true in system prompt case
            self.reuse_cache_ = True
            self.ptuning_ = MultiTaskPtuning(self.config_, self.cache_manager_, prefix_params)
        elif isinstance(prefix_params, PrefixParams):
            self.reuse_cache_ = False
            self.ptuning_ = Ptuning(self.config_, self.cache_manager_, prefix_params)
        else:
            assert prefix_params is None, f"prefix params type {type(prefix_params)} unknown"
            
        logging.info(f"reuse_cache: {self.reuse_cache_}")        
        logging.info(f"block_num after Ptuning: {self.cache_manager_.free_block_nums}")

    def update_prefix(self, stream: GenerateStream):
        if not self.ptuning_:
            ptuning_info = PrefixInfo()
        else:
            ptuning_info = self.ptuning_.get_ptuning_info(stream.generate_config)
        stream.update_prefix(ptuning_info)

    def init_kvcache(self, stream: GenerateStream):
        block_num = self.inital_kvcache_count(stream)
        block_indice = []
        # reuse length represent for ptuning length or kvcache reuse length
        reuse_length = 0
        if self.ptuning_ and isinstance(self.ptuning_, Ptuning):
            block_indice, reuse_length = self.ptuning_.get_block_indice(block_num, stream.generate_config)
        elif self.reuse_cache_:
            block_indice, reuse_length = self.cache_manager_.malloc_with_cache(
                block_num, stream.complete_token_ids[0].numpy().tolist())
        else:
            block_indice = self.cache_manager_.malloc(block_num)
            reuse_length = 0

        stream.set_kvcache([block_indice], reuse_length)
        stream.add_resource_dtor(lambda: self.free_block_cache(stream))

    def incr_kvcache(self, streams: List[GenerateStream]):
        malloc_sizes = self._collect_malloc_sizes(streams)
        for stream, malloc_size in malloc_sizes.items():
            try:
                block_index = [self.cache_manager_.malloc(malloc_size)
                               for _ in range(stream.generate_config.num_beams)]
                stream.add_block_index(block_index)
            except Exception as e:
                stream.stop_and_release('LACK_MEM')
                logging.warning(f"lack of mem, finished. err: {str(e)}")

    def enough_kvcache(self, streams: List[GenerateStream]):
        malloc_sizes = self._collect_malloc_sizes(streams)
        sum_size = sum(malloc_sizes.values())
        return self.cache_manager_.free_block_nums >= sum_size

    def reserve_enough_kvcache(self, streams: List[GenerateStream]):
        malloc_sizes = self._collect_malloc_sizes(streams)
        sum_size = sum(malloc_sizes.values())
        self.cache_manager_.reserve_blocks(sum_size)

    def inital_kvcache_count(self, stream: GenerateStream):
        return (stream.seq_length - 2 + self.gen_num_per_circle) // self.seq_size_per_block_ + 1

    def free_kvcache_count(self):
        return self.cache_manager_.free_block_nums

    def _collect_malloc_sizes(self, streams: List[GenerateStream]) -> Dict[GenerateStream, int]:
        malloc_sizes: Dict[GenerateStream, int] = {}
        for stream in streams:
            malloc_size = self._calc_malloc_size(stream)
            if malloc_size > 0:
                malloc_sizes[stream] = malloc_size
        return malloc_sizes

    def _calc_malloc_size(self, stream: GenerateStream):
        next_length = stream.seq_length + self.gen_num_per_circle - 1
        if not stream.ptuning_info.count_prefix_length:
            next_length += stream.reuse_length
        current_block_length = len(stream.block_indice[0]) * self.seq_size_per_block_
        return (next_length - current_block_length - 1) // self.seq_size_per_block_ + 1

    def free_block_cache(self, stream: GenerateStream):
        block_indice = stream.pop_block_indice()
        if not block_indice:
            return
        if self.ptuning_ and isinstance(self.ptuning_, Ptuning):
            prefix_block_num = self.ptuning_.calc_prefix_block_num(stream.generate_config)
            self.cache_manager_.free([indice[prefix_block_num:] for indice in block_indice])
        elif self.reuse_cache_ and not stream.stopped:
            self.cache_manager_.free_with_cache(block_indice, stream.complete_token_ids[0].numpy().tolist())
        else:
            self.cache_manager_.free(block_indice)

    def get_kv_cache_base(self):
        return self.cache_manager_.get_kv_cache_base()

    def block_used_ratio(self) -> float:
        return self.cache_manager_._block_used_ratio()
