import torch
import logging
from typing import List, Set, Tuple, NamedTuple, Any, Optional
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.utils.util import get_mem_info, get_dtype_size, to_torch_dtype
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

class CacheConfig(NamedTuple):
    layer_num: int
    block_nums: int
    block_size: int
    local_head_num_kv: int
    size_per_head: int
    seq_size_per_block: int
    dtype: torch.dtype

class CacheConfigGenerator(object):
    @staticmethod
    def _create_basic_config(config: GptInitModelParameters) -> CacheConfig:
        if config.head_num_kv > 1:
            local_head_num_kv = config.head_num_kv // g_parallel_info.tp_size
        else:
            local_head_num_kv = config.head_num_kv
        dtype = to_torch_dtype(config.data_type)
        seq_size_per_block = config.seq_size_per_block
        logging.info(f'seq_size_per_block: {seq_size_per_block}')
        scale_size = 0
        if config.int8_kv_cache:
            dtype = torch.int8
            scale_size = 4
        logging.info(f'kv_cache dtype: {dtype}')
        dtype_size = get_dtype_size(dtype)
        block_size = (config.layer_num * local_head_num_kv * (config.size_per_head + scale_size) * dtype_size * seq_size_per_block)
        
        return CacheConfig(config.layer_num, 0, block_size, local_head_num_kv, config.size_per_head, seq_size_per_block, dtype)

    @staticmethod
    def get_free_memory_size(config: GptInitModelParameters) -> int:
        free_gpu_memory_size: int = int(get_mem_info().free) # Byte
        logging.info(f'free kv cache mem size: {free_gpu_memory_size}')
        kv_cache_mem_size: int = free_gpu_memory_size - config.reserve_runtime_mem_mb * 1024 * 1024
        if config.kv_cache_mem_mb > 0:
            kv_cache_mem_size  = config.kv_cache_mem_mb * 1024 * 1024
        logging.info(f'kv cache mem size: {kv_cache_mem_size}')
        return kv_cache_mem_size

    @staticmethod
    def create_config(param: GptInitModelParameters) -> CacheConfig:
        kv_cache_mem_size = CacheConfigGenerator.get_free_memory_size(param)
        config = CacheConfigGenerator._create_basic_config(param)
        block_nums: int = kv_cache_mem_size // config.block_size //  2
        assert block_nums > 0
        return config._replace(block_nums=block_nums)

    @staticmethod
    def create_sp_config(param: GptInitModelParameters, sp_param: GptInitModelParameters) -> Tuple[CacheConfig, CacheConfig]:
        kv_cache_mem_size = CacheConfigGenerator.get_free_memory_size(param)
        config = CacheConfigGenerator._create_basic_config(param)
        sp_config = CacheConfigGenerator._create_basic_config(sp_param)
        block_nums: int = kv_cache_mem_size // (sp_config.block_size + config.block_size) //  2
        assert block_nums > 0

        return config._replace(block_nums=block_nums), sp_config._replace(block_nums=block_nums)