import torch
import logging
from threading import Lock
from typing import List, Set, Tuple, NamedTuple, Any, Optional

from maga_transformer.utils.util import get_mem_info
from maga_transformer.metrics import kmonitor, GaugeMetrics
from maga_transformer.utils.util import get_dtype_size, to_torch_dtype
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.utils.lru_dict import LruDict
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

from maga_transformer.utils.concurrency_controller import ConcurrencyException

class SeqPosition(NamedTuple):
    indice: int
    offset: int

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

class CacheManager:
    def __init__(self, config: CacheConfig, nccl_op: Any) -> None:
        block_nums = config.block_nums
        if g_parallel_info.tp_size > 1:
            block_nums_t = torch.tensor([block_nums], dtype=torch.int32, device="cuda:0")
            nccl_op.broadcast_tp([block_nums_t])
            block_nums = int(block_nums_t[0])
        logging.info(f"block num: {block_nums}")
        self.k_blocks = torch.zeros((config.layer_num, block_nums, config.local_head_num_kv,
                                config.seq_size_per_block, config.size_per_head), dtype=config.dtype, device='cuda:0')
        self.v_blocks = torch.zeros((config.layer_num, block_nums, config.local_head_num_kv, config.seq_size_per_block, config.size_per_head), dtype=config.dtype, device='cuda:0')
        if config.dtype is torch.int8:
            self.k_scale = torch.zeros((config.layer_num, block_nums, config.local_head_num_kv,
                                config.seq_size_per_block), dtype=torch.float32, device='cuda:0')
            self.v_scale = torch.zeros((config.layer_num, block_nums, config.local_head_num_kv, config.seq_size_per_block), dtype=torch.float32, device='cuda:0')
        else:
            self.k_scale = None
            self.v_scale = None
        self.free_blocks_index: Set[int] = set()
        # 0 block for tmp or padding use
        for i in range(1, block_nums):
            self.free_blocks_index.add(i)
        self.seq_size_per_block = config.seq_size_per_block
        self.block_nums = block_nums
        self.lock = Lock()
        self.block_cache = BlockCache()

    def __free(self, indices: List[List[int]]) -> None:
        for indice in indices:
            for i in indice:
                self.free_blocks_index.add(i)

    def free(self, indices: List[List[int]]) -> None:
        if len(indices) == 0:
            return
        with self.lock:
            self.__free(indices)

    def __malloc(self, nums: int) -> List[int]:
        if len(self.free_blocks_index) < nums:
            raise ConcurrencyException(f"failed to malloc {nums} blocks, only {len(self.free_blocks_index)} blocks left")
        else:
            return [self.free_blocks_index.pop() for _ in range(nums)]

    def malloc(self, nums: int = 1) -> List[int]:
        with self.lock:
            self._maybe_free_block_from_cache(nums)
            return self.__malloc(nums)

    def get_kv_cache_base(self):
        return self.k_blocks, self.v_blocks

    def get_kv_cache_scale_base(self):
        return self.k_scale, self.v_scale

    def set_kv_block_value(self, index: int, k_value: torch.Tensor, v_value: torch.Tensor):
        self.k_blocks[ :, index, ...] = k_value.cuda()
        self.v_blocks[ :, index, ...] = v_value.cuda()

    def block_copy(self, src_block_indice: int, dest_block_indice: int):
        self.k_blocks[ :, dest_block_indice, ...] = self.k_blocks[ :, src_block_indice, ...]
        self.v_blocks[ :, dest_block_indice, ...] = self.v_blocks[ :, src_block_indice, ...]

    def _maybe_free_block_from_cache(self, num: int):
        while len(self.free_blocks_index) < num and not self.block_cache.empty():
            indices = self.block_cache.pop()
            self.__free([indices])

    def free_with_cache(self, block_indice: List[List[int]], token_ids: List[int], chat_id: Optional[str] = None) -> None:
        with self.lock:
            if not chat_id:
                self.__free(block_indice)
            # kvcache长度比output token长度少1
            elif len(token_ids) > 1:
                indices = self.block_cache.put(chat_id, token_ids[: -1], block_indice[0])
                self.__free([indices])
                self.__free(block_indice[1: ])

    def malloc_with_cache(self, nums: int, token_ids: List[int], chat_id: Optional[str] = None) -> Tuple[List[int], int]:
        with self.lock:
            reuse_cache, common_length = self.block_cache.match(chat_id, token_ids)
            kmonitor.report(GaugeMetrics.KVCACHE_REUSE_LENGTH_METRIC, common_length)
            reuse_num = len(reuse_cache)
            self._maybe_free_block_from_cache(nums - reuse_num)
            # 如果cache里的block size大于当前block size，就释放多余的部分
            if reuse_num > nums:
                self.__free([reuse_cache[nums: ]])
                reuse_cache = reuse_cache[: nums]
                reuse_num = nums
            try:
                # 这里取min(xxx, len - 1)是一定需要算最后一个token的hidden_states
                return reuse_cache + self.__malloc(nums - reuse_num), min(common_length, len(token_ids) - 1)
            except Exception as e:
                self.__free([reuse_cache])
                raise e

    #TODO 看看能不能和prefix的kvcache block copy合并
    def copy_kvcache_from_seq_idxs(self, block_indice_list: List[int], src_index: List[int], tgt_index: List[int]):
        if (len(src_index) != len(tgt_index)):
            raise Exception("src and tgt length should equal")
        src_seq_positions: List[SeqPosition] = [self._get_seq_position(block_indice_list, x) for x in src_index]
        tgt_seq_positions: List[SeqPosition] = [self._get_seq_position(block_indice_list, x) for x in tgt_index]
        for index in range(len(src_seq_positions)):
            self._copy_kvcache_from_seq_position(src_seq_positions[index], tgt_seq_positions[index])

    def _get_seq_position(self, block_indice_list: List[int], idx: int) -> SeqPosition:
        block_idx = idx // self.seq_size_per_block
        if block_idx >= len(block_indice_list):
            raise Exception("block idx should not >= len(block_indice_list)")
        return SeqPosition(block_indice_list[idx // self.seq_size_per_block], idx % self.seq_size_per_block)

    def _copy_kvcache_from_seq_position(self, src_seq_position: SeqPosition, dst_seq_position: SeqPosition):
        self.k_blocks[:, dst_seq_position.indice, :, dst_seq_position.offset, :].copy_(self.k_blocks[:, src_seq_position.indice, :, src_seq_position.offset, :], non_blocking=True)
        self.v_blocks[:, dst_seq_position.indice, :, dst_seq_position.offset, :].copy_(self.v_blocks[:, src_seq_position.indice, :, src_seq_position.offset, :], non_blocking=True)

    def is_lack_mem(self) -> bool:
        return len(self.free_blocks_index) > 0

    def block_used_ratio(self):
        return 100 * (1 - (len(self.free_blocks_index) + self.block_cache.total_block) / self.block_nums)

class SingleBlock(NamedTuple):
    input_id_list: List[int] = []
    cache_indice: List[int] = []

class BlockCache(object):
    def __init__(self):
        self.cache = LruDict()
        # 为每次插入设置一个unique id
        # 记录所有存储的block
        self.total_block = 0

    @staticmethod
    def prefix_length(left: List[int], right: List[int]):
        max_common_length = min(len(left), len(right))
        for index in range(max_common_length):
            if left[index] != right[index]:
                return index
        return max_common_length

    def match(self, chat_id: Optional[str], token_list: List[int]) -> Tuple[List[int], int]:
        if not chat_id or chat_id not in self.cache:
            return [], 0
        single_block: SingleBlock = self.cache.pop(chat_id)
        common_length = BlockCache.prefix_length(single_block.input_id_list, token_list)
        self.total_block -= len(single_block.cache_indice)
        return single_block.cache_indice, common_length

    def empty(self) -> bool:
        return self.cache.empty()

    def pop_key(self, key: Any) -> List[int]:
        single_block: SingleBlock = self.cache.pop(key)
        self.total_block -= len(single_block.cache_indice)
        return single_block.cache_indice

    def pop(self) -> List[int]:
        single_block: SingleBlock = self.cache.poplast()[1]
        self.total_block -= len(single_block.cache_indice)
        return single_block.cache_indice

    def put(self, chat_id: str, token_list: List[int], block_indice: List[int]) -> List[int]:
        self.total_block += len(block_indice)
        single_block: SingleBlock = SingleBlock()
        if chat_id in self.cache:
            single_block = self.cache.pop(chat_id)
        self.cache[chat_id] = SingleBlock(token_list, block_indice)
        self.total_block -= len(single_block.cache_indice)
        return single_block.cache_indice
