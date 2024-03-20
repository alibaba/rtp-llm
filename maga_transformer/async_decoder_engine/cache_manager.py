import torch
import atexit
import time
import logging
import hashlib
from threading import Lock, Thread
from typing import List, Set, Tuple, NamedTuple, Any, Dict

from maga_transformer.utils.lru_dict import LruDict
from maga_transformer.utils.concurrency_controller import ConcurrencyException
from maga_transformer.metrics import kmonitor, GaugeMetrics
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.cache_config import CacheConfig

class SeqPosition(NamedTuple):
    indice: int
    offset: int
    
class BlockRefCounter:
    def __init__(self, block_nums: int):
        self.ref_counter: Dict[int, int] = {}
        for i in range(1, block_nums):
            self.ref_counter[i] = 0
            
    def get_ref_counter(self, block_index: int) -> int:
        return self.ref_counter[block_index]
    
    def increment_ref_counter(self, block_indice: List[int]):
        for index in block_indice:
            self.ref_counter[index] += 1
            
    def dec_ref_counter(self, block_indice: List[int]):
        for index in block_indice:
            self.ref_counter[index] -= 1
            
class CacheManager:
    k_blocks: torch.Tensor
    v_blocks: torch.Tensor
    def __init__(self, config: CacheConfig, nccl_op: Any) -> None:
        self.config = config
        self.seq_size_per_block = config.seq_size_per_block
        self.__init_free_block(config, nccl_op)
        self.__init_kv_cache(config)
        self.lock = Lock()
        self.start()

    def __init_free_block(self, config: CacheConfig, nccl_op: Any):
        block_nums = config.block_nums
        
        if g_parallel_info.tp_size > 1:
            block_nums_t = torch.tensor([block_nums], dtype=torch.int32, device="cuda:0")
            nccl_op.broadcast_tp([block_nums_t])
            block_nums = int(block_nums_t[0])
        logging.info(f"block num: {block_nums}")
        self.block_nums: int = block_nums
        self.free_blocks_index: Set[int] = set()
        # block 0 is reserved for tmp or padding use
        for i in range(1, block_nums):
            self.free_blocks_index.add(i)
        
        self.block_ref_counter = BlockRefCounter(block_nums)
        self.block_cache = BlockCache()
        
    def __init_kv_cache(self, config: CacheConfig):
        # block num not use config when tp, use sync block num
        # block_nums = config.block_nums
        self.k_blocks = torch.zeros((config.layer_num, self.block_nums, config.local_head_num_kv,
                                config.seq_size_per_block, config.size_per_head), dtype=config.dtype, device='cuda:0')
        self.v_blocks = torch.zeros((config.layer_num, self.block_nums, config.local_head_num_kv, config.seq_size_per_block, config.size_per_head), dtype=config.dtype, device='cuda:0')
        if config.dtype is torch.int8:
            self.k_scale = torch.zeros((config.layer_num, self.block_nums, config.local_head_num_kv,
                                config.seq_size_per_block), dtype=torch.float32, device='cuda:0')
            self.v_scale = torch.zeros((config.layer_num, self.block_nums, config.local_head_num_kv, config.seq_size_per_block), dtype=torch.float32, device='cuda:0')
        else:
            self.k_scale = None
            self.v_scale = None

    def __free(self, indices: List[List[int]]) -> None:
        for indice in indices:
            self.block_ref_counter.dec_ref_counter(indice)
            for i in indice:
                ref_count = self.block_ref_counter.get_ref_counter(i)
                if ref_count == 0:
                    self.free_blocks_index.add(i)

    def free(self, indices: List[List[int]]) -> None:
        if len(indices) == 0:
            return
        with self.lock:
            self.__free(indices)

    def __malloc(self, nums: int) -> List[int]:
        if self.free_block_nums < nums:
            raise ConcurrencyException(f"failed to malloc {nums} blocks, only {self.free_block_nums} blocks left")
        else:
            result = [self.free_blocks_index.pop() for _ in range(nums)]
            
            self.block_ref_counter.increment_ref_counter(result)
                
            return result

    def malloc(self, nums: int = 1) -> List[int]:
        with self.lock:
            self._maybe_free_block_from_cache(nums)
            return self.__malloc(nums)

    def reserve_blocks(self, nums: int):
        with self.lock:
            self._maybe_free_block_from_cache(nums)

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
        while self.free_block_nums < num and not self.block_cache.empty():
            indices = self.block_cache.pop()
            if len(indices) == 0:
                # avoid infinite loop
                break
            self.__free([indices])
            
    def free_with_cache(self, block_indice: List[List[int]], token_ids: List[int], ) -> None:
        self._insert_into_cache(block_indice, token_ids, is_resident=False)
        
    def insert_resident_cache(self, block_indice: List[int], token_ids: List[int]):
        self._insert_into_cache([block_indice], token_ids, is_resident=True)
        
    def _insert_into_cache(self, block_indice: List[List[int]], token_ids: List[int], is_resident: bool) -> None:
        with self.lock:
            # the kvcache length is 1 less than the output token length.
            if len(token_ids) > 1:
                cache_block = block_indice[0]
                cache_len = len(token_ids) - 1
                # only cache aligned block
                block_len = cache_len // self.config.seq_size_per_block
                indices = self.block_cache.put(token_ids[: cache_len], cache_block[: block_len], is_resident)
                self.__free([indices])
                self.__free([cache_block[block_len:]])
                self.__free(block_indice[1: ])
            else:
                self.__free(block_indice)
                pass

    def malloc_with_cache(self, want_block_nums: int, token_ids: List[int]) -> Tuple[List[int], int]:
        with self.lock:
            cache_blocks, common_length = self.block_cache.match(token_ids)
            kmonitor.report(GaugeMetrics.KV_CACHE_REUSE_LENGTH_METRIC, common_length)
            
            cache_block_num = len(cache_blocks)
            # here, select min(xxx, len(token_ids) - 1) is to calculate the hidden_states of the last token.
            reuse_length = min(common_length, len(token_ids) - 1)
            old_reuse_length = reuse_length
            # this is to ensure that the reuse block must be aligned, must be read-only, and cannot be modified.
            reuse_block_num = reuse_length // self.config.seq_size_per_block
            reuse_length = reuse_block_num * self.config.seq_size_per_block
            if reuse_block_num > want_block_nums or reuse_block_num > cache_block_num:
                logging.info(f"token_ids len = {len(token_ids)}, common_length = {common_length}, cache_block_num = {cache_block_num}, \
                    old_reuse_length = {old_reuse_length}, reuse_block_num = {reuse_block_num}, reuse_length = {reuse_length}, \
                    want_block_nums = {want_block_nums}, self.config.seq_size_per_block = {self.config.seq_size_per_block}")
            assert reuse_block_num <= want_block_nums, f"reuse_block_num {reuse_block_num} should <= want_block_nums {want_block_nums}"
            assert reuse_block_num <= cache_block_num, f"reuse_block_num {reuse_block_num} should <= cache_block_num {cache_block_num}"
            reuse_blocks = cache_blocks[: reuse_block_num]
            
            # increase the reference count first to prevent it from being recycled later
            self.block_ref_counter.increment_ref_counter(reuse_blocks)
            self._maybe_free_block_from_cache(want_block_nums - reuse_block_num)
            
            try:
                return reuse_blocks + self.__malloc(want_block_nums - reuse_block_num), reuse_length
            except Exception as e:
                self.__free([reuse_blocks])
                raise e

    #TODO check if it can be merged with the kvcache block copy of prefix
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

    @property
    def free_block_nums(self) -> int:
        return len(self.free_blocks_index)

    @property
    def cache_item_num(self):
        return self.block_cache.item_num()

    def block_used_ratio(self) -> float:
        return 100.0 * (1 - (self.free_block_nums / self.block_nums))

    def start(self):
        self._running = True
        self._thread = Thread(target=self._report_metrics, daemon=True)
        self._thread.start()
        atexit.register(self.stop)

    def stop(self):
        self._running = False
        self._thread.join()

    def _report_metrics(self):
        while self._running:
            with self.lock:
                kmonitor.report(GaugeMetrics.KV_CACHE_MEM_USED_RATIO_METRIC, self.block_used_ratio())
                kmonitor.report(GaugeMetrics.KV_CACHE_ITEM_NUM_METRIC, self.cache_item_num)
            time.sleep(1)

    def clean_cache(self):
        del self.k_blocks, self.v_blocks, self.k_scale, self.v_scale
        self.k_blocks = self.v_blocks = self.k_scale = self.v_scale = torch.empty([1,0])
        torch.cuda.empty_cache()

class CacheItem(NamedTuple):
    token_list: List[int] = []
    block_indice: List[int] = []
    cache_key: str = ""
    is_resident: bool = False

class BlockCache(object):
    def __init__(self):
        self.cache = LruDict()
        
    @staticmethod
    def prefix_length(left: List[int], right: List[int]):
        max_common_length = min(len(left), len(right))
        for index in range(max_common_length):
            if left[index] != right[index]:
                return index
        return max_common_length

    def match(self, token_list: List[int]) -> Tuple[List[int], int]:
        matched_item = CacheItem()
        matched_len = 0
        
        for item in self.cache.items():
            common_length = BlockCache.prefix_length(item[1].token_list, token_list)
            if common_length > matched_len:
                matched_item = item[1]
                matched_len = common_length    
        
        if matched_len != 0:
            # increase the popularity of matched cache items
            self.cache[matched_item.cache_key]
        
        return matched_item.block_indice, matched_len

    def empty(self) -> bool:
        return self.cache.empty()

    def item_num(self) -> int:
        return self.cache.len()

    def pop(self) -> List[int]:
        return_cache_item = CacheItem()
        resident_list: List[CacheItem] = []
        
        while not self.empty():
            cache_item = self.cache.poplast()[1]
            if cache_item.is_resident:
                resident_list.append(cache_item)
            else:
                return_cache_item = cache_item
                break
        
        for resident_cache_item in resident_list:
            self.cache[resident_cache_item.cache_key] = resident_cache_item
    
        return return_cache_item.block_indice

    def hash_key(self, token_list: List[int]) -> str:
        return hashlib.md5(str(token_list).encode()).hexdigest()
    
    def put(self, token_list: List[int], block_indice: List[int], is_resident: bool) -> List[int]:    
        assert len(token_list) > 0, f"token_list shoud not be empty"
        
        if len(block_indice) == 0:
            return []
        
        cache_key = self.hash_key(token_list)
        cache_item: CacheItem = CacheItem(token_list, block_indice, cache_key, is_resident) 
        # if cache has this key, reject put to protect resident item
        if cache_key in self.cache:
            return block_indice
            
        self.cache[cache_key] = cache_item
        return []
    
    def has_key(self, token_list: List[int]):
        cache_key = self.hash_key(token_list)
        return cache_key in self.cache
    
    def is_resident(self, token_list: List[int]):
        if not self.has_key(token_list):
            return False
        cache_key = self.hash_key(token_list)
        return self.cache[cache_key].is_resident
