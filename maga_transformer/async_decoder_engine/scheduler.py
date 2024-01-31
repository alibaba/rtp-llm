import os
import logging
from collections import deque
from typing import Any, List, Optional, Union, Dict
import torch
import traceback
from threading import Lock
from maga_transformer.structure.raw_query import RawQuery, SingleRawQuery
from maga_transformer.utils.model_weight import LoraResource
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.tokenizer.tokenizer_base import TokenizerBase
from maga_transformer.async_decoder_engine.cache_manager import CacheManager, CacheConfig
from maga_transformer.async_decoder_engine.batch_query import BatchQuery, QueryStats
from maga_transformer.async_decoder_engine.ptuning import Ptuning, PrefixParams, MultiTaskPtuning, PrefixType
from maga_transformer.async_decoder_engine.query_resource_manager import QueryResourceManager
from maga_transformer.metrics import kmonitor, AccMetrics

class Scheduler:
    def __init__(self, config: GptInitModelParameters, cache_config: CacheConfig,
                prefix_params: Optional[PrefixParams] = None, gen_num_per_circle: int = 1, nccl_op: Any = None) -> None:
        self.config_ = config
        self.cache_config_ = cache_config
        self.seq_size_per_block_ = self.cache_config_.seq_size_per_block
        self.cache_manager_ = CacheManager(cache_config, nccl_op)
        # TODO(xinfei.sxf) move this config
        self.gen_num_per_circle = gen_num_per_circle
        logging.info(f"model generate length per circle: {self.gen_num_per_circle}")
        
        self.query_resource_manager = QueryResourceManager(self.config_, prefix_params, self.cache_manager_, self.gen_num_per_circle)
        self.count_prefix_length = self.query_resource_manager.count_prefix_length
        
        self.running_query_ = BatchQuery(self.count_prefix_length, gen_num_per_circle, nccl_op)
        self.wait_queries_: deque[QueryStats] = deque()
        self.lock_ = Lock()
        
        self.force_batching = os.environ.get('FORCE_BATCHING') == '1' # for perf_test
        
        #TODO(xinfei.sxf) 含义不太明确
        self.guarante_generate_mem = bool(int(os.environ.get("GUARANTE_GENERATE_MEM", 0)))
        self.generate_reserve_blocks = int(os.environ.get("GENERATE_RESERVE_BLOCKS", 3))
        self.max_attention_mem = self.config_.max_context_batch_size * self.config_.max_seq_len
        
        logging.info("block_size after Ptuning: " + str(len(self.cache_manager_.free_blocks_index)))

    def create_config_json(self) -> Dict[str, Any]:
        config_json = {
            "reuse_cache": self.query_resource_manager.reuse_cache_,
            "use_ptuning": self.query_resource_manager.ptuning_ is not None,
            "gen_num_per_circle": self.gen_num_per_circle,
            "block_num": self.cache_config_.block_nums,
            "seq_size_per_block": self.seq_size_per_block_
        }
        return config_json

    def enqueue(self, raw_query: RawQuery, lora_resource: Optional[LoraResource]):
        queries: List[QueryStats] = []
        try:
            for i in range(raw_query.batch_size):
                single_raw_query = SingleRawQuery(raw_query.get_tokens_id(i), raw_query.images[i],
                                        raw_query.get_adapter_name(i), raw_query.generate_config, raw_query.tokenizer)
                query = self.query_resource_manager.construct_new_query(single_raw_query, lora_resource)
                queries.append(query)
        except Exception as e:
            [q.set_error(str(e)) for q in queries]
            [self.query_resource_manager.release_query_resource(q) for q in queries]
            raise e
        [self.wait_queries_.append(q) for q in queries]
        return queries

    def check_mem_left_v2(self, query: QueryStats, new_queries: List[QueryStats]):
        batch_size = len(self.running_query_.queries) + len(new_queries)
        return len(self.cache_manager_.free_blocks_index) > batch_size * self.generate_reserve_blocks + query.seq_length // self.seq_size_per_block_

    def check_mem_left(self, query: QueryStats, new_queries: List[QueryStats]):
        left_block = len(self.cache_manager_.free_blocks_index)
        left_block -= query.seq_length // self.seq_size_per_block_ + 1
        if left_block < 8:
            return False
        left_block -= 8
        require_lens = [(query.max_new_tokens // self.seq_size_per_block_ + 1, query.context_length + query.max_new_tokens)]
        for query_status in self.running_query_.queries + new_queries:
            if query_status.seq_length <= 0:
                return False
            require_lens.append(((query_status.max_new_tokens - query_status.seq_length + query_status.context_length) // self.seq_size_per_block_ + 2, query_status.max_new_tokens + query_status.context_length))
        require_lens = sorted(require_lens, key=lambda x: x[0])
        for i, lengths in enumerate(require_lens):
            if (len(require_lens) - i) * (lengths[0] + 4) > left_block:
                return False
            left_block += lengths[1] // self.seq_size_per_block_
        return True

    def check_query_to_append(self, query: QueryStats, new_queries: List[QueryStats]) -> bool:
        if self.force_batching:
            return True
        if len(self.running_query_.queries) == 0 and len(new_queries) == 0:
            return True
        self.max_context_len = max(query.seq_length, self.max_context_len)
        if (len(new_queries) + 1) * self.max_context_len > self.max_attention_mem:
            return False
        # For ease of implementing beam search, all queries in a batch must have same beam width.
        if len(self.running_query_.queries) > 0 and self.running_query_.beam_width != query.beam_width:
            return False
        if len(new_queries) > 0 and new_queries[0].beam_width != query.beam_width:
            return False
        if self.guarante_generate_mem and not self.check_mem_left_v2(query, new_queries):
            return False
        return True

    # NOTE: This function is executed in single-thread environment.
    def schedule(self) -> BatchQuery:
        new_queries: List[QueryStats] = []
        # attention buf is special
        self.max_context_len = 0
        while len(self.wait_queries_) > 0:
            query = self.wait_queries_.popleft()
            if query.stop:
                query.set_error("query has been canceled")
                self.query_resource_manager.release_query_resource(query)
                continue
            if query.has_timeout():
                self.query_resource_manager.release_query_resource(query)
                continue
            if self.check_query_to_append(query, new_queries):
                try:
                    self.query_resource_manager.initial_allocate_cache(query)
                except Exception as e:
                    query.set_error(str(e))
                    self.query_resource_manager.release_query_resource(query)
                    continue
                new_queries.append(query)
            else:
                self.wait_queries_.appendleft(query)
                break
        [query.report_wait_time() for query in new_queries]
        self.running_query_.add_new_query(new_queries)
        return self.running_query_

    def allocate_cache_for_next_step(self):
        for i, query in enumerate(self.running_query_.queries[:]):
            if not query.finish:
                self.query_resource_manager.incremental_allocate_cache(query)
                
            if query.finish:
                self.query_resource_manager.release_query_resource(query)
                self.running_query_.queries.remove(query)

    def fallback(self):
        while self.guarante_generate_mem and len(self.cache_manager_.free_blocks_index) < len(self.running_query_.queries):
            query = self.running_query_.queries[-1]
            self.query_resource_manager._free_block_cache(query)
            if query.generate_config.num_beams > 1:
                query.seq_length = query.context_length
            self.wait_queries_.appendleft(query)
            self.running_query_.queries.remove(query)
            logging.info(f"lack mem running query back to wait and context_length:{query.context_length} seq_length:{query.seq_length}")
            kmonitor.report(AccMetrics.FALLBACK_QPS_METRIC, 1)

    def prepare_for_next_step(self):        
        self.running_query_.update_query_output()
        self.allocate_cache_for_next_step()
        self.fallback()

    def set_stop(self, queries: List[QueryStats]):
        [q.set_stop() for q in queries]

    def update_all_errors(self, err: str):
        self.running_query_.update_all_errors(err)
        self._free(self.running_query_.queries)
        self.running_query_.queries.clear()

    def _free(self, queries: List[QueryStats]):
        for query in queries:
            self.query_resource_manager.release_query_resource(query)

    def has_query(self) -> bool:
        return len(self.running_query_.queries) > 0 or len(self.wait_queries_) > 0

    def running_batch_size(self) -> int:
        return self.running_query_.total_batch_size

    def wait_query_size(self) -> int:
        return len(self.wait_queries_)
