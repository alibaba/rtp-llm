import os
import logging
from collections import deque
from typing import Any, List, Optional, Union
import torch
from threading import Lock
from maga_transformer.async_decoder_engine.cache_manager import CacheManager, CacheConfig
from maga_transformer.utils.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.async_decoder_engine.batch_query import BatchQuery, QueryStats
from maga_transformer.async_decoder_engine.ptuning import Ptuning, PrefixParams, MultiTaskPtuning, PrefixType


class QueryManager:
    def __init__(self, config: GptInitModelParameters, cache_config: CacheConfig, prefix_params: Optional[PrefixParams] = None, gen_num_per_circle: int = 1, nccl_op: Any = None) -> None:
        self.config_ = config
        self.cache_config_ = cache_config
        self.cache_manager_ = CacheManager(cache_config, nccl_op)
        self.gen_num_per_circle = gen_num_per_circle
        logging.info(f"model generate length per circle: {self.gen_num_per_circle}")
        self.count_prefix_length = True
        if prefix_params is not None and prefix_params.prefix_type == PrefixType.PTuningV2:
            self.count_prefix_length = False
        self.batch_query_ = BatchQuery(self.count_prefix_length, gen_num_per_circle, nccl_op)
        self.wait_queries_: deque[QueryStats] = deque()
        self.lock_ = Lock()
        self.use_cache_ = os.environ.get('USE_BLOCK_CACHE', None) == '1'
        logging.info(f"USE_BLOCK_CACHE: {self.use_cache_}")
        self.reset_ptuning(prefix_params)
        logging.info("block_size after Ptuning: " + str(len(self.cache_manager_.free_blocks_index)))
        self.max_attention_mem = self.config_.max_context_batch_size * self.config_.max_seq_len

    def reset_ptuning(self, prefix_params: Optional[PrefixParams]):
        if prefix_params is None:
            self.ptuning_ = None
            self.use_cache_ = os.environ.get('USE_BLOCK_CACHE', None) == '1'
            return
        if isinstance(prefix_params.prefix_kvcache, dict):
            assert prefix_params.prefix_tensor is not None
            self.ptuning_ = MultiTaskPtuning(self.config_, self.cache_manager_,
                                             prefix_params.prefix_kvcache, prefix_params.prefix_type, prefix_params.prefix_tensor)
        else:
            assert isinstance(prefix_params.prefix_kvcache, torch.Tensor)
            self.ptuning_ = Ptuning(self.config_, self.cache_manager_, prefix_params.prefix_kvcache, torch.zeros([0]), prefix_params.prefix_type)
        self.use_cache_ = False

    def get_prefix_args(self, batch_query: BatchQuery) -> Union[torch.IntTensor, torch.BoolTensor, torch.IntTensor]:
        if self.ptuning_:
            count_length = torch.BoolTensor([self.ptuning_.count_length()])
            max_length = 0 if batch_query.generate_batch_size == 0 else max(batch_query.reuse_lengths_list[:batch_query.generate_batch_size * batch_query.beam_width])
            max_prefix_length = torch.IntTensor([max_length])
        else:
            count_length = torch.BoolTensor([1])
            max_prefix_length = torch.IntTensor([0])
        prefix_lengths = torch.IntTensor(batch_query.reuse_lengths_list)
        return prefix_lengths, count_length, max_prefix_length

    def put_requests_to_queue(self,
                              input_token_ids: torch.Tensor,
                              context_lengths: torch.Tensor,
                              images: List[List[str]],
                              generate_config: GenerateConfig):
        queries: List[QueryStats] = []
        try:
            for i in range(input_token_ids.shape[0]):
                adapter_name = ""
                if generate_config.adapter_name:
                    if not isinstance(generate_config.adapter_name, list):
                        raise Exception(f"batch query generate config type error {type(generate_config.adapter_name)}")
                    adapter_name = generate_config.adapter_name[i]
                query = self._gen_new_request(input_token_ids[i, :int(context_lengths[i])], images[i], generate_config, adapter_name)

                queries.append(query)
        except Exception as e:
            [q.set_error(str(e)) for q in queries]
            [self._release_query_resource(q) for q in queries]
            raise e
        [self.wait_queries_.append(q) for q in queries]
        return queries

    def _gen_new_request(self, inputs: torch.Tensor, images: List[str], generate_config: GenerateConfig, adapter_name: str):
        seq_length: int = inputs.shape[-1]
        # reuse length represent for ptuning length or kvcache reuse length
        block_size = (seq_length - 2 + self.gen_num_per_circle) // self.cache_config_.seq_size_per_block + 1
        block_indice = []
        slice_length = 0
        try:
            if self.ptuning_:
                block_indice, reuse_length = self.ptuning_.get_block_indice(block_size, generate_config)
                prefix_type, prefix_tensors = self.ptuning_.get_prefix_params(generate_config)
                slice_length = len(prefix_tensors)
                inputs = torch.concat([prefix_tensors, inputs], dim=0)
            elif self.use_cache_:
                block_indice, reuse_length = self.cache_manager_.malloc_with_cache(block_size, inputs.numpy().tolist(), generate_config.chat_id)
            else:
                block_indice = self.cache_manager_.malloc(block_size)
                reuse_length = 0
            # query.length is acutal length for decoder, query.input_length for context_decoder
        except Exception as e:
            self.cache_manager_.free(block_indice)
            raise e
        return QueryStats(input_tokens=inputs,
                          images=images,
                          max_seq_len=self.config_.max_seq_len,
                          reuse_length=reuse_length,
                          block_indice=block_indice,
                          slice_length=slice_length,
                          generate_config=generate_config,
                          adapter_name = adapter_name)

    def has_query(self) -> bool:
        return len(self.batch_query_.queries) > 0 or len(self.wait_queries_) > 0

    def check_query_to_append(self, query: QueryStats, new_queries: List[QueryStats]) -> bool:
        # batch context query has bug, tmp just allow one
        if len(new_queries) > 1:
            return False
        self.max_context_len = max(query.context_length, self.max_context_len)
        if (len(new_queries) + 1) * self.max_context_len > self.max_attention_mem:
            return False
        # For ease of implementing beam search, all queries in a batch must have same beam width.
        if len(self.batch_query_.queries) > 0 and self.batch_query_.beam_width != query.beam_width:
            return False
        if len(new_queries) > 0 and new_queries[0].beam_width != query.beam_width:
            return False
        return True

    # NOTE: This function is executed in single-thread environment.
    def get_batch_request(self) -> BatchQuery:
        new_queries: List[QueryStats] = []
        # attention buf is special
        self.max_context_len = 0
        while len(self.wait_queries_) > 0:
            query = self.wait_queries_.popleft()
            if query.stop:
                query.set_error("query has been canceled")
                continue
            
            if self.check_query_to_append(query, new_queries):
                new_queries.append(query)
            else:
                self.wait_queries_.appendleft(query)
                break
            # use cache has bug when multi context query
        [query.report_wait_time() for query in new_queries]
        self.batch_query_.add_new_query(new_queries)
        return self.batch_query_

    def set_stop(self, queries: List[QueryStats]):
        [q.set_stop() for q in queries]

    def update_batch_query(self):        
        assert (self.batch_query_.finished != None and \
            self.batch_query_.hidden_states != None and \
            self.batch_query_.updated_token_ids != None and \
            self.batch_query_.cum_log_probs != None)
        finished = self.batch_query_.finished.numpy().tolist()
        hidden_states = self.batch_query_.hidden_states
        logits = self.batch_query_.logits
        updated_tokens = self.batch_query_.updated_token_ids
        cum_log_probs = self.batch_query_.cum_log_probs
        beam_width = self.batch_query_.beam_width
        gen_num = self.gen_num_per_circle
        update_length = self.batch_query_.update_length
        medusa_state = self.batch_query_.medusa_state

        for i, query in enumerate(self.batch_query_.queries[:]):
            query.report_first_token_rt()
            start_idx = i * beam_width
            end_idx = (i + 1) * beam_width
            query_update_length = update_length[i]
            query.medusa_state = None if medusa_state is None else medusa_state[i]
            assert query_update_length <= gen_num, "query update length bigger than gen length"
            if beam_width > 1:
                query.output_token_ids_[:, :query.seq_length + gen_num] = \
                    updated_tokens[start_idx: end_idx, :query.seq_length + gen_num]
            new_tokens = self.batch_query_.slice_output_token(start_idx, end_idx, query_update_length).reshape(-1, beam_width)
            for token in new_tokens:
                query.update(
                    hidden_states[start_idx: end_idx],
                    logits[start_idx: end_idx],
                    token,
                    cum_log_probs[start_idx: end_idx],
                )
            if finished[start_idx] or query.need_finish():
                query.finish = True
            else:
                malloc_size = self._calc_malloc_size(query)
                if malloc_size > 0:
                    try:
                        query.add_block_index([self.cache_manager_.malloc(malloc_size) for _ in range(beam_width)])
                    except Exception as e:
                        query.set_error('LACK_MEM')
                        logging.warning(f"lack of mem, finished. err: {str(e)}")
            if query.finish:
                self._release_query_resource(query)
                self.batch_query_.queries.remove(query)

    def _calc_malloc_size(self, query: QueryStats):
        next_length = query.seq_length + self.gen_num_per_circle - 1
        # 在ptuning-v2场景下,seq_len需要加上prefix_length
        if not self.count_prefix_length:
            next_length += query.reuse_length
        current_block_length = len(query.block_indice[0]) * self.config_.seq_size_per_block
        return (next_length - current_block_length - 1) // self.config_.seq_size_per_block + 1

    def _release_query_resource(self, query: QueryStats):
        block_indice = query.pop_block_indice()
        if not block_indice:
            return
        if self.ptuning_:
            prefix_block_size = self.ptuning_.calc_prefix_block_size(query.generate_config)
            self.cache_manager_.free([indice[prefix_block_size:] for indice in block_indice])
        elif self.use_cache_ and not query.has_error():
            self.cache_manager_.free_with_cache(
                block_indice, query.output_token_ids[0].numpy().tolist(), query.chat_id
            )
        else:
            self.cache_manager_.free(block_indice)

    def free(self, queries: List[QueryStats]):
        for query in queries:
            self._release_query_resource(query)

    def get_kv_cache_base(self):
        return self.cache_manager_.get_kv_cache_base()


    def get_kv_cache_scale_base(self):
        return self.cache_manager_.get_kv_cache_scale_base()

    def update_all_errors(self, err: str):
        self.batch_query_.update_all_errors(err)
        self.free(self.batch_query_.queries)
        self.batch_query_.queries.clear()

    def running_batch_size(self) -> int:
        return self.batch_query_.total_batch_size

    def wait_query_size(self) -> int:
        return len(self.wait_queries_)

    def block_used_ratio(self) -> float:
        return self.cache_manager_.block_used_ratio()
