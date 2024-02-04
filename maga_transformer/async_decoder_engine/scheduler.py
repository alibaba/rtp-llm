import os
import logging
from collections import deque
from typing import Any, List, Optional, Union, Dict
import torch
import traceback
from threading import Lock
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.tokenizer.tokenizer_base import TokenizerBase
from maga_transformer.async_decoder_engine.cache_manager import CacheManager, CacheConfig
from maga_transformer.async_decoder_engine.batch_query import BatchQuery
from maga_transformer.async_decoder_engine.generate_stream import GenerateStream
from maga_transformer.async_decoder_engine.ptuning import Ptuning, PrefixParams, MultiTaskPtuning, PrefixType
from maga_transformer.async_decoder_engine.stream_cache_manager import StreamCacheManager
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

        self._stream_cache_manager = StreamCacheManager(self.config_, prefix_params, self.cache_manager_,
                                                        self.gen_num_per_circle)
        self.batch_query = BatchQuery(self._stream_cache_manager.count_prefix_length, gen_num_per_circle, nccl_op)
        self._waiting_streams: deque[GenerateStream] = deque()

        self.force_batching = os.environ.get('FORCE_BATCHING') == '1' # for perf_test
        #TODO(xinfei.sxf) 含义不太明确
        self.guarante_generate_mem = bool(int(os.environ.get("GUARANTE_GENERATE_MEM", 0)))
        self.generate_reserve_blocks = int(os.environ.get("GENERATE_RESERVE_BLOCKS", 3))
        self._max_tokens = self.config_.max_context_batch_size * self.config_.max_seq_len
        logging.info("block_size after Ptuning: " + str(len(self.cache_manager_.free_blocks_index)))

    def create_config_json(self) -> Dict[str, Any]:
        config_json = {
            "reuse_cache": self._stream_cache_manager.reuse_cache_,
            "use_ptuning": self._stream_cache_manager.ptuning_ is not None,
            "gen_num_per_circle": self.gen_num_per_circle,
            "block_num": self.cache_config_.block_nums,
            "seq_size_per_block": self.seq_size_per_block_
        }
        return config_json

    def enqueue(self, stream: GenerateStream):
        self._stream_cache_manager.update_prefix(stream)
        self._waiting_streams.append(stream)

    def check_mem_left_v2(self, stream: GenerateStream, new_streams: List[GenerateStream]):
        batch_size = len(self.batch_query.streams) + len(new_streams)
        return len(self.cache_manager_.free_blocks_index) > batch_size * self.generate_reserve_blocks + stream.seq_length // self.seq_size_per_block_

    def check_stream_to_append(self, stream: GenerateStream, new_streams: List[GenerateStream]) -> bool:
        if self.force_batching:
            return True
        if len(self.batch_query.streams) == 0 and len(new_streams) == 0:
            return True
        total_input_length = sum(stream.input_length for stream in new_streams) + stream.input_length
        if total_input_length > self._max_tokens:
            return False
        # For ease of implementing beam search, all streams in a batch must have same beam width.
        if len(self.batch_query.streams) > 0 and self.batch_query.num_beams != stream.generate_config.num_beams:
            return False
        if len(new_streams) > 0 and new_streams[0].generate_config.num_beams != stream.generate_config.num_beams:
            return False
        if self.guarante_generate_mem and not self.check_mem_left_v2(stream, new_streams):
            return False
        return True

    # NOTE: This function is executed in single-thread environment.
    def schedule(self) -> BatchQuery:
        for stream in self._waiting_streams:
            stream.check_timeout()
        new_streams: List[GenerateStream] = []
        # attention buf is special
        while len(self._waiting_streams) > 0:
            stream = self._waiting_streams.popleft()
            if self.check_stream_to_append(stream, new_streams):
                try:
                    self._stream_cache_manager.init_kvcache(stream)
                except Exception as e:
                    stream.set_stop(str(e))
                    continue
                new_streams.append(stream)
            else:
                self._waiting_streams.appendleft(stream)
                break
        self.batch_query.add_new_stream(new_streams)
        return self.batch_query

    def update_batch_query(self):
        assert (self.batch_query.finished != None and \
            self.batch_query.hidden_states != None and \
            self.batch_query.updated_token_ids != None and \
            self.batch_query.cum_log_probs != None)
        finished = self.batch_query.finished.tolist()
        hidden_states = self.batch_query.hidden_states
        logits = self.batch_query.logits
        updated_tokens = self.batch_query.updated_token_ids
        cum_log_probs = self.batch_query.cum_log_probs
        num_beams = self.batch_query.num_beams
        gen_num = self.gen_num_per_circle
        update_length = self.batch_query.update_length
        medusa_state = self.batch_query.medusa_state

        for i, stream in enumerate(self.batch_query.streams):
            start_idx = i * num_beams
            end_idx = (i + 1) * num_beams
            query_update_length = update_length[i]
            stream.medusa_state = None if medusa_state is None else medusa_state[i]
            assert query_update_length <= gen_num, "query update length bigger than gen length"
            new_tokens = self.batch_query.slice_output_token(
                start_idx, end_idx, query_update_length).reshape(num_beams, -1)
            stream.update(new_tokens,
                          finished[start_idx],
                          hidden_states[start_idx: end_idx],
                          logits[start_idx: end_idx],
                          cum_log_probs[start_idx: end_idx])
            stream.check_timeout()
            if stream.finished or stream.stopped:
                self.batch_query.remove_stream(stream)

        self._maybe_preempt_kvcache()
        self._stream_cache_manager.incr_kvcache(self.batch_query.streams)

    def _maybe_preempt_kvcache(self):
        if not self.guarante_generate_mem:
            return False
        while not self._stream_cache_manager.enough_kvcache(self.batch_query.streams):
            stream = self.batch_query.streams[-1]
            self._stream_cache_manager.free_block_cache(stream)
            if stream.generate_config.num_beams > 1:
                stream.seq_length = stream.input_length
            self._waiting_streams.appendleft(stream)
            self.batch_query.remove_stream(stream)
            logging.info(f"lack mem running query back to wait and input_length:{stream.input_length} seq_length:{stream.seq_length}")
    def update_all_errors(self, err: str):
        self.batch_query.update_all_errors(err)

    def running_batch_size(self) -> int:
        return self.batch_query.total_batch_size

    def wait_stream_size(self) -> int:
        return len(self._waiting_streams)

    def have_streams(self):
        return self.wait_stream_size() > 0 or len(self.batch_query.streams) > 0
