import os
import logging
from typing import List

from maga_transformer.async_decoder_engine.generate_stream import GenerateStream

class BasicScheduleStrategy:
    def __init__(self, config, stream_cache_manager):
        self._stream_cache_manager = stream_cache_manager
        self._generate_reserve_blocks = int(os.environ.get("GENERATE_RESERVE_BLOCKS", 3))
        self._max_tokens = config.max_context_batch_size * config.max_seq_len

    def schedule_new(self, streams: List[GenerateStream]) -> List[GenerateStream]:
        new_streams = []
        total_kvcache = len(streams) * self._generate_reserve_blocks
        total_tokens = 0
        for stream in streams:
            cur_tokens = stream.input_length
            cur_kvcache = self._generate_reserve_blocks + self._stream_cache_manager.inital_kvcache_count(stream)
            if (total_kvcache + cur_kvcache <= self._stream_cache_manager.free_kvcache_count() and
                total_tokens + cur_tokens <= self._max_tokens):
                total_kvcache += cur_kvcache
                total_tokens += cur_tokens
                new_streams.append(stream)
            else:
                break
        return new_streams

    def schedule_current(self, streams: List[GenerateStream]) -> List[GenerateStream]:
        to_remove, to_wait = [], []
        while not self._stream_cache_manager.enough_kvcache(streams):
            stream = streams[-1]
            self._stream_cache_manager.free_block_cache(stream)
            if stream.generate_config.num_beams > 1:
                stream.seq_length = stream.input_length
            streams.remove(stream)
            to_wait.append(stream)
            logging.info(f"lack mem running query back to wait and input_length:{stream.input_length} seq_length:{stream.seq_length}")
        return to_remove, to_wait

class PerfTestScheduleStrategy:
    def __init__(self, config, stream_cache_manager):
        self._stream_cache_manager = stream_cache_manager

    def schedule_new(self, streams: List[GenerateStream]) -> List[GenerateStream]:
        return streams

    def schedule_current(self, streams: List[GenerateStream]) -> List[GenerateStream]:
        return [], []

def create_schedule_strategy(config, stream_cache_manager):
    if os.environ.get('PERF_TEST_SCHEDULE') == '1':
        return PerfTestScheduleStrategy(config, stream_cache_manager)
    return BasicScheduleStrategy(config, stream_cache_manager)
