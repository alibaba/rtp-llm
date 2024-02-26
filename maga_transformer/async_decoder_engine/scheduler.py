import os
import logging
from collections import deque
from typing import Any, List, Optional, Union, Dict

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.tokenizer.tokenizer_base import TokenizerBase
from maga_transformer.async_decoder_engine.batch_query import BatchQuery
from maga_transformer.async_decoder_engine.schedule_strategy import PerfTestScheduleStrategy, create_schedule_strategy
from maga_transformer.async_decoder_engine.generate_stream import GenerateStream
from maga_transformer.async_decoder_engine.ptuning import Ptuning, PrefixParams, MultiTaskPtuning, PrefixType
from maga_transformer.metrics import kmonitor, AccMetrics

class Scheduler:
    def __init__(self,
                 config: GptInitModelParameters,
                 stream_cache_manager,
                 gen_num_per_circle: int = 1,
                 nccl_op: Any = None):
        # TODO(xinfei.sxf) move this config
        self.gen_num_per_circle = gen_num_per_circle
        self._stream_cache_manager = stream_cache_manager
        logging.info(f"model generate length per circle: {self.gen_num_per_circle}")

        self.batch_query = BatchQuery(gen_num_per_circle, nccl_op)
        self._waiting_streams: deque[GenerateStream] = deque()
        self._schedule_strategy = create_schedule_strategy(config, stream_cache_manager)

    # just for perf test
    def enable_perf_test_schedule_strategy(self):
        self._schedule_strategy = PerfTestScheduleStrategy(None, self._stream_cache_manager)

    def create_config_json(self) -> Dict[str, Any]:
        config_json = {
            "gen_num_per_circle": self.gen_num_per_circle,
        }
        return config_json

    def enqueue(self, stream: GenerateStream):
        self._stream_cache_manager.update_prefix(stream)
        self._waiting_streams.append(stream)

    def _schedule_streams(self, streams: List[GenerateStream]) -> List[GenerateStream]:
        new_streams = []
        for stream in streams:
            # For ease of implementing beam search, all streams in a batch must have same beam width.
            if len(self.batch_query.streams) > 0 and self.batch_query.num_beams != stream.generate_config.num_beams:
                continue
            if len(new_streams) > 0 and new_streams[0].generate_config.num_beams != stream.generate_config.num_beams:
                continue
            new_streams.append(stream)
        return new_streams

    # NOTE: This function is executed in single-thread environment.
    def schedule(self) -> BatchQuery:
        waiting_streams = self._waiting_streams.copy()
        for stream in waiting_streams:
            stream.check_timeout()
        new_streams = self._schedule_streams(waiting_streams)
        new_streams = self._schedule_strategy.schedule_new(new_streams)
        for stream in new_streams[:]:
            try:
                self._stream_cache_manager.init_kvcache(stream)
            except Exception as e:
                stream.set_stop(str(e))
            finally:
                if stream.stopped:
                    new_streams.remove(stream)
                self._waiting_streams.remove(stream)
        self.batch_query.add_new_stream(new_streams)
        return self.batch_query

    def prepare_next_step(self):
        self.batch_query.update_streams()
        to_remove, to_wait = self._schedule_strategy.schedule_current(
            self.batch_query.streams)
        for stream in to_remove + to_wait:
            self.batch_query.remove_stream(stream)
        for stream in to_wait:
            self._waiting_streams.appendleft(stream)
        self._stream_cache_manager.incr_kvcache(self.batch_query.streams)

    def update_all_errors(self, err: str):
        self.batch_query.update_all_errors(err)

    def running_batch_size(self) -> int:
        return self.batch_query.total_batch_size

    def wait_stream_size(self) -> int:
        return len(self._waiting_streams)

    def have_streams(self):
        return self.wait_stream_size() > 0 or len(self.batch_query.streams) > 0
