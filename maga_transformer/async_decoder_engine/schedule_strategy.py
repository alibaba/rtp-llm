import logging
from typing import List, Tuple
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.generate_stream import GenerateStream
from maga_transformer.async_decoder_engine.stream_cache_manager import StreamCacheManager

class BasicScheduleStrategy:
    def __init__(self, config: GptInitModelParameters, stream_cache_manager: StreamCacheManager):
        self._stream_cache_manager = stream_cache_manager
        self._max_tokens = config.max_context_batch_size * config.max_seq_len

    def schedule_new(self, streams: List[GenerateStream], force: bool) -> List[GenerateStream]:
        new_streams: List[GenerateStream] = []
        total_tokens = 0
        for stream in streams:
            cur_tokens = stream.input_length
            if total_tokens + cur_tokens > self._max_tokens:
                break
            try:
                self._stream_cache_manager.init_kvcache(stream)
                total_tokens += cur_tokens
                new_streams.append(stream)
                # once a query is successfully scheduled, set force to false
                force = False
            except Exception as e:
                if force:
                    stream.stop_and_release(str(e), e)
                else:
                    break
        return new_streams

    def schedule_current(self, streams: List[GenerateStream]) -> Tuple[List[GenerateStream], List[GenerateStream]]:
        to_remove: List[GenerateStream] = []
        to_wait: List[GenerateStream] = []
        
        self._stream_cache_manager.reserve_enough_kvcache(streams)
        
        while not self._stream_cache_manager.enough_kvcache(streams):    
            if len(streams) > 0:
                stream = streams[-1]
            else:
                break
            self._stream_cache_manager.free_block_cache(stream)
            if stream.generate_config.num_beams > 1:
                stream.seq_length = stream.input_length
            streams.remove(stream)
            to_wait.append(stream)
            logging.info(f"request_id = {stream._stream_id}, lack mem running stream back to wait and input_length:{stream.input_length} seq_length:{stream.seq_length}")
        return to_remove, to_wait
