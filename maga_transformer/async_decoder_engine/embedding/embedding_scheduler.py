import torch
import copy
from typing import List, Deque, Dict, Any
from collections import deque
from threading import Lock
from maga_transformer.utils.time_util import current_time_ms
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingStream, EngineInputs


class EmbeddingScheduler(object):
    def __init__(self, config: GptInitModelParameters):
        self.config_ = config
        self.waiting_streams_: Deque[EmbeddingStream] = deque()
        self.lock_ = Lock()

    def enqueue(self, inputs: EngineInputs) -> EmbeddingStream:
        with self.lock_:                        
            stream = EmbeddingStream(inputs=inputs, begin_time=current_time_ms())
            self.waiting_streams_.append(stream)
        return stream
    
    def wait_queue_size(self):
        return len(self.waiting_streams_)

    def _calc_length(self, stream: EmbeddingStream):
        return stream.inputs.input_length

    def schedule(self) -> List[EmbeddingStream]:
        with self.lock_:
            new_streams: List[EmbeddingStream] = []
            total_len = 0
            for stream in copy.copy(self.waiting_streams_):
                new_length = stream.inputs.input_length
                if total_len + new_length > self.config_.max_context_batch_size * self.config_.max_seq_len:
                    if len(new_streams) == 0:
                        aborted_stream = self.waiting_streams_.popleft()
                        aborted_stream.set_error(f"stream is not schedule since length exceed max length, max_length: {self.config_.max_context_batch_size} * {self.config_.max_seq_len}, acutal: {aborted_stream.inputs.input_length}")
                    break
                # make sure embedding config is the same
                if len(new_streams) > 0 and stream.inputs.config != new_streams[0].inputs.config:
                    break
                new_streams.append(stream)
                total_len += new_length

            for new_stream in new_streams:
                new_stream.set_running()
                self.waiting_streams_.remove(new_stream)        
            return new_streams