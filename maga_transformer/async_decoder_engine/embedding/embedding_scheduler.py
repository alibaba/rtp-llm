import torch
import copy
from typing import List, Deque
from collections import deque
from threading import Lock
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingStream, EmbeddingInput


class EmbeddingScheduler(object):
    def __init__(self, config: GptInitModelParameters):
        self.config_ = config    
        self.waiting_streams_: Deque[EmbeddingStream] = deque()
        self.lock_ = Lock()

    def enqueue(self, inputs: List[EmbeddingInput]) -> List[EmbeddingStream]:
        streams: List[EmbeddingStream] = []
        with self.lock_:
            for input in inputs:
                stream = EmbeddingStream(input=input)
                streams.append(stream)
                self.waiting_streams_.append(stream)
        return streams

    def schedule(self) -> List[EmbeddingStream]:
        with self.lock_:
            new_streams: List[EmbeddingStream] = []
            total_len = 0

            for stream in copy.copy(self.waiting_streams_):
                if total_len + stream.input.input_length > self.config_.max_context_batch_size * self.config_.max_seq_len:
                    break
                # make sure embedding config is the same
                if len(new_streams) > 0 and stream.input.embedding_config != new_streams[0].input.embedding_config:
                    break
                new_streams.append(stream)
                total_len += stream.input.input_length

            for new_stream in new_streams:
                self.waiting_streams_.remove(new_stream)        
            return new_streams