import torch
import copy
from typing import Optional, List, Deque
from collections import deque
from threading import Lock
from maga_transformer.ops.comm.nccl_op import NcclOp
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingStream, EmbeddingInput
from maga_transformer.async_decoder_engine.embedding.embedding_batch_query import EmbeddingBatchQuery

from maga_transformer.utils.thread_safe_deque import ThreadSafeDeque

class EmbeddingScheduler(object):
    def __init__(self, config: GptInitModelParameters):
        self.config_ = config
        self.batch_query_ = EmbeddingBatchQuery(NcclOp())
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

    def schedule(self) -> EmbeddingBatchQuery:
        with self.lock_:
            self._remove_timeout_stream()
            new_streams: List[EmbeddingStream] = []
            total_len = 0

            for stream in copy.copy(self.waiting_streams_):
                if total_len + stream.input.input_length > self.config_.max_context_batch_size * self.config_.max_seq_len:
                    break
                new_streams.append(stream)
                total_len += stream.input.input_length

            for new_stream in new_streams:
                self.waiting_streams_.remove(new_stream)
        self.batch_query_.set_stream(new_streams)
        return self.batch_query_

    def _remove_timeout_stream(self):
        pass
