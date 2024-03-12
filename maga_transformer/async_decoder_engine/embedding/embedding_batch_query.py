import copy
import torch
from pydantic import BaseModel
from typing import Optional, List, Any

from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingStream, EmbeddingOutput

class EmbeddingBatchQuery(object):
    def __init__(self, nccl_op: Any) -> None:
        self.context_streams: List[EmbeddingStream] = []
        self.nccl_op_ = nccl_op
        self.model_output: List[EmbeddingOutput] = []

    def clear(self):
        self.context_lengths_list: List[int] = []
        self.reuse_lengths_list: List[int] = []
        self.combo_tokens: List[int] = []
        self.combo_token_type_ids: List[int] = []
        self.lora_ids: List[int] = []

    def set_stream(self, streams: List[EmbeddingStream]):
        self.context_streams = streams

    @property
    def streams(self) -> List[EmbeddingStream]:
        return copy.copy(self.context_streams)

    def empty(self) -> bool:
        return len(self.context_streams) == 0

    def tp_sync(self) -> None:
        pass
        # raise NotImplementedError("todo!!")

    def update_all_errors(self, err: str):
        for s in self.streams:
            s.set_error(err)

    # create model input based on stream
    def generate_model_input(self) -> None:
        self.clear()
        if g_parallel_info.tp_rank > 0:
            return
        for stream in self.context_streams:
            self.context_lengths_list.append(stream.input.input_length)
            self.reuse_lengths_list.extend([0] * stream.input.input_length)
            self.combo_tokens.extend(stream.input.token_ids)
            self.combo_token_type_ids.extend(stream.input.token_type_ids)
            self.lora_ids.extend([0] * stream.input.input_length)

    @property
    def total_batch_size(self) -> int:
        return len(self.context_streams)

    def update_output(self, model_output: List[EmbeddingOutput]):
        self.model_output = model_output

    # update strem with model output
    def update_streams(self) -> None:
        for i, stream in enumerate(self.streams):
            stream.update(
                self.model_output[i].sentence_embedding,
                self.model_output[i].sparse_embedding,
                self.model_output[i].colbert_embedding
            )

