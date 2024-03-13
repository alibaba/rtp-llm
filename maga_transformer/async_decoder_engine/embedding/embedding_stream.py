import torch
import numpy
from typing import Any, List, Optional

from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.config.base_model_config import PyDanticModelBase

class EmbeddingInput(PyDanticModelBase):
    token_ids: List[int]
    token_type_ids: List[int]
    input_length: int
    generate_config: GenerateConfig

class EmbeddingOutput(PyDanticModelBase):
    sentence_embedding: Optional[torch.Tensor] = None
    sparse_embedding: Optional[torch.Tensor] = None
    colbert_embedding: Optional[torch.Tensor] = None

class EmbeddingStream(PyDanticModelBase):
    input: EmbeddingInput
    output: EmbeddingOutput = EmbeddingOutput()
    error_info: str = ""
    finished: bool = False

    def set_error(self, error: str):
        self.error_info = error

    def update(self,
               embedding_output: EmbeddingOutput):
        self.finished = True
        self.output = embedding_output

class EmbeddingBatchedInput(object):
    def __init__(self, nccl_op: Any) -> None:
        self.nccl_op_ = nccl_op

    def clear(self):
        self.batch_size = 0
        self.token_num = 0
        self.context_lengths_list: List[int] = []
        self.combo_tokens: List[int] = []
        self.combo_token_type_ids: List[int] = []

    def generate_model_input(self, streams: List[EmbeddingStream]):
        self.clear()
        if g_parallel_info.tp_rank > 0:
            return
        for stream in streams:
            self.context_lengths_list.append(stream.input.input_length)
            self.combo_tokens.extend(stream.input.token_ids)
            self.combo_token_type_ids.extend(stream.input.token_type_ids)
        self.batch_size = len(self.context_lengths_list)
        self.token_num = len(self.combo_tokens)

    def tp_sync(self):
        pass