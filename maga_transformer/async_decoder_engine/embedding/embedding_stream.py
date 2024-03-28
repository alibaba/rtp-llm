import torch
from typing import Any, List, Dict, Optional
from maga_transformer.utils.util import to_cuda, to_cpu

from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.embedding.embedding_config import EmbeddingGenerateConfig
from maga_transformer.config.base_model_config import PyDanticModelBase

class EmbeddingInput(PyDanticModelBase):
    token_ids: List[int]
    token_type_ids: List[int]
    input_length: int
    embedding_config: EmbeddingGenerateConfig

class EmbeddingOutput(PyDanticModelBase):
    sentence_embedding: Optional[torch.Tensor] = None
    sparse_embedding: Optional[Dict[str, float]] = None
    colbert_embedding: Optional[torch.Tensor] = None

class EmbeddingStream(PyDanticModelBase):
    input: EmbeddingInput
    output: EmbeddingOutput = EmbeddingOutput()
    error_info: Optional[str] = None
    finished: bool = False

    def set_error(self, error: str):
        self.error_info = error

    def update(self, embedding_output: EmbeddingOutput):
        self.output = embedding_output
        self.finished = True

class EmbeddingBatchedInput(object):
    def __init__(self, nccl_op: Any) -> None:
        self.nccl_op_ = nccl_op

    def clear(self):
        self.batch_size = 0
        self.token_num = 0
        self.context_lengths_list: List[int] = []
        self.combo_tokens: List[int] = []
        self.combo_token_type_ids: List[int] = []
        # no need to broadcast embedding config since only tp=0 will use it
        self.embedding_config = EmbeddingGenerateConfig()

    def generate_model_input(self, streams: List[EmbeddingStream]):
        self.clear()
        if g_parallel_info.tp_rank > 0:
            return
        for stream in streams:
            self.context_lengths_list.append(stream.input.input_length)
            self.combo_tokens.extend(stream.input.token_ids)
            self.combo_token_type_ids.extend(stream.input.token_type_ids)
        self.batch_size = len(self.context_lengths_list)
        self.embedding_config = streams[0].input.embedding_config
        self.token_num = len(self.combo_tokens)

    def tp_sync(self):
        if g_parallel_info.tp_size <= 1:
            return
        check_num: int = 998244352
        check_num2: int = 1000000008
        shape_hints = torch.IntTensor([check_num, self.batch_size, self.token_num, check_num2])
        shape_hints = to_cuda(shape_hints)
        self.nccl_op_.broadcast_tp([shape_hints])
        torch.cuda.current_stream().synchronize()
        shape_hints = shape_hints.cpu().numpy()
        assert shape_hints[0] == check_num and shape_hints[-1] == check_num2, 'check sum error'

        if g_parallel_info.tp_rank == 0:
            context_length_tensor = to_cuda(torch.IntTensor(self.context_lengths_list))
            combo_tokens_tensor = to_cuda(torch.IntTensor(self.combo_tokens))
            combo_token_type_ids_tensor = to_cuda(torch.IntTensor(self.combo_token_type_ids))
        else:
            self.batch_size = shape_hints[1]
            self.token_num = shape_hints[2]
            context_length_tensor = torch.zeros([self.batch_size], dtype=torch.int32, device="cuda:0")
            combo_tokens_tensor = torch.zeros([self.token_num], dtype=torch.int32, device="cuda:0")
            combo_token_type_ids_tensor = torch.zeros([self.token_num], dtype=torch.int32, device="cuda:0")
        self.nccl_op_.broadcast_tp([context_length_tensor, combo_tokens_tensor, combo_token_type_ids_tensor])
        if g_parallel_info.tp_rank > 0:
            self.context_lengths_list = to_cpu(context_length_tensor).tolist()
            self.combo_tokens = to_cpu(combo_tokens_tensor).tolist()
            self.combo_token_type_ids = to_cpu(combo_token_type_ids_tensor).tolist()
