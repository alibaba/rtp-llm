import torch
from typing import Any, List, Dict, Optional, Union

from maga_transformer.utils.util import to_cuda, to_cpu
from maga_transformer.utils.time_util import current_time_ms
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.base_model_config import PyDanticModelBase
from maga_transformer.metrics import kmonitor, GaugeMetrics

class EngineInputs(PyDanticModelBase):
    token_ids: torch.Tensor
    token_type_ids: torch.Tensor
    input_lengths: torch.Tensor
    config: Dict[str, Any] = {}
    
    @property
    def input_length(self):
        return len(self.token_ids)
    
    @property
    def batch_size(self):
        return len(self.input_lengths)


class EngineOutputs(PyDanticModelBase):
    outputs: Union[torch.Tensor, List[Any]]
    input_length: int


class EmbeddingStream(PyDanticModelBase):
    inputs: EngineInputs
    begin_time: float
    outputs: Optional[EngineOutputs] = None
    error_info: Optional[str] = None
    finished: bool = False

    def set_error(self, error: str):
        self.error_info = error
        self.finished = True

    def update(self, embedding_output: Union[torch.Tensor, List[Any]]):
        self.outputs = EngineOutputs(outputs=embedding_output, input_length=self.inputs.input_length)
        self.finished = True

    def set_running(self):
        self._report_wait_time()

    def _report_wait_time(self):
        kmonitor.report(GaugeMetrics.ASYNC_WAIT_WAIT_TIME_METRIC, current_time_ms() - self.begin_time)

class EmbeddingBatchedInput(object):
    def __init__(self, nccl_op: Any) -> None:
        self.nccl_op_ = nccl_op
        self._clear()
        
    def generate_model_input(self, streams: List[EmbeddingStream]):
        self._clear()
        if g_parallel_info.tp_rank == 0:
            self.context_lengths_list = torch.concat([stream.inputs.input_lengths for stream in streams], dim=-1).to(torch.int32)
            self.combo_tokens = torch.concat([stream.inputs.token_ids for stream in streams], dim=-1).to(torch.int32)
            self.combo_token_type_ids = torch.concat([stream.inputs.token_type_ids for stream in streams], dim=-1).to(torch.int32)

            self.batch_size = len(self.context_lengths_list)
            self.config = streams[0].inputs.config
            self.token_num = len(self.combo_tokens)
        self._tp_sync()

    def _clear(self):
        self.batch_size = 0
        self.token_num = 0
        self.context_lengths_list: torch.Tensor = torch.empty(1,0)
        self.combo_tokens: torch.Tensor = torch.empty(1, 0)
        self.combo_token_type_ids: torch.Tensor = torch.empty(1, 0)
        # no need to broadcast embedding config since only tp=0 will use it
        self.config = {}

    def _tp_sync(self):
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
            context_length_tensor = to_cuda(self.context_lengths_list)
            combo_tokens_tensor = to_cuda(self.combo_tokens)
            combo_token_type_ids_tensor = to_cuda(self.combo_token_type_ids)
        else:
            self.batch_size = shape_hints[1]
            self.token_num = shape_hints[2]
            context_length_tensor = torch.zeros([self.batch_size], dtype=torch.int32, device="cuda:0")
            combo_tokens_tensor = torch.zeros([self.token_num], dtype=torch.int32, device="cuda:0")
            combo_token_type_ids_tensor = torch.zeros([self.token_num], dtype=torch.int32, device="cuda:0")
        self.nccl_op_.broadcast_tp([context_length_tensor, combo_tokens_tensor, combo_token_type_ids_tensor])
        if g_parallel_info.tp_rank > 0:
            self.context_lengths_list = to_cpu(context_length_tensor)
            self.combo_tokens = to_cpu(combo_tokens_tensor)
            self.combo_token_type_ids = to_cpu(combo_token_type_ids_tensor)
