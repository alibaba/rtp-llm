from __future__ import annotations
import libth_transformer_config
import torch
import typing
__all__: list[str] = ['EmbeddingCppOutput', 'EngineScheduleInfo', 'EngineTaskInfo', 'KVCacheInfo', 'MultimodalInput', 'RtpEmbeddingOp', 'RtpLLMOp', 'TypedOutput', 'WorkerStatusInfo']
class EmbeddingCppOutput:
    output: TypedOutput
    def __init__(self) -> None:
        ...
    def setMapOutput(self, arg0: list[dict[str, torch.Tensor]]) -> None:
        ...
    def setTensorOutput(self, arg0: torch.Tensor) -> None:
        ...
class EngineScheduleInfo:
    finished_task_info_list: list[EngineTaskInfo]
    last_schedule_delta: int
    running_task_info_list: list[EngineTaskInfo]
    def __init__(self) -> None:
        ...
class EngineTaskInfo:
    end_time_ms: int
    input_length: int
    inter_request_id: int
    iterate_count: int
    prefix_length: int
    request_id: int
    waiting_time_ms: int
    def __init__(self) -> None:
        ...
class KVCacheInfo:
    available_kv_cache: int
    block_size: int
    cached_keys: list[int]
    total_kv_cache: int
    version: int
    def __init__(self) -> None:
        ...
class MultimodalInput:
    mm_type: int
    tensors: list[torch.Tensor]
    url: str
    def __init__(self, url: str, tensors: list[torch.Tensor], mm_type: int) -> None:
        ...
class RtpEmbeddingOp:
    def __init__(self) -> None:
        ...
    def decode(self, token_ids: torch.Tensor, token_type_ids: torch.Tensor, input_lengths: torch.Tensor, request_id: int, multimodal_inputs: list[MultimodalInput]) -> typing.Any:
        ...
    def init(self, model: typing.Any, engine_config: typing.Any, vit_config: typing.Any, mm_process_engine: typing.Any) -> None:
        ...
    def stop(self) -> None:
        ...
class RtpLLMOp:
    def __init__(self) -> None:
        ...
    def init(self, model: typing.Any, engine_config: typing.Any, vit_config: typing.Any, mm_process_engine: typing.Any, propose_model: typing.Any, token_processor: typing.Any) -> None:
        ...
    def start_http_server(self, model_weights_loader: typing.Any, lora_infos: typing.Any, world_info: typing.Any, tokenizer: typing.Any, render: typing.Any) -> None:
        ...
    def stop(self) -> None:
        ...
class TypedOutput:
    isTensor: bool
    def __init__(self) -> None:
        ...
    @property
    def map(self) -> typing.Any:
        ...
    @map.setter
    def map(self, arg1: list[dict[str, torch.Tensor]]) -> None:
        ...
    @property
    def t(self) -> typing.Any:
        ...
    @t.setter
    def t(self, arg1: torch.Tensor) -> None:
        ...
