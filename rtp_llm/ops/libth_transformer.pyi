from __future__ import annotations
import typing
__all__: list[str] = ['EngineScheduleInfo', 'EngineTaskInfo', 'KVCacheInfo', 'MultimodalInput', 'RtpEmbeddingOp', 'RtpLLMOp', 'WorkerStatusInfo', 'EmbeddingCppOutput', 'TypedOutput']
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
    tensor: torch.Tensor
    url: str
    def __init__(self, url: str, tensor: torch.Tensor, mm_type: int) -> None:
        ...
class RtpEmbeddingOp:
    def __init__(self) -> None:
        ...
    def decode(self, token_ids: torch.Tensor, token_type_ids: torch.Tensor, input_lengths: torch.Tensor, request_id: int, multimodal_inputs: list[MultimodalInput]) -> typing.Any:
        ...
    def init(self, model: typing.Any, mm_process_engine: typing.Any) -> None:
        ...
    def stop(self) -> None:
        ...

class TypedOutput:
    isTensor: bool
    t: torch.Tensor
    map: list[dict[str, torch.Tensor]]
    def __init__(self) -> None:
        ...
    def setTensorOutput(self, tensor: torch.Tensor) -> None:
        ...
    def setMapOutput(self, tensor_map: list[dict[str, torch.Tensor]]) -> None:
        ...
        
class EmbeddingQueryOutput:
    output: TypedOutput
    error_info: typing.Any
    def __init__(self) -> None:
        ...
    def setMapOutput(self, output: typing.Any) -> None:
        ...
    def setTensorOutput(self, output: typing.Any) -> None:
        ...
    def setErrorInfo(self, error_info: typing.Any) -> None:
        ...
    
class RtpLLMOp:
    def __init__(self) -> None:
        ...
    def add_lora(self, adapter_name: str, lora_a_weights: typing.Any, lora_b_weights: typing.Any) -> None:
        ...
    def get_cache_status_info(self, latest_cache_version: int) -> KVCacheInfo:
        ...
    def get_engine_schedule_info(self, arg0: int) -> EngineScheduleInfo:
        ...
    def get_worker_status_info(self, latest_finished_version: int) -> WorkerStatusInfo:
        ...
    def init(self, model: typing.Any, mm_process_engine: typing.Any, propose_model: typing.Any, token_processor: typing.Any) -> None:
        ...
    def pause(self) -> None:
        ...
    def remove_lora(self, adapter_name: str) -> None:
        ...
    def restart(self) -> None:
        ...
    def start_http_server(self, model_weights_loader: typing.Any, lora_infos: typing.Any, gang_info: typing.Any, tokenizer: typing.Any, render: typing.Any) -> None:
        ...
    def stop(self) -> None:
        ...
    def update_eplb_config(self, config: ...) -> bool:
        ...
    def update_scheduler_info(self, scheduler_info: str) -> None:
        ...
class WorkerStatusInfo:
    alive: bool
    dp_rank: int
    dp_size: int
    engine_schedule_info: EngineScheduleInfo
    precision: str
    role: str
    status_version: int
    tp_size: int
    def __init__(self) -> None:
        ...
