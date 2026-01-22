from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from rtp_llm.config.generate_config import RoleType


class WorkerStatusRequest(BaseModel):
    latest_cache_version: Optional[int] = -1


class TaskInfo(BaseModel):
    request_id: int
    inter_request_id: int
    prefix_length: int  # cache hit len
    input_length: int
    waiting_time_ms: int  # for master check server is hang or not
    iterate_count: int
    end_time_ms: int = -1
    dp_rank: int


class CacheStatus(BaseModel):
    available_kv_cache: int = -1
    total_kv_cache: int = -1
    block_size: int = -1
    version: int = -1
    cached_keys: Optional[List[int]] = None


class ProfileMeta(BaseModel):
    profile_time: Dict[int, int]  # {token_size, time}


class WorkStatus(BaseModel):
    role: str  # prefill, decode, vit
    server_port: Optional[int] = None
    http_port: Optional[int] = None
    grpc_port: Optional[int] = None

    running_task_info: List[TaskInfo]  # 当前running 的task信息
    finished_task_list: List[TaskInfo]  # 窗口内已经完成的任务状态

    dp_size: int
    tp_size: int
    alive: bool
    precision: str = "fp16"
    status_version: Optional[int] = -1  # 时间戳

    profile_meta: Optional[ProfileMeta] = None  # 统计的处理数据


class DebugInfo(BaseModel):
    running_batch_size: int
    queue_size: int
    waiting_time_ms: int
    available_kv_cache_len: int
    estimate_ttft_ms: int
    estimate_tpot_ms: int
    hit_cache_len: int


class ServerStatus(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: RoleType
    server_ip: str
    http_port: int
    grpc_port: int
    debug_info: Optional[DebugInfo] = None

    @model_validator(mode="before")
    def validate_role(cls, values: Dict[str, Any]):
        role = values.get("role")
        if isinstance(role, str):
            values["role"] = getattr(RoleType, role)
        elif isinstance(role, int):
            values["role"] = RoleType(role)
        else:
            raise ValueError(f"Invalid role: {role}")
        return values


class ScheduleMeta(BaseModel):
    server_status: List[ServerStatus]
    cache_local: int = 0  # 0: LOCAL, 1: REMOTE
    inter_request_id: int
    code: int = 200  # 200: OK
