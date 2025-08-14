from abc import abstractmethod
from typing import Any, AsyncGenerator, Dict

from pydantic import BaseModel

from rtp_llm.ops import EngineScheduleInfo, LoadBalanceInfo, WorkerStatusInfo, CacheStatusInfo


class BaseEngine:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def ready(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, input: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError()

    @abstractmethod
    def get_load_balance_info(self, latest_cache_version: int) -> LoadBalanceInfo:
        raise NotImplementedError()
    
    @abstractmethod
    def get_worker_status_info(
        self, latest_cache_version: int, latest_finished_version: int
    ) -> WorkerStatusInfo:
        raise NotImplementedError()
    
    @abstractmethod
    def get_cache_status_info(self, latest_cache_version: int) -> CacheStatusInfo:
        raise NotImplementedError()

    @abstractmethod
    def get_engine_schedule_info(
        self, latest_finised_version: int
    ) -> EngineScheduleInfo:
        raise NotImplementedError()

    @abstractmethod
    def update_scheduler_info(self, scheduler_info: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def update_eplb_config(self, req: Dict[str, str]) -> bool:
        raise NotImplementedError()
