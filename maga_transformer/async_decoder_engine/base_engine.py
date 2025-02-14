from abc import abstractmethod
from typing import AsyncGenerator, Any, Dict
from pydantic import BaseModel
from maga_transformer.ops import LoadBalanceInfo, EngineScheduleInfo


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
    def get_load_balance_info(self) -> LoadBalanceInfo:
        raise NotImplementedError()

    @abstractmethod
    def get_engine_schedule_info(self) -> EngineScheduleInfo:
        raise NotImplementedError()
