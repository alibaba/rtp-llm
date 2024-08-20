from abc import abstractmethod
from typing import AsyncGenerator, Any, Dict
from pydantic import BaseModel


class KVCacheInfo(BaseModel):
    available_kv_cache: int = 0
    total_kv_cache: int = 0


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
    def decode(self, input: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError()

    @abstractmethod
    def get_kv_cache_info(self) -> KVCacheInfo:
        raise NotImplementedError()
