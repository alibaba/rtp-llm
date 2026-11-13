from abc import abstractmethod
from typing import Callable, Dict, Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.base_model import BaseModel


class BaseEngine:
    def __init__(self, model: BaseModel) -> None:
        self.started = False
        self.model: BaseModel = model
        self.config: ModelConfig = model.model_config

    def start(self) -> None:
        self._start()
        self.started = True

    @abstractmethod
    def _start(self) -> None:
        raise NotImplementedError()

    def stop(
        self, before_rpc_shutdown: Optional[Callable[[], None]] = None
    ) -> None:
        self.started = False
        if before_rpc_shutdown is None:
            self._stop()
        else:
            self._stop(before_rpc_shutdown)

    @abstractmethod
    def _stop(
        self, before_rpc_shutdown: Optional[Callable[[], None]] = None
    ) -> None:
        raise NotImplementedError()

    def ready(self) -> bool:
        return self.started

    @property
    def task_type(self):
        # Task type is stored on ModelConfig; use config to avoid depending on
        # model implementations exposing a `task_type` attribute.
        return self.config.task_type

    @property
    def default_generate_config(self):
        return self.model.default_generate_config
