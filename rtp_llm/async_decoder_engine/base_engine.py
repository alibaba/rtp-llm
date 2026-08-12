from abc import abstractmethod
from typing import Dict

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

    def stop(self) -> None:
        self.started = False
        self._stop()

    def prepare_stop(self, coordinated: bool = True, target_step: int = -1) -> None:
        """Stop engine work while keeping process-wide resources alive."""

    def completed_steps(self) -> int:
        """Return a conservative completed-step snapshot for shutdown coordination."""
        return -1

    def arm_stop(self, target_step: int) -> None:
        """Install a coordinated stop target without joining the engine loop."""

    def cancel_armed_stop(self) -> None:
        """Cancel an uncommitted coordinated stop target."""

    @abstractmethod
    def _stop(self) -> None:
        raise NotImplementedError()

    def ready(self) -> bool:
        return self.started

    def onflight_request_num(self) -> int:
        """Return this rank's native RPC request count when supported."""
        return 0

    @property
    def task_type(self):
        # Task type is stored on ModelConfig; use config to avoid depending on
        # model implementations exposing a `task_type` attribute.
        return self.config.task_type

    @property
    def default_generate_config(self):
        return self.model.default_generate_config
