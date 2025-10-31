from abc import abstractmethod
from typing import Any, AsyncGenerator, Dict

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.config.task_type import TaskType
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import EngineScheduleInfo, KVCacheInfo, WorkerStatusInfo


class BaseEngine:
    def __init__(self, model: BaseModel) -> None:
        self.started = False
        self.model: BaseModel = model
        self.config: GptInitModelParameters = model.config
        self.role_type = str(model.config.role_type)
        pass

    def start(self) -> None:
        self._start()
        self.started = True

    @abstractmethod
    def _start(self) -> None:
        raise NotImplementedError()

    def stop(self) -> None:
        self.started = False
        self._stop()

    @abstractmethod
    def _stop(self) -> None:
        raise NotImplementedError()

    def ready(self) -> bool:
        return self.started

    @abstractmethod
    def decode(self, input: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError()

    @abstractmethod
    def get_worker_status_info(self, latest_finished_version: int) -> WorkerStatusInfo:
        raise NotImplementedError()

    @abstractmethod
    def get_cache_status_info(self, latest_cache_version: int) -> KVCacheInfo:
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

    def pause(self) -> None:
        self.started = False
        self._pause()

    def _pause(self) -> None:
        """Pauses the engine's execution.

        When called, this method sets the `pause_` flag to true. The engine's
        `step` method checks this flag and sleeps when it's true, effectively
        pausing execution. This is necessary for tasks like updating model weights
        or clearing GPU memory, which require the engine to be inactive. The `pause_`
        parameter is modified only by this interface, so it doesn't need to be
        thread-safe.
        """
        raise NotImplementedError()

    def restart(self) -> None:
        self.started = False
        self._restart()
        self.started = True

    @abstractmethod
    def _restart(self) -> None:
        """Restarts the engine's execution."""
        raise NotImplementedError()

    @property
    def task_type(self) -> TaskType:
        return self.model.task_type

    @property
    def default_generate_config(self) -> GenerateConfig:
        return self.model.default_generate_config
