import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtp_llm.config.py_config_modules import PyEnvConfigs


class StartupPhase(Enum):
    MODULE_INIT = "rtp_llm_init"
    BACKEND_SERVER_PROCESS_LAUNCH = "backend_server_process_launch"
    RANK_PROCESS_LAUNCH = "rank_process_launch"
    BACKEND_MANAGER_START = "backend_manager_start"
    DISTRIBUTED_SERVER_START = "distributed_server_start"
    DISTRIBUTED_ENVIRONMENT_INIT = "distributed_environment_init"
    DEEPEP_INIT = "deepep_init"
    MODEL_LOAD = "model_load"
    SP_MODEL_LOAD = "sp_model_load"
    WEIGHT_LOAD = "weight_load"
    SP_WEIGHT_LOAD = "sp_weight_load"
    ASYNC_ENGINE_START = "async_engine_start"
    MAIN_TOTAL = "main_total"


@dataclass
class _StartupPhaseRecord:
    start: float | None = None
    end: float | None = None
    tags: dict[str, str] | None = None

    @property
    def cost(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start


@dataclass
class ProcessLaunchTimestamp:
    main_enrty_time: float = 0
    backend_server_spawn_time: float = 0
    backend_server_entry_time: float = 0
    rank_spawn_time: float = 0


class StartupTimeline:
    _instance: "StartupTimeline | None" = None
    STARTUP_LATENCY_METRIC: str = "rtp_startup_latency"

    def __init__(self):
        self.phases: dict[StartupPhase, _StartupPhaseRecord] = {}
        self.process_launch_timestamp: ProcessLaunchTimestamp = ProcessLaunchTimestamp()

    @classmethod
    def mark_phase(
        cls,
        phase: StartupPhase,
        start_time: float,
        cost_time: float,
        tags: dict[str, str] | None = None,
    ):
        instance = cls._get_instance()
        if phase not in instance.phases:
            instance.phases[phase] = _StartupPhaseRecord()
        instance.phases[phase].start = start_time
        instance.phases[phase].end = start_time + cost_time
        if tags is not None:
            instance.phases[phase].tags = tags

    @classmethod
    def mark_main_entry(cls):
        cls._get_instance().process_launch_timestamp.main_enrty_time = time.time()
        logging.info(f"[StartupTimeline] Start server")

    @classmethod
    def mark_backend_server_spawn(cls):
        cls._get_instance().process_launch_timestamp.backend_server_spawn_time = (
            time.time()
        )
        logging.info(f"[StartupTimeline] Start backend server process outer")

    @classmethod
    def mark_backend_server_entry(cls, timestamp: ProcessLaunchTimestamp):
        cls._get_instance().process_launch_timestamp = timestamp
        now = time.time()
        cls._get_instance().process_launch_timestamp.backend_server_entry_time = now
        logging.info(f"[StartupTimeline] Start backend server process")
        start = cls._get_instance().process_launch_timestamp.backend_server_spawn_time
        StartupTimeline.mark_phase(
            StartupPhase.BACKEND_SERVER_PROCESS_LAUNCH, start, now - start
        )

    @classmethod
    def mark_rank_spawn(cls):
        cls._get_instance().process_launch_timestamp.rank_spawn_time = time.time()
        logging.info(f"[StartupTimeline] Start rank process outer")

    @classmethod
    def mark_rank_entry(cls, timestamp: ProcessLaunchTimestamp | None):
        if timestamp is not None:
            cls._get_instance().process_launch_timestamp = timestamp
            start = cls._get_instance().process_launch_timestamp.rank_spawn_time
            StartupTimeline.mark_phase(
                StartupPhase.RANK_PROCESS_LAUNCH, start, time.time() - start
            )
        logging.info(f"[StartupTimeline] Start local rank process")

    @classmethod
    def _get_instance(cls) -> "StartupTimeline":
        """获取当前进程的单例实例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_startup_timestamp(cls) -> ProcessLaunchTimestamp:
        return cls._get_instance().process_launch_timestamp

    @classmethod
    def report(cls, process_name: str, py_env_configs: PyEnvConfigs):
        from rtp_llm.aios.kmonitor.python_client.kmonitor.kmonitor import KMonitor

        kmonitor = KMonitor()
        default_tags = {
            "process_name": process_name,
            "role_type": py_env_configs.role_config.role_type.name,
            "main_start_time": str(
                int(cls._get_instance().process_launch_timestamp.main_enrty_time)
            ),
        }
        metirc = kmonitor.register_gauge_metric(
            cls.STARTUP_LATENCY_METRIC, default_tags
        )
        for phase, record in cls._get_instance().phases.items():
            if record.cost is not None:
                metirc.report(
                    record.cost,
                    {
                        "phase": str(phase.value),
                        "start_time": str(record.start),
                        "end_time": str(record.end),
                    },
                )


@contextmanager
def startup_phase(phase: StartupPhase, tags: dict[str, str] | None = None):
    start = time.time()
    try:
        yield
    finally:
        cost_time = time.time() - start
        logging.info(f"[StartupTimeline] {phase.value} took: {cost_time:.2f}s")
        StartupTimeline.mark_phase(phase, start, cost_time, tags)
