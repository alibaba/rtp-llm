import os
from typing import Optional

from .alog import configureNaviSchedAlog
from .util import (
    DEFAULT_MASTER_RUNTIME_NAME,
    DEFAULT_MASTER_SNAPSHOT_DIR,
    DEFAULT_MASTER_SNAPSHOT_KEEP_COUNT,
    RoleFacade,
    makeRuntimeConfig,
    makeServiceConfig,
    naviSchedModule,
)


class NaviSchedMaster(RoleFacade):
    def __init__(
        self,
        biz_name: str,
        runtime_config=None,
        max_wait_schedule_count: int = 30,
        max_wait_schedule_time_ms: int = 10,
        queue_overload_start_cycle_multiplier: int = 10,
        queue_overload_full_cycle_multiplier: int = 30,
        scheduler_name: str = "random",
        scheduler_config: str = "",
        time_compression: float = 1.0,
        latency_model_pool_state_file: Optional[str] = None,
        latency_model_pool_source_min_revision: int = 1024,
        latency_model_pool_max_candidates: int = 4,
        latency_model_pool_persist_interval_ms: int = 10000,
        name: str = DEFAULT_MASTER_RUNTIME_NAME,
        collector_thread_num: int = 0,
        collector_queue_count: int = 0,
        **options,
    ):
        configureNaviSchedAlog()
        navi_sched = naviSchedModule()
        options.setdefault("snapshot_dir", DEFAULT_MASTER_SNAPSHOT_DIR)
        options.setdefault(
            "snapshot_keep_count", DEFAULT_MASTER_SNAPSHOT_KEEP_COUNT
        )
        if latency_model_pool_state_file is None:
            latency_model_pool_state_file = os.path.join(
                str(options["snapshot_dir"]),
                str(name),
                "latency_model_pool.state",
            )
        service_config = makeServiceConfig(navi_sched, name, options)
        config = makeRuntimeConfig(
            navi_sched,
            navi_sched.NaviSchedMasterConfig,
            biz_name,
            runtime_config,
        )
        config.service = service_config
        config.collector_thread_num = int(collector_thread_num)
        config.collector_queue_count = int(collector_queue_count)
        config.max_wait_schedule_count = int(max_wait_schedule_count)
        config.max_wait_schedule_time_ms = int(max_wait_schedule_time_ms)
        config.queue_overload_start_cycle_multiplier = int(
            queue_overload_start_cycle_multiplier
        )
        config.queue_overload_full_cycle_multiplier = int(
            queue_overload_full_cycle_multiplier
        )
        config.scheduler_name = str(scheduler_name)
        config.scheduler_config = str(scheduler_config)
        config.time_compression = float(time_compression)
        config.latency_model_pool_state_file = str(
            latency_model_pool_state_file
        )
        config.latency_model_pool_source_min_revision = int(
            latency_model_pool_source_min_revision
        )
        config.latency_model_pool_max_candidates = int(
            latency_model_pool_max_candidates
        )
        config.latency_model_pool_persist_interval_ms = int(
            latency_model_pool_persist_interval_ms
        )
        service = navi_sched.NaviSchedMaster(config)
        super().__init__(navi_sched, service)

__all__ = ["NaviSchedMaster"]
