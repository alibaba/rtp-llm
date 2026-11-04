import json
import os

from navi_sched import NaviSchedMaster
from navi_sched._navi_sched_core import get_simulated_execution_stats
from navi_sched.master_process import (
    MasterProcessConfig,
    load_master_config,
    make_runtime_config,
    make_scheduler_config,
    wait_for_stop,
)


def main() -> None:
    config = MasterProcessConfig.from_env()
    config.validate()
    role_options = load_master_config(config.config_file)
    role_options.pop("biz_name", None)
    scheduler_config = make_scheduler_config(role_options)
    runtime_config = make_runtime_config(role_options)
    with NaviSchedMaster(
        config.master_biz_name,
        name=config.name,
        gig={
            "grpc_port": config.grpc_port,
            "snapshot_log_dir": config.snapshot_dir,
        },
        log_file_name=config.log_file_name,
        snapshot_dir=config.snapshot_dir,
        enable_kmon=config.enable_kmon,
        wait_no_query=False,
        runtime_config=runtime_config,
        scheduler_config=scheduler_config,
        time_compression=role_options.pop("time_compression", 1.0),
        **role_options,
    ) as master:
        master.init()
        wait_for_stop()
        stats_file = os.environ.get("NAVI_SCHED_SIMULATE_STATS_FILE", "")
        if stats_file:
            with open(stats_file, "w", encoding="utf-8") as output:
                json.dump(get_simulated_execution_stats(master), output)
