import logging
import os
import sys
import time
import traceback

from setproctitle import setproctitle

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.worker_info import WorkerInfo

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import setup_logging
from rtp_llm.ops import RoleType
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    set_global_controller,
)

setup_logging()


def start_frontend_server(
    rank_id: int,
    server_id: int,
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    worker_info: WorkerInfo,
):
    # Set rank_id and server_id on the passed config
    logging.info(
        f"[PROCESS_START]Start frontend server process rank_{rank_id}_server_{server_id}"
    )
    start_time = time.time()
    from rtp_llm.distribute.worker_info import FrontendServerInfo
    from rtp_llm.frontend.frontend_app import FrontendApp

    if rank_id == 0 and server_id == 0:
        logging.info(f"import FrontendApp took {time.time() - start_time:.2f}s")

    py_env_configs.server_config.frontend_server_id = server_id
    py_env_configs.server_config.rank_id = rank_id
    setproctitle(f"rtp_llm_frontend_server_rank_{rank_id}_server_{server_id}")
    g_frontend_server_info = FrontendServerInfo(
        py_env_configs.server_config.frontend_server_id
    )
    logging.info(
        f"start_frontend_server, before update_worker_info: {worker_info}, {py_env_configs.parallelism_config.to_string()}"
    )
    # Update port configuration based on rank_id
    worker_info.adjust_ports_by_rank_id(
        rank_id,
        py_env_configs.server_config.worker_info_port_num,
    )
    # Set master info so EngineConfig.create (in FrontendWorker) gets correct NCCL ports
    parallelism_config = py_env_configs.parallelism_config
    if parallelism_config.world_size > 1:
        from rtp_llm.distribute.distributed_server import get_master

        master_ip, master_server_port = get_master(
            py_env_configs.distribute_config, parallelism_config
        )
        base_port = (
            int(master_server_port)
            if master_server_port
            else WorkerInfo.server_port_offset(
                local_rank=0,
                server_port=py_env_configs.server_config.start_port,
            )
        )
        worker_info.update_master_info(master_ip, base_port, parallelism_config)
    logging.info(
        f"start_frontend_server, after update_worker_info: {worker_info}, {parallelism_config.to_string()}"
    )
    try:
        logging.info(f"g_frontend_server_info = {g_frontend_server_info}")
        set_global_controller(global_controller)
        separated_frontend = py_env_configs.role_config.role_type == RoleType.FRONTEND
        app = FrontendApp(py_env_configs, worker_info, separated_frontend)
        app.start()
    except BaseException as e:
        logging.error(
            f"start frontend server error: {e}, trace: {traceback.format_exc()}"
        )
        raise e
    return app
