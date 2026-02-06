import logging
import os
import sys
import time
import traceback

from setproctitle import setproctitle

from rtp_llm.config.py_config_modules import PyEnvConfigs

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import setup_logging
from rtp_llm.distribute.worker_info import WorkerInfo
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
    worker_info: WorkerInfo,
    py_env_configs: PyEnvConfigs,
):
    # Set rank_id and server_id on the passed config
    logging.info(
        f"[PROCESS_START]Start frontend server process rank_{rank_id}_server_{server_id}"
    )
    start_time = time.time()
    from rtp_llm.frontend.frontend_app import FrontendApp

    if rank_id == 0 and server_id == 0:
        logging.info(f"import FrontendApp took {time.time() - start_time:.2f}s")

    py_env_configs.server_config.frontend_server_id = server_id
    py_env_configs.server_config.rank_id = rank_id
    setproctitle(f"rtp_llm_frontend_server_rank_{rank_id}_server_{server_id}")

    try:
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
