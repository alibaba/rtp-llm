import logging
import os
import sys
import time
import traceback

from setproctitle import setproctitle

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import set_parallelism_config

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import setup_logging
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    set_global_controller,
)

setup_logging()


def start_dash_sc_server(
    rank_id: int,
    server_id: int,
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    pipe_writer=None,
):
    logging.info(
        f"[PROCESS_START]Start dash_sc server process rank_{rank_id}_server_{server_id}"
    )
    start_time = time.time()
    from rtp_llm.dash_sc import DashScApp

    if rank_id == 0 and server_id == 0:
        logging.info(f"import DashScApp took {time.time() - start_time:.2f}s")

    py_env_configs.server_config.frontend_server_id = server_id
    set_parallelism_config(
        py_env_configs.parallelism_config,
        rank_id,
        py_env_configs.ffn_disaggregate_config,
        py_env_configs.prefill_cp_config,
    )
    py_env_configs.server_config.set_local_rank(
        py_env_configs.parallelism_config.local_rank
    )
    py_env_configs.distribute_config.set_local_rank(
        py_env_configs.parallelism_config.local_rank
    )
    setproctitle(f"rtp_llm_dash_sc_server_rank_{rank_id}_server_{server_id}")

    try:
        set_global_controller(global_controller)
        app = DashScApp(py_env_configs)
        app.start(ready_pipe_writer=pipe_writer)
    except BaseException as e:
        logging.error(
            f"start dash_sc server error: {e}, trace: {traceback.format_exc()}"
        )
        raise e
