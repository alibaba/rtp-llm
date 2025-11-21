import logging
import logging.config
import os
import sys
import traceback

from setproctitle import setproctitle

from rtp_llm.config.py_config_modules import PyEnvConfigs

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.distribute.worker_info import FrontendServerInfo
from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    set_global_controller,
)


def start_frontend_server(rank_id: int, server_id: int, global_controller: ConcurrencyController, ssl: bool = False):
    ## collect all args and envs.
    py_env_configs = PyEnvConfigs()
    py_env_configs.update_from_env()
    py_env_configs.server_config.frontend_server_id = server_id
    py_env_configs.server_config.rank_id = rank_id
    setproctitle(f"rtp_llm_frontend_server_rank_{rank_id}_server_{server_id}_ssl_{ssl}")
    app = None
    g_frontend_server_info = FrontendServerInfo(
        py_env_configs.server_config.frontend_server_id
    )
    try:
        logging.info(f"g_frontend_server_info = {g_frontend_server_info} {ssl}")
        set_global_controller(global_controller)
        separated_frontend = os.environ.get("ROLE_TYPE", "") == "FRONTEND"
        app = FrontendApp(py_env_configs, separated_frontend, ssl)
        app.start()
    except BaseException as e:
        logging.error(
            f"start frontend server error: {e}, trace: {traceback.format_exc()}"
        )
        raise e
    return app
