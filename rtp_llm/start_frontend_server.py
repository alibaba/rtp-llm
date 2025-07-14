import json
import logging
import logging.config
import os
import sys
import time
import traceback
from typing import Any, Dict, Generator, List, Union

from setproctitle import setproctitle

from rtp_llm.config.py_config_modules import PyEnvConfigs

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import LOGGING_CONFIG
from rtp_llm.distribute.worker_info import FrontendServerInfo
from rtp_llm.server.frontend_app import FrontendApp
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    init_controller,
    set_global_controller,
)


def start_frontend_server(server_id: int, global_controller: ConcurrencyController):
    ## collect all args and envs.
    py_env_configs = PyEnvConfigs()
    py_env_configs.update_from_env()
    py_env_configs.server_config.fronted_server_id = server_id
    setproctitle(f"maga_ft_frontend_server_{server_id}")
    app = None
    g_frontend_server_info = FrontendServerInfo(
        py_env_configs.server_config.fronted_server_id
    )
    try:
        logging.info(f"g_frontend_server_info = {g_frontend_server_info}")
        set_global_controller(global_controller)
        app = FrontendApp(py_env_configs)
        app.start()
    except BaseException as e:
        logging.error(
            f"start frontend server error: {e}, trace: {traceback.format_exc()}"
        )
        raise e
    return app
