import os
import sys
import json
import time
import logging
import logging.config
import traceback
from typing import Generator, Union, Any, Dict, List
from setproctitle import setproctitle

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), '..'))

from rtp_llm.config.log_config import LOGGING_CONFIG
from rtp_llm.server.frontend_app import FrontendApp
from rtp_llm.distribute.worker_info import g_frontend_server_info
from rtp_llm.utils.concurrency_controller import ConcurrencyController, init_controller, set_global_controller

def start_frontend_server(server_id: int, global_controller: ConcurrencyController):
    setproctitle(f"maga_ft_frontend_server_{server_id}")
    app = None
    try:
        g_frontend_server_info.reload()
        logging.info(f"g_frontend_server_info = {g_frontend_server_info}")
        set_global_controller(global_controller)
        app = FrontendApp()
        app.start()
    except BaseException as e:
        logging.error(f'start frontend server error: {e}, trace: {traceback.format_exc()}')
        raise e
    return app