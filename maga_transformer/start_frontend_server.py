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

from maga_transformer.config.log_config import LOGGING_CONFIG
from maga_transformer.server.frontend_app import FrontendApp
from maga_transformer.utils.concurrency_controller import ConcurrencyController, init_controller, set_global_controller

def start_frontend_server(server_id: int, global_controller: ConcurrencyController):
    setproctitle(f"frontend_server_{server_id}")     
    app = None
    try:
        set_global_controller(global_controller)
        app = FrontendApp()
        app.start()
    except BaseException as e:
        logging.error(f'start frontend server error: {e}, trace: {traceback.format_exc()}')
        raise e
    return app

if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    start_frontend_server(init_controller())
