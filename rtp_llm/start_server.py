import json
import logging
import logging.config
import multiprocessing
import os
import signal
import sys
import time
import traceback
from typing import Any, Dict, Generator, List, Union

import requests

from rtp_llm.config.py_config_modules import PyEnvConfigs, ServerConfig
from rtp_llm.metrics import kmonitor
from rtp_llm.ops import ProfilingDebugLoggingConfig

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import LOGGING_CONFIG
from rtp_llm.distribute.worker_info import DEFAULT_START_PORT, WorkerInfo
from rtp_llm.server.server_args import setup_args
from rtp_llm.start_backend_server import start_backend_server
from rtp_llm.start_frontend_server import start_frontend_server
from rtp_llm.utils.concurrency_controller import ConcurrencyController, init_controller


def check_server_health(server_port):
    try:
        response = requests.get(f"http://localhost:{server_port}/health", timeout=60)
        logging.info(
            f"response status_code = {response.status_code}, text = {response.text}, len = {len(response.text)}"
        )
        if response.status_code == 200 and response.text.strip() == '"ok"':
            return True
        else:
            logging.info(f"health check is not ready")
            return False
    except BaseException as e:
        logging.debug(f"health check is not ready, {str(e)}")
        return False


def start_backend_server_impl(global_controller):
    profiling_debug_config = ProfilingDebugLoggingConfig()
    profiling_debug_config.update_from_env()
    # only for debug
    if profiling_debug_config.debug_load_server:
        start_backend_server(global_controller)
        os._exit(-1)
    backend_process = multiprocessing.Process(
        target=start_backend_server, args=(global_controller,), name="backend_server"
    )
    backend_process.start()

    retry_interval_seconds = 5
    server_config = ServerConfig()
    server_config.update_from_env()
    start_port = server_config.start_port
    backend_server_port = WorkerInfo.backend_server_port_offset(0, start_port)
    while True:
        if not backend_process.is_alive():
            monitor_and_release_process(backend_process, None)
            raise Exception("backend server is not alive")

        try:
            if check_server_health(backend_server_port):
                logging.info(f"backend server is ready")
                break
            else:
                time.sleep(retry_interval_seconds)
        except Exception as e:
            logging.info(f"backend server is not ready")
            time.sleep(retry_interval_seconds)

    return backend_process


def start_frontend_server_impl(global_controller, backend_process):
    server_config = ServerConfig()
    server_config.update_from_env()
    frontend_server_count = server_config.fronted_server_count
    if frontend_server_count < 1:
        logging.info(
            "frontend server's count is {frontend_server_count}, this may be a mistake"
        )

    frontend_processes = []

    for i in range(frontend_server_count):
        kmonitor.frontend_server_id = i
        process = multiprocessing.Process(
            target=start_frontend_server,
            args=(i, global_controller),
            name=f"frontend_server_{i}",
        )
        frontend_processes.append(process)
        process.start()

    retry_interval_seconds = 5
    start_port = server_config.start_port
    while True:
        if not all(proc.is_alive() for proc in frontend_processes):
            monitor_and_release_process(backend_process, frontend_processes)
            raise Exception("frontend server is not alive")

        try:
            check_server_health(start_port)
            logging.info(f"frontend server is ready")
            break
        except Exception as e:
            # 如果连接失败，等待一段时间后重试
            time.sleep(retry_interval_seconds)

    return frontend_processes


def main():
    setup_args()

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logging.warn(str(e))
        pass

    global_controller = init_controller()

    backend_process = None
    frontend_process = None

    try:
        logging.info("start backend server")
        backend_process = start_backend_server_impl(global_controller)
        logging.info(f"backend server process = {backend_process}")

        logging.info("start frontend server")
        frontend_process = start_frontend_server_impl(
            global_controller, backend_process
        )
        logging.info(f"frontend server process = {frontend_process}")

        logging.info(f"后端RPC 服务监听的ip为 0.0.0.0，ip/ip段可自定义为所需范围")

    finally:
        monitor_and_release_process(backend_process, frontend_process)


def monitor_and_release_process(backend_process, frontend_process):
    all_process = []
    if backend_process:
        all_process.append(backend_process)
    if frontend_process:
        all_process.extend(frontend_process)
    logging.info(f"all process = {all_process}")

    while any(proc.is_alive() for proc in all_process):
        if not all(proc.is_alive() for proc in all_process):
            logging.error(f"server monitor : some process is not alive, exit!")
            for proc in all_process:
                try:
                    proc.terminate()
                except Exception as e:
                    logging.error(f"catch exception when process terminate : {str(e)}")
        time.sleep(1)
    [proc.join() for proc in all_process]

    logging.info("all process exit")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    main()
