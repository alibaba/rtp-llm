import logging
import multiprocessing
import os
import signal
import sys
import time
import traceback

import requests
import torch

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.distribute.gang_info import get_gang_info
from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info
from rtp_llm.ops import RoleType, VitSeparation
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.concurrency_controller import init_controller
from rtp_llm.utils.process_manager import ProcessManager

setup_logging()


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
        logging.debug("health check is not ready, %s", str(e))
        return False


def start_backend_server_impl(global_controller, py_env_configs):
    from rtp_llm.start_backend_server import start_backend_server

    # only for debug
    if py_env_configs.profiling_debug_logging_config.debug_load_server:
        start_backend_server(global_controller, py_env_configs)
        os._exit(-1)
    backend_process = multiprocessing.Process(
        target=start_backend_server,
        args=(global_controller, py_env_configs),
        name="backend_server",
    )
    backend_process.start()

    retry_interval_seconds = 5
    start_port = py_env_configs.server_config.start_port
    backend_server_port = WorkerInfo.backend_server_port_offset(0, start_port)
    while True:
        if not backend_process.is_alive():
            logging.error("Backend server is not alive")
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


def start_vit_server_impl(py_env_configs: PyEnvConfigs):
    from rtp_llm.start_backend_server import vit_start_server

    vit_process = multiprocessing.Process(
        target=vit_start_server, args=(py_env_configs,), name="vit_server"
    )
    vit_process.start()

    retry_interval_seconds = 5
    server_config = py_env_configs.server_config
    start_port = server_config.start_port
    vit_server_port = (
        WorkerInfo.vit_http_server_port_offset(0, start_port)
        if py_env_configs.role_config.role_type == RoleType.VIT
        else WorkerInfo.vit_http_server_port_offset(0, start_port)
    )
    while True:
        if not vit_process.is_alive():
            logging.error("vit server is not alive")
            raise Exception("vit server is not alive")

        try:
            if check_server_health(vit_server_port):
                logging.info(f"vit server is ready")
                break
            else:
                time.sleep(retry_interval_seconds)
        except Exception as e:
            logging.info(f"vit server is not ready")
            time.sleep(retry_interval_seconds)
    return vit_process


def start_frontend_server_impl(global_controller, py_env_configs):
    from rtp_llm.start_frontend_server import start_frontend_server

    frontend_server_count = py_env_configs.server_config.frontend_server_count
    if frontend_server_count < 1:
        logging.info(
            "frontend server's count is {frontend_server_count}, this may be a mistake"
        )

    frontend_processes = []

    # tmp code
    local_world_size = g_parallel_info.world_size
    if "LOCAL_WORLD_SIZE" in os.environ:
        logging.info(
            f"multi rank starts with local world size specified in env: {os.environ['LOCAL_WORLD_SIZE']}"
        )
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        logging.info(
            f"multi rank starts with default local world size: {local_world_size}, world size = {g_parallel_info.world_size}"
        )

    for rank in range(local_world_size):
        for i in range(frontend_server_count):
            process = multiprocessing.Process(
                target=start_frontend_server,
                args=(rank, i, global_controller, py_env_configs),
                name=f"frontend_server_{i}",
            )
            frontend_processes.append(process)
            process.start()

    retry_interval_seconds = 5
    start_port = py_env_configs.server_config.start_port

    while True:
        if not all(proc.is_alive() for proc in frontend_processes):
            logging.error("Frontend server is not alive")
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
    py_env_configs: PyEnvConfigs = setup_args()
    setup_and_configure_server(py_env_configs)
    start_server(py_env_configs)


def start_server(py_env_configs):
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logging.warning(str(e))

    global_controller = init_controller(
        py_env_configs.concurrency_config, dp_size=g_parallel_info.dp_size
    )

    # Create process manager with config values
    process_manager = ProcessManager(
        shutdown_timeout=py_env_configs.server_config.shutdown_timeout,
        monitor_interval=py_env_configs.server_config.monitor_interval,
    )

    # Initialize backend_process to None in case role_type is FRONTEND
    backend_process = None

    try:
        if (
            py_env_configs.role_config.role_type != RoleType.FRONTEND
            and py_env_configs.vit_config.vit_separation
            != VitSeparation.VIT_SEPARATION_REMOTE
        ):
            logging.info("start vit server")
            vit_process = start_vit_server_impl(py_env_configs)
            process_manager.add_process(vit_process)
            logging.info(f"vit server process = {vit_process}")

        if (
            py_env_configs.role_config.role_type != RoleType.FRONTEND
            and py_env_configs.role_config.role_type != RoleType.VIT
        ):
            # vit and frontend role do not start backend server
            logging.info("start backend server")
            backend_process = start_backend_server_impl(
                global_controller, py_env_configs
            )
            process_manager.add_process(backend_process)

        if py_env_configs.role_config.role_type != RoleType.VIT:
            # vit has its own frontend server
            logging.info("start frontend server")
            frontend_process = start_frontend_server_impl(
                global_controller, py_env_configs
            )
            process_manager.add_processes(frontend_process)

        logging.info(f"后端RPC 服务监听的ip为 0.0.0.0，ip/ip段可自定义为所需范围")
    except Exception as e:
        logging.error(f"start failed, trace: {traceback.format_exc()}")
        # Trigger graceful shutdown on any exception
        process_manager.graceful_shutdown()
    finally:
        process_manager.monitor_and_release_processes()


if __name__ == "__main__":
    main()
