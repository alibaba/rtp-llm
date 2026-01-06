import asyncio
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
import traceback

import requests
import torch

from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.ops import ProfilingDebugLoggingConfig, RoleType
from rtp_llm.tools.api.hf_model_helper import get_hf_model_info

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info, g_worker_info, update_worker_info
from rtp_llm.ops import RoleType
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.concurrency_controller import init_controller
from rtp_llm.utils.process_manager import ProcessManager

setup_logging()

def check_server_health(server_port):
    try:
        response = requests.get(f"http://localhost:{server_port}/health", timeout=60)
        if response.status_code == 200 and response.json().get("status", "") == "ok":
            logging.info(
                f"{server_port}/health, response status_code = {response.status_code}, text = {response.text}, len = {len(response.text)}"
            )
            return True
        else:
            return False
    except BaseException as e:
        return False

def start_backend_server_impl(
    global_controller,
    py_env_configs: PyEnvConfigs,
    process_manager: ProcessManager = None,
):
    from rtp_llm.start_backend_server import start_backend_server

    # only for debug
    if py_env_configs.profiling_debug_logging_config.debug_load_server:
        start_backend_server(global_controller, py_env_configs)
        os._exit(-1)

    backend_process = multiprocessing.Process(
        target=start_backend_server,
        args=(global_controller, py_env_configs),
        name="backend_manager",
    )
    backend_process.start()
    process_manager.monitor_and_release_processes()


def start_frontend_server_impl(
    global_controller,
    backend_process,
    py_env_configs: PyEnvConfigs,
    process_manager=None,
):
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

    retry_interval_seconds = 1
    start_port = py_env_configs.server_config.start_port

    # Check ProcessManager availability (includes shutdown signal and process health)
    if process_manager and not process_manager.is_available():
        logging.info("ProcessManager is not available, aborting startup")
        raise Exception("ProcessManager not available during startup")
    
    # Set timeout to 1 hour (3600 seconds)
    timeout_seconds = 3600
    start_time = time.time()
    while not check_server_health(start_port):
        elapsed_time = time.time() - start_time
        if elapsed_time >= timeout_seconds:
            raise TimeoutError(
                f"Frontend server health check timeout after {timeout_seconds} seconds "
                f"({timeout_seconds // 60} minutes). Server at port {start_port} did not become healthy."
            )
        time.sleep(retry_interval_seconds)
    logging.info(f"frontend server is ready")
    return frontend_processes


def main():
    py_env_configs: PyEnvConfigs = setup_args()
    setup_and_configure_server(py_env_configs)
    start_server(py_env_configs)


def start_server(py_env_configs: PyEnvConfigs):
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
    # Get number of nodes
    try:
        world_info = get_world_info(
            py_env_configs.server_config, py_env_configs.distribute_config
        )
        num_nodes = world_info.num_nodes
    except Exception:
        # If get_world_info fails, estimate from world_size
        # Assuming 8 GPUs per node
        num_nodes = (g_parallel_info.world_size + 7) // 8
        logging.info(
            f"Failed to get world_info, estimated num_nodes={num_nodes} from world_size={g_parallel_info.world_size}"
        )

    try:
        if py_env_configs.role_config.role_type != RoleType.FRONTEND:
            logging.info("start backend server")
            backend_process = start_backend_server_impl(
                global_controller, py_env_configs, process_manager
            )
            process_manager.add_process(backend_process)

        logging.info("start frontend server")
        frontend_process = start_frontend_server_impl(
            global_controller, backend_process, py_env_configs, process_manager
        )
        process_manager.add_processes(frontend_process)

        logging.info(
            f"Backend RPC service is listening on 0.0.0.0, IP/IP range can be customized as needed"
        )
    except Exception as e:
        logging.error(f"start failed, trace: {traceback.format_exc()}")
        # Trigger graceful shutdown on any exception
        process_manager.graceful_shutdown()
    finally:
        process_manager.monitor_and_release_processes()


if __name__ == "__main__":
    main()
