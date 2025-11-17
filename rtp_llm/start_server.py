import logging
import multiprocessing
import os
import signal
import sys
import time
import traceback
import json
import requests
import torch

from rtp_llm.config.py_config_modules import ServerConfig, StaticConfig
from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.ops import ProfilingDebugLoggingConfig, RoleType
from rtp_llm.tools.api.hf_model_helper import get_hf_model_info

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info
from rtp_llm.distribute.gang_info import get_gang_info
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.concurrency_controller import init_controller
from rtp_llm.utils.process_manager import ProcessManager
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.ops import RoleType

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


def start_backend_server_impl(
    global_controller, py_env_configs,  process_manager: ProcessManager = None
):
    from rtp_llm.start_backend_server import start_backend_server

    # only for debug
    if py_env_configs.profiling_debug_logging_config.debug_load_server:
        start_backend_server(global_controller, py_env_configs, None)
        os._exit(-1)

    # Create pipe for subprocess startup status communication
    pipe_reader, pipe_writer = multiprocessing.Pipe(duplex=False)

    backend_process = multiprocessing.Process(
        target=start_backend_server,
        args=(global_controller, py_env_configs pipe_writer),
        name="backend_manager",
    )
    backend_process.start()
    pipe_writer.close()  # Parent process closes write end

    # Wait for subprocess to send startup status, maximum 3600 seconds
    max_wait_seconds = 60 * 60
    logging.info(
        f"Waiting for backend server startup status (timeout: {max_wait_seconds}s)..."
    )
    try:
        # 使用 poll 检查是否有数据可读，设置超时
        if pipe_reader.poll(timeout=max_wait_seconds):
            status_msg = pipe_reader.recv()
            if status_msg.get("status") == "success":
                logging.info(
                    f"Backend server started successfully: {status_msg.get('message', '')}"
                )
                return backend_process

            # Startup failed
            error_msg = status_msg.get("message", "Unknown error")
            traceback_info = status_msg.get("traceback", "")
            if traceback_info:
                logging.error(f"Traceback: {traceback_info}")

            # Unified failure handling
            logging.error(f"Backend server failed to start: {error_msg}")
            process_manager.monitor_and_release_processes()
            raise Exception(f"Backend server start failed: {error_msg}")
        else:
            # 超时情况
            logging.error(
                f"Backend server startup timeout after {max_wait_seconds} seconds"
            )
            process_manager.monitor_and_release_processes()
            raise Exception(
                f"Backend server startup timeout after {max_wait_seconds} seconds"
            )
    finally:
        pipe_reader.close()


def start_frontend_server_impl(
    global_controller, backend_process, py_env_configs, process_manager=None
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
            # If connection fails, wait and retry
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

    global_controller = init_controller(py_env_configs.concurrency_config,
                                        dp_size=g_parallel_info.dp_size)
    
    # Create process manager with config values
    process_manager = ProcessManager(
        shutdown_timeout=py_env_configs.server_config.shutdown_timeout,
        monitor_interval=py_env_configs.server_config.monitor_interval
    )

    # Initialize backend_process to None in case role_type is FRONTEND
    backend_process = None
        # Get number of nodes
    try:
        world_info = get_world_info()
        num_nodes = world_info.num_nodes
    except Exception:
        # If get_world_info fails, estimate from world_size
        # Assuming 8 GPUs per node
        num_nodes = (world_size + 7) // 8
        logging.info(
            f"Failed to get world_info, estimated num_nodes={num_nodes} from world_size={world_size}"
        )

    try:
        if py_env_configs.role_config.role_type != RoleType.FRONTEND:
            logging.info("start backend server")
            backend_process = start_backend_server_impl(global_controller, py_env_configs)
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
