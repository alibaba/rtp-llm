import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time
import traceback

import requests

from rtp_llm.config.py_config_modules import ServerConfig, StaticConfig
from rtp_llm.metrics import kmonitor
from rtp_llm.ops import ProfilingDebugLoggingConfig
from rtp_llm.tools.api.hf_model_helper import get_hf_model_info

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info
from rtp_llm.server.server_args.server_args import EnvArgumentParser, setup_args
from rtp_llm.utils.concurrency_controller import init_controller


class ProcessManager:
    """Manages server processes and handles graceful shutdown"""

    def __init__(self):
        self.backend_process = None
        self.frontend_processes = []
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully"""
        logging.info(f"Received signal {signum}, starting graceful shutdown...")
        self._graceful_shutdown()
        sys.exit(0)

    def _graceful_shutdown(self, timeout=30):
        """Perform graceful shutdown of all managed processes"""
        all_processes = []
        if self.backend_process:
            all_processes.append(self.backend_process)
        all_processes.extend(self.frontend_processes)

        if not all_processes:
            return

        # Send SIGTERM to all processes
        for proc in all_processes:
            if proc.is_alive():
                logging.info(f"Sending SIGTERM to process {proc.pid}")
                proc.terminate()

        # Wait for graceful shutdown
        start_time = time.time()
        while (
            any(proc.is_alive() for proc in all_processes)
            and (time.time() - start_time) < timeout
        ):
            time.sleep(1)

        # Force kill remaining processes
        for proc in all_processes:
            if proc.is_alive():
                logging.warning(f"Force killing process {proc.pid}")
                proc.kill()

        logging.info("Graceful shutdown completed")

    def set_backend_process(self, process):
        """Set the backend process"""
        self.backend_process = process

    def set_frontend_processes(self, processes):
        """Set the frontend processes"""
        self.frontend_processes = processes if processes else []

    def monitor_and_release_processes(self):
        """Monitor all managed processes and handle failures"""
        all_processes = []
        if self.backend_process:
            all_processes.append(self.backend_process)
        all_processes.extend(self.frontend_processes)

        logging.info(f"all process = {all_processes}")

        while any(proc.is_alive() for proc in all_processes):
            if not all(proc.is_alive() for proc in all_processes):
                logging.error(f"server monitor : some process is not alive, exit!")
                for proc in all_processes:
                    try:
                        proc.terminate()
                    except Exception as e:
                        logging.error(
                            f"catch exception when process terminate : {str(e)}"
                        )
            time.sleep(1)

        # Join all processes
        for proc in all_processes:
            proc.join()

        logging.info("all process exit")


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


def start_backend_server_impl(global_controller):
    from rtp_llm.start_backend_server import start_backend_server

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
    from rtp_llm.start_frontend_server import start_frontend_server

    server_config = ServerConfig()
    server_config.update_from_env()
    frontend_server_count = server_config.frontend_server_count
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
                args=(rank, i, global_controller),
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


def get_model_type_and_update_env(parser: EnvArgumentParser, args: argparse.Namespace):
    if (
        hasattr(args, "checkpoint_path")
        and args.checkpoint_path is not None
        and args.checkpoint_path != ""
    ):
        model_path = args.checkpoint_path
        current_model_type = os.environ.get(
            "MODEL_TYPE", StaticConfig.model_config.model_type
        )
        if current_model_type is None or current_model_type == "":
            if (
                hasattr(args, "model_type")
                and args.model_type is not None
                and args.model_type != ""
            ):
                config_model_type = args.model_type
            else:
                model_info = get_hf_model_info(model_path)
                config_model_type = model_info.ft_model_type
                setattr(args, "model_type", config_model_type)
            if config_model_type is not None and config_model_type != "":
                EnvArgumentParser.update_env_from_args(parser, "model_type", args)
    StaticConfig.update_from_env()


def main():
    parser, args = setup_args()

    start_server(parser, args)


def start_server(parser: EnvArgumentParser, args: argparse.Namespace):
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logging.warn(str(e))

    global_controller = init_controller()
    process_manager = ProcessManager()
    get_model_type_and_update_env(parser, args)
    try:
        backend_process = None
        frontend_process = None

        if os.environ.get("ROLE_TYPE", "") != "FRONTEND":
            logging.info("start backend server")
            backend_process = start_backend_server_impl(global_controller)
            process_manager.set_backend_process(backend_process)
            logging.info(f"backend server process = {backend_process}")

        logging.info("start frontend server")
        frontend_process = start_frontend_server_impl(
            global_controller, backend_process
        )
        process_manager.set_frontend_processes(frontend_process)
        logging.info(f"frontend server process = {frontend_process}")

        logging.info(f"后端RPC 服务监听的ip为 0.0.0.0，ip/ip段可自定义为所需范围")
    except Exception as e:
        logging.error(f"start failed, trace: {traceback.format_exc()}")
    finally:
        process_manager.monitor_and_release_processes()


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    main()
