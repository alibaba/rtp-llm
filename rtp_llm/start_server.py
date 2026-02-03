import logging
import multiprocessing
import os
import sys
import time
import traceback

import requests

from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.utils.time_util import timer_wrapper

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.distribute.worker_info import WorkerInfo
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


@timer_wrapper(description="start backend server")
def start_backend_server_impl(
    global_controller,
    py_env_configs: PyEnvConfigs,
    worker_info: WorkerInfo,
    process_manager: ProcessManager = None,
):
    from rtp_llm.start_backend_server import start_backend_server

    # only for debug
    if py_env_configs.profiling_debug_logging_config.debug_load_server:
        start_backend_server(global_controller, py_env_configs, worker_info, None)
        os._exit(-1)

    # Create pipe for subprocess startup status communication
    pipe_reader, pipe_writer = multiprocessing.Pipe(duplex=False)
    logging.info(f"[PROCESS_SPAWN]Start backend server process outer")

    backend_process = multiprocessing.Process(
        target=start_backend_server,
        args=(global_controller, py_env_configs, worker_info, pipe_writer),
        name="backend_manager",
    )
    backend_process.start()
    pipe_writer.close()  # Parent process closes write end

    # Create check_ready_fn for pipe-based health check
    max_wait_seconds = 60 * 60
    startup_status = {"ready": False, "error": None}

    def check_backend_ready():
        """Check if backend server is ready via pipe communication"""
        if startup_status["ready"]:
            return True

        if startup_status["error"]:
            raise Exception(startup_status["error"])

        # Non-blocking check if data is available
        if pipe_reader.poll(timeout=0):
            try:
                status_msg = pipe_reader.recv()
                if status_msg.get("status") == "success":
                    logging.info(
                        f"Backend server started successfully: {status_msg.get('message', '')}"
                    )
                    startup_status["ready"] = True
                    pipe_reader.close()
                    return True
                else:
                    # Startup failed
                    error_msg = status_msg.get("message", "Unknown error")
                    traceback_info = status_msg.get("traceback", "")
                    if traceback_info:
                        logging.error(f"Traceback: {traceback_info}")

                    error = f"Backend server start failed: {error_msg}"
                    startup_status["error"] = error
                    pipe_reader.close()
                    raise Exception(error)
            except EOFError:
                error = "Backend server pipe closed unexpectedly"
                startup_status["error"] = error
                pipe_reader.close()
                raise Exception(error)

        return False

    # Register health check with ProcessManager using custom check_ready_fn
    if process_manager:
        process_manager.register_health_check(
            processes=[backend_process],
            process_name="backend_server",
            check_ready_fn=check_backend_ready,
            retry_interval_seconds=0.1,
        )

    return backend_process


@timer_wrapper(description="start frontend server")
def start_frontend_server_impl(
    global_controller,
    py_env_configs: PyEnvConfigs,
    worker_info: WorkerInfo,
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
    parallelism_config = py_env_configs.parallelism_config
    local_world_size = parallelism_config.world_size
    if "LOCAL_WORLD_SIZE" in os.environ:
        logging.info(
            f"multi rank starts with local world size specified in env: {os.environ['LOCAL_WORLD_SIZE']}"
        )
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        logging.info(
            f"multi rank starts with default local world size: {local_world_size}, world size = {parallelism_config.world_size}"
        )

    # To reduce the number of frontend servers, we only start those with tp_rank=0;
    # however, since k8s needs to check machine heartbeat, rank 0 on each machine also needs to be started.
    for rank in range(local_world_size):
        for i in range(frontend_server_count):
            if (
                rank == 0
                or (parallelism_config.world_rank + rank) % parallelism_config.tp_size
                == 0
            ):
                logging.info(
                    f"[PROCESS_SPAWN]Start frontend server process rank_{rank}_server_{i} outer"
                )
                process = multiprocessing.Process(
                    target=start_frontend_server,
                    args=(rank, i, global_controller, py_env_configs, worker_info),
                    name=f"frontend_server_{i}",
                )
                frontend_processes.append(process)
                process.start()
            else:
                logging.info(
                    f"rank {parallelism_config.world_rank + rank} skipping frontend startup"
                    f"rank {parallelism_config.world_rank + rank} skipping frontend startup"
                )

    if process_manager and frontend_processes:
        # Register health check with ProcessManager for the first frontend server
        def check_frontend_ready():
            return check_server_health(py_env_configs.server_config.start_port)

        process_manager.register_health_check(
            processes=frontend_processes,
            process_name="frontend_server",
            check_ready_fn=check_frontend_ready,
            retry_interval_seconds=0.1,
        )

    return frontend_processes


def main():
    py_env_configs: PyEnvConfigs = setup_args()

    worker_info = WorkerInfo.from_parallelism_config(
        py_env_configs.parallelism_config,
        start_port=py_env_configs.server_config.start_port,
        remote_server_port=py_env_configs.distribute_config.remote_server_port,
        worker_info_port_num=py_env_configs.server_config.worker_info_port_num,
    )

    setup_and_configure_server(py_env_configs, worker_info)
    start_server(py_env_configs, worker_info)


def start_server(py_env_configs: PyEnvConfigs, worker_info: WorkerInfo):
    logging.info(f"[PROCESS_START]Start server")
    start_time = time.time()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logging.warning(str(e))

    parallelism_config = py_env_configs.parallelism_config

    global_controller = init_controller(
        py_env_configs.concurrency_config, dp_size=parallelism_config.dp_size
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
            py_env_configs.server_config,
            py_env_configs.distribute_config,
            py_env_configs.parallelism_config,
            worker_info=worker_info,
        )
        num_nodes = world_info.num_nodes
    except Exception:
        # If get_world_info fails, estimate from world_size
        # Assuming 8 GPUs per node
        num_nodes = (parallelism_config.world_size + 7) // 8
        logging.info(
            f"Failed to get world_info, estimated num_nodes={num_nodes} from world_size={parallelism_config.world_size}"
        )

    try:
        if py_env_configs.role_config.role_type != RoleType.FRONTEND:
            logging.info("start backend server")
            backend_process = start_backend_server_impl(
                global_controller, py_env_configs, worker_info, process_manager
            )
            process_manager.add_process(backend_process)

        logging.info("start frontend server")
        frontend_process = start_frontend_server_impl(
            global_controller, py_env_configs, worker_info, process_manager
        )
        process_manager.add_processes(frontend_process)

        # Start parallel health checks and wait for completion
        if not process_manager.run_health_checks():
            logging.error("Health checks failed")
            raise Exception("Health checks failed")

        logging.info(
            f"Backend RPC service is listening on 0.0.0.0, IP/IP range can be customized as needed"
        )
        consume_s = time.time() - start_time
        logging.info(f"start server took {consume_s:.2f}s")
    except Exception as e:
        logging.error(f"start failed, trace: {traceback.format_exc()}")
        # Trigger graceful shutdown on any exception
        process_manager.graceful_shutdown()
    finally:
        process_manager.monitor_and_release_processes()


if __name__ == "__main__":
    main()
