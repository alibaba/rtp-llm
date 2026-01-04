import glob
import json
import logging
import logging.config
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import traceback
from multiprocessing import Process
from typing import List, Optional

import torch
import zmq
from setproctitle import setproctitle

from rtp_llm.config.py_config_modules import DistributeConfig, PyEnvConfigs, VitConfig
from rtp_llm.server.backend_manager import BackendManager

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.worker_info import (
    g_parallel_info,
    g_worker_info,
    update_worker_info,
)
from rtp_llm.ops import VitSeparation
from rtp_llm.server.vit_rpc_server import vit_start_server
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    set_global_controller,
)
from rtp_llm.utils.process_manager import ProcessManager
from rtp_llm.utils.util import copy_gemm_config

setup_logging()


def local_rank_start(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    zmq_address: Optional[str] = None,
):
    """Start local rank with proper signal handling for graceful shutdown"""
    backend_manager = None

    def signal_handler(signum, frame):
        logging.info(
            f"Local rank received signal {signum}, shutting down gracefully..."
        )
        if backend_manager is not None:
            try:
                backend_manager.request_shutdown()
            except Exception as e:
                logging.error(f"Error during backend manager shutdown: {e}")

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    copy_gemm_config()

    try:
        # avoid multiprocessing load failed
        update_worker_info(
            py_env_configs.server_config.start_port,
            py_env_configs.server_config.worker_info_port_num,
            py_env_configs.distribute_config.remote_server_port,
        )
        if g_parallel_info.world_size > 1:
            setproctitle(f"rtp_llm_rank-{g_parallel_info.local_rank}")
        logging.info(f"start local {g_worker_info}, {g_parallel_info}")
        set_global_controller(global_controller)
        backend_manager = BackendManager(py_env_configs)
        backend_manager.start()
        logging.info("Backend server initialized successfully, sending ready status")

        # Send startup success message via ZMQ
        if zmq_address is not None:
            try:
                context = zmq.Context()
                push_socket = context.socket(zmq.PUSH)
                push_socket.connect(zmq_address)
                logging.info(f"send ready status to ZMQ address: {zmq_address}")

                message = json.dumps(
                    {
                        "status": "success",
                        "rank_id": g_parallel_info.local_rank,
                        "message": f"rank {g_parallel_info.local_rank} started successfully",
                    }
                )
                push_socket.send_string(message)
                push_socket.close()
                context.term()
                logging.info("Successfully sent ready status via ZMQ")
            except Exception as e:
                logging.warning(f"Failed to send success status via ZMQ: {e}")

        # Enter service loop to keep the process alive
        logging.info("Entering service loop to keep backend_manager alive")
        backend_manager.serve_forever()

    except BaseException as e:
        error_msg = f"start server error: {e}"
        error_trace = traceback.format_exc()
        logging.error(f"{error_msg}, trace: {error_trace}")

        # Send startup failure message via ZMQ
        if zmq_address is not None:
            try:
                context = zmq.Context()
                push_socket = context.socket(zmq.PUSH)
                push_socket.connect(zmq_address)

                message = json.dumps(
                    {
                        "status": "failed",
                        "rank_id": g_parallel_info.local_rank,
                        "message": error_msg,
                        "traceback": error_trace,
                    }
                )
                push_socket.send_string(message)
                push_socket.close()
                context.term()
            except Exception as zmq_error:
                logging.warning(f"Failed to send error status via ZMQ: {zmq_error}")
        raise e


def _get_local_world_size() -> int:
    """Calculate local world size based on environment and hardware"""
    local_world_size = min(torch.cuda.device_count(), g_parallel_info.world_size)
    if "LOCAL_WORLD_SIZE" in os.environ:
        logging.info(
            f"multi rank starts with local world size specified in env: {os.environ['LOCAL_WORLD_SIZE']}"
        )
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        logging.info(
            f"multi rank starts with default local world size: {local_world_size}, "
            f"device count = {torch.cuda.device_count()}, world size = {g_parallel_info.world_size}"
        )
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
    return local_world_size


def _get_cuda_device_list() -> List[str]:
    """Get CUDA device list from environment or hardware detection"""
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    return (
        cuda_devices.split(",")
        if cuda_devices is not None
        else [str(i) for i in range(torch.cuda.device_count())]
    )


def _validate_dp_configuration():
    """Validate data parallelism configuration"""
    if g_parallel_info.dp_size > 1:
        # tp must on one device when dp
        assert g_parallel_info.world_rank % g_parallel_info.tp_size == 0


def _create_rank_processes(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    zmq_address: Optional[str] = None,
):
    """Create and start rank processes, returns processes list"""
    local_world_size = _get_local_world_size()
    cuda_device_list = _get_cuda_device_list()
    _validate_dp_configuration()

    processes = []

    for _, world_rank in enumerate(
        range(g_parallel_info.world_rank, g_parallel_info.world_rank + local_world_size)
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_device_list)
        os.environ["WORLD_RANK"] = str(world_rank)
        proc = Process(
            target=local_rank_start,
            args=(global_controller, py_env_configs, zmq_address),
            name=f"rank-{world_rank}",
        )
        processes.append(proc)
        proc.start()

    return processes


def multi_rank_start(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    zmq_address: Optional[str] = None,
):
    """Start multi-rank backend server with proper process management"""
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logging.warning(str(e))

    # Create ZMQ PULL socket to receive status from rank processes
    context = None
    pull_socket = None
    if zmq_address is not None:
        context = zmq.Context()
        pull_socket = context.socket(zmq.PULL)
        # Find an available port for rank communication
        rank_zmq_port = py_env_configs.server_config.start_port + 6
        rank_zmq_address = f"tcp://127.0.0.1:{rank_zmq_port}"
        pull_socket.bind(rank_zmq_address)
        logging.info(f"ZMQ PULL socket for ranks bound to {rank_zmq_address}")
    else:
        rank_zmq_address = None

    # Create processes
    processes = _create_rank_processes(
        global_controller, py_env_configs, rank_zmq_address
    )
    local_world_size = len(processes)

    if py_env_configs.distribute_config.fake_gang_env:
        if pull_socket:
            pull_socket.close()
            context.term()
        return processes

    # Wait for all ranks to report startup status via ZMQ
    logging.info(
        f"Waiting for all {local_world_size} ranks to report startup status..."
    )
    all_success = True
    error_messages = []
    timeout = 3600
    start_time = time.time()
    ready_count = 0
    results = {}

    if pull_socket:
        pull_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout for polling

    while ready_count < local_world_size and (time.time() - start_time) < timeout:
        if pull_socket:
            try:
                message = pull_socket.recv_string(zmq.NOBLOCK)
                status_msg = json.loads(message)
                process_id = status_msg.get("rank_id", ready_count)

                if status_msg.get("status") == "success":
                    logging.info(f"[Parent] Received success from rank-{process_id}")
                    results[process_id] = True
                    ready_count += 1
                else:
                    error_msg = status_msg.get("message", "Unknown error")
                    error_messages.append(f"Rank {process_id}: {error_msg}")
                    results[process_id] = False
                    ready_count += 1
            except zmq.Again:
                # No message available, continue polling
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Error receiving ZMQ message: {e}")
                time.sleep(0.1)
        else:
            # No ZMQ, just wait a bit
            time.sleep(5)

    # Verify results
    all_success = ready_count == local_world_size and all(results.values())

    # Send overall status to parent process via ZMQ
    if zmq_address is not None:
        try:
            push_context = zmq.Context()
            push_socket = push_context.socket(zmq.PUSH)
            push_socket.connect(zmq_address)

            if all_success:
                message = json.dumps(
                    {
                        "status": "success",
                        "message": f"All {local_world_size} backend ranks started successfully",
                    }
                )
                logging.info(f"All {local_world_size} ranks started successfully")
            else:
                error_msg = (
                    "; ".join(error_messages) if error_messages else "Some ranks failed"
                )
                message = json.dumps(
                    {
                        "status": "failed",
                        "message": f"Some ranks failed to start: {error_msg}",
                        "traceback": "",
                    }
                )
                logging.error(f"Some ranks failed: {error_msg}")

            push_socket.send_string(message)
            push_socket.close()
            push_context.term()
        except Exception as e:
            logging.warning(f"Failed to send status via ZMQ to parent: {e}")

    if not all_success:
        # Terminate all processes if any rank failed
        logging.error("Terminating all ranks due to startup failures")
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            proc.join(timeout=5)
        error_msg = "; ".join(error_messages) if error_messages else "Unknown error"
        raise Exception(f"Multi-rank startup failed: {error_msg}")

    # Clean up ZMQ socket
    if pull_socket:
        pull_socket.close()
    if context:
        context.term()

    # After successful startup, monitor processes
    manager = ProcessManager(
        shutdown_timeout=py_env_configs.server_config.shutdown_timeout,
        monitor_interval=py_env_configs.server_config.monitor_interval,
    )
    manager.set_processes(processes)
    manager.monitor_and_release_processes()

    return processes


def load_gpu_nic_affinity():
    if os.environ.get("ACCL_NIC_GPU_AFFINITY") != None:
        return True
    # 检查 /usr/local/bin/run_affinity 是否存在
    run_affinity_path = "/usr/local/bin/run_affinity"
    if not os.path.exists(run_affinity_path):
        logging.info(f"get gpu nic affinity failed, {run_affinity_path} not exist")
        return False

    try:
        # 执行 run_affinity 文件
        result = subprocess.run(
            [run_affinity_path],
            check=True,  # 如果返回非零退出码则抛出异常
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        # 执行失败
        logging.info(
            f"get gpu nic affinity failed, run {run_affinity_path} failed, exception is {e}"
        )
        return False

    # 检查当前目录是否存在 npu_nic_affinity.json
    json_path = "npu_nic_affinity.json"
    if not os.path.exists(json_path):
        logging.info(f"get gpu nic affinity failed, {json_path} 不存在")
        return False

    try:
        # 读取 JSON 文件内容
        with open(json_path, "r") as f:
            content = f.read().strip()  # 读取并去除首尾空白
        # 将内容存入环境变量
        os.environ["ACCL_NIC_GPU_AFFINITY"] = content
        logging.info(
            f"get gpu nic affinity success, set env ACCL_NIC_GPU_AFFINITY to {content}"
        )
        return True
    except Exception as e:
        logging.info(
            f"get gpu nic affinity failed, load {json_path} failed, exception is {e}"
        )
        return False


def clear_jit_filelock():
    # check whether exists jit dir
    if os.path.exists("deep_gemm_runtime"):
        files = glob.glob("./deep_gemm_runtime/**/*_lock", recursive=True)
        for file in files:
            os.remove(file)


def start_backend_server(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    zmq_address: Optional[str] = None,
):
    setproctitle("rtp_llm_backend_server")
    os.makedirs("logs", exist_ok=True)
    load_gpu_nic_affinity()

    clear_jit_filelock()

    update_worker_info(
        py_env_configs.server_config.start_port,
        py_env_configs.server_config.worker_info_port_num,
        py_env_configs.distribute_config.remote_server_port,
    )

    # TODO(xinfei.sxf) fix this
    if py_env_configs.vit_config.vit_separation == VitSeparation.VIT_SEPARATION_ROLE:
        return vit_start_server()

    if not torch.cuda.is_available():
        return local_rank_start(global_controller, py_env_configs, zmq_address)

    if (
        g_parallel_info.world_size % torch.cuda.device_count() != 0
        and g_parallel_info.world_size > torch.cuda.device_count()
    ):
        raise Exception(
            f"result: {g_parallel_info.world_size % torch.cuda.device_count()} \
            not support WORLD_SIZE {g_parallel_info.world_size} for {torch.cuda.device_count()} local gpu"
        )

    if torch.cuda.device_count() > 1 and g_parallel_info.world_size > 1:
        return multi_rank_start(global_controller, py_env_configs, zmq_address)
    else:
        return local_rank_start(global_controller, py_env_configs, zmq_address)


def main():
    return start_backend_server(None)


if __name__ == "__main__":
    main()
