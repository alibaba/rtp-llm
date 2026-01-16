import glob
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
from typing import List

import torch
from setproctitle import setproctitle

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import (
    set_parallelism_config,
    setup_cuda_device_and_accl_env,
)
from rtp_llm.ops import VitSeparation
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    set_global_controller,
)
from rtp_llm.utils.process_manager import ProcessManager

setup_logging()


def local_rank_start(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    world_rank: int = 0,
    pipe_writer=None,
):
    """Start local rank with proper signal handling for graceful shutdown"""
    backend_manager = None
    logging.info(f"[PROCESS_START]Start local rank process")
    start_time = time.time()
    from rtp_llm.server.backend_manager import BackendManager
    from rtp_llm.utils.util import copy_gemm_config

    logging.info(f"import BackendManager took {time.time()- start_time:.2f}s")

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
        set_parallelism_config(
            py_env_configs.parallelism_config,
            world_rank,
            py_env_configs.ffn_disaggregate_config,
            py_env_configs.prefill_cp_config,
        )
        py_env_configs.server_config.set_local_rank(
            py_env_configs.parallelism_config.local_rank
        )
        py_env_configs.distribute_config.set_local_rank(
            py_env_configs.parallelism_config.local_rank, world_rank
        )
        setup_cuda_device_and_accl_env(world_rank)
        if py_env_configs.parallelism_config.world_size > 1:
            setproctitle(f"rtp_llm_rank-{py_env_configs.parallelism_config.local_rank}")
        set_global_controller(global_controller)
        backend_manager = BackendManager(py_env_configs)
        backend_manager.start()
        logging.info("Backend server initialized successfully, sending ready status")

        # Send startup success message
        if pipe_writer is not None:
            try:
                pipe_writer.send(
                    {
                        "status": "success",
                        "message": f"Backend server started successfully on rank {py_env_configs.parallelism_config.local_rank}",
                    }
                )
                pipe_writer.close()
            except Exception as e:
                logging.warning(f"Failed to send success status via pipe: {e}")

        # Enter service loop to keep the process alive
        logging.info("Entering service loop to keep backend_manager alive")
        backend_manager.serve_forever()

    except BaseException as e:
        error_msg = f"start server error: {e}"
        error_trace = traceback.format_exc()
        logging.error(f"{error_msg}, trace: {error_trace}")

        # Send startup failure message
        if pipe_writer is not None:
            try:
                pipe_writer.send(
                    {"status": "failed", "message": error_msg, "traceback": error_trace}
                )
                pipe_writer.close()
            except Exception as pipe_error:
                logging.warning(f"Failed to send error status via pipe: {pipe_error}")
        raise e


def _get_local_world_size(py_env_configs: PyEnvConfigs) -> int:
    """Calculate local world size based on environment and hardware"""
    world_size = py_env_configs.parallelism_config.world_size
    local_world_size = min(torch.cuda.device_count(), world_size)
    if "LOCAL_WORLD_SIZE" in os.environ:
        logging.info(
            f"multi rank starts with local world size specified in env: {os.environ['LOCAL_WORLD_SIZE']}"
        )
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        logging.info(
            f"multi rank starts with default local world size: {local_world_size}, "
            f"device count = {torch.cuda.device_count()}, world size = {world_size}"
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


def _validate_dp_configuration(py_env_configs: PyEnvConfigs):
    """Validate data parallelism configuration"""
    pc = py_env_configs.parallelism_config
    if pc.dp_size > 1:
        # tp must on one device when dp
        assert pc.world_rank % pc.tp_size == 0


def _create_rank_processes(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
):
    """Create and start rank processes, returns (processes, rank_pipe_readers)"""
    pc = py_env_configs.parallelism_config
    local_world_size = _get_local_world_size(py_env_configs)
    cuda_device_list = _get_cuda_device_list()
    _validate_dp_configuration(py_env_configs)

    processes = []
    rank_pipe_readers = []  # Store pipe readers for each rank

    for _, world_rank in enumerate(
        range(pc.world_rank, pc.world_rank + local_world_size)
    ):
        reader, writer = multiprocessing.Pipe(duplex=False)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_device_list)
        os.environ["WORLD_RANK"] = str(world_rank)

        proc = Process(
            target=local_rank_start,
            args=(global_controller, py_env_configs, world_rank, writer),
            name=f"rank-{world_rank}",
        )
        proc.start()
        writer.close()  # Parent process closes write end
        processes.append(proc)
        rank_pipe_readers.append(reader)

    return processes, rank_pipe_readers


def multi_rank_start(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    pipe_writer=None,
):
    """Start multi-rank backend server with proper process management"""
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logging.warning(str(e))

    # Create processes and get pipe readers
    processes, rank_pipe_readers = _create_rank_processes(
        global_controller, py_env_configs
    )
    local_world_size = len(processes)

    if py_env_configs.distribute_config.fake_gang_env:
        return processes

    # Wait for all ranks to report startup status via pipe
    logging.info(
        f"Waiting for all {local_world_size} ranks to report startup status..."
    )
    all_success = True
    error_messages = []

    for i, reader in enumerate(rank_pipe_readers):
        try:
            data = reader.recv()  # Block and wait for status from each rank
            if data.get("status") == "success":
                logging.info(
                    f"Rank {i} started successfully: {data.get('message', '')}"
                )
            else:
                all_success = False
                error_msg = data.get("message", "Unknown error")
                error_messages.append(f"Rank {i}: {error_msg}")
                traceback_info = data.get("traceback", "")
                if traceback_info:
                    logging.error(f"Rank {i} traceback: {traceback_info}")
        except Exception as e:
            all_success = False
            error_messages.append(f"Rank {i}: Failed to receive status - {e}")
            logging.error(f"Failed to receive status from rank {i}: {e}")
        finally:
            reader.close()

    # Report overall status via external pipe
    if pipe_writer is not None:
        try:
            if all_success:
                pipe_writer.send(
                    {
                        "status": "success",
                        "message": f"All {local_world_size} backend ranks started successfully",
                    }
                )
                logging.info(f"All {local_world_size} ranks started successfully")
            else:
                error_msg = "; ".join(error_messages)
                pipe_writer.send(
                    {
                        "status": "failed",
                        "message": f"Some ranks failed to start: {error_msg}",
                        "traceback": "",
                    }
                )
                logging.error(f"Some ranks failed: {error_msg}")
            pipe_writer.close()
        except Exception as e:
            logging.warning(f"Failed to send status via pipe: {e}")

    if not all_success:
        # Terminate all processes if any rank failed
        logging.error("Terminating all ranks due to startup failures")
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            proc.join(timeout=5)
        raise Exception(f"Multi-rank startup failed: {'; '.join(error_messages)}")

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
    pipe_writer=None,
):
    logging.info(f"[PROCESS_START]Start backend server process")
    setproctitle("rtp_llm_backend_server")
    os.makedirs("logs", exist_ok=True)
    load_gpu_nic_affinity()

    clear_jit_filelock()

    if py_env_configs.vit_config.vit_separation == VitSeparation.VIT_SEPARATION_ROLE:
        from rtp_llm.server.vit_rpc_server import vit_start_server

        return vit_start_server()

    if not torch.cuda.is_available():
        return local_rank_start(global_controller, py_env_configs)

    pc = py_env_configs.parallelism_config
    if (
        pc.world_size % torch.cuda.device_count() != 0
        and pc.world_size > torch.cuda.device_count()
    ):
        raise Exception(
            f"result: {pc.world_size % torch.cuda.device_count()} \
            not support WORLD_SIZE {pc.world_size} for {torch.cuda.device_count()} local gpu"
        )

    if torch.cuda.device_count() > 1 and pc.world_size > 1:
        return multi_rank_start(global_controller, py_env_configs, pipe_writer)
    else:
        return local_rank_start(global_controller, py_env_configs, 0, pipe_writer)


def main():
    return start_backend_server(None)


if __name__ == "__main__":
    main()
