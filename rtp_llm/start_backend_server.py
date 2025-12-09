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

from rtp_llm.config.py_config_modules import GangConfig, PyEnvConfigs, VitConfig

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.distribute.worker_info import g_parallel_info, g_worker_info
from rtp_llm.server.backend_app import BackendApp
from rtp_llm.server.vit_rpc_server import vit_start_server
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    set_global_controller,
)
from rtp_llm.utils.process_manager import ProcessManager
from rtp_llm.utils.util import copy_gemm_config


def local_rank_start(global_controller: ConcurrencyController):
    """Start local rank with proper signal handling for graceful shutdown"""
    app = None

    def signal_handler(signum, frame):
        logging.info(
            f"Local rank received signal {signum}, shutting down gracefully..."
        )
        if app and hasattr(app, "shutdown"):
            try:
                app.shutdown()
            except Exception as e:
                logging.error(f"Error during app shutdown: {e}")

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    copy_gemm_config()
    ## collect all args and envs.
    py_env_configs = PyEnvConfigs()
    py_env_configs.update_from_env()
    try:
        # avoid multiprocessing load failed
        # reload for multiprocessing.start_method == fork
        g_parallel_info.reload()
        g_worker_info.reload()
        if g_parallel_info.world_size > 1:
            setproctitle(f"rtp_llm_rank-{g_parallel_info.local_rank}")
        logging.info(f"start local {g_worker_info}, {g_parallel_info}")
        set_global_controller(global_controller)
        app = BackendApp(py_env_configs)
        app.start(g_worker_info)
    except BaseException as e:
        logging.error(f"start server error: {e}, trace: {traceback.format_exc()}")
        raise e

    if not torch.cuda.is_available():
        logging.info("GPU not found: using CPU")


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


def _create_rank_processes(global_controller: ConcurrencyController) -> List[Process]:
    """Create and start rank processes"""
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
            args=(global_controller,),
            name=f"rank-{world_rank}",
        )
        proc.start()
        processes.append(proc)

    return processes


def multi_rank_start(global_controller: ConcurrencyController):
    """Start multi-rank backend server with proper process management"""
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logging.warn(str(e))

    # Create processes
    processes = _create_rank_processes(global_controller)

    # Check for fake gang environment
    gang_config = GangConfig()
    gang_config.update_from_env()
    if gang_config.fake_gang_env:
        return processes

    # Create manager and monitor processes
    manager = ProcessManager()
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


def start_backend_server(global_controller: ConcurrencyController):
    setproctitle("rtp_llm_backend_server")
    os.makedirs("logs", exist_ok=True)
    load_gpu_nic_affinity()

    clear_jit_filelock()

    ## collect all args and envs.
    vit_config = VitConfig()
    vit_config.update_from_env()
    # TODO(xinfei.sxf) fix this
    if vit_config.vit_separation == 1:
        return vit_start_server()

    if not torch.cuda.is_available():
        return local_rank_start(global_controller)

    if (
        g_parallel_info.world_size % torch.cuda.device_count() != 0
        and g_parallel_info.world_size > torch.cuda.device_count()
    ):
        raise Exception(
            f"result: {g_parallel_info.world_size % torch.cuda.device_count()} \
            not support WORLD_SIZE {g_parallel_info.world_size} for {torch.cuda.device_count()} local gpu"
        )

    if torch.cuda.device_count() > 1 and g_parallel_info.world_size > 1:
        return multi_rank_start(global_controller)
    else:
        return local_rank_start(global_controller)


def main():
    return start_backend_server(None)


if __name__ == "__main__":
    main()
