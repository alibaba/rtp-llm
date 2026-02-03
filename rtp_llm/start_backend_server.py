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
from rtp_llm.distribute.worker_info import WorkerInfo
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
    worker_info: WorkerInfo,
    pipe_writer=None,
):
    """
    Start local rank with proper signal handling for graceful shutdown.

    Args:
        global_controller: Concurrency controller
        py_env_configs: Environment configurations
        worker_info: Worker information (contains world_rank and local_world_size)
        pipe_writer: Optional pipe writer for status communication
    """
    # WORLD_RANK environment variable is already set before process spawn
    # This is needed because setup_logging() is called during module import in subprocess
    # and it reads WORLD_RANK to configure log file names

    # Get world_rank and local_world_size from worker_info
    # CRITICAL: Verify worker_info is correctly set, with fallback to environment variables
    world_rank = worker_info.world_rank
    local_world_size = worker_info.local_world_size

    # Defensive check: If worker_info values are invalid, try to recover from environment
    # This ensures rank configuration is always correct even if worker_info serialization fails
    env_world_rank = os.environ.get("WORLD_RANK")
    env_local_world_size = os.environ.get("LOCAL_WORLD_SIZE")

    if world_rank is None or world_rank < 0:
        if env_world_rank is not None:
            world_rank = int(env_world_rank)
            worker_info.world_rank = world_rank
            logging.warning(
                f"worker_info.world_rank was invalid, recovered from environment: {world_rank}"
            )
        else:
            raise ValueError(
                f"worker_info.world_rank is invalid ({world_rank}) and WORLD_RANK environment variable is not set"
            )

    if local_world_size is None or local_world_size <= 0:
        if env_local_world_size is not None:
            local_world_size = int(env_local_world_size)
            worker_info.local_world_size = local_world_size
            logging.warning(
                f"worker_info.local_world_size was invalid, recovered from environment: {local_world_size}"
            )
        else:
            raise ValueError(
                f"worker_info.local_world_size is invalid ({local_world_size}) and LOCAL_WORLD_SIZE environment variable is not set"
            )

    # Ensure environment variables are set in this process (defensive programming)
    # Even though they should be inherited from parent, setting them here ensures
    # they are available for any code that runs before the parent's env is fully propagated
    os.environ["WORLD_RANK"] = str(world_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

    # Final verification: Log the final configuration for debugging
    logging.info(
        f"[PROCESS_START]Rank configuration verified: world_rank={world_rank}, "
        f"local_rank={worker_info.local_rank}, local_world_size={local_world_size}, "
        f"worker_info.world_rank={worker_info.world_rank}, worker_info.local_world_size={worker_info.local_world_size}"
    )

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
        # Update parallelism_config with the world_rank for this process
        parallelism_config = py_env_configs.parallelism_config

        logging.info(
            f"local_rank_start, before update_worker_info, {parallelism_config.to_string()}, worker_info: {worker_info}"
        )

        _update_parallelism_config_from_world_rank(
            parallelism_config, world_rank=world_rank
        )

        # Configure worker_info ports first (offset by local_rank * worker_info_port_num)
        # so each rank gets distinct ports; then log the final worker_info
        worker_info.configure_ports(
            parallelism_config.local_rank,
            parallelism_config.world_rank,
            py_env_configs.server_config.start_port,
            py_env_configs.distribute_config.remote_server_port,
            py_env_configs.server_config.worker_info_port_num,
            local_world_size,
        )

        logging.info(
            f"local_rank_start, after update_worker_info, {parallelism_config.to_string()}, worker_info: {worker_info}"
        )

        if parallelism_config.world_size > 1:
            setproctitle(f"rtp_llm_rank-{parallelism_config.local_rank}")
        logging.info(f"start local {worker_info}, {parallelism_config}")
        set_global_controller(global_controller)
        backend_manager = BackendManager(py_env_configs, worker_info)
        backend_manager.start()
        logging.info("Backend server initialized successfully, sending ready status")

        # Send startup success message
        if pipe_writer is not None:
            try:
                pipe_writer.send(
                    {
                        "status": "success",
                        "message": f"Backend server started successfully on rank {parallelism_config.local_rank}",
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


def _update_parallelism_config_from_world_rank(parallelism_config, world_rank):
    """
    Update parallelism_config with per-process values based on world_rank.

    Only updates values that depend on WORLD_RANK:
    - world_rank (from parameter or env var as fallback)
    - tp_rank, dp_rank, ep_rank, local_rank, ffn_tp_rank (calculated from world_rank)

    Other values (world_size, tp_size, dp_size, pp_size, ep_size, local_world_size, ffn_sp_size)
    should already be set correctly in start_server.py.

    Args:
        parallelism_config: ParallelismConfig instance to update
        world_rank: World rank for this process. If None, will try to read from
                   parallelism_config.world_rank, or calculate from world_index.
        world_index: Optional world index (node index). If provided and world_rank is None,
                    calculates world_rank as world_index * local_world_size.
    """
    # Update world_rank
    parallelism_config.world_rank = world_rank

    parallelism_config.tp_rank = world_rank % parallelism_config.tp_size
    parallelism_config.dp_rank = world_rank // parallelism_config.tp_size
    parallelism_config.ep_rank = world_rank % parallelism_config.ep_size
    parallelism_config.local_rank = world_rank % parallelism_config.local_world_size
    parallelism_config.ffn_tp_rank = (
        parallelism_config.tp_rank % parallelism_config.ffn_tp_size
    )


def _get_local_world_size(parallelism_config, local_world_size_override=None) -> int:
    """
    Calculate local world size based on parallelism_config and hardware.

    Args:
        parallelism_config: ParallelismConfig instance (already has local_world_size set in start_server.py)
        local_world_size_override: Optional override value. If None, uses parallelism_config.local_world_size
                                  or calculates from hardware.

    Returns:
        Local world size for this node
    """
    if local_world_size_override is not None:
        local_world_size = local_world_size_override
        logging.info(f"Using provided local_world_size: {local_world_size}")
    elif (
        parallelism_config.local_world_size is not None
        and parallelism_config.local_world_size > 0
    ):
        local_world_size = parallelism_config.local_world_size
        logging.info(
            f"Using local_world_size from parallelism_config: {local_world_size}"
        )
    else:
        # local_world_size should be the number of ranks in the local node
        # Use min(device_count, world_size) as the original logic before refactoring
        device_count = (
            torch.cuda.device_count()
            if torch.cuda.is_available()
            else parallelism_config.world_size
        )
        local_world_size = min(device_count, parallelism_config.world_size)
        logging.info(
            f"Calculated local world size: {local_world_size}, "
            f"device count = {device_count}, world size = {parallelism_config.world_size}"
        )
    return local_world_size


def _get_cuda_device_list(cuda_visible_devices=None) -> List[str]:
    """
    Get CUDA device list from parameter or hardware detection.

    Args:
        cuda_visible_devices: Optional comma-separated string of device IDs (e.g., "0,1,2,3").
                             If None, uses all available devices.

    Returns:
        List of device ID strings
    """
    if cuda_visible_devices is not None:
        return cuda_visible_devices.split(",")
    else:
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return [str(i) for i in range(device_count)]


def _validate_dp_configuration(parallelism_config):
    """Validate data parallelism configuration"""
    if parallelism_config.dp_size > 1:
        # tp must on one device when dp
        assert parallelism_config.world_rank % parallelism_config.tp_size == 0


def _create_rank_processes(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    worker_info: WorkerInfo,
    cuda_visible_devices=None,
    local_world_size_override=None,
):
    """
    Create and start rank processes, returns (processes, rank_pipe_readers)

    Args:
        global_controller: Concurrency controller
        py_env_configs: Environment configurations
        worker_info: Worker information
        cuda_visible_devices: Optional CUDA device list string (e.g., "0,1,2,3").
                             If None, will be detected from environment or hardware.
        local_world_size_override: Optional override for local_world_size.

    Returns:
        Tuple of (processes list, rank_pipe_readers list)
    """
    parallelism_config = py_env_configs.parallelism_config
    local_world_size = _get_local_world_size(
        parallelism_config, local_world_size_override
    )

    # Get CUDA device list from parameter or environment (for backward compatibility)
    # Note: We still check environment as fallback for external callers
    if cuda_visible_devices is None:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    cuda_device_list = _get_cuda_device_list(cuda_visible_devices)

    _validate_dp_configuration(parallelism_config)

    processes = []
    rank_pipe_readers = []  # Store pipe readers for each rank

    if len(cuda_device_list) > local_world_size:
        # Limit to local_world_size devices to match DeepEP's strict equality requirement
        actual_device_list = cuda_device_list[:local_world_size]
        logging.info(
            f"Limiting CUDA_VISIBLE_DEVICES from {len(cuda_device_list)} to {local_world_size} devices for DeepEP compatibility"
        )
    else:
        # Use full list if it's already <= local_world_size
        actual_device_list = cuda_device_list

    for idx, world_rank in enumerate(
        range(
            parallelism_config.world_rank,
            parallelism_config.world_rank + local_world_size,
        )
    ):
        reader, writer = multiprocessing.Pipe(duplex=False)

        # Update worker_info with rank-specific values before spawning
        # Note: In spawn mode, each process gets a serialized copy of worker_info,
        # so updating it here is safe - each process will have its own copy with correct values
        local_rank = world_rank % local_world_size
        worker_info.world_rank = world_rank
        worker_info.local_rank = local_rank
        worker_info.local_world_size = local_world_size

        # Verify worker_info is correctly set before spawning (critical for rank configuration)
        if worker_info.world_rank != world_rank:
            raise ValueError(
                f"worker_info.world_rank ({worker_info.world_rank}) does not match expected world_rank ({world_rank})"
            )
        if worker_info.local_world_size != local_world_size:
            raise ValueError(
                f"worker_info.local_world_size ({worker_info.local_world_size}) does not match expected local_world_size ({local_world_size})"
            )

        logging.debug(
            f"[PROCESS_SPAWN]worker_info configured: world_rank={worker_info.world_rank}, "
            f"local_rank={worker_info.local_rank}, local_world_size={worker_info.local_world_size}"
        )

        # Set environment variables for this specific rank before spawning
        # Note: In spawn mode, each process inherits parent's environment at spawn time
        # CUDA_VISIBLE_DEVICES: Required by CUDA runtime library to determine visible GPUs
        # WORLD_RANK: Required by setup_logging() which is called during module import in subprocess
        # LOCAL_WORLD_SIZE: Required by parallelism_config / server_args and by NCCL/DeepEP
        #   when inferring local size; missing it can cause wrong process group and collective timeout.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(actual_device_list)
        os.environ["WORLD_RANK"] = str(world_rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

        logging.info(
            f"[PROCESS_SPAWN]Start local rank outer {world_rank}, local_world_size={local_world_size}, CUDA_VISIBLE_DEVICES={','.join(actual_device_list)}"
        )
        proc = Process(
            target=local_rank_start,
            args=(global_controller, py_env_configs, worker_info, writer),
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
    worker_info: WorkerInfo,
    pipe_writer=None,
):
    """Start multi-rank backend server with proper process management"""
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logging.warning(str(e))

    # Create processes and get pipe readers
    processes, rank_pipe_readers = _create_rank_processes(
        global_controller, py_env_configs, worker_info
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


def load_gpu_nic_affinity(accl_nic_gpu_affinity=None):
    """
    Load GPU NIC affinity configuration.

    Args:
        accl_nic_gpu_affinity: Optional pre-configured affinity string.
                              If None, will try to read from environment or file.

    Returns:
        Tuple of (success: bool, affinity_content: str or None)
        If success is True, affinity_content contains the affinity configuration.
        The caller is responsible for setting it in environment if needed for external libraries.
    """
    # Check if already configured
    if accl_nic_gpu_affinity is not None:
        logging.info(f"Using provided ACCL_NIC_GPU_AFFINITY: {accl_nic_gpu_affinity}")
        return True

    # Check environment variable (for backward compatibility)
    if os.environ.get("ACCL_NIC_GPU_AFFINITY") is not None:
        content = os.environ.get("ACCL_NIC_GPU_AFFINITY")
        logging.info(f"Found ACCL_NIC_GPU_AFFINITY in environment")
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
        # Set in environment for external libraries that may need it
        # But we also return it so caller can use it directly
        os.environ["ACCL_NIC_GPU_AFFINITY"] = content
        logging.info(f"get gpu nic affinity success, loaded from {json_path}")
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
    worker_info: WorkerInfo,
    pipe_writer=None,
):
    logging.info(f"[PROCESS_START]Start backend server process")
    setproctitle("rtp_llm_backend_server")
    os.makedirs("logs", exist_ok=True)
    # Load GPU NIC affinity (returns content but also sets env var for external libraries)
    load_gpu_nic_affinity()

    clear_jit_filelock()

    parallelism_config = py_env_configs.parallelism_config
    logging.info(
        f"start_backend_server, after update_worker_info: {parallelism_config.to_string()}, worker_info: {worker_info}"
    )
    logging.info(
        f"start_backend_server, after update_worker_info: {parallelism_config.to_string()}, worker_info: {worker_info}"
    )
    # TODO(xinfei.sxf) fix this
    if py_env_configs.vit_config.vit_separation == VitSeparation.VIT_SEPARATION_ROLE:
        from rtp_llm.server.vit_rpc_server import vit_start_server

        return vit_start_server(worker_info)

    if not torch.cuda.is_available():
        return local_rank_start(
            global_controller,
            py_env_configs,
            worker_info,
            None,
            parallelism_config.world_rank,
        )

    if (
        parallelism_config.world_size % torch.cuda.device_count() != 0
        and parallelism_config.world_size > torch.cuda.device_count()
    ):
        raise Exception(
            f"result: {parallelism_config.world_size % torch.cuda.device_count()} \
            not support WORLD_SIZE {parallelism_config.world_size} for {torch.cuda.device_count()} local gpu"
        )

    if torch.cuda.device_count() > 1 and parallelism_config.world_size > 1:
        return multi_rank_start(
            global_controller, py_env_configs, worker_info, pipe_writer
        )
    else:
        return local_rank_start(
            global_controller,
            py_env_configs,
            worker_info,
            pipe_writer,
        )


def main():
    return start_backend_server(None)


if __name__ == "__main__":
    main()
