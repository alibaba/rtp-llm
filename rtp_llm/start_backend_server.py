import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing import Process
from pathlib import Path
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
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    set_global_controller,
)
from rtp_llm.utils.process_manager import ProcessManager
from rtp_llm.utils.util import copy_gemm_config

setup_logging()

JIT_CACHE_SETUP_TIMEOUT_S = 120


def _send_pipe_status(pipe_writer, status: str, message: str, traceback: str = ""):
    if pipe_writer is None:
        return
    try:
        pipe_writer.send({"status": status, "message": message, "traceback": traceback})
        pipe_writer.close()
    except Exception as e:
        logging.warning(f"Failed to send status via pipe: {e}")


def _setup_jit_cache(remote_jit_dir: str, local_rank: int, jit_cache_ready):
    from rtp_llm.utils import jit_cache_manager as jit
    from rtp_llm.utils.jit_cache_store import restore_lock

    components, compatible = jit.setup_jit_cache_env()
    if local_rank != 0:
        # Only wait when rank 0 may restore the shared tree.
        if remote_jit_dir and jit_cache_ready is not None:
            if not jit_cache_ready.wait(timeout=JIT_CACHE_SETUP_TIMEOUT_S + 5):
                logging.warning("JIT cache setup wait timed out; continuing")
        return None

    # Fail-open throughout: a broken cache tree or lock must never block startup.
    try:
        if not remote_jit_dir:
            return None
        if not compatible:
            logging.warning("JIT remote cache disabled: incomplete scope or directory")
            return None
        with restore_lock(Path(jit.LOCAL_JIT_DIR)):
            manager_out, commit_lock, cancel = None, threading.Lock(), threading.Event()

            def _worker():
                nonlocal manager_out
                manager = None
                try:
                    remote_root = jit.resolve_remote_root(remote_jit_dir)
                    if not remote_root or cancel.is_set():
                        return
                    manager = jit.JitCacheManager(remote_root, components)
                    manager.start_background_sync(cancel=cancel, commit=commit_lock)
                    with commit_lock:
                        if not cancel.is_set():
                            manager_out = manager
                            manager = None
                except Exception:
                    logging.exception(
                        "JIT cache setup failed; continuing without remote cache"
                    )
                if manager is not None:
                    manager.stop()

            worker = threading.Thread(
                target=_worker, name="jit-cache-setup", daemon=True
            )
            worker.start()
            worker.join(JIT_CACHE_SETUP_TIMEOUT_S)
            with commit_lock:
                if worker.is_alive():
                    cancel.set()
                    logging.warning(
                        "JIT cache setup timed out; continuing without remote cache"
                    )
                return manager_out
    except Exception:
        logging.exception("JIT cache setup failed; continuing without remote cache")
        return None
    finally:
        if jit_cache_ready is not None:
            jit_cache_ready.set()


def local_rank_start(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    world_rank: int = 0,
    pipe_writer=None,
    jit_cache_ready=None,
):
    """Start local rank with proper signal handling for graceful shutdown"""
    backend_manager = None
    jit_cache_manager = None
    shutdown_requested = False
    logging.info(f"[PROCESS_START]Start local rank process")

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logging.info(
            f"Local rank received signal {signum}, shutting down gracefully..."
        )
        if backend_manager is None:
            return
        shutdown_requested = True
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
        local_rank = py_env_configs.parallelism_config.local_rank
        py_env_configs.server_config.set_local_rank(local_rank)
        py_env_configs.distribute_config.set_local_rank(local_rank)
        setup_cuda_device_and_accl_env(local_rank)
        if py_env_configs.parallelism_config.world_size > 1:
            setproctitle(f"rtp_llm_rank-{local_rank}")
        set_global_controller(global_controller)
        jit_cache_manager = _setup_jit_cache(
            str(py_env_configs.jit_config.remote_jit_dir or "").strip(),
            local_rank,
            jit_cache_ready,
        )

        # Import after rank 0 finished/abandoned setup: BackendManager pulls in
        # JIT-producing deps (e.g. TVM FFI).
        from rtp_llm.server.backend_manager import BackendManager

        backend_manager = BackendManager(py_env_configs)
        backend_manager.start()
        # Engine startup overwrites SIGTERM/SIGINT; restore Python handlers so
        # the finally block can stop JIT cache workers on shutdown.
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logging.info("Backend server initialized successfully, sending ready status")

        # Send startup success message
        _send_pipe_status(
            pipe_writer,
            "success",
            f"Backend server started successfully on rank {py_env_configs.parallelism_config.local_rank}",
        )

        # Enter service loop to keep the process alive
        logging.info("Entering service loop to keep backend_manager alive")
        backend_manager.serve_forever()

    except BaseException as e:
        error_msg = f"start server error: {e}"
        error_trace = traceback.format_exc()
        logging.error(f"{error_msg}, trace: {error_trace}")

        # Send startup failure message
        _send_pipe_status(pipe_writer, "failed", error_msg, error_trace)
        raise e
    finally:
        # Best-effort cleanup: log failures but never skip the hard-exit below.
        try:
            if jit_cache_manager:
                jit_cache_manager.stop()
        except Exception:
            logging.exception("JIT cache stop failed during shutdown")
        # Hard-exit to skip pybind destructors unsafe after GIL release.
        if shutdown_requested:
            # os._exit skips atexit, so unmount FUSE/NFS first. Only on shutdown:
            # doing so on a startup exception would yank mounts from loading ranks.
            try:
                from rtp_llm.utils.fuser import umount_all

                umount_all()
            except Exception:
                logging.exception("umount_all failed during shutdown")
            os._exit(0)


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
    ctx,
    jit_cache_ready,
):
    """Create and start rank processes."""
    pc = py_env_configs.parallelism_config
    local_world_size = _get_local_world_size(py_env_configs)
    cuda_device_list = _get_cuda_device_list()
    _validate_dp_configuration(py_env_configs)
    processes, rank_pipe_readers = [], []

    for world_rank in range(pc.world_rank, pc.world_rank + local_world_size):
        reader, writer = ctx.Pipe(duplex=False)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_device_list)
        os.environ["WORLD_RANK"] = str(world_rank)
        proc = ctx.Process(
            target=local_rank_start,
            args=(
                global_controller,
                py_env_configs,
                world_rank,
                writer,
                jit_cache_ready,
            ),
            name=f"rank-{world_rank}",
        )
        proc.start()
        writer.close()  # Parent process closes write end
        processes.append(proc)
        rank_pipe_readers.append(reader)
    return processes, rank_pipe_readers


def _wait_for_ranks_startup(
    processes: List[Process],
    rank_pipe_readers: List[multiprocessing.Pipe],
    local_world_size: int,
):
    """
    Wait for all ranks to report startup status via pipe.

    Args:
        processes: List of rank processes
        rank_pipe_readers: List of pipe readers for each rank
        local_world_size: Total number of ranks

    Raises:
        Exception: If any rank fails to start or times out
    """
    logging.info(
        f"Waiting for all {local_world_size} ranks to report startup status..."
    )

    # Track which ranks have reported
    ranks_received = [False] * local_world_size
    poll_timeout = 0.5  # seconds per poll
    max_wait_time = 3600  # Maximum 1 hour wait
    start_time = time.time()

    try:
        # Wait for all ranks to report or until timeout/failure
        while not all(ranks_received):
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Check if timeout
            if elapsed_time > max_wait_time:
                raise Exception(f"Ranks startup timeout: {elapsed_time:.1f}s")

            # Check if all processes are still alive
            for proc_idx, proc in enumerate(processes):
                if not proc.is_alive() or proc.exitcode is not None:
                    logging.error("At least one process died, terminating wait")
                    raise Exception(
                        f"Rank {proc_idx} process died unexpectedly with exit code {proc.exitcode} is_alive: {proc.is_alive()}"
                    )

            # Check each reader for available data
            for i, reader in enumerate(rank_pipe_readers):
                if ranks_received[i]:
                    continue

                try:
                    # Non-blocking check if data is available
                    if reader.poll(timeout=poll_timeout):
                        data = reader.recv()
                        ranks_received[i] = True
                        if data.get("status") == "success":
                            logging.info(
                                f"Rank {i} started successfully: {data.get('message', '')}"
                            )
                        else:
                            error_msg = data.get("message", "Unknown error")
                            traceback_info = data.get("traceback", "")
                            if traceback_info:
                                logging.error(f"Rank {i} traceback: {traceback_info}")
                            raise Exception(f"Rank {i} startup failed: {error_msg}")
                except EOFError:
                    # Pipe closed unexpectedly (process died)
                    if not ranks_received[i]:
                        error_msg = f"Rank {i}: Pipe closed unexpectedly (process may have died)"
                        logging.error(error_msg)
                        raise Exception(error_msg)
                except Exception as e:
                    if not ranks_received[i]:
                        logging.error(f"Failed to receive status from rank {i}: {e}")
                        raise
            time.sleep(5)

        logging.info(f"All {local_world_size} ranks started successfully")
    finally:
        # Always close all readers
        for reader in rank_pipe_readers:
            try:
                reader.close()
            except Exception:
                pass


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

    ctx = multiprocessing.get_context("spawn")
    jit_cache_ready = ctx.Event()
    processes, rank_pipe_readers = _create_rank_processes(
        global_controller, py_env_configs, ctx, jit_cache_ready
    )
    local_world_size = len(processes)

    if py_env_configs.distribute_config.fake_gang_env:
        return processes

    # Wait for all ranks to report startup status
    try:
        _wait_for_ranks_startup(processes, rank_pipe_readers, local_world_size)

        # Report success via external pipe
        _send_pipe_status(
            pipe_writer,
            "success",
            f"All {local_world_size} backend ranks started successfully",
        )
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Multi-rank startup failed: {error_msg}")

        # Report failure via external pipe
        _send_pipe_status(pipe_writer, "failed", error_msg)

        # Terminate all processes if any rank failed
        logging.error("Terminating all ranks due to startup failures")
        for proc in processes:
            if proc.is_alive():
                proc.terminate()

        # timeout join + kill to avoid terminate failed
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                logging.warning(f"Force killing process {proc.name} (pid={proc.pid})")
                proc.kill()
                proc.join(timeout=2)

        # os._exit to avoid atexit deadlock
        alive_procs = [p for p in processes if p.is_alive()]
        if alive_procs:
            logging.error(
                f"{len(alive_procs)} processes still alive after kill, using os._exit to avoid atexit deadlock"
            )
            os._exit(1)
        else:
            raise Exception("Multi-rank startup failed")

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


def start_backend_server(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    pipe_writer=None,
):
    logging.info(f"[PROCESS_START]Start backend server process")
    setproctitle("rtp_llm_backend_server")
    os.makedirs("logs", exist_ok=True)
    load_gpu_nic_affinity()

    if not torch.cuda.is_available():
        return local_rank_start(
            global_controller,
            py_env_configs,
        )

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
        return local_rank_start(
            global_controller,
            py_env_configs,
            0,
            pipe_writer,
        )


def main():
    return start_backend_server(None)


if __name__ == "__main__":
    main()
