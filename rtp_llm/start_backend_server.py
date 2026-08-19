import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
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
from rtp_llm.utils.oom_diag import install_oom_dump
from rtp_llm.utils.process_manager import ProcessManager
from rtp_llm.utils.util import copy_gemm_config

setup_logging()


def _install_hot_hook_runtime(role: str) -> None:
    try:
        from rtp_llm.utils.hot_hook_runtime import install_if_enabled

        if install_if_enabled():
            logging.info("RTP hot hook runtime installed for %s", role)
    except Exception as e:
        logging.error("failed to install RTP hot hook runtime for %s: %s", role, e)


def _send_pipe_status(pipe_writer, status: str, message: str, error_trace: str = ""):
    if pipe_writer is None:
        return
    try:
        pipe_writer.send(
            {"status": status, "message": message, "traceback": error_trace}
        )
        pipe_writer.close()
    except Exception as e:
        logging.warning(f"Failed to send status via pipe: {e}")


def local_rank_start(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    world_rank: int = 0,
    pipe_writer=None,
    shutdown_ready_event=None,
):
    """Start local rank with proper signal handling for graceful shutdown"""
    _install_hot_hook_runtime(f"backend_rank_{world_rank}")
    backend_manager = None
    logging.info(f"[PROCESS_START]Start local rank process")

    def signal_handler(signum, frame):
        logging.info(
            f"Local rank received signal {signum}, shutting down gracefully..."
        )
        if backend_manager is not None:
            # Let teardown failures reach the outer BaseException handler so
            # this rank exits nonzero and its owning manager reports failure.
            backend_manager.request_shutdown()

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
        install_oom_dump()
        from rtp_llm.server.backend_manager import BackendManager

        backend_manager = BackendManager(
            py_env_configs, shutdown_ready_event=shutdown_ready_event
        )
        backend_manager.start()
        # Defer these handlers until model loading completes.
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logging.info("Backend server initialized successfully, sending ready status")

        _send_pipe_status(
            pipe_writer,
            "success",
            f"Backend server started successfully on rank {py_env_configs.parallelism_config.local_rank}",
        )
        pipe_writer = None  # success closed it

        logging.info("Entering service loop to keep backend_manager alive")
        backend_manager.serve_forever()

    except BaseException as e:
        error_msg = f"start server error: {e}"
        error_trace = traceback.format_exc()
        logging.error(f"{error_msg}, trace: {error_trace}")

        _send_pipe_status(pipe_writer, "failed", error_msg, error_trace)
        raise


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


def _shutdown_order_indices(
    local_world_size: int, first_world_rank: int, tp_size: int
):
    """Return local follower/leader indices for two-phase engine shutdown."""
    if local_world_size <= 0:
        return [], []
    if tp_size > 1:
        follower_indices = [
            index
            for index in range(local_world_size)
            if (first_world_rank + index) % tp_size != 0
        ]
        leader_indices = [
            index
            for index in range(local_world_size)
            if (first_world_rank + index) % tp_size == 0
        ]
        return follower_indices, leader_indices
    if local_world_size > 1:
        # TP1 ranks can still share DP/EP collectives. Keep the first local rank
        # alive until every other local rank has stopped its scheduler.
        return list(range(1, local_world_size)), [0]
    return [], [0]


def _create_rank_processes(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    ctx,
    processes: List[BaseProcess],
    rank_pipe_readers: List[Connection],
    shutdown_ready_events: List,
):
    """Create and start rank processes. Each proc is appended before start() so a
    mid-loop abort still leaves every spawned object in the caller's list; teardown
    guards proc.pid to skip any that never started."""
    pc = py_env_configs.parallelism_config
    local_world_size = _get_local_world_size(py_env_configs)
    cuda_device_list = _get_cuda_device_list()
    _validate_dp_configuration(py_env_configs)

    follower_indices, _ = _shutdown_order_indices(
        local_world_size, pc.world_rank, pc.tp_size
    )
    follower_index_set = set(follower_indices)

    for local_index, world_rank in enumerate(
        range(pc.world_rank, pc.world_rank + local_world_size)
    ):
        reader, writer = ctx.Pipe(duplex=False)
        rank_pipe_readers.append(reader)
        shutdown_ready_event = (
            ctx.Event() if local_index in follower_index_set else None
        )
        shutdown_ready_events.append(shutdown_ready_event)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_device_list)
        os.environ["WORLD_RANK"] = str(world_rank)
        proc = ctx.Process(
            target=local_rank_start,
            args=(
                global_controller,
                py_env_configs,
                world_rank,
                writer,
                shutdown_ready_event,
            ),
            name=f"rank-{world_rank}",
        )
        processes.append(proc)
        try:
            proc.start()
        finally:
            writer.close()  # drop parent copy so reader EOFs when the rank dies


def _close_readers(readers: List[Connection]) -> None:
    for reader in readers:
        with suppress(Exception):
            reader.close()


def _wait_for_ranks_startup(
    processes: List[BaseProcess],
    rank_pipe_readers: List[Connection],
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

                # The try only guards poll/recv: a failed status must raise
                # below, outside any handler that could swallow it.
                try:
                    if not reader.poll(timeout=poll_timeout):
                        continue
                    data = reader.recv()
                except EOFError:
                    # Pipe closed unexpectedly (process died)
                    error_msg = (
                        f"Rank {i}: Pipe closed unexpectedly (process may have died)"
                    )
                    logging.error(error_msg)
                    raise Exception(error_msg)
                except Exception as e:
                    logging.error(f"Failed to receive status from rank {i}: {e}")
                    raise
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
            if all(ranks_received):
                break
            time.sleep(5)

        logging.info(f"All {local_world_size} ranks started successfully")
    finally:
        _close_readers(rank_pipe_readers)


def multi_rank_start(
    global_controller: ConcurrencyController,
    py_env_configs: PyEnvConfigs,
    pipe_writer=None,
    cleanup=None,
):
    """Start multi-rank backend server with proper process management"""
    ctx = multiprocessing.get_context("spawn")
    processes, rank_pipe_readers, shutdown_ready_events = [], [], []
    try:
        _create_rank_processes(
            global_controller,
            py_env_configs,
            ctx,
            processes,
            rank_pipe_readers,
            shutdown_ready_events,
        )
        local_world_size = len(processes)

        if py_env_configs.distribute_config.fake_gang_env:
            # Test-only path: returning releases the manager, which publishes one
            # last snapshot; the caller-owned ranks keep serving from the local tree.
            _close_readers(rank_pipe_readers)
            return processes

        # Wait for all ranks to report startup status
        _wait_for_ranks_startup(processes, rank_pipe_readers, local_world_size)

        manager = ProcessManager(
            shutdown_timeout=py_env_configs.server_config.shutdown_timeout,
            monitor_interval=py_env_configs.server_config.monitor_interval,
            allow_defer_first_sigterm=True,
            pre_exit_cleanup=cleanup,
            termination_ack_timeout=10.0,
        )
        follower_indices, leader_indices = _shutdown_order_indices(
            len(processes),
            py_env_configs.parallelism_config.world_rank,
            py_env_configs.parallelism_config.tp_size,
        )
        shutdown_order = follower_indices + leader_indices
        manager.set_processes(
            [processes[index] for index in shutdown_order],
            [
                shutdown_ready_events[index]
                if index in follower_indices
                else None
                for index in shutdown_order
            ],
            shutdown_group="backend",
        )
        _send_pipe_status(
            pipe_writer,
            "success",
            f"All {local_world_size} backend ranks started successfully",
        )
        pipe_writer = None  # success closed it; a later abort speaks via exit code
    except BaseException as e:
        error_msg = str(e) or type(e).__name__
        error_trace = traceback.format_exc()
        logging.error(f"Multi-rank startup failed: {error_msg}")

        # Report failure via external pipe
        _send_pipe_status(pipe_writer, "failed", error_msg, error_trace)

        _close_readers(rank_pipe_readers)

        # Terminate all processes if any rank failed
        logging.error("Terminating all ranks due to startup failures")
        for proc in processes:
            if proc.pid is not None and proc.is_alive():
                proc.terminate()

        # timeout join + kill to avoid terminate failed
        for proc in processes:
            if proc.pid is None:
                continue
            proc.join(timeout=5)
            if proc.is_alive():
                logging.warning(f"Force killing process {proc.name} (pid={proc.pid})")
                proc.kill()
                proc.join(timeout=2)

        alive_procs = [p for p in processes if p.pid is not None and p.is_alive()]
        if alive_procs:
            logging.error(f"{len(alive_procs)} processes still alive after kill")
            if cleanup:
                try:
                    cleanup()  # JitCacheManager.stop() is bounded.
                except Exception:
                    logging.exception("JIT cache cleanup failed before hard exit")
            os._exit(1)
        if isinstance(e, Exception):
            raise Exception(f"Multi-rank startup failed: {error_msg}") from e
        raise

    # After successful startup, monitor processes
    if not manager.monitor_and_release_processes():
        raise RuntimeError("one or more backend ranks exited abnormally")

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
    # Startup window only: turn SIGTERM/SIGINT into an exception so the teardown
    # below runs (a defaulted SIGTERM would kill the process with no cleanup);
    # local_rank_start / ProcessManager install the runtime handlers later.
    def abort(signum, frame):
        # A second signal must kill immediately, not be swallowed by teardown.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt(f"signal {signum} during startup")

    signal.signal(signal.SIGTERM, abort)
    signal.signal(signal.SIGINT, abort)

    logging.info(f"[PROCESS_START]Start backend server process")
    setproctitle("rtp_llm_backend_server")
    os.makedirs("logs", exist_ok=True)
    load_gpu_nic_affinity()

    if not torch.cuda.is_available():
        return local_rank_start(global_controller, py_env_configs, 0, pipe_writer)

    pc = py_env_configs.parallelism_config
    if (
        pc.world_size % torch.cuda.device_count() != 0
        and pc.world_size > torch.cuda.device_count()
    ):
        raise Exception(
            f"result: {pc.world_size % torch.cuda.device_count()} \
            not support WORLD_SIZE {pc.world_size} for {torch.cuda.device_count()} local gpu"
        )

    manager = None
    try:
        try:
            from rtp_llm.utils.jit_cache_manager import start_from_config

            manager = start_from_config(py_env_configs.jit_config)
        except Exception:  # cold start; a signal instead unwinds to the finally
            logging.exception("JIT_CACHE_FAIL_OPEN: setup failed; cold start")
        if torch.cuda.device_count() > 1 and pc.world_size > 1:
            return multi_rank_start(
                global_controller,
                py_env_configs,
                pipe_writer,
                cleanup=manager.stop if manager else None,
            )
        return local_rank_start(global_controller, py_env_configs, 0, pipe_writer)
    finally:
        if manager:
            manager.stop()
