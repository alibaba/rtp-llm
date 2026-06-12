import logging
import multiprocessing
import os
import sys
import threading
import time
import traceback

from rtp_llm.utils.time_util import timer_wrapper

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import (
    maybe_write_jit_cache_to_remote,
    setup_and_configure_server,
)
from rtp_llm.ops import RoleType, SpeculativeType
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.concurrency_controller import init_controller
from rtp_llm.utils.process_manager import (
    DEFER_FIRST_SIGTERM_ENV,
    DEFER_FIRST_SIGTERM_SECONDS_ENV,
    DEFER_FIRST_SIGTERM_VALUE,
    ProcessManager,
)

setup_logging()


def _install_hot_hook_runtime(role: str) -> None:
    try:
        from rtp_llm.utils.hot_hook_runtime import install_if_enabled

        if install_if_enabled():
            logging.info("RTP hot hook runtime installed for %s", role)
    except Exception as e:
        logging.error("failed to install RTP hot hook runtime for %s: %s", role, e)


STARTUP_WARMUP_HEALTH_GATE_FILE_ENV = "RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE"
STARTUP_REAL_WARMUP_MIN_TOKEN_LEN = 2
STARTUP_REAL_WARMUP_TIMEOUT_S = 600.0
STARTUP_REAL_WARMUP_MAX_NEW_TOKENS = 1
STARTUP_REAL_WARMUP_TOKEN_ID = 100


class StartupRealWarmupAddressResolutionError(RuntimeError):
    pass


def check_server_health(server_port, path="/health"):
    try:
        import requests

        response = requests.get(f"http://localhost:{server_port}{path}", timeout=60)
        health_ok = False
        if response.status_code == 200:
            try:
                health_body = response.json()
                if isinstance(health_body, dict):
                    health_ok = health_body.get("status", "") == "ok"
                elif isinstance(health_body, str):
                    health_ok = health_body == "ok"
            except ValueError:
                health_ok = response.text.strip() == "ok"
        if health_ok:
            logging.info(
                f"{server_port}{path}, response status_code = {response.status_code}, text = {response.text}, len = {len(response.text)}"
            )
            return True
        else:
            return False
    except BaseException as e:
        return False


def _backend_deferred_sigterm_seconds(py_env_configs: PyEnvConfigs) -> str:
    timeout = ProcessManager.normalize_shutdown_timeout_seconds(
        py_env_configs.server_config.shutdown_timeout
    )
    return str(ProcessManager.deferred_group_shutdown_timeout_seconds(timeout))


def _sync_server_shutdown_timeout(py_env_configs: PyEnvConfigs):
    py_env_configs.server_config.shutdown_timeout = (
        ProcessManager.sync_shutdown_timeout_env(
            py_env_configs.server_config.shutdown_timeout
        )
    )


@timer_wrapper(description="start backend server")
def start_backend_server_impl(
    global_controller,
    py_env_configs: PyEnvConfigs,
    process_manager: ProcessManager = None,
):
    from rtp_llm.start_backend_server import start_backend_server

    # only for debug
    if py_env_configs.profiling_debug_logging_config.debug_load_server:
        start_backend_server(global_controller, py_env_configs, None)
        os._exit(-1)

    # Create pipe for subprocess startup status communication
    pipe_reader, pipe_writer = multiprocessing.Pipe(duplex=False)
    logging.info(f"[PROCESS_SPAWN]Start backend server process outer")

    old_defer = os.environ.get(DEFER_FIRST_SIGTERM_ENV)
    old_defer_seconds = os.environ.get(DEFER_FIRST_SIGTERM_SECONDS_ENV)
    _sync_server_shutdown_timeout(py_env_configs)
    os.environ[DEFER_FIRST_SIGTERM_ENV] = DEFER_FIRST_SIGTERM_VALUE
    os.environ[DEFER_FIRST_SIGTERM_SECONDS_ENV] = _backend_deferred_sigterm_seconds(
        py_env_configs
    )
    try:
        backend_process = multiprocessing.Process(
            target=start_backend_server,
            args=(global_controller, py_env_configs, pipe_writer),
            name="backend_manager",
        )
        backend_process.start()
    finally:
        if old_defer is None:
            os.environ.pop(DEFER_FIRST_SIGTERM_ENV, None)
        else:
            os.environ[DEFER_FIRST_SIGTERM_ENV] = old_defer
        if old_defer_seconds is None:
            os.environ.pop(DEFER_FIRST_SIGTERM_SECONDS_ENV, None)
        else:
            os.environ[DEFER_FIRST_SIGTERM_SECONDS_ENV] = old_defer_seconds
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


def _iter_serving_ranks(py_env_configs: PyEnvConfigs):
    """Yield (rank, local_world_size) for each rank that should host frontend/dash_sc.

    Matches the historical frontend filter: rank 0 on every node (for heartbeat) plus
    any tp_rank==0 rank.
    """
    pc = py_env_configs.parallelism_config
    local_world_size = pc.world_size
    if "LOCAL_WORLD_SIZE" in os.environ:
        logging.info(
            f"multi rank starts with local world size specified in env: {os.environ['LOCAL_WORLD_SIZE']}"
        )
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        logging.info(
            f"multi rank starts with default local world size: {local_world_size}, world size = {pc.world_size}"
        )
    for rank in range(local_world_size):
        if rank == 0 or (pc.world_rank + rank) % pc.tp_size == 0:
            yield rank
        else:
            logging.info(
                f"rank {pc.world_rank + rank} skipping frontend/dash_sc startup"
            )


@timer_wrapper(description="start dash_sc server")
def start_dash_sc_server_impl(
    global_controller,
    py_env_configs: PyEnvConfigs,
    process_manager=None,
):
    from rtp_llm.start_dash_sc_server import start_dash_sc_server

    dash_sc_processes = []
    dash_sc_pipe_readers = []

    pc = py_env_configs.parallelism_config
    frontend_server_count = py_env_configs.server_config.frontend_server_count

    for rank in _iter_serving_ranks(py_env_configs):
        for i in range(frontend_server_count):
            pipe_reader, pipe_writer = multiprocessing.Pipe(duplex=False)
            logging.info(
                f"[PROCESS_SPAWN]Start dash_sc server process rank_{rank}_server_{i} outer"
            )
            process = multiprocessing.Process(
                target=start_dash_sc_server,
                args=(
                    rank,
                    i,
                    global_controller,
                    py_env_configs,
                    pipe_writer,
                ),
                name=f"dash_sc_server_{rank}_{i}",
            )
            process.start()
            pipe_writer.close()
            dash_sc_processes.append(process)
            dash_sc_pipe_readers.append(pipe_reader)

    if not dash_sc_processes:
        return dash_sc_processes

    startup_status = {"remaining": set(range(len(dash_sc_pipe_readers)))}

    def check_dash_sc_ready():
        if not startup_status["remaining"]:
            return True
        # Poll each outstanding reader non-blocking; raise on any failure.
        done_now = []
        for idx in list(startup_status["remaining"]):
            reader = dash_sc_pipe_readers[idx]
            try:
                if not reader.poll(timeout=0):
                    continue
                msg = reader.recv()
            except EOFError:
                raise Exception(
                    f"DashSc server rank_idx={idx} pipe closed unexpectedly"
                )
            if msg.get("status") == "success":
                logging.info(
                    f"DashSc server rank_idx={idx} ready: {msg.get('message', '')}"
                )
                done_now.append(idx)
                try:
                    reader.close()
                except Exception:
                    pass
            else:
                error_msg = msg.get("message", "Unknown error")
                traceback_info = msg.get("traceback", "")
                if traceback_info:
                    logging.error(f"DashSc server traceback: {traceback_info}")
                raise Exception(
                    f"DashSc server rank_idx={idx} start failed: {error_msg}"
                )
        for idx in done_now:
            startup_status["remaining"].discard(idx)
        return not startup_status["remaining"]

    if process_manager:
        process_manager.register_health_check(
            processes=dash_sc_processes,
            process_name="dash_sc_server",
            check_ready_fn=check_dash_sc_ready,
            retry_interval_seconds=0.1,
        )

    return dash_sc_processes


@timer_wrapper(description="start frontend server")
def start_frontend_server_impl(
    global_controller,
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

    pc = py_env_configs.parallelism_config
    local_world_size = pc.world_size
    if "LOCAL_WORLD_SIZE" in os.environ:
        logging.info(
            f"multi rank starts with local world size specified in env: {os.environ['LOCAL_WORLD_SIZE']}"
        )
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        logging.info(
            f"multi rank starts with default local world size: {local_world_size}, world size = {pc.world_size}"
        )

    # To reduce the number of frontend servers, we only start those with tp_rank=0;
    # however, since k8s needs to check machine heartbeat, rank 0 on each machine also needs to be started.
    for rank in range(local_world_size):
        for i in range(frontend_server_count):
            if rank == 0 or (pc.world_rank + rank) % pc.tp_size == 0:
                logging.info(
                    f"[PROCESS_SPAWN]Start frontend server process rank_{rank}_server_{i} outer"
                )
                process = multiprocessing.Process(
                    target=start_frontend_server,
                    args=(
                        rank,
                        i,
                        global_controller,
                        py_env_configs,
                    ),
                    name=f"frontend_server_{i}",
                )
                frontend_processes.append(process)
                process.start()
            else:
                logging.info(f"rank {pc.world_rank + rank} skipping frontend startup")

    if process_manager and frontend_processes:
        # Register health check with ProcessManager for the first frontend server
        def check_frontend_ready():
            return check_server_health(
                py_env_configs.server_config.start_port, path="/frontend_health"
            )

        process_manager.register_health_check(
            processes=frontend_processes,
            process_name="frontend_server",
            check_ready_fn=check_frontend_ready,
            retry_interval_seconds=0.1,
        )

    return frontend_processes


def _role_is_prefill(py_env_configs: PyEnvConfigs) -> bool:
    role_type = py_env_configs.role_config.role_type
    role_value = getattr(role_type, "value", role_type)
    prefill_value = getattr(RoleType.PREFILL, "value", RoleType.PREFILL)
    role_is_prefill = role_type == RoleType.PREFILL or str(role_type).endswith(
        "PREFILL"
    )
    try:
        role_is_prefill = role_is_prefill or int(role_value) == int(prefill_value)
    except Exception:
        pass
    return role_is_prefill


def _is_startup_real_warmup_entry_rank(py_env_configs: PyEnvConfigs) -> bool:
    parallelism_config = py_env_configs.parallelism_config
    world_rank = int(parallelism_config.world_rank)
    world_size = int(parallelism_config.world_size)
    tp_size = int(parallelism_config.tp_size)
    if world_size <= 1:
        return True
    if tp_size <= 0:
        raise ValueError(
            f"parallelism_config.tp_size should be positive, got {tp_size}"
        )
    return world_rank % tp_size == 0


def _should_run_startup_real_warmup(py_env_configs: PyEnvConfigs) -> bool:
    flag = os.environ.get("DSV4_STARTUP_REAL_WARMUP", "auto").strip().lower()
    if flag in ("0", "false", "off", "no"):
        return False

    role_is_prefill = _role_is_prefill(py_env_configs)
    if not role_is_prefill:
        return False

    if not _is_startup_real_warmup_entry_rank(py_env_configs):
        parallelism_config = py_env_configs.parallelism_config
        logging.info(
            "skip DSV4 startup real warmup on non-entry rank, "
            "world_rank=%s, tp_size=%s, world_size=%s",
            parallelism_config.world_rank,
            parallelism_config.tp_size,
            parallelism_config.world_size,
        )
        return False

    model_type = getattr(py_env_configs.model_args, "model_type", "")
    if flag in ("1", "true", "on", "yes", "force"):
        return True
    return model_type == "deepseek_v4"


def _setup_startup_warmup_health_gate(py_env_configs: PyEnvConfigs):
    if not _should_run_startup_real_warmup(py_env_configs):
        os.environ.pop(STARTUP_WARMUP_HEALTH_GATE_FILE_ENV, None)
        return None

    gate_file = os.path.join(
        "/tmp",
        "rtp_llm_startup_warmup_ready_"
        f"{os.getpid()}_{int(py_env_configs.server_config.start_port)}",
    )
    try:
        os.remove(gate_file)
    except FileNotFoundError:
        pass
    os.environ[STARTUP_WARMUP_HEALTH_GATE_FILE_ENV] = gate_file
    logging.info("startup warmup health gate enabled, gate_file=%s", gate_file)
    return gate_file


def _mark_startup_warmup_health_gate_ready(gate_file):
    if not gate_file:
        return
    try:
        with open(gate_file, "w") as f:
            f.write("ready\n")
        logging.info("startup warmup health gate marked ready, gate_file=%s", gate_file)
    except Exception:
        logging.error(
            "failed to mark startup warmup health gate ready, gate_file=%s, trace=%s",
            gate_file,
            traceback.format_exc(),
        )
        raise


def _start_post_startup_jit_cache_writer(
    py_env_configs: PyEnvConfigs, startup_warmup_succeeded: bool
):
    remote_write_dir = (
        py_env_configs.jit_config.warm_up_jit_and_write_remote or ""
    ).strip()
    if not remote_write_dir:
        return

    def _write_remote_jit_cache():
        try:
            maybe_write_jit_cache_to_remote(py_env_configs, startup_warmup_succeeded)
        except Exception:
            logging.error(
                "post-startup remote JIT cache publishing failed, trace=%s",
                traceback.format_exc(),
            )

    writer = threading.Thread(
        target=_write_remote_jit_cache,
        name="post_startup_jit_cache_writer",
        daemon=True,
    )
    writer.start()
    logging.info(
        "post-startup remote JIT cache writer started for WARM_UP_JIT_AND_WRITE_REMOTE=%s",
        remote_write_dir,
    )


def main():
    _install_hot_hook_runtime("main")
    py_env_configs: PyEnvConfigs = setup_args()
    setup_and_configure_server(py_env_configs)
    start_server(py_env_configs)


def start_server(py_env_configs: PyEnvConfigs):
    logging.info(f"[PROCESS_START]Start server")
    start_time = time.time()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logging.warning(str(e))

    logging.info(
        f"dp_size:  parallelism_config={py_env_configs.parallelism_config.dp_size}"
    )
    global_controller = init_controller(
        py_env_configs.concurrency_config,
        dp_size=py_env_configs.parallelism_config.dp_size,
    )
    _sync_server_shutdown_timeout(py_env_configs)

    # Create process manager with config values
    process_manager = ProcessManager(
        shutdown_timeout=py_env_configs.server_config.shutdown_timeout,
        monitor_interval=py_env_configs.server_config.monitor_interval,
    )
    # Initialize backend_process to None in case role_type is FRONTEND
    backend_process = None
    startup_warmup_gate_file = _setup_startup_warmup_health_gate(py_env_configs)

    try:
        if py_env_configs.role_config.role_type != RoleType.FRONTEND:
            logging.info("start backend server")
            backend_process = start_backend_server_impl(
                global_controller, py_env_configs, process_manager
            )
            process_manager.add_process(backend_process, shutdown_group="backend")

        logging.info("start frontend server")
        frontend_process = start_frontend_server_impl(
            global_controller, py_env_configs, process_manager
        )
        process_manager.add_processes(frontend_process, shutdown_group="frontend")

        if py_env_configs.role_config.role_type != RoleType.VIT:
            logging.info("start dash_sc server")
            dash_sc_processes = start_dash_sc_server_impl(
                global_controller, py_env_configs, process_manager
            )
            if dash_sc_processes:
                process_manager.add_processes(
                    dash_sc_processes, shutdown_group="frontend"
                )

        # Start parallel health checks and wait for completion
        if not process_manager.run_health_checks():
            logging.error("Health checks failed")
            raise Exception("Health checks failed")

        startup_warmup_succeeded = _maybe_run_startup_real_warmup(py_env_configs)
        _mark_startup_warmup_health_gate_ready(startup_warmup_gate_file)
        _start_post_startup_jit_cache_writer(py_env_configs, startup_warmup_succeeded)

        logging.info(
            f"Backend RPC service is listening on 0.0.0.0, IP/IP range can be customized as needed"
        )
        consume_s = time.time() - start_time
        logging.info(f"start server took {consume_s:.2f}s")
    except Exception as e:
        logging.error(f"start failed, trace: {traceback.format_exc()}")
        # If a SIGTERM/SIGINT already triggered shutdown before this exception,
        # the exception is a side-effect of the signal (health check tripped on
        # shutdown_requested), not a real failure — preserve graceful exit
        # semantics. Otherwise mark failure so the process manager uses bounded
        # timeouts and the parent exits non-zero.
        if not process_manager.shutdown_requested:
            process_manager.request_failure_shutdown()
    finally:
        process_manager.monitor_and_release_processes()


def _get_startup_real_warmup_pow2_lens(max_len: int):
    if max_len < STARTUP_REAL_WARMUP_MIN_TOKEN_LEN:
        raise ValueError(
            "model_args.max_seq_len should be at least "
            f"{STARTUP_REAL_WARMUP_MIN_TOKEN_LEN}, got {max_len}"
        )
    lens = []
    value = STARTUP_REAL_WARMUP_MIN_TOKEN_LEN
    while value <= max_len:
        lens.append(value)
        value *= 2
    if lens[-1] != max_len:
        lens.append(max_len)
    return lens


def _get_startup_real_warmup_max_len(py_env_configs: PyEnvConfigs):
    model_max_len = int(getattr(py_env_configs.model_args, "max_seq_len", None) or 0)
    if model_max_len <= 0:
        raise ValueError(
            f"model_args.max_seq_len should be positive, got {model_max_len}"
        )
    logging.info(
        "DSV4 startup real warmup max len = model max_seq_len = %d",
        model_max_len,
    )
    return model_max_len


def _get_startup_real_warmup_token_lens(py_env_configs: PyEnvConfigs):
    max_len = _get_startup_real_warmup_max_len(py_env_configs)
    token_lens = _get_startup_real_warmup_pow2_lens(max_len)
    logging.info(
        "DSV4 startup real warmup uses fixed pow2 token lens through max_seq_len=%d: %s",
        max_len,
        token_lens,
    )
    return token_lens


def _get_startup_real_warmup_grpc_addresses(py_env_configs: PyEnvConfigs):
    from rtp_llm.distribute.distributed_server import (
        get_dp_addrs_from_world_info,
        get_world_info,
    )

    parallelism_config = py_env_configs.parallelism_config
    resolve_trace = None
    try:
        world_info = get_world_info(
            server_config=py_env_configs.server_config,
            distribute_config=py_env_configs.distribute_config,
            parallelism_config=parallelism_config,
        )
        addrs = get_dp_addrs_from_world_info(world_info, parallelism_config)
        if addrs:
            return addrs
    except Exception:
        resolve_trace = traceback.format_exc()

    world_size = int(parallelism_config.world_size)
    if world_size > 1:
        if resolve_trace:
            logging.warning(
                "failed to resolve DSV4 startup real warmup grpc addrs from world info, "
                "trace=%s",
                resolve_trace,
            )
        raise StartupRealWarmupAddressResolutionError(
            "failed to resolve DSV4 startup real warmup grpc entry address "
            "in multi-rank mode; refusing to fallback to local rpc_server_port"
        )
    if resolve_trace:
        logging.warning(
            "failed to resolve DSV4 startup real warmup grpc addrs from world info, "
            "fallback to local rpc_server_port, trace=%s",
            resolve_trace,
        )
    return [f"127.0.0.1:{int(py_env_configs.server_config.rpc_server_port)}"]


def _new_startup_real_warmup_request_id(index: int) -> int:
    return (int(time.time() * 1000000) + index) & 0x7FFFFFFFFFFFFFFF


def _get_startup_real_warmup_speculative_reserve_step(
    py_env_configs: PyEnvConfigs,
) -> int:
    sp_config = getattr(py_env_configs, "sp_config", None)
    if sp_config is None:
        return 0
    sp_type = getattr(sp_config, "type", SpeculativeType.NONE)
    if sp_type in (None, "", SpeculativeType.NONE):
        return 0
    return int(getattr(sp_config, "gen_num_per_cycle", 0) or 0) + 1


def _get_startup_real_warmup_request_token_len(
    token_len: int, max_len: int, reserve_step: int = 0
) -> int:
    max_request_token_len = max_len - STARTUP_REAL_WARMUP_MAX_NEW_TOKENS
    if reserve_step > 0:
        if max_len <= reserve_step:
            raise ValueError(
                "model_args.max_seq_len should be greater than speculative "
                f"reserve_step, got max_seq_len={max_len}, reserve_step={reserve_step}"
            )
        max_request_token_len = min(max_request_token_len, max_len - reserve_step)
    if max_request_token_len <= 0:
        raise ValueError(
            "startup real warmup request token len should be positive, got "
            f"max_seq_len={max_len}, reserve_step={reserve_step}, "
            f"max_new_tokens={STARTUP_REAL_WARMUP_MAX_NEW_TOKENS}"
        )
    return min(token_len, max_request_token_len)


def _get_startup_real_warmup_max_new_tokens() -> int:
    return STARTUP_REAL_WARMUP_MAX_NEW_TOKENS


def _get_startup_real_warmup_timeout_s() -> float:
    timeout_s = float(
        os.environ.get(
            "DSV4_STARTUP_REAL_WARMUP_TIMEOUT_S", STARTUP_REAL_WARMUP_TIMEOUT_S
        )
    )
    if timeout_s <= 0:
        raise ValueError(
            f"DSV4_STARTUP_REAL_WARMUP_TIMEOUT_S should be positive, got {timeout_s}"
        )
    return timeout_s


def _run_startup_real_warmup_async(coroutine):
    import asyncio

    if hasattr(asyncio, "run"):
        return asyncio.run(coroutine)
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


async def _run_startup_real_warmup_grpc(py_env_configs: PyEnvConfigs):
    import torch

    from rtp_llm.config.generate_config import GenerateConfig
    from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient
    from rtp_llm.utils.base_model_datatypes import GenerateInput

    token_lens = _get_startup_real_warmup_token_lens(py_env_configs)
    max_len = _get_startup_real_warmup_max_len(py_env_configs)
    reserve_step = _get_startup_real_warmup_speculative_reserve_step(py_env_configs)
    addresses = _get_startup_real_warmup_grpc_addresses(py_env_configs)
    timeout_s = _get_startup_real_warmup_timeout_s()
    timeout_ms = int(timeout_s * 1000)

    client_config = (
        py_env_configs.grpc_config.get_client_config()
        if py_env_configs.grpc_config is not None
        else {}
    )
    logging.info(
        "running DSV4 startup real warmup via backend grpc, addrs=%s, "
        "token_lens=%s, token_id=%d, max_new_tokens=%d, reserve_step=%d, timeout=%.1fs",
        addresses,
        token_lens,
        STARTUP_REAL_WARMUP_TOKEN_ID,
        STARTUP_REAL_WARMUP_MAX_NEW_TOKENS,
        reserve_step,
        timeout_s,
    )

    begin_all = time.time()
    total_requests = 0
    for addr_idx, addr in enumerate(addresses):
        client = ModelRpcClient(
            addresses=[addr],
            client_config=client_config,
            max_rpc_timeout_ms=timeout_ms,
        )
        try:
            for len_idx, token_len in enumerate(token_lens):
                total_requests += 1
                request_id = _new_startup_real_warmup_request_id(
                    addr_idx * len(token_lens) + len_idx
                )
                request_token_len = _get_startup_real_warmup_request_token_len(
                    token_len, max_len, reserve_step
                )
                max_new_tokens = _get_startup_real_warmup_max_new_tokens()
                generate_config = GenerateConfig(
                    max_new_tokens=max_new_tokens,
                    top_k=1,
                    top_p=1.0,
                    temperature=0.0,
                    do_sample=False,
                    can_use_pd_separation=False,
                    reuse_cache=False,
                    enable_device_cache=False,
                    enable_memory_cache=False,
                    enable_remote_cache=False,
                    aux_info=True,
                    timeout_ms=timeout_ms,
                )
                generate_input = GenerateInput(
                    request_id=request_id,
                    token_ids=torch.full(
                        (request_token_len,),
                        STARTUP_REAL_WARMUP_TOKEN_ID,
                        dtype=torch.int32,
                    ),
                    mm_inputs=[],
                    generate_config=generate_config,
                )

                begin = time.time()
                last_aux = None
                chunk_count = 0
                logging.info(
                    "DSV4 startup grpc warmup request begin, "
                    "addr=%s, request_id=%d, target_token_len=%d, "
                    "request_token_len=%d, max_new_tokens=%d, reserve_step=%d",
                    addr,
                    request_id,
                    token_len,
                    request_token_len,
                    max_new_tokens,
                    reserve_step,
                )
                async for outputs in client.enqueue(generate_input):
                    chunk_count += 1
                    if outputs.generate_outputs:
                        last_aux = outputs.generate_outputs[0].aux_info
                if last_aux is not None:
                    logging.info(
                        "DSV4 startup grpc warmup request finished, addr=%s, request_id=%d, "
                        "target_token_len=%d, request_token_len=%d, max_new_tokens=%d, "
                        "chunks=%d, input_len=%s, "
                        "reuse_len=%s, output_len=%s, cost=%.2fs",
                        addr,
                        request_id,
                        token_len,
                        request_token_len,
                        max_new_tokens,
                        chunk_count,
                        getattr(last_aux, "input_len", None),
                        getattr(last_aux, "reuse_len", None),
                        getattr(last_aux, "output_len", None),
                        time.time() - begin,
                    )
                else:
                    logging.info(
                        "DSV4 startup grpc warmup request finished, addr=%s, request_id=%d, "
                        "target_token_len=%d, request_token_len=%d, max_new_tokens=%d, "
                        "chunks=%d, aux_info=None, cost=%.2fs",
                        addr,
                        request_id,
                        token_len,
                        request_token_len,
                        max_new_tokens,
                        chunk_count,
                        time.time() - begin,
                    )
        finally:
            try:
                await client.close()
            except Exception:
                logging.warning(
                    "failed to close DSV4 startup grpc warmup client, addr=%s, trace=%s",
                    addr,
                    traceback.format_exc(),
                )

    logging.info(
        "DSV4 startup grpc warmup finished, requests=%d, addrs=%d, token_lens=%d, cost=%.2fs",
        total_requests,
        len(addresses),
        len(token_lens),
        time.time() - begin_all,
    )


def _maybe_run_startup_real_warmup(py_env_configs: PyEnvConfigs) -> bool:
    if not _should_run_startup_real_warmup(py_env_configs):
        return False

    try:
        _run_startup_real_warmup_async(_run_startup_real_warmup_grpc(py_env_configs))
        return True
    except StartupRealWarmupAddressResolutionError:
        logging.error(
            "DSV4 startup real warmup address resolution failed, trace: %s",
            traceback.format_exc(),
        )
        raise
    except Exception:
        logging.error(
            "DSV4 startup real warmup failed, trace: %s",
            traceback.format_exc(),
        )
        return False


if __name__ == "__main__":
    main()
