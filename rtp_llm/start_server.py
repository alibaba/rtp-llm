import logging
import multiprocessing
import os
import sys
import time
import traceback

import requests

from rtp_llm.distribute.distributed_server import (
    get_dp_addrs_from_world_info,
    get_world_info,
)
from rtp_llm.utils.time_util import timer_wrapper

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.ops import RoleType
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.concurrency_controller import init_controller
from rtp_llm.utils.process_manager import ProcessManager

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


def check_server_health(server_port, path="/health"):
    try:
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

    backend_process = multiprocessing.Process(
        target=start_backend_server,
        args=(global_controller, py_env_configs, pipe_writer),
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
    role_is_prefill = role_type == RoleType.PREFILL or str(role_type).endswith("PREFILL")
    try:
        role_is_prefill = role_is_prefill or int(role_value) == int(prefill_value)
    except Exception:
        pass
    return role_is_prefill


def _should_run_startup_real_warmup(py_env_configs: PyEnvConfigs) -> bool:
    model_type = getattr(py_env_configs.model_args, "model_type", "")
    return model_type == "deepseek_v4" and _role_is_prefill(py_env_configs)


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
            process_manager.add_process(backend_process)

        logging.info("start frontend server")
        frontend_process = start_frontend_server_impl(
            global_controller, py_env_configs, process_manager
        )
        process_manager.add_processes(frontend_process)

        if py_env_configs.role_config.role_type != RoleType.VIT:
            logging.info("start dash_sc server")
            dash_sc_processes = start_dash_sc_server_impl(
                global_controller, py_env_configs, process_manager
            )
            if dash_sc_processes:
                process_manager.add_processes(dash_sc_processes)

        # Start parallel health checks and wait for completion
        if not process_manager.run_health_checks():
            logging.error("Health checks failed")
            raise Exception("Health checks failed")

        _maybe_run_startup_real_warmup(py_env_configs)
        _mark_startup_warmup_health_gate_ready(startup_warmup_gate_file)

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
    max_len = int(getattr(py_env_configs.model_args, "max_seq_len", None) or 0)
    if max_len <= 0:
        raise ValueError(
            f"model_args.max_seq_len should be positive, got {max_len}"
        )
    return max_len


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
    try:
        world_info = get_world_info(
            server_config=py_env_configs.server_config,
            distribute_config=py_env_configs.distribute_config,
            parallelism_config=py_env_configs.parallelism_config,
        )
        addrs = get_dp_addrs_from_world_info(
            world_info, py_env_configs.parallelism_config
        )
        if addrs:
            return addrs
    except Exception:
        logging.warning(
            "failed to resolve DSV4 startup real warmup grpc addrs from world info, "
            "fallback to local rpc_server_port, trace=%s",
            traceback.format_exc(),
        )

    return [f"127.0.0.1:{int(py_env_configs.server_config.rpc_server_port)}"]


def _new_startup_real_warmup_request_id(index: int) -> int:
    return (int(time.time() * 1000000) + index) & 0x7FFFFFFFFFFFFFFF


def _get_startup_real_warmup_request_token_len(token_len: int, max_len: int) -> int:
    if token_len >= max_len:
        return max(1, max_len - STARTUP_REAL_WARMUP_MAX_NEW_TOKENS)
    return token_len


def _get_startup_real_warmup_max_new_tokens() -> int:
    return STARTUP_REAL_WARMUP_MAX_NEW_TOKENS


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
    addresses = _get_startup_real_warmup_grpc_addresses(py_env_configs)
    timeout_ms = int(STARTUP_REAL_WARMUP_TIMEOUT_S * 1000)

    client_config = (
        py_env_configs.grpc_config.get_client_config()
        if py_env_configs.grpc_config is not None
        else {}
    )
    logging.info(
        "running DSV4 startup real warmup via backend grpc, addrs=%s, "
        "token_lens=%s, token_id=%d, max_new_tokens=%d, timeout=%.1fs",
        addresses,
        token_lens,
        STARTUP_REAL_WARMUP_TOKEN_ID,
        STARTUP_REAL_WARMUP_MAX_NEW_TOKENS,
        STARTUP_REAL_WARMUP_TIMEOUT_S,
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
                    token_len, max_len
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
                    "request_token_len=%d, max_new_tokens=%d",
                    addr,
                    request_id,
                    token_len,
                    request_token_len,
                    max_new_tokens,
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

def _maybe_run_startup_real_warmup(py_env_configs: PyEnvConfigs):
    if not _should_run_startup_real_warmup(py_env_configs):
        return

    try:
        _run_startup_real_warmup_async(_run_startup_real_warmup_grpc(py_env_configs))
    except Exception:
        logging.error(
            "DSV4 startup real warmup failed, trace: %s",
            traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
