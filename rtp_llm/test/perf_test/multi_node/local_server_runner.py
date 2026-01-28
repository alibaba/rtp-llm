import os

for key, value in os.environ.items():
    print(f"start env {key}={value}")

import json
import logging
import pathlib
import signal
import socket
import sys
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional

import requests

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.parent.parent.absolute()))
from rtp_llm.utils.import_util import has_internal_source

if has_internal_source():
    from internal_source.rtp_llm.test.util.set_internal_env import (
        configure_optional_env,
    )

    configure_optional_env()

import torch.distributed as dist
from torch.distributed import TCPStore

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.distribute.distributed_server import members_from_test_env
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.test.perf_test.batch_decode_test import run_single
from rtp_llm.test.perf_test.test_util import create_query
from rtp_llm.test.utils.maga_server_manager import MagaServerManager

# from uvicorn.loops.uvloop import uvloop_setup
# uvloop_setup()

_SERVER_STARTUP_STORE_KEY_FAILED = "server_startup_failed"
_SERVER_STARTUP_STORE_KEY_OK_PREFIX = "server_startup_ok_"
_SERVER_STARTUP_BARRIER_PREFIX = "barrier/"
_SERVER_STARTUP_BARRIER_RELEASE_KEY = "barrier/release"


def patch_logging_stream_handler():
    """
    Add a StreamHandler to stdout to ensure logs can be captured by subprocess.run to test.log.

    The setup_logging() function configures logging via dictConfig which only outputs to files,
    not to stdout/stderr. However, test.log is captured through subprocess.run(..., stdout=f, stderr=subprocess.STDOUT),
    so we need to add a StreamHandler to stdout to make logging.info() output visible in test.log.

    Returns:
        bool: True if handler was added, False if it already exists.
    """
    root_logger = logging.getLogger()

    # If there's already a StreamHandler writing to stdout/stderr, do NOT add another.
    # In our test harness, subprocess.run redirects stderr to stdout, so having both
    # stdout+stderr stream handlers will cause duplicated lines in test.log.
    has_console_stream_handler = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)
            if stream in (sys.stdout, sys.stderr):
                has_console_stream_handler = True
                break

    if has_console_stream_handler:
        logging.debug("Console StreamHandler already exists, skipping")
        return False

    try:
        # Create and configure StreamHandler for stdout
        stream_handler = logging.StreamHandler(sys.stdout)

        # Use the same log level as root logger
        stream_handler.setLevel(
            root_logger.level if root_logger.level else logging.INFO
        )

        # Use the same formatter format as setup_logging() for consistency
        formatter = logging.Formatter(
            "[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(formatter)

        # Add handler to root logger
        root_logger.addHandler(stream_handler)
        logging.debug("Successfully added StreamHandler to stdout")
        return True
    except Exception as e:
        # Log error but don't raise, as this is a non-critical enhancement
        logging.warning(f"Failed to add StreamHandler to stdout: {e}")
        return False


def _init_startup_store(
    py_env_configs: PyEnvConfigs,
    world_rank: int,
    timeout: int,
) -> TCPStore:
    """
    A lightweight cross-node coordination channel for perf tests.

    We intentionally DO NOT reuse the server's own TCPStore port (master_port-1),
    because the real distributed server will also bind it. Use master_port-12 instead.
    """
    try:
        dist_config_str = os.environ.get(
            "GANG_CONFIG_STRING", py_env_configs.distribute_config.gang_config_string
        )
        if not dist_config_str:
            raise RuntimeError(
                "no gang config string (GANG_CONFIG_STRING), unexpected!"
            )
        dist_members = members_from_test_env(dist_config_str)
        master_member = dist_members[0]
        master_ip = master_member.ip
        master_port = int(master_member.server_port)
        store_port = master_port - 12
        if store_port <= 0:
            raise RuntimeError(f"invalid startup store port: {store_port}")

        store_timeout = timedelta(seconds=timeout)
        logging.info(
            f"init startup store {master_ip}:{store_port}, world_size={len(dist_members)}, world_rank={world_rank}"
        )
        store = dist.TCPStore(
            host_name=master_ip,
            port=store_port,
            world_size=len(dist_members),
            is_master=(world_rank == 0),
            wait_for_workers=False,
            timeout=store_timeout,
        )
        return store
    except Exception as e:
        logging.warning(
            f"failed to init startup store, fallback to timeout wait. err={e}"
        )
        return None


def _store_set_safe(store: Optional[Any], key: str, value: str) -> None:
    if store is None:
        return
    try:
        store.set(key, value)
    except Exception as e:
        logging.warning(f"startup store set failed, key={key}, err={e}")


def _store_check_failed(store: Optional[Any]) -> Optional[str]:
    if store is None:
        return None
    try:
        if store.check([_SERVER_STARTUP_STORE_KEY_FAILED]):
            v = store.get(_SERVER_STARTUP_STORE_KEY_FAILED)
            try:
                return str(v, encoding="utf-8")
            except Exception:
                return str(v)
    except Exception:
        return None
    return None


def _store_check_ok(store: Optional[Any], key: str) -> bool:
    if store is None:
        return False
    try:
        v = store.get(key)
        v_str = str(v, encoding="utf-8")
        logging.info(f"store key {key} value bytes={v}, value str={v_str}")
        return v_str == "ok"
    except Exception:
        return False


def _store_barrier(store: TCPStore, node_rank: int, node_world_size: int) -> None:
    """
    A pure TCPStore-based barrier (no dist.init_process_group required).

    Protocol:
    - Each rank: store.set(f"barrier/{rank}", "1")
    - Master(rank=0): store.get("barrier/0..N-1") then store.set("barrier/release", "1")
    - Others: store.get("barrier/release") then continue
    """
    try:
        store.set(f"{_SERVER_STARTUP_BARRIER_PREFIX}{node_rank}", "1")
    except Exception as e:
        logging.warning(f"store barrier set failed, node_rank={node_rank}, err={e}")
        raise

    if node_rank == 0:
        store.wait(
            [f"{_SERVER_STARTUP_BARRIER_PREFIX}{i}" for i in range(node_world_size)]
        )
        store.set(_SERVER_STARTUP_BARRIER_RELEASE_KEY, "1")
    else:
        store.wait([_SERVER_STARTUP_BARRIER_RELEASE_KEY])


def wait_master_done(
    env_dict: Dict[str, str] = {},
    world_rank: int = 0,
    py_env_configs: PyEnvConfigs = None,
    retry_interval: int = 1,
    retry_times: int = 3,
    heartbeat_interval: int = 10,
    check_connection_timeout: int = 10,
) -> None:
    # Get gang_config_string from environment variable or env_dict
    dist_config_str = env_dict.get(
        "GANG_CONFIG_STRING", py_env_configs.distribute_config.gang_config_string
    )
    if not dist_config_str:
        raise RuntimeError("no gang config string, unexpected!")
    dist_members = members_from_test_env(dist_config_str)
    master_member = dist_members[0]
    master_host = master_member.ip
    master_port = master_member.server_port
    while True:
        logging.info(
            f"rank [{world_rank}] waiting for master {master_host}:{master_port} done"
        )
        time.sleep(heartbeat_interval)

        retry_count = 0
        connection_failed = True
        while True:
            try:
                sock = socket.create_connection(
                    (master_host, master_port), timeout=check_connection_timeout
                )
                sock.close()
                connection_failed = False
                break
            except (socket.error, ConnectionRefusedError) as e:
                retry_count += 1
                if retry_count >= retry_times:
                    break
                logging.info(
                    f"rank [{world_rank}] connection attempt {retry_count} failed, retrying... Error: {e}"
                )
                time.sleep(retry_interval)

        if connection_failed:
            break

    logging.info(
        f"rank [{world_rank}] master {master_host}:{master_port} done, worker exit!"
    )
    return


def wait_world_server_startup(
    tcp_store: TCPStore,
    py_env_configs: PyEnvConfigs,
    check_interval: int = 3,
    check_connection_timeout: int = 10,
    server_startup_timeout: int = 1600,
):
    """
    Wait for all servers to complete startup.
    """
    start_time = time.time()

    dist_config_str = py_env_configs.distribute_config.gang_config_string
    if not dist_config_str:
        raise RuntimeError("no gang config string, unexpected!")
    dist_members = members_from_test_env(dist_config_str)
    targets = [(m.name, m.ip, int(m.server_port)) for m in dist_members]
    logging.info(
        f"waiting all servers startup, targets={targets}, check_interval={check_interval}s, check_connection_timeout={check_connection_timeout}s, server_startup_timeout={server_startup_timeout}s"
    )

    def _is_ready(node_rank: int, host: str, port: int) -> bool:
        try:
            health_resp = requests.get(
                f"http://{host}:{port}/health", timeout=check_connection_timeout
            )
            health_resp_status_code = health_resp.status_code
            health_resp_status_text = health_resp.json()
            health_ok = (
                health_resp_status_code == 200 and health_resp_status_text == "ok"
            )
            update_scheduler_info_resp = requests.post(
                f"http://{host}:{port}/update_scheduler_info",
                json={"batch_size": 1, "mode": "decode"},
                timeout=check_connection_timeout,
            )
            update_scheduler_info_resp_status_code = (
                update_scheduler_info_resp.status_code
            )
            update_scheduler_info_resp_status_text = (
                update_scheduler_info_resp.json().get("status", "not ok")
            )
            update_scheduler_info_ok = (
                update_scheduler_info_resp_status_code == 200
                and update_scheduler_info_resp_status_text == "ok"
            )
            store_key_ok = _store_check_ok(
                tcp_store, f"{_SERVER_STARTUP_STORE_KEY_OK_PREFIX}{node_rank}"
            )
            return health_ok and update_scheduler_info_ok and store_key_ok
        except Exception as e:
            logging.warning(
                f"node rank {node_rank} health check failed, host={host}, port={port}, error={str(e)}"
            )
            return False

    while True:
        failed = _store_check_failed(tcp_store)
        if failed:
            logging.warning(f"other node server startup failed: {failed}")
            return False

        all_ready = True
        for node_rank, (_, ip, port) in enumerate(targets):
            if not _is_ready(node_rank, ip, port):
                all_ready = False
                break
        if all_ready:
            logging.info("all servers are ready")
            return True

        if time.time() - start_time > server_startup_timeout:
            logging.warning(f"waiting all servers startup timeout")
            return False

        time.sleep(check_interval)


def script_exit(pgrp_set: bool = False):
    sys.stdout.flush()
    if pgrp_set:
        os.killpg(0, signal.SIGKILL)
        os._exit(0)
    else:
        os._exit(0)


def try_upload_log(log_dir_path: str, upload_path: str):
    import shutil

    if not os.path.exists(log_dir_path) or not os.path.isdir(log_dir_path):
        logging.info(f"{log_dir_path} not exist, skip upload")
        return
    if not os.listdir(log_dir_path):
        logging.info(f"{log_dir_path} is empty, skip upload")
        return

    zip_path = f"{log_dir_path}.zip"
    shutil.make_archive(log_dir_path, "zip", log_dir_path)
    logging.info(f"zip {log_dir_path} to {zip_path}")

    logging.info(f"upload {zip_path} ...")
    os.system(f"osscmd put {zip_path} {upload_path}/{zip_path}")

    # shutil.rmtree(log_dir_path)
    os.remove(zip_path)


if __name__ == "__main__":
    # Setup logging
    setup_logging()
    # Patch logging to also output to stdout so it can be captured by subprocess.run
    patch_logging_stream_handler()

    # Get test config from environment variables
    batch_size_list = json.loads(os.environ.get("BATCH_SIZE_LIST", "[1,4,8]"))
    input_len_list = json.loads(os.environ.get("INPUT_LEN_LIST", "[2048, 4096, 8192]"))
    is_decode = os.environ.get("IS_DECODE", "1") == "1"
    decode_test_length = int(os.environ.get("DECODE_TEST_LENGTH", 10))
    max_seq_len = max(input_len_list) + decode_test_length + 1
    # Set some environment variables for test
    os.environ["GEN_TIMELINE_SYNC"] = "1"
    os.environ["MAX_SEQ_LEN"] = str(max_seq_len)
    os.environ["FAKE_BALANCE_EXPERT"] = "1"
    os.environ.setdefault("WORKER_INFO_PORT_NUM", "10")
    os.environ["USE_BATCH_DECODE_SCHEDULER"] = "1"
    # Initialize py_env_configs
    py_env_configs: PyEnvConfigs = setup_args()
    setup_and_configure_server(py_env_configs)

    # Set test results and logs save directory
    port = py_env_configs.server_config.start_port
    world_rank = py_env_configs.parallelism_config.world_rank
    local_world_size = py_env_configs.parallelism_config.local_world_size
    log_dir_name = (
        f"test_output_{py_env_configs.model_args.model_type}_{py_env_configs.parallelism_config.dp_size}"
        f"_{py_env_configs.parallelism_config.tp_size}_{py_env_configs.parallelism_config.world_rank}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}"
    ).upper()
    log_dir_path = os.path.abspath(log_dir_name)
    os.makedirs(log_dir_path, exist_ok=True)
    # Update profiling debug logging config
    os.environ["TORCH_CUDA_PROFILER_DIR"] = log_dir_path
    # Get node world size and rank
    dist_config_str = os.environ.get(
        "GANG_CONFIG_STRING", py_env_configs.distribute_config.gang_config_string
    )
    if not dist_config_str:
        raise RuntimeError("no gang config string (GANG_CONFIG_STRING), unexpected!")
    node_world_size = len(members_from_test_env(dist_config_str))
    node_rank = world_rank // local_world_size

    # Get tokenizer path
    tokenizer_path = py_env_configs.model_args.tokenizer_path
    if tokenizer_path is None:
        raise RuntimeError(
            f"fetch tokenizer path failed, tokenizer_path: {py_env_configs.model_args.tokenizer_path}"
        )
    # Create query data for test
    input_query_dict = create_query(
        model_type=py_env_configs.model_args.model_type,
        tokenizer_path=tokenizer_path,
        input_len_list=input_len_list,
    )

    pgrp_set = False
    try:
        os.setpgrp()
        pgrp_set = True
    except Exception as e:
        logging.info(f"setpgrp error: {e}")

    # Set timeout time
    request_tpot = 100  # ms
    bootstrap_timeout = 10  # s
    server_startup_timeout = 1600  # s
    # Set connection retry interval and retry times
    retry_interval = 0.5  # s
    retry_times = 3
    # Set check interval and check timeout
    check_interval = 3  # s
    check_connection_timeout = 10  # s
    # Set heartbeat interval and heartbeat times
    heartbeat_interval = 10  # s

    # Initialize MagaServerManager
    os.environ["MAGA_SERVER_WORK_DIR"] = os.getcwd()
    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = log_dir_path
    server = MagaServerManager(port=str(port))
    # Initialize tcp store
    tcp_store = _init_startup_store(
        py_env_configs, world_rank, timeout=bootstrap_timeout
    )
    if tcp_store is None:
        raise Exception("failed to init tcp store")
    # Set startup store key to starting for current node
    ok_key = f"{_SERVER_STARTUP_STORE_KEY_OK_PREFIX}{node_rank}"
    logging.info(f"set startup store key {ok_key} to starting")
    _store_set_safe(tcp_store, ok_key, "starting")

    try:
        # Start multiple frontend and backend servers for current node
        if not server.start_server(
            retry_interval=check_interval,
            check_connection_timeout=check_connection_timeout,
            timeout=server_startup_timeout,
        ):
            _store_set_safe(
                tcp_store,
                _SERVER_STARTUP_STORE_KEY_FAILED,
                json.dumps(
                    {
                        "world_rank": world_rank,
                        "host": socket.gethostname(),
                        "ip": socket.gethostbyname(socket.gethostname()),
                        "start_port": port,
                        "reason": "server.start_server() returned False (timeout or process exited)",
                    },
                    ensure_ascii=False,
                ),
            )
            server.print_process_log()
            raise Exception("server start failed")
        _store_set_safe(tcp_store, ok_key, "ok")

        # Wait for all servers to complete startup.
        if not wait_world_server_startup(
            tcp_store=tcp_store,
            py_env_configs=py_env_configs,
            check_interval=check_interval,
            check_connection_timeout=check_connection_timeout,
            server_startup_timeout=server_startup_timeout,
        ):
            raise Exception("wait world server startup failed")
        # Store-based barrier (sync all nodes before starting servers).
        _store_barrier(tcp_store, node_rank=node_rank, node_world_size=node_world_size)

        # The master node and other nodes perform different tasks respectively.
        if node_rank:
            # The other nodes continue to attempt to connect to the master node until
            # the task on the master node is completed, the connection attempts fail.
            logging.info(f"node rank non-zero: {node_rank}, wait for main.")
            wait_master_done(
                world_rank=world_rank,
                py_env_configs=py_env_configs,
                retry_interval=retry_interval,
                retry_times=retry_times,
                heartbeat_interval=heartbeat_interval,
                check_connection_timeout=check_connection_timeout,
            )

        else:
            # After all nodes complete startup, the master node performs the test task.
            logging.info(f"world rank zero: {world_rank}, start test")
            run_single(
                base_port=port,
                dp_size=py_env_configs.parallelism_config.dp_size,
                tp_size=py_env_configs.parallelism_config.tp_size,
                batch_size_list=batch_size_list,
                input_len_list=input_len_list,
                input_query_dict=input_query_dict,
                gang_config_string=py_env_configs.distribute_config.gang_config_string,
                local_world_size=py_env_configs.parallelism_config.local_world_size,
                request_tpot=request_tpot,
                connection_timeout=check_connection_timeout,
                retry_times=retry_times,
                retry_interval=retry_interval,
                is_decode=is_decode,
                dump_json_path=log_dir_path,
                decode_test_length=decode_test_length,
                is_speculative=False,
                propose_step=0,
                generate_config={},
            )

    finally:
        # Stop the servers after the test task is completed.
        server.stop_server()
        # Upload log to OSS
        upload_path = os.environ.get("UPLOAD_OSS_PATH", "")
        if upload_path != "":
            logging.info(f"upload log to {upload_path}")
            try_upload_log(log_dir_path, upload_path)
        script_exit(pgrp_set)
