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
from rtp_llm.test.perf_test.multi_node.perf_runner import run_single
from rtp_llm.test.perf_test.multi_node.perf_util import create_query
from rtp_llm.test.perf_test.multi_node.server_manager import LocalServerManager

# from uvicorn.loops.uvloop import uvloop_setup
# uvloop_setup()

_SERVER_STARTUP_STORE_KEY_FAILED = "server_startup_failed"
_SERVER_STARTUP_STORE_KEY_OK_PREFIX = "server_startup_ok_"
_SERVER_STARTUP_BARRIER_PREFIX = "barrier/"
_SERVER_STARTUP_BARRIER_RELEASE_KEY = "barrier/release"


def patch_logging_stream_handler():
    root_logger = logging.getLogger()

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
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(
            root_logger.level if root_logger.level else logging.INFO
        )
        formatter = logging.Formatter(
            "[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
        logging.debug("Successfully added StreamHandler to stdout")
        return True
    except Exception as e:
        logging.warning(f"Failed to add StreamHandler to stdout: {e}")
        return False


def _init_startup_store(
    py_env_configs: PyEnvConfigs,
    world_rank: int,
    timeout: int,
) -> TCPStore:
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

    os.remove(zip_path)


if __name__ == "__main__":
    setup_logging()
    patch_logging_stream_handler()

    batch_size_list = json.loads(os.environ.get("BATCH_SIZE_LIST", "[1,4,8]"))
    input_len_list = json.loads(os.environ.get("INPUT_LEN_LIST", "[2048, 4096, 8192]"))
    is_decode = os.environ.get("IS_DECODE", "1") == "1"
    decode_test_length = int(os.environ.get("DECODE_TEST_LENGTH", 10))
    max_seq_len = max(input_len_list) + decode_test_length + 1

    os.environ["GEN_TIMELINE_SYNC"] = "1"
    os.environ["MAX_SEQ_LEN"] = str(max_seq_len)
    os.environ["FAKE_BALANCE_EXPERT"] = "1"
    os.environ.setdefault("WORKER_INFO_PORT_NUM", "10")
    os.environ["USE_BATCH_DECODE_SCHEDULER"] = "1"

    py_env_configs: PyEnvConfigs = setup_args()
    setup_and_configure_server(py_env_configs)

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

    os.environ["TORCH_CUDA_PROFILER_DIR"] = log_dir_path

    dist_config_str = os.environ.get(
        "GANG_CONFIG_STRING", py_env_configs.distribute_config.gang_config_string
    )
    if not dist_config_str:
        raise RuntimeError("no gang config string (GANG_CONFIG_STRING), unexpected!")
    node_world_size = len(members_from_test_env(dist_config_str))
    node_rank = world_rank // local_world_size

    tokenizer_path = py_env_configs.model_args.tokenizer_path
    if tokenizer_path is None:
        raise RuntimeError(
            f"fetch tokenizer path failed, tokenizer_path: {py_env_configs.model_args.tokenizer_path}"
        )

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

    request_tpot = 100
    bootstrap_timeout = 10
    server_startup_timeout = 1600
    retry_interval = 0.5
    retry_times = 3
    check_interval = 3
    check_connection_timeout = 10
    heartbeat_interval = 10

    os.environ["MAGA_SERVER_WORK_DIR"] = os.getcwd()
    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = log_dir_path
    server = LocalServerManager(port=port, log_dir=log_dir_path)

    tcp_store = _init_startup_store(
        py_env_configs, world_rank, timeout=bootstrap_timeout
    )
    if tcp_store is None:
        raise Exception("failed to init tcp store")

    ok_key = f"{_SERVER_STARTUP_STORE_KEY_OK_PREFIX}{node_rank}"
    logging.info(f"set startup store key {ok_key} to starting")
    _store_set_safe(tcp_store, ok_key, "starting")

    try:
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

        if not wait_world_server_startup(
            tcp_store=tcp_store,
            py_env_configs=py_env_configs,
            check_interval=check_interval,
            check_connection_timeout=check_connection_timeout,
            server_startup_timeout=server_startup_timeout,
        ):
            raise Exception("wait world server startup failed")

        _store_barrier(tcp_store, node_rank=node_rank, node_world_size=node_world_size)

        if node_rank:
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
        server.stop_server()
        upload_path = os.environ.get("UPLOAD_OSS_PATH", "")
        if upload_path != "":
            logging.info(f"upload log to {upload_path}")
            try_upload_log(log_dir_path, upload_path)
        script_exit(pgrp_set)
