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
from typing import Dict, List

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.parent.parent.absolute()))
from rtp_llm.utils.import_util import has_internal_source

if has_internal_source():
    from internal_source.rtp_llm.test.util.set_internal_env import (
        configure_optional_env,
    )

    configure_optional_env()

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import members_from_test_env
from rtp_llm.test.perf_test.batch_decode_test import run_single
from rtp_llm.test.perf_test.test_util import create_query
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.utils.fuser import fetch_remote_file_to_local

# from uvicorn.loops.uvloop import uvloop_setup
# uvloop_setup()


def wait_master_done(env_dict: Dict[str, str] = {}, world_rank: int = 0) -> None:
    # Get gang_config_string from environment variable or env_dict
    dist_config_str = env_dict.get(
        "GANG_CONFIG_STRING", PyEnvConfigs.distribute_config.gang_config_string
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
        time.sleep(10)

        # 添加3次重试机制
        retry_count = 0
        max_retries = 5
        connection_failed = True

        while retry_count < max_retries and connection_failed:
            try:
                sock = socket.create_connection(
                    (master_host, master_port), timeout=1000
                )
                sock.close()
                connection_failed = False
            except (socket.error, ConnectionRefusedError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    logging.info(
                        f"rank [{world_rank}] connection attempt {retry_count} failed, retrying... Error: {e}"
                    )
                    time.sleep(5)  # 重试前等待2秒
                else:
                    logging.info(
                        f"rank [{world_rank}] all {max_retries} connection attempts failed, master is done"
                    )
                    break

        if connection_failed:
            break

    logging.info(
        f"rank [{world_rank}] master {master_host}:{master_port} done, this worker exit!"
    )
    return


def script_exit(pgrp_set: bool = False):
    sys.stdout.flush()
    if pgrp_set:
        os.killpg(0, signal.SIGKILL)
        os._exit(0)
    else:
        os._exit(0)


def test_main(
    port: int,
    py_env_configs: PyEnvConfigs,
    batch_size_list: List[int],
    input_len_list: List[int],
    input_query_dict: Dict[int, str],
    is_decode: bool,
    log_dir_path: str,
    decode_test_length: int,
):
    run_single(
        port,
        py_env_configs.parallelism_distributed_config.dp_size,
        py_env_configs.parallelism_distributed_config.tp_size,
        batch_size_list,
        input_len_list,
        input_query_dict,
        is_decode,
        log_dir_path,
        decode_test_length,
    )


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
    batch_size_list = json.loads(os.environ.get("BATCH_SIZE_LIST", "[1,4,8]"))
    input_len_list = json.loads(os.environ.get("INPUT_LEN_LIST", "[2048, 4096, 8192]"))
    is_decode = os.environ.get("IS_DECODE", "1") == "1"
    decode_test_length = int(os.environ.get("DECODE_TEST_LENGTH", 10))
    max_seq_len = max(input_len_list) + decode_test_length

    os.environ["USE_BATCH_DECODE_SCHEDULER"] = "1"
    os.environ["FAKE_BALANCE_EXPERT"] = "1"
    os.environ["MAX_SEQ_LEN"] = str(max_seq_len + 20)

    py_env_configs = PyEnvConfigs()
    port = py_env_configs.server_config.start_port
    world_rank = py_env_configs.parallelism_distributed_config.world_rank
    log_dir_name = (
        f"test_output_{py_env_configs.model_args.model_type}_{py_env_configs.parallelism_distributed_config.dp_size}"
        f"_{py_env_configs.parallelism_distributed_config.tp_size}_{py_env_configs.parallelism_distributed_config.world_rank}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}"
    ).upper()
    log_dir_path = os.path.abspath(log_dir_name)
    os.makedirs(log_dir_path, exist_ok=True)
    os.environ["TORCH_CUDA_PROFILER_DIR"] = log_dir_path
    # for maga server files
    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = log_dir_path
    os.environ["MAGA_SERVER_WORK_DIR"] = os.getcwd()

    pgrp_set = False
    try:
        os.setpgrp()
        pgrp_set = True
    except Exception as e:
        logging.info(f"setpgrp error: {e}")

    tokenizer_path = fetch_remote_file_to_local(
        py_env_configs.model_args.tokenizer_path
    )
    if tokenizer_path is None:
        raise RuntimeError(
            f"fetch tokenizer path failed, tokenizer_path: {py_env_configs.model_args.tokenizer_path}"
        )

    input_query_dict = create_query(
        py_env_configs.model_args.model_type,
        tokenizer_path,
        input_len_list,
    )

    server = MagaServerManager(port=str(port))
    try:
        if not server.start_server():
            server.print_process_log()
            raise Exception("server start failed")
        if world_rank:
            logging.info(f"world rank non-zero: {world_rank}, wait for main.")
            wait_master_done(world_rank=world_rank)
        else:
            test_main(
                port,
                py_env_configs,
                batch_size_list,
                input_len_list,
                input_query_dict,
                is_decode,
                log_dir_path,
                decode_test_length,
            )
    finally:
        server.stop_server()
        upload_path = os.environ.get("UPLOAD_OSS_PATH", "")
        if upload_path != "":
            logging.info(f"upload log to {upload_path}")
            try_upload_log(log_dir_path, upload_path)
        script_exit(pgrp_set)
