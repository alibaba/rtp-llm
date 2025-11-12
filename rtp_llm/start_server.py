import logging
import multiprocessing
import os
import sys
import time

import requests

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

import json

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info, update_worker_info
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.concurrency_controller import init_controller
from rtp_llm.utils.fuser import fetch_remote_file_to_local


def fetch_model_files_to_local(py_env_configs: PyEnvConfigs):
    """Fetch remote model files to local and update py_env_configs in place."""
    # Fetch checkpoint_path from model_args
    model_args = py_env_configs.model_args
    if model_args.ckpt_path:
        model_args.ckpt_path = fetch_remote_file_to_local(
            model_args.ckpt_path
        )
    
    # Fetch tokenizer_path from model_args
    tokenizer_path = model_args.tokenizer_path
    if not tokenizer_path:
        tokenizer_path = model_args.ckpt_path
    if tokenizer_path:
        model_args.tokenizer_path = fetch_remote_file_to_local(tokenizer_path)
    
    # Fetch extra_data_path from model_args
    if model_args.extra_data_path:
        local_extra_data_path = fetch_remote_file_to_local(
            model_args.extra_data_path
        )
        model_args.local_extra_data_path = local_extra_data_path
    
    # Fetch ptuning_path from model_args
    if model_args.ptuning_path:
        model_args.ptuning_path = fetch_remote_file_to_local(
            model_args.ptuning_path
        )
    
    # Fetch lora paths
    lora_config = py_env_configs.lora_config
    if lora_config.lora_info:
        try:
            lora_infos = json.loads(lora_config.lora_info)
            for lora_name, lora_path in lora_infos.items():
                lora_infos[lora_name] = fetch_remote_file_to_local(lora_path)
            # Update lora_info back to string format
            lora_config.lora_info = json.dumps(lora_infos)
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"Failed to parse lora_info: {e}, skipping lora path fetching")
    
    # Fetch sp_checkpoint_path if exists
    sp_config = py_env_configs.sp_config
    if sp_config.sp_checkpoint_path:
        sp_config.sp_checkpoint_path = fetch_remote_file_to_local(
            sp_config.sp_checkpoint_path
        )
    
    logging.info(
        f"Fetched model files - checkpoint_path: {model_args.ckpt_path}, "
        f"tokenizer_path: {model_args.tokenizer_path}, "
        f"ptuning_path: {model_args.ptuning_path}, "
        f"extra_data_path: {model_args.local_extra_data_path}"
    )


def check_server_health(server_port):
    try:
        response = requests.get(f"http://localhost:{server_port}/health", timeout=60)
        logging.info(
            f"response status_code = {response.status_code}, text = {response.text}, len = {len(response.text)}"
        )
        if response.status_code == 200 and response.text.strip() == '"ok"':
            return True
        else:
            logging.info(f"health check is not ready")
            return False
    except BaseException as e:
        logging.debug("health check is not ready, %s", str(e))
        return False


def start_backend_server_impl(global_controller, py_env_configs):
    from rtp_llm.start_backend_server import start_backend_server

    # only for debug
    if py_env_configs.profiling_debug_config.debug_load_server:
        start_backend_server(global_controller, py_env_configs)
        os._exit(-1)
    backend_process = multiprocessing.Process(
        target=start_backend_server, args=(global_controller, py_env_configs), name="backend_server"
    )
    backend_process.start()

    retry_interval_seconds = 5
    start_port = py_env_configs.server_config.start_port
    backend_server_port = WorkerInfo.backend_server_port_offset(0, start_port)
    while True:
        if not backend_process.is_alive():
            monitor_and_release_process(backend_process, None)
            raise Exception("backend server is not alive")

        try:
            if check_server_health(backend_server_port):
                logging.info(f"backend server is ready")
                break
            else:
                time.sleep(retry_interval_seconds)
        except Exception as e:
            logging.info(f"backend server is not ready")
            time.sleep(retry_interval_seconds)

    return backend_process


def start_frontend_server_impl(global_controller, backend_process, py_env_configs):
    from rtp_llm.start_frontend_server import start_frontend_server

    frontend_server_count = py_env_configs.server_config.frontend_server_count
    if frontend_server_count < 1:
        logging.info(
            "frontend server's count is {frontend_server_count}, this may be a mistake"
        )

    frontend_processes = []

    # tmp code
    local_world_size = g_parallel_info.world_size
    if "LOCAL_WORLD_SIZE" in os.environ:
        logging.info(
            f"multi rank starts with local world size specified in env: {os.environ['LOCAL_WORLD_SIZE']}"
        )
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        logging.info(
            f"multi rank starts with default local world size: {local_world_size}, world size = {g_parallel_info.world_size}"
        )

    for rank in range(local_world_size):
        for i in range(frontend_server_count):
            process = multiprocessing.Process(
                target=start_frontend_server,
                args=(rank, i, global_controller, py_env_configs),
                name=f"frontend_server_{i}",
            )
            frontend_processes.append(process)
            process.start()

    retry_interval_seconds = 5
    start_port = py_env_configs.server_config.start_port

    while True:
        if not all(proc.is_alive() for proc in frontend_processes):
            monitor_and_release_process(backend_process, frontend_processes)
            raise Exception("frontend server is not alive")

        try:
            check_server_health(start_port)
            logging.info(f"frontend server is ready")
            break
        except Exception as e:
            # 如果连接失败，等待一段时间后重试
            time.sleep(retry_interval_seconds)

    return frontend_processes


def monitor_and_release_process(backend_process, frontend_process):
    all_process = []
    if backend_process:
        all_process.append(backend_process)
    if frontend_process:
        all_process.extend(frontend_process)
    logging.info(f"all process = {all_process}")

    while any(proc.is_alive() for proc in all_process):
        if not all(proc.is_alive() for proc in all_process):
            logging.error(f"server monitor : some process is not alive, exit!")
            for proc in all_process:
                try:
                    proc.terminate()
                except Exception as e:
                    logging.error(f"catch exception when process terminate : {str(e)}")
        time.sleep(1)
    [proc.join() for proc in all_process]

    logging.info("all process exit")


def main():
    py_env_configs: PyEnvConfigs = setup_args()
    fetch_model_files_to_local(py_env_configs)
    update_worker_info(py_env_configs.server_config.start_port, py_env_configs.worker_config.worker_info_port_num)

    start_server(py_env_configs)


def start_server(py_env_configs):
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logging.warning(str(e))
    global_controller = init_controller(py_env_configs.concurrency_config,
                                        dp_size=g_parallel_info.dp_size)
    backend_process = None
    frontend_process = None
    try:
        if os.environ.get("ROLE_TYPE", "") != "FRONTEND":
            logging.info("start backend server")
            backend_process = start_backend_server_impl(global_controller, py_env_configs)
            logging.info(f"backend server process = {backend_process}")

        logging.info("start frontend server")
        frontend_process = start_frontend_server_impl(
            global_controller, backend_process, py_env_configs
        )
        logging.info(f"frontend server process = {frontend_process}")

        logging.info(f"后端RPC 服务监听的ip为 0.0.0.0，ip/ip段可自定义为所需范围")
    except Exception as e:
        import traceback
        logging.error(f"start failed, {str(e)}")
        logging.error(f"Traceback:\n{traceback.format_exc()}")
    finally:
        monitor_and_release_process(backend_process, frontend_process)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    main()
