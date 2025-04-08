import os
import sys
import json
import time
import logging
import logging.config
import traceback
import requests
import signal
import multiprocessing
from typing import Generator, Union, Any, Dict, List

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), '..'))

from maga_transformer.config.log_config import LOGGING_CONFIG
from maga_transformer.distribute.worker_info import WorkerInfo, DEFAULT_START_PORT
from maga_transformer.start_backend_server import start_backend_server
from maga_transformer.start_frontend_server import start_frontend_server
from maga_transformer.utils.concurrency_controller import ConcurrencyController, init_controller

def check_server_health(server_port):
    try:
        response = requests.get(f'http://localhost:{server_port}/health', timeout=5)
        logging.info(f"response status_code = {response.status_code}, text = {response.text}, len = {len(response.text)}")
        if response.status_code == 200 and response.text.strip() == '"ok"':
            return True
        else:
            logging.info(f"health check is not ready")
            return False
    except BaseException as e:
        logging.info(f"health check is not ready, {str(e)}")
        return False

def start_backend_server_impl(global_controller):
    # only for debug
    if os.environ.get('DEBUG_LOAD_SERVER', None) == '1':
        start_backend_server(global_controller)
        os._exit(-1)
    backend_process = multiprocessing.Process(target=start_backend_server, args=(global_controller, ), name="backend_server")
    backend_process.start()
    
    retry_interval_seconds = 5
    start_port = int(os.environ.get('START_PORT', DEFAULT_START_PORT))
    backend_server_port = WorkerInfo.backend_server_port_offset(0, start_port)
    while True:
        if not backend_process.is_alive():
            monitor_and_release_process(backend_process, None)
            raise Exception("backend server is not alive")

        try:
            if check_server_health(backend_server_port):
                logging.info(f'backend server is ready')
                break
            else:
                time.sleep(retry_interval_seconds)
        except Exception as e:
            logging.info(f'backend server is not ready')
            time.sleep(retry_interval_seconds)
    
    return backend_process

def start_frontend_server_impl(global_controller, backend_process):
    frontend_server_count = int(os.environ.get('FRONTEND_SERVER_COUNT', 4))
    if frontend_server_count < 1:
        logging.info("frontend server's count is {frontend_server_count}, this may be a mistake")
    
    frontend_processes = []
    
    for i in range(frontend_server_count) :
        os.environ['FRONTEND_SERVER_ID'] = str(i)
        process = multiprocessing.Process(target=start_frontend_server,
                args=(i, global_controller), name=f"frontend_server_{i}")
        frontend_processes.append(process)
        process.start()

    retry_interval_seconds = 5
    start_port = int(os.environ.get('START_PORT', DEFAULT_START_PORT))
    while True:
        if not all(proc.is_alive() for proc in frontend_processes):
            monitor_and_release_process(backend_process, frontend_processes)
            raise Exception("frontend server is not alive")

        try:
            check_server_health(start_port)
            logging.info(f'frontend server is ready')
            break
        except Exception as e:
            # 如果连接失败，等待一段时间后重试
            time.sleep(retry_interval_seconds)

    return frontend_processes

def main():
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError as e:
        logging.warn(str(e))
        pass
    
    global_controller = init_controller()
    
    backend_process = None
    frontend_process = None
    
    try:
        logging.info("start backend server")
        backend_process = start_backend_server_impl(global_controller)
        logging.info(f"backend server process = {backend_process}")
        
        logging.info("start frontend server")
        frontend_process = start_frontend_server_impl(global_controller, backend_process)
        logging.info(f"frontend server process = {frontend_process}")
    finally:
        monitor_and_release_process(backend_process, frontend_process)
    
def monitor_and_release_process(backend_process, frontend_process):
    all_process = []
    if backend_process:
        all_process.append(backend_process)
    if frontend_process:
        all_process.extend(frontend_process)
    logging.info(f"all process = {all_process}")
    
    while any(proc.is_alive() for proc in all_process):
        if not all(proc.is_alive() for proc in all_process):
            logging.error(f'server monitor : some process is not alive, exit!')
            if backend_process:
                try:
                    os.killpg(os.getpgid(backend_process.pid), signal.SIGTERM)
                except Exception as e:
                    logging.error(f"catch exception when kill backend process : {str(e)}")
        
            for proc in all_process:
                try:
                    proc.terminate()
                except Exception as e:
                    logging.error(f"catch exception when process terminate : {str(e)}")
        time.sleep(1)
    [proc.join() for proc in all_process]

    logging.info("all process exit")

if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    main()
