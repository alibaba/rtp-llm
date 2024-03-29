import time
import requests
import copy
import os
import traceback
import datetime
import logging
import uvicorn
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from fastapi import FastAPI
from datetime import timedelta

import torch.distributed as dist
from maga_transformer.config.uvicorn_config import UVICORN_LOGGING_CONFIG
from maga_transformer.distribute.worker_info import g_worker_info, g_parallel_info, g_master_info, update_master_info
from maga_transformer.distribute.gang_info import get_gang_info, GangInfo

# for ut
from maga_transformer.distribute.gang_test_util import create_store, store_based_barrier

def http_post_with_retry(url: str, data: Dict[str, Any], retry_limit: int = 3):
    retry_time = 0
    ret = None
    while retry_time < retry_limit:
        retry_time += 1
        try:
            ret = requests.post(url, json=data, timeout=10)
            if ret.status_code == 200:
                return ret
        except:
            if retry_time == retry_limit:
                raise
    return ret

class GangServer:
    def __init__(self):
        self._initialized: bool = False
        self._gang_status: Dict[str, str] = {}
        self._gang_info = None
        self._request_threadpool = ThreadPoolExecutor(g_parallel_info.world_size)

    def _start_server(self):
        app = FastAPI()

        @app.post("/heartbeat")
        def heartbeat(req: Dict[str, str]):
            name = req['name']
            ip = req['ip']
            if not self._initialized:
                logging.info(f'server member recv: {name} {ip}')
                self._gang_status[name] = ip
            return {'initializing':not self._initialized}

        try:
            uvicorn.run(app, host="0.0.0.0", port=g_worker_info.gang_hb_port, log_config=UVICORN_LOGGING_CONFIG)
        except:
            logging.error(traceback.format_exc())
            os._exit(-1)

    def _start(self):
        t = Thread(target=self._start_server)
        t.daemon = True
        t.start()

    def _check_gang_info(self, gang_info: GangInfo):
        if gang_info.master is None or gang_info.self is None:
            raise Exception(f"gang_info master {gang_info.master}, gang_info self {gang_info.self}")
        if len(gang_info.members) != g_parallel_info.world_size:
            raise Exception(f'gang_size({gang_info.members}) != world_size({g_parallel_info.world_size})')
        for member in gang_info.members:
            if not member.ip or not member.name:
                raise Exception(f'gang_info not complete: {gang_info}')

    def _exchange_gang_info(self, gang_info: GangInfo):
        for member in gang_info.members:
            url = f'http://{member.ip}:{member.gang_hb_port}/heartbeat'
            try:
                result = requests.post(url, json={"name": gang_info.self.name, "ip": gang_info.self.ip}, timeout=1)
            except Exception as e:
                logging.warning(f'request {url} failed, {str(e)}')
                continue
            if result.status_code == 200:
                response = result.json()
                if response['initializing'] == True:
                    logging.info(f'client member recv: {member.name} {member.ip}')
                    self._gang_status[member.name] = member.ip

    def _check_ready(self):
        for member in self._gang_info.members:
            actual_ip = self._gang_status.get(member.name)
            if actual_ip != member.ip:
                raise Exception(f'gang member {member.name} expect {member.ip} actual {actual_ip}')

    def _wait_ready(self):
        timeout_minutes = int(os.environ.get('GANG_TIMEOUT_MIN', '30'))
        sleep_time = int(os.environ.get('GANG_SLEEP_TIME', '10'))
        start_time = datetime.datetime.now()
        retry_time = 0
        while True:
            try:
                self._gang_info = get_gang_info()
                self._check_gang_info(self._gang_info)
                self._exchange_gang_info(self._gang_info)
                self._check_ready()
                return
            except Exception as e:
                logging.warning(f"gang cluster is not complete, will retry, retry times: {retry_time}, error_msg: {str(e)}")
                cur_time = datetime.datetime.now()
                if cur_time - start_time > datetime.timedelta(minutes=timeout_minutes):
                    raise Exception("failed to start gang server")
                retry_time += 1
                time.sleep(sleep_time)    
    
    def wait_infernece_server_ready(self):
        for member in self._gang_info.workers():                
            url = f'http://{member.ip}:{member.server_port}/health'
            try:
                resposne = requests.get(url, timeout=1)
            except:                
                raise Exception(f"failed to check member {member.ip}:{member.server_port}/health")
            if resposne.status_code != 200:
                raise Exception(f"member {member.ip}:{member.server_port} /health status_code: {resposne.status_code}, is not ready")

    def request_workers(self, req: Dict[str, Any], uri: str = 'inference_internal'):
        req = copy.deepcopy(req)
        def curl_impl(url: str):
            _ = requests.post(url, json=req)
        for member in self._gang_info.workers():
            url = f'http://{member.ip}:{member.server_port}/{uri}'
            self._request_threadpool.submit(curl_impl, url)

    def _health_check_impl(self):
        for member in self._gang_info.members:
            url = f'http://{member.ip}:{member.gang_hb_port}/heartbeat'
            try:
                res = http_post_with_retry(url, {"name": "", "ip": ""})
            except:
                logging.error(f"Gang server {member.ip} heartbeat loss, do abort")
                os._exit(-1)
            if res.status_code != 200:
                logging.error(f"Gang server {member.ip} status code is not 200, do abort")
                os._exit(-1)
            if res.json()['initializing'] == True:
                logging.error(f"Gang server {member.ip} is restarted, do abort")
                os._exit(-1)

    def _start_health_check(self):
        sleep_time = int(os.environ.get('GANG_SLEEP_TIME', '10'))
        def wrapper():            
            while True:
                self._health_check_impl()
                time.sleep(sleep_time)
        t = Thread(target=wrapper)
        t.daemon = True
        t.start()
     
    # for ut
    # c10d自带tcp_barrier依赖group已经创建，所以需要重新实现一个
    def memory_barrier(self, master_url: str):
        self.store = create_store(master_url, g_parallel_info.world_rank, g_parallel_info.world_size)
        store_based_barrier(g_parallel_info.world_rank, g_parallel_info.world_size, self.store, timeout=timedelta(seconds=10))
        
    def init_process_with_timeout(self, master_url: str, world_rank: int , world_size: int, timeout_seconds: int):
        def init_func():
            dist.init_process_group(backend=dist.Backend.NCCL, init_method=master_url, rank=world_rank, world_size=world_size)
            logging.info('torch dist init done')
        timeout = datetime.timedelta(seconds=timeout_seconds)
        start_time = datetime.datetime.now()
        t = Thread(target=init_func, daemon=True)
        t.start()
        while t.is_alive():        
            now = datetime.datetime.now()
            if now - start_time > timeout:
                raise Exception(f"timeout, expect init in {timeout_seconds}s, acutal use time: {now - start_time}")
            time.sleep(0.1)

    def start(self):
        if g_parallel_info.world_size == 1:
            logging.info("world_size==1, do not start gang_server")
            return
        self._start()
        self._wait_ready()
        self._initialized = True
        update_master_info(
            self._gang_info.master.ip,
            self._gang_info.master.server_port)
        master_url = f"tcp://{g_master_info.ip}:{g_master_info.th_nccl_port}"
        logging.info(f'gang info exchange done, master_url: {master_url} gang_info: {self._gang_info}')
        # init_process_group会去检查gpu num > 0, 所以测试环境不希望init_process_group，使用tcp barrier作为代替
        if os.environ.get('FAKE_GANG_ENV', None) == None:
            init_process_timeout = int(os.environ.get('INI_PROCESS_WITH_TIMEOUT', 300))
            self.init_process_with_timeout(master_url, g_parallel_info.world_rank, g_parallel_info.world_size, init_process_timeout)
        else:
            self.memory_barrier(master_url)
        self._start_health_check()
        logging.info(f'gang init done')
