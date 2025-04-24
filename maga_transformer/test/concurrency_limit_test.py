import os
import random
import time
import requests
import logging
import unittest
from unittest import TestCase, main
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import asyncio
from pydantic import BaseModel, Field
import uvicorn
from fastapi import FastAPI

from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.server.frontend_server import FrontendWorker, FrontendServer
from maga_transformer.server.frontend_app import FrontendApp
from maga_transformer.server.backend_server import BackendServer
from maga_transformer.server.backend_app import BackendApp
from maga_transformer.distribute.worker_info import g_worker_info, g_parallel_info
from maga_transformer.test.utils.port_util import get_consecutive_free_ports
from maga_transformer.utils.complete_response_async_generator import CompleteResponseAsyncGenerator
from maga_transformer.ops import LoadBalanceInfo

def fake_init(self, *args, **kwargs):
    self.model_config = None
    self.tokenizer = None
    self.model_cls = None
    self.pipeline = None
    self.backend_rpc_server_visitor = None

def fake_start(self):
    pass

def fake_ready(self):
    return True

def _exception_infer_impl(self, *args, **kwargs):
    raise Exception("exception in infer impl function")

async def _exception_func(self, *args, **kwargs):
    raise Exception("exception in function")

class FakePipelineResponse(BaseModel):
    hello: str

def fake_inference(*args, **kwargs):
    async def response_generator():
        for _ in range(5):
            await asyncio.sleep(1)
            yield FakePipelineResponse(hello="gg")
    return CompleteResponseAsyncGenerator(response_generator(), CompleteResponseAsyncGenerator.get_last_value)

def fake_load_balance_info(*args, **kwargs):
    load_balance = LoadBalanceInfo()
    load_balance.step_latency_us = 1000
    load_balance.iterate_count = 10
    load_balance.step_per_minute = 60
    return load_balance

FrontendWorker.__init__ = fake_init
FrontendWorker.inference = fake_inference

BackendServer.start = fake_start
BackendServer.ready = fake_ready
BackendServer.get_load_balance_info = fake_load_balance_info
BackendServer.model_runtime_meta = lambda x: "fake_model"

OpenaiEndopoint.__init__ = fake_init
OpenaiEndopoint.chat_completion = fake_inference

class ConcurrencyLimitTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.port = random.randint(20000, 30000)
        os.environ['CONCURRENCY_LIMIT'] = '16'
        os.environ['START_PORT'] = str(self.port)
        g_worker_info.reload()
        self.frontend_app = FrontendApp()
        self.backend_app = BackendApp()

    def start_frontend_server(self):
        t = Thread(target=self.frontend_app.start, daemon=True)
        t.start()
    
    def start_backend_server(self):
        t = Thread(target=self.backend_app.start, daemon=True)
        t.start()

    def wait_server_start(self, port):
        timeout = 60
        while timeout > 0:
            try:
                res = requests.get(f"http://localhost:{port}/health", timeout=1)
                if res.status_code == 200:
                    break
            except Exception as e:
                print(e)
            timeout -= 5
            time.sleep(5)
        if timeout <= 0:
            raise Exception("faile to start server")

    def curl(self):
        res = requests.post(f"http://localhost:{g_worker_info.server_port}", json={"prompt": "gg!"}, timeout=60)
        logging.info(f"result:{res.status_code} {res.text}")
        self.assertTrue(res.status_code == 200)

    def curl_exception(self, is_streaming = False):
        res = requests.post(f"http://localhost:{g_worker_info.server_port}",
            json={"prompt": "gg!", "generate_config": {"is_streaming": is_streaming}}, timeout=60)
        logging.info(f"result:{res.status_code} {res.text}")
        self.assertTrue(res.status_code != 200)

    def chat_completion(self):
        res = requests.post(f"http://localhost:{g_worker_info.server_port}/chat/completions", json={"messages": []}, timeout=60)
        logging.info(f"result:{res.status_code} {res.text}")
        self.assertTrue(res.status_code == 200)

    def get_available_concurrency(self):
        res = requests.get(f"http://localhost:{g_worker_info.server_port}/worker_status", timeout=1)
        logging.info(f"result:{res.status_code} {res.text}")
        self.assertTrue(res.status_code == 200)
        return res.json()['frontend_available_concurrency']

    def get_backend_available_concurrency(self):
        res = requests.get(f"http://localhost:{g_worker_info.server_port}/worker_status", timeout=1)
        logging.info(f"result:{res.status_code} {res.text}")
        self.assertTrue(res.status_code == 200)
        return res.json()['available_concurrency']

    def get_worker_status(self):
        res = requests.get(f"http://localhost:{g_worker_info.server_port}/worker_status", timeout=1)
        logging.info(f"result:{res.status_code} {res.text}")
        self.assertTrue(res.status_code == 200)
        return res.json()

    # 直接测端到端的结果
    @unittest.skip("Temporarily disabled test case")
    def test_simple(self):
        executor = ThreadPoolExecutor(100)
        self.start_frontend_server()
        self.start_backend_server()
        time.sleep(6)
        self.wait_server_start(g_worker_info.server_port)
        self.wait_server_start(g_worker_info.backend_server_port)
        self.curl()
        for i in range(10):
            executor.submit(self.curl)
        time.sleep(1)
        self.assertEqual(self.get_available_concurrency(), 6)
        time.sleep(10)
        self.assertEqual(self.get_available_concurrency(), 16)

        for i in range(10):
            executor.submit(self.chat_completion)
        time.sleep(1)
        self.assertEqual(self.get_available_concurrency(), 6)
        time.sleep(10)
        self.assertEqual(self.get_available_concurrency(), 16)

        os.environ['LOAD_BALANCE'] = "1"
        self.assertEqual(self.get_backend_available_concurrency(), 60)
        load_balance_info = fake_load_balance_info()
        excepted = {
            "available_concurrency": load_balance_info.step_per_minute,
            "available_kv_cache": 0,
            "total_kv_cache": 0,
            "step_latency_ms": load_balance_info.step_latency_us / 1000,
            "step_per_minute": load_balance_info.step_per_minute,
            "onflight_requests": load_balance_info.onflight_requests,
            "iterate_count": load_balance_info.iterate_count,
            "version": 1,
            "alive": True,
            'finished_task_list': [],
            'last_schedule_delta': 0,
            'running_task_list': [],
            'machine_info': 'fake_model',
            "frontend_available_concurrency": 16
        }
        logging.info(f"self.get_worker_status() = {self.get_worker_status()}")
        logging.info(f"excepted = {excepted}")
        self.assertEqual(self.get_worker_status(), excepted)

    @unittest.skip("Temporarily disabled test case")
    def test_exception(self):
        self.start_frontend_server()
        self.start_backend_server()
        time.sleep(6)
        self.wait_server_start(g_worker_info.server_port)
        self.wait_server_start(g_worker_info.backend_server_port)

        origin_func = FrontendServer._infer_impl
        FrontendServer._infer_impl = _exception_infer_impl
        self.curl_exception(False)
        self.curl_exception(True)
        self.assertEqual(self.get_available_concurrency(), 16)
        FrontendServer._infer_impl = origin_func
        
        origin_func = FrontendServer._collect_complete_response_and_record_access_log
        FrontendServer._collect_complete_response_and_record_access_log = _exception_func
        self.curl_exception(False)
        self.curl_exception(True)
        self.assertEqual(self.get_available_concurrency(), 16)
        FrontendServer._collect_complete_response_and_record_access_log =origin_func

if __name__ == '__main__':
    main()
