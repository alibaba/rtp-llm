import asyncio
import logging
import os
import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Any
from unittest import TestCase, main

import requests
from pydantic import BaseModel

from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.distribute.worker_info import g_worker_info
from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.frontend.frontend_server import FrontendServer, FrontendWorker
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.server.backend_manager import BackendManager
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)


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

    return CompleteResponseAsyncGenerator(
        response_generator(), CompleteResponseAsyncGenerator.get_last_value
    )


FrontendWorker.__init__ = fake_init
FrontendWorker.inference = fake_inference

BackendManager.start = fake_start
BackendManager.ready = fake_ready

OpenaiEndpoint.__init__ = fake_init
OpenaiEndpoint.chat_completion = fake_inference


class ConcurrencyLimitTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.port = random.randint(20000, 30000)
        os.environ["CONCURRENCY_LIMIT"] = "16"
        os.environ["START_PORT"] = str(self.port)
        g_worker_info.reload()
        self.frontend_app = FrontendApp()
        self.backend_manager = BackendManager(StaticConfig)

    def start_frontend_server(self):
        t = Thread(target=self.frontend_app.start, daemon=True)
        t.start()

    def start_backend_server(self):
        t = Thread(target=self.backend_manager.start, args=(), daemon=True)
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
        res = requests.post(
            f"http://localhost:{g_worker_info.server_port}",
            json={"prompt": "gg!"},
            timeout=60,
        )
        logging.info(f"result:{res.status_code} {res.text}")
        self.assertTrue(res.status_code == 200)

    def curl_exception(self, is_streaming=False):
        res = requests.post(
            f"http://localhost:{g_worker_info.server_port}",
            json={"prompt": "gg!", "generate_config": {"is_streaming": is_streaming}},
            timeout=60,
        )
        logging.info(f"result:{res.status_code} {res.text}")
        self.assertTrue(res.status_code != 200)

    def chat_completion(self):
        res = requests.post(
            f"http://localhost:{g_worker_info.server_port}/chat/completions",
            json={"messages": []},
            timeout=60,
        )
        logging.info(f"result:{res.status_code} {res.text}")
        self.assertTrue(res.status_code == 200)

    def get_available_concurrency(self):
        res = requests.get(
            f"http://localhost:{g_worker_info.server_port}/worker_status", timeout=1
        )
        logging.info(f"result:{res.status_code} {res.text}")
        self.assertTrue(res.status_code == 200)
        return res.json()["frontend_available_concurrency"]

    def get_backend_available_concurrency(self):
        res = requests.get(
            f"http://localhost:{g_worker_info.server_port}/worker_status", timeout=1
        )
        logging.info(f"result:{res.status_code} {res.text}")
        self.assertTrue(res.status_code == 200)
        return res.json()["available_concurrency"]

    def get_worker_status(self):
        res = requests.get(
            f"http://localhost:{g_worker_info.server_port}/worker_status", timeout=1
        )
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

        self.assertEqual(self.get_backend_available_concurrency(), 60)
        excepted = {
            "available_kv_cache": 0,
            "total_kv_cache": 0,
            "version": 1,
            "alive": True,
            "finished_task_list": [],
            "last_schedule_delta": 0,
            "running_task_list": [],
            "machine_info": "fake_model",
            "frontend_available_concurrency": 16,
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
        FrontendServer._collect_complete_response_and_record_access_log = (
            _exception_func
        )
        self.curl_exception(False)
        self.curl_exception(True)
        self.assertEqual(self.get_available_concurrency(), 16)
        FrontendServer._collect_complete_response_and_record_access_log = origin_func


if __name__ == "__main__":
    main()
