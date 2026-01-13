import asyncio
import logging
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from unittest import TestCase, main
from unittest.mock import patch

import requests
from pydantic import BaseModel

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.worker_info import g_worker_info
from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.frontend.frontend_server import FrontendServer, FrontendWorker
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.server.backend_manager import BackendManager
from pytest import mark
from rtp_llm.test.utils.port_util import PortManager
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


@mark.gpu
@mark.A10
@mark.cuda
class ConcurrencyLimitTest(TestCase):
    def _compute_worker_base_ports(self) -> tuple[int, int]:
        """
        Recover (start_port, remote_server_port) that can restore g_worker_info via reload().
        """
        local_rank = int(getattr(g_worker_info, "local_rank", 0))
        worker_info_port_num = int(getattr(g_worker_info, "worker_info_port_num", 8))
        start_port = int(g_worker_info.server_port) - local_rank * worker_info_port_num
        remote_server_port = (
            int(g_worker_info.remote_rpc_server_port) - local_rank * worker_info_port_num - 1
        )
        return start_port, remote_server_port

    def setUp(self) -> None:
        super().setUp()

        # Patch only within this test case lifecycle (avoid global side effects on import).
        patchers = [
            patch.object(FrontendWorker, "__init__", new=fake_init),
            patch.object(FrontendWorker, "inference", new=fake_inference),
            patch.object(BackendManager, "start", new=fake_start),
            patch.object(BackendManager, "ready", new=fake_ready),
            patch.object(OpenaiEndpoint, "__init__", new=fake_init),
            patch.object(OpenaiEndpoint, "chat_completion", new=fake_inference),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

        # Isolate env changes.
        old_env = {
            "CONCURRENCY_LIMIT": os.environ.get("CONCURRENCY_LIMIT"),
            "START_PORT": os.environ.get("START_PORT"),
        }

        def _restore_env() -> None:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(_restore_env)

        # Isolate g_worker_info changes.
        old_start_port, old_remote_server_port = self._compute_worker_base_ports()
        self.addCleanup(lambda: g_worker_info.reload(old_start_port, old_remote_server_port))

        # Allocate a consecutive port range using project PortManager to avoid conflicts across tests.
        # For local_rank=0, g_worker_info uses start_port..start_port+worker_info_port_num-1.
        worker_info_port_num = g_worker_info.worker_info_port_num
        ports, locks = PortManager().get_consecutive_ports(worker_info_port_num)
        self.addCleanup(lambda: [lock.__exit__(None, None, None) for lock in locks])
        self.port = int(ports[0])
        os.environ["CONCURRENCY_LIMIT"] = "16"
        os.environ["START_PORT"] = str(self.port)
        g_worker_info.reload(self.port, self.port + 1)

        py_env_configs = PyEnvConfigs()
        self.frontend_app = FrontendApp(py_env_configs)
        self.backend_manager = BackendManager(py_env_configs)

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

        with patch.object(FrontendServer, "_infer_impl", new=_exception_infer_impl):
            self.curl_exception(False)
            self.curl_exception(True)
            self.assertEqual(self.get_available_concurrency(), 16)

        with patch.object(
            FrontendServer,
            "_collect_complete_response_and_record_access_log",
            new=_exception_func,
        ):
            self.curl_exception(False)
            self.curl_exception(True)
            self.assertEqual(self.get_available_concurrency(), 16)


if __name__ == "__main__":
    main()
