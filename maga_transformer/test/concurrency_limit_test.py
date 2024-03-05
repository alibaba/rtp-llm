import os
import random
import time
import requests
import logging
from unittest import TestCase, main
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import asyncio
from pydantic import BaseModel, Field


from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.server.inference_server import InferenceWorker
from maga_transformer.server.inference_app import InferenceApp
from maga_transformer.distribute.worker_info import g_worker_info, g_parallel_info
from maga_transformer.utils.complete_response_async_generator import CompleteResponseAsyncGenerator
def fake_init(self, *args, **kwargs):
    self.model = None

class FakePipelineResponse(BaseModel):
    hello: str

def fake_inference(*args, **kwargs):
    async def response_generator():
        for _ in range(5):
            await asyncio.sleep(1)
            yield FakePipelineResponse(hello="gg")
    return CompleteResponseAsyncGenerator(response_generator(), CompleteResponseAsyncGenerator.get_last_value)

InferenceWorker.__init__ = fake_init
InferenceWorker.inference = fake_inference

OpenaiEndopoint.__init__ = fake_init
OpenaiEndopoint.chat_completion = fake_inference

# import maga_transformer.start_server
# maga_transformer.start_server.InferenceWorker = FakeInferenceWorker

class ConcurrencyLimitTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.inference_app = InferenceApp()
        self.port = random.randint(20000, 30000)
        g_worker_info.server_port = self.port
        os.environ['CONCURRENCY_LIMIT'] = '16'

    def start_in_thread(self):
        t = Thread(target=self.inference_app.start, daemon=True)
        t.start()

    def wait_server_start(self):
        timeout = 60
        while timeout > 0:
            try:
                res = requests.get(f"http://localhost:{self.port}/health", timeout=1)
                if res.status_code == 200:
                    break
            except Exception as e:
                print(e)
            timeout -= 5
            time.sleep(5)
        if timeout <= 0:
            raise Exception("faile to start server")

    def curl(self):
        res = requests.post(f"http://localhost:{self.port}", json={"prompt": "gg!"}, timeout=60)
        logging.info(f"resutl:{res.text}, {dir(res)}")
        self.assertTrue(res.status_code == 200)

    def chat_completion(self):
        res = requests.post(f"http://localhost:{self.port}/chat/completions", json={"messages": []}, timeout=60)
        self.assertTrue(res.status_code == 200)

    def get_available_concurrency(self):
        res = requests.get(f"http://localhost:{self.port}/worker_status", timeout=1)
        self.assertTrue(res.status_code == 200)
        return res.json()['available_concurrency']

    # 直接测端到端的结果
    def test_simple(self):
        executor = ThreadPoolExecutor(100)
        self.start_in_thread()
        self.wait_server_start()
        self.curl()
        for i in range(10):
            executor.submit(self.curl)
        time.sleep(1)
        # print("bbbb")
        self.assertEqual(self.get_available_concurrency(), 6)
        time.sleep(10)
        self.assertEqual(self.get_available_concurrency(), 16)

        for i in range(10):
            executor.submit(self.chat_completion)
        time.sleep(1)
        self.assertEqual(self.get_available_concurrency(), 6)
        time.sleep(10)
        self.assertEqual(self.get_available_concurrency(), 16)

    def test_openai(self):
        pass

if __name__ == '__main__':
    main()