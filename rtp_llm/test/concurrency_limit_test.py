import asyncio
import unittest
from typing import Any

import pytest
from pydantic import BaseModel

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyException,
    init_controller,
    set_global_controller,
)

pytestmark = [pytest.mark.gpu(type="A10")]


class _Response(BaseModel):
    value: str


class _RawRequest:
    headers: dict[str, str] = {}

    async def is_disconnected(self) -> bool:
        return False


class _GatedWorker:
    def __init__(self, started: asyncio.Event, release: asyncio.Event, count: int):
        self.started = started
        self.release = release
        self.target_count = count
        self.current_count = 0

    def inference(self, prompt: str, *args: Any, **kwargs: Any):
        async def response_generator():
            self.current_count += 1
            if self.current_count == self.target_count:
                self.started.set()
            await self.release.wait()
            yield _Response(value=prompt)

        return CompleteResponseAsyncGenerator(
            response_generator(), CompleteResponseAsyncGenerator.get_last_value
        )

    def is_streaming(self, *args: Any, **kwargs: Any) -> bool:
        return False


class _FailingWorker:
    def inference(self, *args: Any, **kwargs: Any):
        raise RuntimeError("inference failed")

    def is_streaming(self, *args: Any, **kwargs: Any) -> bool:
        return False


class ConcurrencyLimitTest(unittest.IsolatedAsyncioTestCase):
    def _make_server(self, limit: int, worker: Any):
        py_env_configs = PyEnvConfigs()
        py_env_configs.concurrency_config.concurrency_limit = limit
        controller = init_controller(py_env_configs.concurrency_config)
        set_global_controller(controller)
        server = FrontendServer(
            rank_id=0,
            server_id=0,
            py_env_configs=py_env_configs,
        )
        server._frontend_worker = worker
        return server, controller

    async def test_simple(self):
        request_count = 10
        started = asyncio.Event()
        release = asyncio.Event()
        worker = _GatedWorker(started, release, request_count)
        server, controller = self._make_server(16, worker)

        requests = [
            asyncio.create_task(
                server.inference({"prompt": str(i)}, raw_request=_RawRequest())
            )
            for i in range(request_count)
        ]
        await asyncio.wait_for(started.wait(), timeout=5)
        self.assertEqual(controller.get_available_concurrency(), 6)

        release.set()
        responses = await asyncio.gather(*requests)
        self.assertEqual(len(responses), request_count)
        self.assertEqual(controller.get_available_concurrency(), 16)

    async def test_exception(self):
        server, controller = self._make_server(2, _FailingWorker())

        response = await server.inference(
            {"prompt": "fails"}, raw_request=_RawRequest()
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(controller.get_available_concurrency(), 2)
        controller.increment()
        controller.increment()
        with self.assertRaises(ConcurrencyException):
            controller.increment()
        controller.decrement()
        controller.decrement()


if __name__ == "__main__":
    unittest.main()
