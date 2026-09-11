import asyncio
import json
from typing import Any
from unittest import TestCase, main

from pydantic import BaseModel

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.structure.request_constants import request_id_field_name
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import init_controller, set_global_controller


class FakePipelinResponse(BaseModel):
    res: str


class FakeFrontendWorker(object):
    class FakeBackendRpcServerVisitor:
        def __init__(self):
            self.refresh_calls = []

        def is_backend_service_ready(self, refresh: bool = False):
            self.refresh_calls.append(refresh)
            return True

    def __init__(self):
        self.backend_rpc_server_visitor = self.FakeBackendRpcServerVisitor()

    def inference(self, prompt: str, *args: Any, **kwargs: Any):
        response_generator = self._inference(prompt, *args, **kwargs)
        return CompleteResponseAsyncGenerator(
            response_generator, CompleteResponseAsyncGenerator.get_last_value
        )

    def tokenizer_encode(self, prompt: str):
        return [1, 2, 3, 4], ["b", "c", "d", "e"]

    async def _inference(self, prompt: str, *args: Any, **kwargs: Any):
        yield FakePipelinResponse(res=prompt)

    def is_streaming(self, *args: Any, **kwargs: Any):
        return False


class FakeRawRequest(object):
    headers = {}

    async def is_disconnected(self):
        return False


class FrontendServerTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Create PyEnvConfigs with default values for testing
        py_env_configs = PyEnvConfigs()
        set_global_controller(init_controller(py_env_configs.concurrency_config))
        py_env_configs.server_config.start_port = 0
        py_env_configs.server_config.rank_id = 0
        self.frontend_server = FrontendServer(
            rank_id=0,
            server_id=0,
            py_env_configs=py_env_configs,
        )
        self.frontend_server._frontend_worker = FakeFrontendWorker()

    async def _async_run(self, *args: Any, **kwargs: Any):
        res = await self.frontend_server.inference(*args, **kwargs)
        return res

    def test_simple(self):
        loop = asyncio.new_event_loop()
        res = loop.run_until_complete(
            self._async_run(req={"prompt": "hello"}, raw_request=FakeRawRequest())
        )
        self.assertEqual(
            res.body.decode("utf-8"), '{"res":"hello"}', res.body.decode("utf-8")
        )
        res = loop.run_until_complete(
            self._async_run(req='{"prompt": "hello"}', raw_request=FakeRawRequest())
        )
        self.assertEqual(
            res.body.decode("utf-8"), '{"res":"hello"}', res.body.decode("utf-8")
        )

    def test_encode(self):
        res = self.frontend_server.tokenizer_encode('{"prompt": "b c d e"}')
        self.assertEqual(
            res.body.decode("utf-8"),
            '{"token_ids":[1,2,3,4],"tokens":["b","c","d","e"],"error":""}',
        )
        # test error input
        res = self.frontend_server.tokenizer_encode('{"text": "b c d e"}')
        self.assertEqual(json.loads(res.body.decode("utf-8"))["error_code"], 514)

    def test_check_health_uses_cached_service_discovery(self):
        self.assertTrue(self.frontend_server.check_health())
        visitor = self.frontend_server._frontend_worker.backend_rpc_server_visitor
        self.assertEqual(visitor.refresh_calls, [False])

    def test_stream_completion_markers(self):
        async def collect(openai):
            async def generate():
                yield FakePipelinResponse(res="first")
                yield FakePipelinResponse(res="last")

            response = CompleteResponseAsyncGenerator(
                generate(), CompleteResponseAsyncGenerator.get_last_value
            )
            self.frontend_server._global_controller.increment()
            return [
                chunk
                async for chunk in self.frontend_server.stream_response(
                    {"stream": openai, request_id_field_name: 1}, response
                )
            ]

        self.assertEqual(
            asyncio.run(collect(True)),
            [
                'data: {"res":"first"}\r\n\r\n',
                'data: {"res":"last"}\r\n\r\n',
                "data: [DONE]\r\n\r\n",
            ],
        )
        self.assertEqual(
            asyncio.run(collect(False)),
            [
                'data:{"res":"first"}\r\n\r\n',
                'data:{"res":"last"}\r\n\r\n',
                "data:[done]\r\n\r\n",
            ],
        )

    def test_failed_stream_has_no_success_marker(self):
        async def collect():
            async def generate():
                yield FakePipelinResponse(res="partial")
                raise RuntimeError("generation failed")

            response = CompleteResponseAsyncGenerator(
                generate(), CompleteResponseAsyncGenerator.get_last_value
            )
            self.frontend_server._global_controller.increment()
            return [
                chunk
                async for chunk in self.frontend_server.stream_response(
                    {"stream": True, request_id_field_name: 2}, response
                )
            ]

        chunks = asyncio.run(collect())
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], 'data: {"res":"partial"}\r\n\r\n')
        self.assertNotIn("[DONE]", "".join(chunks))
        self.assertIn("error_code", json.loads(chunks[1][6:]))

    def test_failed_response_aggregation_has_no_success_marker(self):
        async def collect(openai):
            async def generate():
                yield FakePipelinResponse(res="partial")

            async def fail_aggregation(_):
                raise RuntimeError("response aggregation failed")

            response = CompleteResponseAsyncGenerator(generate(), fail_aggregation)
            self.frontend_server._global_controller.increment()
            return [
                chunk
                async for chunk in self.frontend_server.stream_response(
                    {"stream": openai, request_id_field_name: 3}, response
                )
            ]

        for openai in (True, False):
            with self.subTest(openai=openai):
                chunks = asyncio.run(collect(openai))
                self.assertEqual(len(chunks), 2)
                self.assertNotIn("[done]", "".join(chunks).lower())
                self.assertIn("error_code", json.loads(chunks[1].split(":", 1)[1]))


main()
