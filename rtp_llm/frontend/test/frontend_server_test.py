import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any
from unittest import TestCase, main
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.frontend.shutdown_manager import FrontendShutdownManager
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.openai.renderers.deepseekv41_renderer import DeepseekV41Renderer
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

    def test_tokenize_route_offloads_v41_and_preserves_native_and_legacy_dispatch(self):
        async def run():
            server = self.frontend_server
            owner = FrontendApp.__new__(FrontendApp)
            owner.frontend_server = server
            owner.shutdown_manager = FrontendShutdownManager()
            owner.separated_frontend = True
            owner.server_config = SimpleNamespace(http_port=0)
            owner.grpc_client = None
            endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
            renderer = DeepseekV41Renderer.__new__(DeepseekV41Renderer)
            legacy = BasicRenderer.__new__(BasicRenderer)
            endpoint.chat_renderer = renderer
            endpoint.template_renderer = legacy
            server._openai_endpoint = endpoint
            loop = asyncio.get_running_loop()
            loop_thread = threading.get_ident()
            started = asyncio.Event()
            release = threading.Event()
            render_threads = []
            native_prompts = []

            def slow_render(req):
                render_threads.append(threading.get_ident())
                loop.call_soon_threadsafe(started.set)
                if not release.wait(10):
                    raise TimeoutError("test did not release image preparation")
                return RenderedInputs(input_ids=[129264, 129264])

            def native_encode(prompt):
                self.assertEqual(threading.get_ident(), loop_thread)
                native_prompts.append(prompt)
                return [1, 2, 3]

            def legacy_render(req):
                self.assertEqual(threading.get_ident(), loop_thread)
                return RenderedInputs(input_ids=[7, 8])

            image_request = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://image.invalid/slow.png"},
                            }
                        ],
                    }
                ]
            }
            with patch.object(
                renderer, "render_chat", side_effect=slow_render
            ), patch.object(
                legacy, "render_chat", side_effect=legacy_render
            ), patch.object(
                server._frontend_worker,
                "pipeline",
                SimpleNamespace(encode=native_encode),
                create=True,
            ):
                async with AsyncClient(
                    transport=ASGITransport(app=owner.create_app()),
                    base_url="http://test",
                ) as client:
                    slow = asyncio.create_task(
                        client.post("/tokenize", json=image_request)
                    )
                    try:
                        await asyncio.wait_for(started.wait(), timeout=5)
                        for payload in (
                            {"prompt": "hello"},
                            json.dumps({"prompt": "hello"}),
                        ):
                            response = await asyncio.wait_for(
                                client.post("/tokenize", json=payload), timeout=5
                            )
                            self.assertEqual(response.status_code, 200)
                            self.assertEqual(response.json(), {"token_ids": [1, 2, 3]})
                        response = await asyncio.wait_for(
                            client.post(
                                "/tokenize",
                                json={
                                    "messages": [{"role": "user", "content": "hello"}],
                                    "user_template": "{{ messages[0].content }}",
                                },
                            ),
                            timeout=5,
                        )
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json(), {"token_ids": [7, 8]})
                        self.assertFalse(slow.done())
                    finally:
                        release.set()
                        image_response = await asyncio.wait_for(slow, timeout=5)

                    self.assertEqual(image_response.status_code, 200)
                    self.assertEqual(
                        image_response.json(), {"token_ids": [129264, 129264]}
                    )
                    self.assertNotIn(loop_thread, render_threads)
                    self.assertEqual(native_prompts, ["hello", "hello"])

                    endpoint.chat_renderer = legacy
                    for payload in (
                        {"messages": [{"role": "user", "content": "hello"}]},
                        json.dumps(
                            {"messages": [{"role": "user", "content": "hello"}]}
                        ),
                    ):
                        response = await client.post("/tokenize", json=payload)
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json(), {"token_ids": [7, 8]})
                        self.assertEqual(
                            json.loads(server.tokenize(payload).body), response.json()
                        )

                    self.assertEqual(
                        json.loads(server.tokenize({"prompt": "hello"}).body),
                        {"token_ids": [1, 2, 3]},
                    )
                    for payload in ({"text": "missing prompt"}, "{"):
                        response = await client.post("/tokenize", json=payload)
                        self.assertEqual(response.status_code, 500)
                        self.assertIn("error_code", response.json())
                    owner.shutdown_manager.start_draining("tokenize test")
                    response = await client.post("/tokenize", json={"prompt": "hello"})
                    self.assertEqual(response.status_code, 503)

        asyncio.run(run())

    def test_engine_unavailable_http_contract(self):
        for openai in (False, True):
            for streaming in (False, True):
                with self.subTest(openai=openai, streaming=streaming):
                    error = FtRuntimeException(
                        ExceptionType.ENGINE_UNAVAILABLE,
                        "engine is SLEEPING; retry elsewhere",
                    )

                    async def generate():
                        raise error
                        yield  # make this an async generator; it never emits a token

                    def response(*args, **kwargs):
                        return CompleteResponseAsyncGenerator(
                            generate(), CompleteResponseAsyncGenerator.get_last_value
                        )

                    async def run():
                        worker = self.frontend_server._frontend_worker
                        with patch.object(
                            worker, "inference", side_effect=response
                        ), patch.object(worker, "is_streaming", return_value=streaming):
                            if openai:
                                self.frontend_server._openai_endpoint = MagicMock()
                                self.frontend_server._openai_endpoint.chat_completion_async = AsyncMock(
                                    side_effect=response
                                )
                                result = await self.frontend_server.chat_completion(
                                    ChatCompletionRequest(
                                        messages=[{"role": "user", "content": "hello"}],
                                        stream=streaming,
                                    ),
                                    FakeRawRequest(),
                                )
                            else:
                                result = await self.frontend_server.inference(
                                    {"prompt": "hello", "stream": streaming},
                                    FakeRawRequest(),
                                )
                            if streaming:
                                # Headers are already committed. Preserve the SSE
                                # error contract rather than pretending to send 503.
                                self.assertEqual(result.status_code, 200)
                                chunks = [chunk async for chunk in result.body_iterator]
                                self.assertEqual(len(chunks), 1)
                                body = json.loads(chunks[0].split(":", 1)[1])
                            else:
                                self.assertEqual(result.status_code, 503)
                                body = json.loads(result.body)
                            self.assertEqual(body["error_code"], 8600)
                            self.assertEqual(
                                body["error_code_str"], "8600_ENGINE_UNAVAILABLE"
                            )

                    asyncio.run(run())

    def test_openai_stream_emits_done_sentinel(self):
        async def generate():
            yield FakePipelinResponse(res="hello")

        response = CompleteResponseAsyncGenerator(
            generate(), CompleteResponseAsyncGenerator.get_last_value
        )

        async def run():
            chunks = [
                chunk
                async for chunk in self.frontend_server.stream_response(
                    {"stream": True, request_id_field_name: 1}, response
                )
            ]
            self.assertEqual(chunks[-1], "data: [DONE]\r\n\r\n")

        asyncio.run(run())

    def test_other_errors_keep_existing_http_status(self):
        for error in (
            RuntimeError("internal failure"),
            FtRuntimeException(ExceptionType.UNKNOWN_ERROR, "internal failure"),
        ):
            with self.subTest(error=type(error).__name__):
                result = self.frontend_server._handle_exception(
                    {request_id_field_name: 1}, error
                )
                self.assertEqual(result.status_code, 500)


if __name__ == "__main__":
    main()
