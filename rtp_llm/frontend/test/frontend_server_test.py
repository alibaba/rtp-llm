import asyncio
import json
from typing import Any
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.openai.api_datatype import ChatCompletionRequest
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
                                self.frontend_server._openai_endpoint.chat_completion.side_effect = (
                                    response
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
