import asyncio
import json
from typing import Any
from unittest import TestCase, main

from pydantic import BaseModel

from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)


class FakePipelinResponse(BaseModel):
    res: str


class FakeTokenizer(object):
    def encode(self, text: str):
        return [1, 2, 3, 4]

    def decode(self, token_id: int):
        token_map = {1: "b", 2: "c", 3: "d", 4: "e"}
        return token_map.get(token_id, "")


class FakeAccessLogger(object):
    def log_query_access(self, request):
        pass

    def log_success_access(self, request, response):
        pass

    def log_exception_access(self, request, exception):
        pass


class FakeGlobalController(object):
    def __init__(self):
        self.counter = 0

    def increment(self):
        self.counter += 1
        return self.counter

    def decrement(self):
        self.counter -= 1


class FakeFrontendWorker(object):
    def inference(self, prompt: str, *args: Any, **kwargs: Any):
        response_generator = self._inference(prompt, *args, **kwargs)
        return CompleteResponseAsyncGenerator(
            response_generator, CompleteResponseAsyncGenerator.get_last_value
        )

    async def _inference(self, prompt: str, *args: Any, **kwargs: Any):
        yield FakePipelinResponse(res=prompt)

    def is_streaming(self, *args: Any, **kwargs: Any):
        return False


class FakeRawRequest(object):
    async def is_disconnected(self):
        return False


class FrontendServerTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Create FrontendServer without initializing tokenizer for testing
        self.frontend_server = object.__new__(FrontendServer)
        self.frontend_server._frontend_worker = FakeFrontendWorker()
        self.frontend_server._tokenizer = FakeTokenizer()
        self.frontend_server._access_logger = FakeAccessLogger()
        self.frontend_server._openai_endpoint = None
        self.frontend_server._embedding_endpoint = None
        self.frontend_server.thread_lock_ = None
        self.frontend_server._global_controller = FakeGlobalController()
        self.frontend_server.separated_frontend = False
        self.frontend_server.rank_id = "0"
        self.frontend_server.server_id = "0"
        self.frontend_server.is_embedding = False

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


main()
