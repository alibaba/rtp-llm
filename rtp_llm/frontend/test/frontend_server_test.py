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


class FakeFrontendWorker(object):
    def generate_response(self, prompt: str, *args: Any, **kwargs: Any):
        response_generator = self._create_generation_streams(prompt, *args, **kwargs)
        return CompleteResponseAsyncGenerator(
            response_generator, CompleteResponseAsyncGenerator.get_last_value
        )

    def tokenizer_encode(self, prompt: str):
        return [1, 2, 3, 4], ["b", "c", "d", "e"]

    async def _create_generation_streams(self, prompt: str, *args: Any, **kwargs: Any):
        yield FakePipelinResponse(res=prompt)

    def check_streaming_mode(self, *args: Any, **kwargs: Any):
        return False


class FakeRawRequest(object):
    async def is_disconnected(self):
        return False


class FrontendServerTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.frontend_server = FrontendServer()
        self.frontend_server._frontend_worker = FakeFrontendWorker()

    async def _async_run(self, *args: Any, **kwargs: Any):
        res = await self.frontend_server.generate_response(*args, **kwargs)
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
