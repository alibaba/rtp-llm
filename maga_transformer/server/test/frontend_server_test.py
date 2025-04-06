import os
import json
import asyncio
from unittest import TestCase, main
from fastapi import Request as RawRequest

from maga_transformer.server.frontend_server import FrontendServer
from maga_transformer.utils.complete_response_async_generator import CompleteResponseAsyncGenerator

from typing import Any
from pydantic import BaseModel, Field

class FakePipelinResponse(BaseModel):
    res: str

class FakeFrontendWorker(object):
    def inference(self, prompt: str, *args: Any, **kwargs: Any):
        response_generator = self._inference(prompt, *args, **kwargs)
        return CompleteResponseAsyncGenerator(response_generator, CompleteResponseAsyncGenerator.get_last_value)

    def tokenizer_encode(self, prompt: str):
        return [1,2,3,4], ['b', 'c', 'd', 'e']

    async def _inference(self, prompt: str, *args: Any, **kwargs: Any):
        yield FakePipelinResponse(res=prompt)

    def is_streaming(self, *args: Any, **kawrgs: Any):
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
        res = await self.frontend_server.inference(*args, **kwargs)
        return res

    def test_simple(self):
        loop = asyncio.new_event_loop()
        res = loop.run_until_complete(self._async_run(req={"prompt": "hello"}, raw_request=FakeRawRequest()))
        self.assertEqual(res.body.decode('utf-8'), '{"res":"hello"}', res.body.decode('utf-8'))
        res = loop.run_until_complete(self._async_run(req='{"prompt": "hello"}',raw_request=FakeRawRequest()))
        self.assertEqual(res.body.decode('utf-8'), '{"res":"hello"}', res.body.decode('utf-8'))

    def test_encode(self):
        res = self.frontend_server.tokenizer_encode('{"prompt": "b c d e"}')
        self.assertEqual(res.body.decode('utf-8'), '{"token_ids":[1,2,3,4],"tokens":["b","c","d","e"],"error":""}')
        # test error input
        res = self.frontend_server.tokenizer_encode('{"text": "b c d e"}')
        self.assertEqual(json.loads(res.body.decode('utf-8'))['error_code'], 514)

main()
