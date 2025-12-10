import asyncio
import json
import os
from typing import Any
from unittest import TestCase, main
from unittest.mock import Mock, patch

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

    async def _create_generation_streams(self, prompt: str, *args: Any, **kwargs: Any):
        yield FakePipelinResponse(res=prompt)


class FakeRawRequest(object):
    async def is_disconnected(self):
        return False


class FrontendServerTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # 创建 mock tokenizer，添加所有必要的属性
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4])
        mock_tokenizer.decode = Mock(return_value="test")
        mock_tokenizer.convert_ids_to_tokens = Mock(return_value=["b", "c", "d", "e"])
        # 添加 update_tokenizer_special_tokens 需要的属性
        mock_tokenizer.stop_words_id_list = []
        mock_tokenizer.stop_words_str_list = []
        # 添加其他常用属性
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.pad_token_id = 0

        # 使用 patch 来 mock TokenizerFactory.create_from_env
        with patch(
            "rtp_llm.frontend.frontend_server.TokenizerFactory.create_from_env",
            return_value=mock_tokenizer,
        ):
            self.frontend_server = FrontendServer()

        self.frontend_server._frontend_worker = FakeFrontendWorker()

    async def _async_run(self, *args: Any, **kwargs: Any):
        res = await self.frontend_server.generate(*args, **kwargs)
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


main()
