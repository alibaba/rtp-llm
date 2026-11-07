"""The renderer must honor per-request ignore_eos without changing other requests."""

import asyncio
import json
from unittest import IsolatedAsyncioTestCase, main

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest, ChatMessage, RoleEnum
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


class Tokenizer:
    pieces = {1: "a", 2: "b", 9: "<eos>", 7: "<turn_end>", 8: "<stop>"}
    config_json = {"eos_token_id": [9, 7]}

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[int(token)] for token in ids)

    def convert_tokens_to_ids(self, text):
        return {piece: token for token, piece in self.pieces.items()}.get(text, 1)

    def encode(self, text, **kwargs):
        return [self.convert_tokens_to_ids(text.strip())]

    def __len__(self):
        return 32


class RendererIgnoreEosTest(IsolatedAsyncioTestCase):
    def setUp(self):
        self.renderer = CustomChatRenderer(
            Tokenizer(), RendererParams("test", 128, 9, [[9], [7], [8]]),
            GenerateEnvConfig(),
        )

    async def render(self, ignore_eos, tokens=(1, 9, 2)):
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            stream=True, max_tokens=3,
            extra_configs=GenerateConfig(ignore_eos=ignore_eos),
        )
        config = GenerateConfig(
            max_new_tokens=3, is_streaming=True, ignore_eos=ignore_eos,
            stop_words_list=[[9], [7], [8]],
            stop_words_str=["<eos>", "<turn_end>", "<stop>"],
        )

        async def outputs():
            for index, token in enumerate(tokens, 1):
                yield GenerateOutputs(generate_outputs=[GenerateOutput(
                    output_ids=torch.tensor([[token]], dtype=torch.int32),
                    finished=index == len(tokens),
                    aux_info=AuxInfo(input_len=1, output_len=index),
                )])

        responses = [response async for response in self.renderer.render_response_stream(
            outputs(), request, config
        )]
        return responses[-1]

    async def test_ignore_eos_reaches_requested_length(self):
        response = await self.render(True)
        self.assertEqual(response.usage.completion_tokens, 3)
        self.assertEqual(response.choices[0].finish_reason, "length")

    async def test_normal_request_still_stops_at_eos(self):
        response = await self.render(False)
        self.assertEqual(response.usage.completion_tokens, 2)
        self.assertEqual(response.choices[0].finish_reason, "stop")

    async def test_ignore_eos_includes_additional_model_eos_ids(self):
        response = await self.render(True, tokens=(1, 7, 2))
        self.assertEqual(response.usage.completion_tokens, 3)
        self.assertEqual(response.choices[0].finish_reason, "length")

    async def test_normal_request_still_stops_at_additional_eos(self):
        response = await self.render(False, tokens=(1, 7, 2))
        self.assertEqual(response.usage.completion_tokens, 2)
        self.assertEqual(response.choices[0].finish_reason, "stop")

    async def test_ignore_eos_preserves_non_eos_stop_words(self):
        response = await self.render(True, tokens=(1, 8, 2))
        self.assertEqual(response.usage.completion_tokens, 2)
        self.assertEqual(response.choices[0].finish_reason, "stop")

    async def test_request_setting_does_not_leak(self):
        await self.render(True)
        response = await self.render(False)
        self.assertEqual(response.usage.completion_tokens, 2)

    async def test_concurrent_requests_keep_separate_eos_policy(self):
        ignored, normal = await asyncio.gather(self.render(True), self.render(False))
        self.assertEqual(ignored.usage.completion_tokens, 3)
        self.assertEqual(normal.usage.completion_tokens, 2)

    def test_backend_config_filters_all_model_eos_but_keeps_other_stops(self):
        endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
        endpoint.chat_renderer = self.renderer
        endpoint.tokenizer = self.renderer.tokenizer
        endpoint.stop_words_id_list = [[9], [7], [8]]
        endpoint.stop_words_str_list = ["<eos>", "<turn_end>", "<stop>"]
        endpoint.generate_env_config = GenerateEnvConfig()
        for ignore_eos in (True, False):
            request = ChatCompletionRequest.model_validate({
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 3, "enable_thinking": False,
                "extra_configs": {"ignore_eos": ignore_eos},
            })
            config = endpoint._extract_generation_config(request)
            self.assertIn([8], config.stop_words_list)
            self.assertIn("<stop>", config.stop_words_str)
            for token in (9, 7):
                self.assertEqual([token] in config.stop_words_list, not ignore_eos)
                self.assertEqual(
                    self.renderer.tokenizer.decode([token]) in config.stop_words_str,
                    not ignore_eos,
                )

    async def test_sync_renderer_honors_ignore_eos(self):
        for ignore_eos, expected in ((True, "length"), (False, "stop")):
            statuses = self.renderer._create_status_list_sync(
                1, json.dumps({
                    "messages": [{"role": "user", "content": "test"}],
                    "extra_configs": {"ignore_eos": ignore_eos},
                })
            )
            for index, token in enumerate((1, 9, 2), 1):
                self.renderer.render_stream_response_refactor(
                    statuses, [1], [index], [0], [None],
                    [torch.tensor([[token]], dtype=torch.int32)], 3,
                    ["<eos>", "<stop>"], True,
                )
                if statuses[0].finish_reason is not None:
                    break
            self.assertEqual(statuses[0].finish_reason, expected)


if __name__ == "__main__":
    main()
