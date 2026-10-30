"""Regression coverage for request-specific stop token boundaries."""

import asyncio
import unittest

import torch

from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.config.py_config_modules import GenerateEnvConfig, VitConfig
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    FinisheReason,
)
from rtp_llm.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


class _Tokenizer:
    def __init__(self):
        self.vocab = [chr(index) for index in range(128)]
        self.vocab[0] = "<eos>"
        self.ids = {text: index for index, text in enumerate(self.vocab)}
        self.pieces = sorted(self.ids, key=len, reverse=True)

    def encode(self, text, **_kwargs):
        result = []
        while text:
            piece = next(piece for piece in self.pieces if text.startswith(piece))
            result.append(self.ids[piece])
            text = text[len(piece) :]
        return result

    def decode(self, token_ids):
        return "".join(self.vocab[int(token)] for token in token_ids)


class FrontendStopBoundaryReproTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tokenizer = _Tokenizer()

    def _renderer(self):
        return CustomChatRenderer(
            self.tokenizer,
            RendererParams(
                model_type="test",
                max_seq_len=256,
                eos_token_id=0,
                stop_word_ids_list=[],
            ),
            GenerateEnvConfig(),
            vit_config=VitConfig(),
        )

    async def _render(
        self,
        chunks,
        *,
        stop_ids=(),
        stop_strings=(),
        max_new_tokens=128,
    ):
        renderer = self._renderer()
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            stream=True,
            enable_thinking=False,
            stop=list(stop_strings),
        )
        config = GenerateConfig(
            max_new_tokens=max_new_tokens,
            is_streaming=True,
            stop_words_list=list(stop_ids)
            + [self.tokenizer.encode(word) for word in stop_strings],
            stop_words_str=list(stop_strings),
            in_think_mode=False,
            thinking_mode=ThinkingMode.DISABLED,
        )

        async def outputs():
            total = 0
            for text in chunks:
                await asyncio.sleep(0)
                tokens = self.tokenizer.encode(text)
                total += len(tokens)
                yield GenerateOutputs(
                    generate_outputs=[
                        GenerateOutput(
                            output_ids=torch.tensor(tokens),
                            aux_info=AuxInfo(input_len=4, output_len=total),
                        )
                    ]
                )

        return [
            response
            async for response in renderer.render_response_stream(
                outputs(), request, config
            )
        ]

    @staticmethod
    def _content(responses):
        return "".join(
            choice.delta.content or ""
            for response in responses
            for choice in response.choices
        )

    def _assert_finished(self, responses, reason):
        self.assertEqual(responses[-1].choices[0].finish_reason, reason)
        self.assertIsNotNone(responses[-1].usage)

    async def test_id_only_stop_prefix_is_buffered(self):
        responses = await self._render(
            ["A", "S", "T", "OP"],
            stop_ids=[self.tokenizer.encode("STOP")],
        )
        self.assertEqual(self._content(responses), "A")
        self._assert_finished(responses, FinisheReason.stop)

    async def test_id_only_stop_in_one_frame_drops_stop_and_trailing_text(self):
        responses = await self._render(
            ["ASTOPB"],
            stop_ids=[self.tokenizer.encode("STOP")],
        )
        self.assertEqual(self._content(responses), "A")
        self._assert_finished(responses, FinisheReason.stop)

    async def test_failed_id_stop_prefix_is_released(self):
        responses = await self._render(
            ["A", "S", "T", "X", "<eos>"],
            stop_ids=[self.tokenizer.encode("STOP")],
        )
        self.assertEqual(self._content(responses), "ASTX")
        self._assert_finished(responses, FinisheReason.stop)

    async def test_length_limit_releases_incomplete_id_stop_prefix(self):
        responses = await self._render(
            ["A", "S", "T"],
            stop_ids=[self.tokenizer.encode("STOP")],
            max_new_tokens=3,
        )
        self.assertEqual(self._content(responses), "AST")
        self._assert_finished(responses, FinisheReason.length)


if __name__ == "__main__":
    unittest.main()
