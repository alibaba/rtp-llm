from unittest import IsolatedAsyncioTestCase, main

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest, ChatMessage, RoleEnum
from rtp_llm.openai.renderers.custom_renderer import RendererParams
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.token_normalizer import _MAX_UTF8_WINDOW
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


class Utf8BoundaryTokenizer:
    """Minimal byte-level tokenizer for the production MTP boundary."""

    _TOKEN_BYTES = {
        9001: b"X",
        9000: b"\xf0\x9f\x98",  # First three bytes of the emoji.
        235: b"\x8a",  # Final emoji byte; invalid without the prior chunk.
        271: b"\n\n",
        4000: b"\xe4",  # First byte of 你.
        4001: b"\xbd",  # Second byte of 你.
        4002: b"\xa0",  # Final byte of 你.
        10805: "我是".encode("utf-8"),
        53091: b"Deep",
        4374: b"Se",
        1465: b"ek",
        303: "，".encode("utf-8"),
        1057: "一个".encode("utf-8"),
        9999: b"\xff",  # Permanently invalid UTF-8 byte.
    }

    def __init__(self):
        self.decode_calls = []

    def decode(self, token_ids):
        self.decode_calls.append(list(token_ids))
        token_bytes = b"".join(self._TOKEN_BYTES[token_id] for token_id in token_ids)
        return token_bytes.decode("utf-8", errors="replace")


class FrontendStreamingUtf8ReplayTest(IsolatedAsyncioTestCase):
    """Pure-text renderer coverage with deterministic UTF-8 boundary tokens.

    Focused coverage of TokenNormalizer's replacement-character predicate lives
    in token_normalizer_test.py. This integration suite verifies MTP delta assembly
    without detector/parser state or an external tokenizer model.
    """

    def setUp(self):
        class TestRenderer(ReasoningToolBaseRenderer):
            def _setup_chat_template(self):
                self.chat_template = "test"

            def in_think_mode(self, request: ChatCompletionRequest):
                return False

        self.tokenizer = Utf8BoundaryTokenizer()
        self.renderer = TestRenderer(
            tokenizer=self.tokenizer,
            renderer_params=RendererParams(
                model_type="deepseek_v4",
                max_seq_len=2048,
                eos_token_id=1,
                stop_word_ids_list=[],
            ),
            generate_env_config=GenerateEnvConfig(),
        )
        self.request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="你好")],
        )

    async def _render_mtp_chunks(self, mtp_chunks):
        status = (await self.renderer._create_status_list(1, self.request))[0]

        def create_output(token_ids):
            aux_info = AuxInfo()
            aux_info.input_len = 5
            aux_info.output_len = len(status.output_ids_list) + len(token_ids)
            aux_info.reuse_len = 0

            output = GenerateOutput()
            output.output_ids = torch.tensor([token_ids], dtype=torch.int64)
            output.aux_info = aux_info
            return output

        streamed_deltas = []
        for chunk in mtp_chunks:
            delta = await self.renderer._update_single_status(
                status,
                create_output(chunk),
                max_new_tokens=100,
                stop_words_str=[],
                stop_word_slice_list=[],
                is_streaming=True,
            )
            streamed_deltas.append(delta.output_str)

        all_token_ids = [token_id for chunk in mtp_chunks for token_id in chunk]
        one_shot_text = self.tokenizer.decode(all_token_ids)
        streamed_text = "".join(streamed_deltas)
        return streamed_deltas, streamed_text, one_shot_text

    async def _assert_mtp_chunks_match_one_shot(self, mtp_chunks, expected_text):
        deltas, streamed_text, one_shot_text = await self._render_mtp_chunks(mtp_chunks)
        self.assertEqual(one_shot_text, expected_text)
        self.assertEqual(
            streamed_text,
            one_shot_text,
            "frontend replayed an already-emitted prefix: "
            f"chunks={mtp_chunks!r}, deltas={deltas!r}, streamed={streamed_text!r}",
        )
        return deltas

    async def test_two_token_mtp_batches_match_one_shot_decode(self):
        """Every callback carries two tokens, including the UTF-8 boundary."""
        await self._assert_mtp_chunks_match_one_shot(
            [
                [9001, 9000],
                [235, 271],
                [10805, 53091],
                [4374, 1465],
                [303, 1057],
            ],
            "X😊\n\n我是DeepSeek，一个",
        )

    async def test_mixed_two_and_three_token_mtp_batches_match_one_shot_decode(self):
        """Vary MTP acceptance length across consecutive callbacks."""
        await self._assert_mtp_chunks_match_one_shot(
            [
                [9001, 9000],
                [235, 271, 10805],
                [53091, 4374, 1465],
                [303, 1057],
            ],
            "X😊\n\n我是DeepSeek，一个",
        )

    async def test_production_four_token_mtp_batches_do_not_replay_prefix(self):
        """Regression for the production ``DeepSe\n我是DeepSeek`` replay."""
        mtp_chunks = [
            [9001, 9000],
            [235, 271, 10805, 53091],
            [4374, 1465, 303, 1057],
        ]
        deltas = await self._assert_mtp_chunks_match_one_shot(
            mtp_chunks, "X😊\n\n我是DeepSeek，一个"
        )
        self.assertEqual(deltas, ["X", "😊\n\n我是Deep", "Seek，一个"])

    async def test_previous_mtp_batch_can_start_and_end_inside_utf8(self):
        """Handle leading and trailing incomplete UTF-8 in one previous batch.

        The second callback starts by completing the emoji from the first callback,
        then ends with the first byte of 你. The third callback must finish 你 without
        replaying the newlines that were already emitted by the second callback.
        """
        await self._assert_mtp_chunks_match_one_shot(
            [
                [9001, 9000],
                [235, 271, 4000],
                [4001, 4002, 53091, 4374],
                [1465, 303, 1057],
            ],
            "X😊\n\n你DeepSeek，一个",
        )

    async def test_unresolved_replacement_character_keeps_decode_window_bounded(self):
        """An unresolved token is consumed instead of replayed forever."""
        status = (await self.renderer._create_status_list(1, self.request))[0]
        self.tokenizer.decode_calls.clear()

        for step in range(1, 129):
            aux_info = AuxInfo(input_len=5, output_len=step, reuse_len=0)
            output = GenerateOutput(
                output_ids=torch.tensor([[9999]], dtype=torch.int64),
                finished=False,
                aux_info=aux_info,
            )
            delta = await self.renderer._update_single_status(
                status,
                output,
                max_new_tokens=256,
                stop_words_str=[],
                stop_word_slice_list=[],
                is_streaming=True,
            )

            self.assertEqual(delta.output_str, "")
            self.assertEqual(len(status.last_output_ids), step)
            self.assertLessEqual(status.last_token_length, _MAX_UTF8_WINDOW)

        self.assertLessEqual(
            max(len(token_ids) for token_ids in self.tokenizer.decode_calls),
            _MAX_UTF8_WINDOW + 1,
        )

    async def test_non_streaming_decodes_only_completed_response(self):
        """stream=false must not decode every incremental backend callback."""
        token_ids = [9001, 53091, 4374, 1465]

        async def output_generator():
            for index, token_id in enumerate(token_ids):
                aux_info = AuxInfo(
                    input_len=5,
                    output_len=index + 1,
                    reuse_len=0,
                )
                yield GenerateOutputs(
                    generate_outputs=[
                        GenerateOutput(
                            output_ids=torch.tensor([[token_id]], dtype=torch.int64),
                            finished=index == len(token_ids) - 1,
                            aux_info=aux_info,
                        )
                    ]
                )

        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="你好")],
            stream=False,
        )
        generate_config = GenerateConfig(is_streaming=False, max_new_tokens=100)
        self.tokenizer.decode_calls.clear()

        responses = [
            response
            async for response in self.renderer.render_response_stream(
                output_generator(), request, generate_config
            )
        ]
        content = "".join(
            choice.delta.content or ""
            for response in responses
            for choice in response.choices
        )

        self.assertEqual(content, "XDeepSeek")
        self.assertEqual(self.tokenizer.decode_calls, [token_ids])

    async def test_non_streaming_final_decode_keeps_literal_replacement_character(self):
        """A final U+FFFD is output data, not a renderer stop condition."""

        async def output_generator():
            yield GenerateOutputs(
                generate_outputs=[
                    GenerateOutput(
                        output_ids=torch.tensor([[9999]], dtype=torch.int64),
                        finished=True,
                        aux_info=AuxInfo(input_len=5, output_len=1, reuse_len=0),
                    )
                ]
            )

        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="你好")],
            stream=False,
        )
        self.tokenizer.decode_calls.clear()
        responses = [
            response
            async for response in self.renderer.render_response_stream(
                output_generator(),
                request,
                GenerateConfig(is_streaming=False, max_new_tokens=100),
            )
        ]
        content = "".join(
            choice.delta.content or ""
            for response in responses
            for choice in response.choices
        )

        self.assertEqual(content, "\uFFFD")
        self.assertEqual(self.tokenizer.decode_calls, [[9999]])


if __name__ == "__main__":
    main()
