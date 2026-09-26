import asyncio
import math
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest, ChatMessage, RoleEnum
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    OutputDelta,
    RendererParams,
    StreamStatus,
    ThinkStatus,
)
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput


class _Tokenizer:
    def decode(self, token_ids):
        return "".join({1: "A", 2: "B", 3: "<eos>"}[int(i)] for i in token_ids)


def _output(token_ids, probability_rows):
    return SimpleNamespace(
        output_ids=torch.tensor([token_ids]),
        all_probs=torch.tensor([probability_rows], dtype=torch.float32),
        aux_info=SimpleNamespace(
            input_len=4, output_len=len(token_ids), reuse_len=0, multimodal_lengths={}
        ),
    )


class MtpLogprobsRendererTest(unittest.TestCase):
    def setUp(self):
        self.renderer = CustomChatRenderer.__new__(CustomChatRenderer)
        self.renderer.tokenizer = _Tokenizer()
        self.status = StreamStatus(SimpleNamespace(logprobs=True, top_logprobs=2))

    def _update(self, output, strip_eos=False):
        self.status.update_output(
            output,
            lambda _ids, _input_len: None,
            (lambda ids, _delta: ids[:-1] if ids[-1] == 3 else ids)
            if strip_eos
            else lambda ids, _delta: ids,
        )

    def _render_probs(self, output):
        return asyncio.run(self.renderer._generate_log_probs(self.status, output))

    def test_two_accepted_tokens_use_their_own_target_rows(self):
        output = _output(
            [1, 2],
            [[0.1, 0.7, 0.2, 0.0], [0.05, 0.1, 0.85, 0.0]],
        )
        self._update(output)
        probs = self._render_probs(output)

        self.assertEqual([item.token for item in probs], ["A", "B"])
        self.assertAlmostEqual(probs[0].logprob, math.log(0.7), places=5)
        self.assertAlmostEqual(probs[1].logprob, math.log(0.85), places=5)
        self.assertEqual(
            [[top.token for top in item.top_logprobs] for item in probs],
            [["A", "B"], ["B", "A"]],
        )

        response = asyncio.run(
            self.renderer._generate_stream_response(
                [OutputDelta("AB", probs, 4, 2, 0)], [ThinkStatus()]
            )
        )
        self.assertEqual(response.choices[0].logprobs.content, probs)

    def test_buffered_token_keeps_its_probability_until_emitted(self):
        first = _output([1], [[0.1, 0.7, 0.2, 0.0]])
        second = _output([2], [[0.05, 0.1, 0.85, 0.0]])
        self._update(first)
        self._update(second)

        self.assertEqual(
            [item.token for item in self._render_probs(second)], ["A", "B"]
        )
        self.assertEqual(self._render_probs(second), [])

    def test_stop_token_is_not_reported_as_visible_content(self):
        output = _output(
            [1, 2, 3],
            [[0.1, 0.7, 0.2, 0.0], [0.05, 0.1, 0.85, 0.0], [0, 0, 0, 1]],
        )
        self._update(output, strip_eos=True)

        self.assertEqual(
            [item.token for item in self._render_probs(output)], ["A", "B"]
        )
        self.assertEqual(len(self.status.pending_token_probs), 1)

    def test_token_probability_row_mismatch_is_rejected(self):
        output = _output([1, 2], [[0.1, 0.7, 0.2, 0.0]])
        with self.assertRaisesRegex(
            ValueError, "target probability rows must match output tokens"
        ):
            self._update(output)


class MtpReasoningRendererLogprobsTest(unittest.IsolatedAsyncioTestCase):
    async def test_multi_token_chunk_preserves_every_probability_when_deltas_merge(self):
        class Tokenizer:
            def decode(self, token_ids):
                return "".join({1: "A", 2: "B", 3: "C"}[int(i)] for i in token_ids)

        class Renderer(ReasoningToolBaseRenderer):
            def _setup_chat_template(self):
                self.chat_template = "test"

        renderer = Renderer(
            tokenizer=Tokenizer(),
            renderer_params=RendererParams(
                model_type="test",
                max_seq_len=128,
                eos_token_id=9,
                stop_word_ids_list=[],
            ),
            generate_env_config=GenerateEnvConfig(),
        )
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            logprobs=True,
            top_logprobs=2,
        )
        status = (await renderer._create_status_list(1, request))[0]
        output = GenerateOutput()
        output.output_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        output.all_probs = torch.tensor(
            [
                [
                    [0.1, 0.7, 0.2, 0.0],
                    [0.05, 0.1, 0.85, 0.0],
                    [0.05, 0.1, 0.15, 0.7],
                ]
            ],
            dtype=torch.float32,
        )
        output.aux_info = AuxInfo()
        output.aux_info.input_len = 4
        output.aux_info.output_len = 3

        delta = await renderer._update_single_status(
            status,
            output,
            max_new_tokens=16,
            stop_words_str=[],
            stop_word_slice_list=[],
            is_streaming=True,
        )

        self.assertEqual(delta.output_str, "ABC")
        self.assertEqual([item.token for item in delta.logprobs], ["A", "B", "C"])
        self.assertEqual(
            [round(math.exp(item.logprob), 2) for item in delta.logprobs],
            [0.7, 0.85, 0.7],
        )


if __name__ == "__main__":
    unittest.main()
