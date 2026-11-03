import asyncio
import math
import unittest
from types import SimpleNamespace

import torch
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionTokenLogprob,
    ChatMessage,
    FinisheReason,
    RoleEnum,
)
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    OutputDelta,
    RendererParams,
    StreamStatus,
    ThinkStatus,
)
from rtp_llm.openai.renderers.kimi_k3_renderer import (
    KimiK3Renderer,
    _KimiK3StreamStatus,
)
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


class _Tokenizer:
    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join({1: "A", 2: "B", 3: "<eos>"}[int(i)] for i in token_ids)


def _output(token_ids, probability_rows):
    return SimpleNamespace(
        output_ids=torch.tensor(token_ids),
        all_probs=torch.tensor(probability_rows, dtype=torch.float32),
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
            (
                (lambda ids, _delta: ids[:-1] if ids[-1] == 3 else ids)
                if strip_eos
                else lambda ids, _delta: ids
            ),
        )

    def _render_probs(self, output):
        return asyncio.run(self.renderer._generate_log_probs(self.status, output))

    def test_single_token_frames_use_their_own_target_rows(self):
        outputs = [
            _output([1], [[0.1, 0.7, 0.2, 0.0]]),
            _output([2], [[0.05, 0.1, 0.85, 0.0]]),
        ]
        for output, token, probability in zip(outputs, ["A", "B"], [0.7, 0.85]):
            self._update(output)
            result = self._render_probs(output)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].token, token)
            self.assertAlmostEqual(result[0].logprob, math.log(probability), places=5)
            response = asyncio.run(
                self.renderer._generate_stream_response(
                    [OutputDelta(token, result, 4, 1, 0)], [ThinkStatus()]
                )
            )
            self.assertEqual(response.choices[0].logprobs.content, result)

    def test_zero_top_logprobs_preserves_selected_token_probability(self):
        self.status.request.top_logprobs = 0
        output = _output([1], [[0.1, 0.7, 0.2, 0.0]])
        self._update(output)
        content = self._render_probs(output)[0]
        self.assertIsInstance(content, ChatCompletionTokenLogprob)
        self.assertEqual(content.token, "A")
        self.assertAlmostEqual(content.logprob, math.log(0.7), places=5)
        self.assertEqual(content.top_logprobs, [])

        sync = self.renderer._generate_log_probs_sync(
            self.status, output.all_probs, torch.tensor([1])
        )
        self.assertEqual(sync.token, "A")
        self.assertAlmostEqual(sync.logprob, math.log(0.7), places=5)
        self.assertEqual(sync.top_logprobs, [])

        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        renderer.tokenizer = _Tokenizer()
        status = SimpleNamespace(
            request=self.status.request,
            pending_token_probs=[(1, output.all_probs.reshape(-1))],
        )
        content = renderer._drain_token_probabilities(status, 1)
        self.assertEqual(content[0].token, "A")
        self.assertAlmostEqual(content[0].logprob, math.log(0.7), places=5)
        self.assertEqual(content[0].top_logprobs, [])
        self.assertEqual(status.pending_token_probs, [])

    def test_k3_keeps_buffered_probabilities_for_xtml_decoding(self):
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            logprobs=True,
            top_logprobs=2,
        )
        self.status = _KimiK3StreamStatus(request, thinking=False)
        self._update(_output([1], [[0.1, 0.7, 0.2, 0.0]]))
        self._update(_output([2], [[0.05, 0.1, 0.85, 0.0]]))
        self._update(_output([3], [[0.0, 0.0, 0.0, 1.0]]), strip_eos=True)
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        renderer.tokenizer = _Tokenizer()
        content = renderer._drain_token_probabilities(
            self.status, len(self.status.output_ids)
        )
        self.assertEqual([item.token for item in content], ["A", "B"])
        self.assertAlmostEqual(content[0].logprob, math.log(0.7), places=5)
        self.assertAlmostEqual(content[1].logprob, math.log(0.85), places=5)
        self.assertEqual(len(self.status.pending_token_probs), 1)

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


class MtpReasoningRendererLogprobsTest(unittest.IsolatedAsyncioTestCase):
    async def test_single_token_frames_preserve_each_probability(self):
        class Tokenizer:
            def decode(self, token_ids):
                if isinstance(token_ids, int):
                    token_ids = [token_ids]
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
        content = []
        rows = [
            [0.1, 0.7, 0.2, 0.0],
            [0.05, 0.1, 0.85, 0.0],
            [0.05, 0.1, 0.15, 0.7],
        ]
        for token_id, row in enumerate(rows, 1):
            output = GenerateOutput()
            output.output_ids = torch.tensor([token_id], dtype=torch.int64)
            output.all_probs = torch.tensor([row], dtype=torch.float32)
            output.aux_info = AuxInfo()
            output.aux_info.input_len = 4
            output.aux_info.output_len = token_id
            delta = await renderer._update_single_status(
                status,
                output,
                max_new_tokens=16,
                stop_words_str=[],
                stop_word_slice_list=[],
                is_streaming=True,
            )
            self.assertEqual(len(delta.logprobs), 1)
            content.extend(delta.logprobs)
        self.assertEqual([item.token for item in content], ["A", "B", "C"])
        self.assertEqual(
            [round(math.exp(item.logprob), 2) for item in content], [0.7, 0.85, 0.7]
        )


class StopSetLogprobsRendererTest(unittest.IsolatedAsyncioTestCase):
    class Tokenizer:
        def decode(self, ids):
            return "".join(
                {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 9: "<eos>"}[int(i)]
                for i in ids
            )

    def setUp(self):
        self.renderer = CustomChatRenderer(
            self.Tokenizer(),
            RendererParams("test", 128, 9, [[2, 3, 4]]),
            GenerateEnvConfig(),
        )
        self.renderer.add_extra_stop_word_ids([[5, 6]])

    async def _render(self, token_ids, split, stop_ids, stop_strings, streaming=True):
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            stream=streaming,
            logprobs=True,
            top_logprobs=0,
        )
        config = GenerateConfig(
            max_new_tokens=32,
            stop_words_list=stop_ids,
            stop_words_str=stop_strings,
            is_streaming=streaming,
        )

        async def source():
            offset = 0
            for chunk in (
                [token_ids] if not streaming else (token_ids[:split], token_ids[split:])
            ):
                if not chunk:
                    continue
                output = GenerateOutput()
                output.output_ids = torch.tensor([chunk], dtype=torch.int32)
                output.all_probs = torch.zeros((len(chunk), 10), dtype=torch.float32)
                for row, token_id in enumerate(chunk):
                    probability = 0.5 + 0.01 * (offset + row + 1)
                    output.all_probs[row, token_id] = probability
                    output.all_probs[row, 0] = 1 - probability
                output.aux_info = AuxInfo()
                output.aux_info.input_len = 4
                offset += len(chunk)
                output.aux_info.output_len = offset
                yield GenerateOutputs(generate_outputs=[output])

        outputs = source()
        if not streaming:
            outputs = await self.renderer._merge_non_streaming_outputs(outputs)
        responses = [
            response
            async for response in self.renderer.render_response_stream(
                outputs, request, config
            )
        ]
        text, probabilities = "", []
        for response in responses:
            for choice in response.choices:
                text += choice.delta.content or ""
                if choice.logprobs is not None:
                    probabilities.extend(choice.logprobs.content)
        return text, probabilities, responses[-1].choices[0].finish_reason

    async def test_forced_stop_set_replaces_stale_default_and_extra_ids(self):
        # Endpoint force_stop_words replaces the effective request set. Renderer
        # construction-time default/extra IDs must no longer stop this request.
        for streaming in (False, True):
            for split in range(1, 8):
                with self.subTest(streaming=streaming, split=split):
                    text, probabilities, reason = await self._render(
                        [1, 2, 3, 4, 5, 6, 1, 9], split, [], [], streaming
                    )
                    self.assertEqual(text, "ABCDEFA")
                    self.assertEqual(
                        [item.token for item in probabilities], list("ABCDEFA")
                    )
                    self.assertEqual(reason, FinisheReason.stop)

    async def test_effective_stop_id_prefixes_are_held_at_every_split(self):
        # ID-only extra_configs stops need buffering even if no equivalent
        # request string was supplied, including stops split across frames.
        for stop_strings in ([], ["BCD"]):
            for streaming in (False, True):
                for split in range(1, 4):
                    with self.subTest(
                        strings=stop_strings, streaming=streaming, split=split
                    ):
                        text, probabilities, reason = await self._render(
                            [1, 2, 3, 4], split, [[2, 3, 4]], stop_strings, streaming
                        )
                        self.assertEqual(text, "A")
                        self.assertEqual([item.token for item in probabilities], ["A"])
                        self.assertAlmostEqual(
                            probabilities[0].logprob, math.log(0.51), places=5
                        )
                        self.assertEqual(reason, FinisheReason.stop)

    async def test_request_stop_ids_use_the_resolved_set(self):
        for split in range(1, 4):
            with self.subTest(split=split):
                text, probabilities, _ = await self._render(
                    [1, 2, 3, 4], split, [[3, 4]], ["CD"]
                )
                self.assertEqual(text, "AB")
                self.assertEqual([item.token for item in probabilities], ["A", "B"])

    async def test_unmatched_prefix_releases_tokens_with_original_probabilities(self):
        text, probabilities, _ = await self._render(
            [1, 2, 3, 1, 9], 3, [[2, 3, 4]], ["BCD"]
        )
        self.assertEqual(text, "ABCA")
        self.assertEqual([item.token for item in probabilities], list("ABCA"))
        self.assertEqual(
            [round(math.exp(item.logprob), 2) for item in probabilities],
            [0.51, 0.52, 0.53, 0.54],
        )

    async def test_effective_stops_are_isolated_between_requests(self):
        await self._render([1, 2, 3, 4, 9], 2, [], [])
        text, probabilities, _ = await self._render(
            [1, 2, 3, 4], 2, [[2, 3, 4]], ["BCD"]
        )
        self.assertEqual(text, "A")
        self.assertEqual([item.token for item in probabilities], ["A"])


if __name__ == "__main__":
    unittest.main()
