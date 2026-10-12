from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.generate_config import (
    GenerateConfig,
    ReturnAllProbsMode,
    ThinkingMode,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.chatglm45_renderer import ChatGlm45Renderer
from rtp_llm.openai.renderers.custom_renderer import CustomChatRenderer as Renderer
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolStreamStatus,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


class Tokenizer:
    def __len__(self):
        return 8

    def decode(self, ids, **kwargs):
        return "".join(
            {
                1: "<think>",
                2: "x",
                3: "</think>",
                4: "A",
                5: "B",
                6: "STOP",
                7: "\ufffd",
            }.get(int(i), "?")
            for i in ids
        )


class CompactRendererTest(unittest.TestCase):
    def setUp(self):
        self.renderer = Renderer.__new__(Renderer)
        self.renderer.tokenizer = Tokenizer()

    def status(self, ids, last=()):
        return SimpleNamespace(
            request=SimpleNamespace(logprobs=True, top_logprobs=0),
            output_ids_list=list(ids),
            output_ids=list(ids),
            last_output_ids=list(last),
            pending_logprobs=[],
            pending_logprob_positions=[],
            compact_logprobs=False,
            seen_logprob_positions=set(),
            finish_reason=None,
        )

    def payload(self, offset, count):
        return (
            torch.full((count,), -1.25),
            torch.empty((count, 0), dtype=torch.int32),
            torch.empty((count, 0)),
            offset,
            count,
        )

    def test_reasoning_chunk_requires_no_tensors(self):
        s = self.status([1, 2])
        self.renderer._accumulate_compact_logprobs(
            s, torch.tensor([1, 2]), (None, None, None, 2, 0)
        )
        self.assertEqual(s.pending_logprobs, [])

    def test_cross_phase_mtp_chunk_aligns_suffix(self):
        s = self.status([1, 2, 3, 4, 5])
        self.renderer._accumulate_compact_logprobs(
            s, torch.tensor([3, 4, 5]), self.payload(1, 2)
        )
        self.assertEqual(s.pending_logprob_positions, [3, 4])
        self.assertEqual([r.token for r in s.pending_logprobs], ["A", "B"])

    def test_stop_removes_probabilities_for_hidden_tokens(self):
        s = self.status([4, 5, 6])
        s.output_ids = [4, 5]
        self.renderer._accumulate_compact_logprobs(
            s, torch.tensor([4, 5, 6]), self.payload(0, 3)
        )
        self.renderer._trim_compact_logprobs(s, "A")
        self.assertEqual([r.token for r in s.pending_logprobs], ["A"])

    def test_buffer_flush_is_once(self):
        s = self.status([4, 5])
        self.renderer._accumulate_compact_logprobs(
            s, torch.tensor([4, 5]), self.payload(0, 2)
        )
        self.renderer._accumulate_compact_logprobs(
            s, torch.tensor([4, 5]), self.payload(0, 2)
        )
        self.assertEqual(len(self.renderer._take_compact_logprobs(s)), 2)
        self.assertIsNone(self.renderer._take_compact_logprobs(s))

    def test_utf8_waits_until_terminal(self):
        s = self.status([7])
        self.assertEqual(
            self.renderer._prepare_decoded_string_for_emit(s, "\ufffd", True),
            (None, False),
        )
        s.finish_reason = "stop"
        self.assertEqual(
            self.renderer._prepare_decoded_string_for_emit(s, "\ufffd", True),
            ("", True),
        )

    def test_nonstream_merge_preserves_suffix_offset(self):
        chunks = []
        for ids, offset, count in [([1, 2], 2, 0), ([3, 4], 1, 1), ([5], 0, 1)]:
            values = self.payload(offset, count)
            out = SimpleNamespace(
                output_ids=torch.tensor([ids]),
                token_logprobs=values[0] if count else None,
                top_logprob_token_ids=values[1] if count else None,
                top_logprobs=values[2] if count else None,
                logprobs_offset=offset,
                logprobs_count=count,
            )
            chunks.append(GenerateOutputs(generate_outputs=[out]))
        out = self.renderer._merge_generate_outputs(chunks).generate_outputs[0]
        self.assertEqual((out.logprobs_offset, out.logprobs_count), (3, 2))
        self.assertEqual(out.token_logprobs.numel(), 2)

    def test_nonstream_merge_preserves_absent_legacy_metadata(self):
        chunks = [
            GenerateOutputs(
                generate_outputs=[GenerateOutput(output_ids=torch.tensor([[token]]))]
            )
            for token in [4, 5]
        ]
        out = self.renderer._merge_generate_outputs(chunks).generate_outputs[0]
        self.assertEqual(out.output_ids.tolist(), [[4, 5]])
        self.assertIsNone(out.logprobs_count)
        self.assertIsNone(out.logprobs_offset)
        self.assertIsNone(self.renderer._compact_payload(out))

    def test_empty_legacy_frame_does_not_change_compact_suffix(self):
        empty = GenerateOutput(output_ids=torch.empty((1, 0), dtype=torch.int32))
        token_probs, top_ids, top_probs, offset, count = self.payload(0, 1)
        content = GenerateOutput(
            output_ids=torch.tensor([[4]]),
            token_logprobs=token_probs,
            top_logprob_token_ids=top_ids,
            top_logprobs=top_probs,
            logprobs_offset=offset,
            logprobs_count=count,
        )
        out = self.renderer._merge_generate_outputs(
            [
                GenerateOutputs(generate_outputs=[empty]),
                GenerateOutputs(generate_outputs=[content]),
            ]
        ).generate_outputs[0]
        self.assertEqual((out.logprobs_offset, out.logprobs_count), (0, 1))
        self.assertEqual(out.token_logprobs.tolist(), [-1.25])

    def test_partial_compact_metadata_is_rejected(self):
        self.assertRaises(
            RuntimeError,
            self.renderer._compact_payload,
            GenerateOutput(logprobs_count=1),
        )

    def test_bad_suffix_is_rejected(self):
        s = self.status([4, 5])
        self.assertRaises(
            RuntimeError,
            self.renderer._accumulate_compact_logprobs,
            s,
            torch.tensor([4, 5]),
            self.payload(1, 2),
        )


class CompactReasoningRendererTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.renderer = ChatGlm45Renderer.__new__(ChatGlm45Renderer)
        self.renderer.tokenizer = Tokenizer()
        self.renderer._check_finish_reason = lambda *args, **kwargs: None
        self.renderer._remove_stop_word_ids = lambda ids, delta: ids
        self.renderer._process_stop_words = lambda text, *args: (text, False)
        self.status = ReasoningToolStreamStatus(
            ChatCompletionRequest(messages=[], logprobs=True, top_logprobs=0),
            detector=None,
            reasoning_parser=None,
        )

    def output(self, ids, *, count=None):
        count = len(ids) if count is None else count
        return GenerateOutput(
            output_ids=torch.tensor([ids], dtype=torch.int32),
            aux_info=AuxInfo(input_len=1, output_len=len(ids)),
            token_logprobs=torch.full((count,), -1.25) if count else None,
            top_logprob_token_ids=(
                torch.empty((count, 0), dtype=torch.int32) if count else None
            ),
            top_logprobs=torch.empty((count, 0)) if count else None,
            logprobs_offset=len(ids) - count,
            logprobs_count=count,
        )

    async def update(self, output):
        return await self.renderer._update_single_status(
            self.status, output, 64, [], [], True
        )

    async def test_mtp_chunk_returns_each_compact_record_once(self):
        first = await self.update(self.output([4, 5]))
        self.assertEqual([record.token for record in first.logprobs], ["A", "B"])
        second = await self.update(self.output([4]))
        self.assertEqual([record.token for record in second.logprobs], ["A"])
        self.assertEqual(self.status.pending_logprobs, [])

    async def test_reasoning_only_chunk_needs_no_full_probabilities(self):
        result = await self.update(self.output([1, 2], count=0))
        self.assertIsNone(result.logprobs)
        self.assertTrue(self.status.compact_logprobs)

    async def test_logprobs_keeps_the_reasoning_parser(self):
        parser = object()
        self.renderer.in_think_mode = lambda request: True
        self.renderer._create_detector = lambda request: None
        self.renderer._create_reasoning_parser = lambda request: parser
        statuses = await self.renderer._create_status_list(1, self.status.request)
        self.assertIsInstance(statuses[0], ReasoningToolStreamStatus)
        self.assertIs(statuses[0].reasoning_parser, parser)

    async def test_buffered_text_retains_previous_chunk_probabilities(self):
        self.renderer._process_stop_words = lambda text, *args: (text, True)
        first = await self.update(self.output([4]))
        self.assertIsNone(first.logprobs)
        self.renderer._process_stop_words = lambda text, *args: (text, False)
        second = await self.update(self.output([5]))
        self.assertEqual([record.token for record in second.logprobs], ["A", "B"])

    async def test_split_utf8_retains_both_token_records(self):
        class SplitTokenizer(Tokenizer):
            def decode(self, ids, **kwargs):
                if list(ids) == [7, 8]:
                    return "é"
                if list(ids) == [8]:
                    return "\ufffd"
                return super().decode(ids, **kwargs)

            def token_id_to_bytes(self, token_id):
                return {7: b"\xc3", 8: b"\xa9"}[token_id]

        self.renderer.tokenizer = SplitTokenizer()
        first = await self.update(self.output([7]))
        self.assertIsNone(first.logprobs)
        second = await self.update(self.output([8]))
        self.assertEqual([record.bytes for record in second.logprobs], [[195], [169]])


class CompactEndpointConfigTest(unittest.TestCase):
    def config(self, *, thinking_mode=ThinkingMode.DISABLED, **kwargs):
        endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
        endpoint.tokenizer = Tokenizer()
        endpoint.chat_renderer = SimpleNamespace(
            tokenize_words=lambda words: [],
            resolve_thinking_mode=lambda request: thinking_mode,
        )
        endpoint.stop_words_str_list = []
        endpoint.stop_words_id_list = []
        endpoint.generate_env_config = SimpleNamespace()
        endpoint._reasoning_format_for_prompt = lambda *args: None
        request = ChatCompletionRequest(messages=[], **kwargs)
        with patch.object(GenerateConfig, "add_thinking_params"):
            return endpoint._extract_generation_config(request)

    def test_standard_and_original_logprobs_use_compact_payload(self):
        for mode in (None, "original"):
            for n in (None, 1):
                with self.subTest(mode=mode, n=n):
                    config = self.config(
                        logprobs=True, top_logprobs=5, logprobs_mode=mode, n=n
                    )
                    self.assertTrue(config.return_logprobs)
                    self.assertEqual(config.top_logprobs, 5)
                    self.assertEqual(config.return_all_probs, ReturnAllProbsMode.NONE)

    def test_explicit_default_probabilities_keep_legacy_sampling_semantics(self):
        config = self.config(logprobs=True, top_logprobs=5, logprobs_mode="default")
        self.assertFalse(config.return_logprobs)
        self.assertEqual(config.top_logprobs, 0)
        self.assertEqual(config.return_all_probs, ReturnAllProbsMode.DEFAULT)

    def test_explicit_full_probabilities_keep_legacy_path(self):
        config = self.config(
            logprobs=True,
            top_logprobs=5,
            extra_configs=GenerateConfig(return_all_probs=ReturnAllProbsMode.ORIGINAL),
        )
        self.assertFalse(config.return_logprobs)
        self.assertEqual(config.top_logprobs, 0)
        self.assertEqual(config.return_all_probs, ReturnAllProbsMode.ORIGINAL)

    def test_multiple_sequences_keep_legacy_path(self):
        config = self.config(logprobs=True, n=2, logprobs_mode="original")
        self.assertFalse(config.return_logprobs)
        self.assertEqual(config.return_all_probs, ReturnAllProbsMode.ORIGINAL)

    def test_disabling_logprobs_clears_both_payloads(self):
        config = self.config(
            logprobs=False,
            extra_configs=GenerateConfig(
                return_all_probs=ReturnAllProbsMode.ORIGINAL,
            ),
        )
        self.assertFalse(config.return_logprobs)
        self.assertEqual(config.top_logprobs, 0)
        self.assertEqual(config.return_all_probs, ReturnAllProbsMode.NONE)

    def test_inconsistent_extra_logprobs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.config(
                logprobs=False,
                extra_configs=GenerateConfig(return_logprobs=True, top_logprobs=5),
            )


if __name__ == "__main__":
    unittest.main()
