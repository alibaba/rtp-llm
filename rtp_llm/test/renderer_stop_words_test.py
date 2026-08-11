from typing import List
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, Mock

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    FinisheReason,
    RoleEnum,
)
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RendererParams,
    StreamStatus,
    StreamStatusSync,
)
from rtp_llm.openai.renderers.qwen_renderer import (
    QwenRenderer,
    QwenStreamStatus,
    QwenStreamStatusSync,
)
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs
from rtp_llm.utils.word_util import get_stop_word_slices


def _make_chat_request(**kwargs):
    return ChatCompletionRequest(
        messages=[ChatMessage(role=RoleEnum.user, content="test")], **kwargs
    )


class _AppendOnlyTokenList(list):
    def __add__(self, other):
        raise AssertionError("cumulative token history must not be copied with +")

    def __deepcopy__(self, memo):
        raise AssertionError("cumulative token history must not be deep-copied")


class _SliceProbeList(list):
    def __init__(self, values):
        super().__init__(values)
        self.slice_ranges = []

    def __getitem__(self, key):
        if isinstance(key, slice):
            self.slice_ranges.append(key.indices(len(self)))
        return super().__getitem__(key)


class StreamStatusTest(TestCase):
    def _assert_instances_are_isolated(self, status_type):
        first_request = _make_chat_request()
        second_request = _make_chat_request()
        first = status_type(first_request)
        second = status_type(second_request)

        first.index = 3
        first.output_ids.append(10)
        first.output_ids_list.append(11)
        first.last_output_ids.append(12)
        first.last_token_length = 1
        first.processed_token_count = 1
        first.output_rewound = True
        first.pending_stop_text = "partial"
        first.pending_stop_token_cursor = 1
        first.finish_reason = FinisheReason.stop
        first.tokenizer = object()
        first.responded_string = "first"
        first.delta_output_string = "delta"

        self.assertIs(first.request, first_request)
        self.assertIs(second.request, second_request)
        self.assertEqual(second.index, 0)
        self.assertEqual(second.output_ids, [])
        self.assertEqual(second.output_ids_list, [])
        self.assertEqual(second.last_output_ids, [])
        self.assertEqual(second.last_token_length, 0)
        self.assertEqual(second.processed_token_count, 0)
        self.assertFalse(second.output_rewound)
        self.assertEqual(second.pending_stop_text, "")
        self.assertEqual(second.pending_stop_token_cursor, 0)
        self.assertIsNone(second.finish_reason)
        self.assertIsNone(second.tokenizer)
        self.assertEqual(second.responded_string, "")
        self.assertEqual(second.delta_output_string, "")
        self.assertIsNot(first.output_ids, second.output_ids)
        self.assertIsNot(first.output_ids_list, second.output_ids_list)
        self.assertIsNot(first.last_output_ids, second.last_output_ids)

    def test_stream_status_instances_are_isolated(self):
        self._assert_instances_are_isolated(StreamStatus)

    def test_stream_status_sync_instances_are_isolated(self):
        self._assert_instances_are_isolated(StreamStatusSync)

    def test_update_output_sync_uses_the_passed_tensor(self):
        status = StreamStatusSync(_make_chat_request())
        append_only_history = _AppendOnlyTokenList()
        status.output_ids_list = append_only_history
        check_finish = Mock(return_value=None)
        remove_stop_words = Mock(side_effect=lambda token_ids, _: token_ids)

        status.update_output_sync(
            torch.tensor([[10, 11]], dtype=torch.int32),
            4,
            check_finish,
            remove_stop_words,
        )

        self.assertEqual(status.index, 1)
        self.assertIs(status.output_ids_list, append_only_history)
        self.assertEqual(status.output_ids_list, [10, 11])
        self.assertEqual(status.output_ids, [10, 11])
        check_finish.assert_called_once_with([10, 11], 4)
        remove_stop_words.assert_called_once_with([10, 11], [10, 11])

    def test_update_output_keeps_append_only_history_and_bounded_context(self):
        status = StreamStatus(_make_chat_request())
        append_only_history = _AppendOnlyTokenList()
        status.output_ids_list = append_only_history
        check_finish = Mock(return_value=None)
        remove_stop_words = Mock(side_effect=lambda token_ids, _: token_ids)

        status.update_output(
            GenerateOutput(
                output_ids=torch.tensor([[10, 11]], dtype=torch.int32),
                aux_info=AuxInfo(input_len=4),
            ),
            check_finish,
            remove_stop_words,
        )
        status.update_result()
        status.update_output(
            GenerateOutput(
                output_ids=torch.tensor([[12, 13]], dtype=torch.int32),
                aux_info=AuxInfo(input_len=4),
            ),
            check_finish,
            remove_stop_words,
        )

        self.assertIs(status.output_ids_list, append_only_history)
        self.assertEqual(status.output_ids_list, [10, 11, 12, 13])
        self.assertEqual(status.processed_token_count, 2)
        self.assertEqual(status.last_output_ids, [10, 11])
        self.assertEqual(status.tokens_to_decode, [10, 11, 12, 13])

    def test_sync_two_chunk_cursor_matches_async_and_keeps_bounded_context(self):
        async_status = StreamStatus(_make_chat_request())
        sync_status = StreamStatusSync(_make_chat_request())

        def check_finish(*_):
            return None

        def keep_all_tokens(token_ids, _):
            return token_ids

        processed_count = 0
        for chunk in ([10, 11], [12, 13, 14]):
            async_status.update_output(
                GenerateOutput(
                    output_ids=torch.tensor([chunk], dtype=torch.int32),
                    aux_info=AuxInfo(input_len=4),
                ),
                check_finish,
                keep_all_tokens,
            )
            sync_status.update_output_sync(
                torch.tensor([chunk], dtype=torch.int32),
                4,
                check_finish,
                keep_all_tokens,
            )

            self.assertEqual(
                sync_status.tokens_to_decode,
                async_status.tokens_to_decode,
            )
            async_status.update_result()
            sync_status.update_result()
            processed_count += len(chunk)

            self.assertEqual(sync_status.processed_token_count, processed_count)
            self.assertEqual(
                sync_status.processed_token_count,
                async_status.processed_token_count,
            )
            self.assertEqual(sync_status.last_output_ids, list(chunk))
            self.assertEqual(
                sync_status.last_output_ids,
                async_status.last_output_ids,
            )
            self.assertEqual(sync_status.last_token_length, len(chunk))

        self.assertEqual(sync_status.output_ids, [10, 11, 12, 13, 14])
        self.assertEqual(sync_status.last_output_ids, [12, 13, 14])

    def _assert_qwen_update_result_advances_cursor(self, status_type):
        status = status_type(_make_chat_request())
        status.output_ids = [10, 11]
        status.delta_output_string = "ignored-by-qwen"
        status.total_output_string = "first\nAction:"

        status.update_result()

        self.assertEqual(status.processed_token_count, 2)
        self.assertEqual(status.last_output_ids, [10, 11])
        self.assertEqual(status.responded_string, "first")
        self.assertEqual(status.responded_length, len("first"))

        status.output_ids = [10, 11, 12]
        status.delta_output_string = "also-ignored-by-qwen"
        status.total_output_string = "first second\nAction:"

        status.update_result()

        self.assertEqual(status.processed_token_count, 3)
        self.assertEqual(status.last_token_length, 1)
        self.assertEqual(status.last_output_ids, [12])
        self.assertEqual(status.responded_string, "first second")
        self.assertEqual(status.responded_length, len("first second"))

    def test_qwen_stream_status_update_result_advances_cursor(self):
        self._assert_qwen_update_result_advances_cursor(QwenStreamStatus)

    def test_qwen_stream_status_sync_update_result_advances_cursor(self):
        self._assert_qwen_update_result_advances_cursor(QwenStreamStatusSync)


class RemoveStopWordIdsTest(TestCase):
    """Test _remove_stop_word_ids method which truncates token sequences at stop words."""

    def setUp(self):
        # Create a minimal mock renderer with necessary attributes
        self.renderer = Mock(spec=CustomChatRenderer)
        self.renderer.eos_token_id = 2
        self.renderer.stop_words_id_list = [[151643], [151644], [151645]]
        self.renderer.get_all_extra_stop_word_ids_list = Mock(return_value=[])

        # Bind the actual method to our mock
        self.renderer._remove_stop_word_ids = (
            CustomChatRenderer._remove_stop_word_ids.__get__(self.renderer)
        )

    def test_truncate_at_eos(self):
        # EOS token in middle of sequence - should truncate
        output_ids = [100, 101, 2, 103, 104]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101])

    def test_truncate_at_eos_multiple(self):
        # Multiple EOS tokens - should truncate at FIRST
        output_ids = [100, 2, 102, 2, 104]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100])

    def test_eos_at_beginning(self):
        # EOS at beginning - should return empty
        output_ids = [2, 100, 101]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [])

    def test_eos_at_end(self):
        # EOS at end - should truncate
        output_ids = [100, 101, 2]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101])

    def test_no_eos(self):
        # No EOS token - should return unchanged
        output_ids = [100, 101, 102]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101, 102])

    def test_truncate_at_stop_word_sequence(self):
        # Stop word sequence in middle
        output_ids = [100, 101, 151643, 103, 104]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101])

    def test_truncate_at_multi_token_stop_word(self):
        # Multi-token stop word sequence
        self.renderer.stop_words_id_list = [[200, 201, 202]]
        output_ids = [100, 200, 201, 202, 103]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100])

    def test_multiple_stop_words_truncate_at_first(self):
        # Multiple different stop words - should truncate at earliest
        self.renderer.stop_words_id_list = [[151643], [151644]]
        output_ids = [100, 101, 151644, 102, 151643, 103]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101])

    def test_stop_word_at_beginning(self):
        # Stop word at beginning - should return empty
        output_ids = [151643, 100, 101]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [])

    def test_stop_word_at_end(self):
        # Stop word at end - should truncate
        output_ids = [100, 101, 151643]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101])

    def test_no_stop_words(self):
        # No stop words in sequence
        output_ids = [100, 101, 102, 103]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101, 102, 103])

    def test_eos_before_stop_word(self):
        # Both EOS and stop word, EOS comes first - should truncate at EOS
        output_ids = [100, 2, 151643, 103]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100])

    def test_stop_word_before_eos(self):
        # Both stop word and EOS, stop word comes first - should truncate at stop word
        output_ids = [100, 151643, 2, 103]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100])  # Truncates at stop word position 1

    def test_partial_stop_word_match(self):
        # Partial match of multi-token stop word - should NOT truncate
        self.renderer.stop_words_id_list = [[200, 201, 202]]
        output_ids = [100, 200, 201, 999]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 200, 201, 999])

    def test_overlapping_stop_words(self):
        # Overlapping stop word sequences
        self.renderer.stop_words_id_list = [[200, 201], [201, 202]]
        output_ids = [100, 200, 201, 202, 103]
        # Should match [200, 201] first at position 1
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100])

    def test_empty_sequence(self):
        # Empty output_ids
        output_ids = []
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [])

    def test_mtp_scenario(self):
        # Simulate MTP (Multiple Token Prediction) where 3 tokens generated at once
        # Stop word appears in the middle of the 3-token chunk
        self.renderer.stop_words_id_list = [[151643]]
        # Generated tokens: [100, 151643, 102] - stop word in middle
        output_ids = [98, 99, 100, 151643, 102]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        # Should truncate at position 3 (where stop word starts)
        self.assertEqual(result, [98, 99, 100])

    def test_extra_stop_words(self):
        # Test with extra stop words from get_all_extra_stop_word_ids_list
        self.renderer.get_all_extra_stop_word_ids_list = Mock(return_value=[[300, 301]])
        output_ids = [100, 300, 301, 102]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100])

    def test_incremental_scan_only_checks_new_suffix_boundary(self):
        self.renderer.stop_words_id_list = [[200, 201, 202]]
        old_prefix = [100] * 4094 + [200, 201]
        output_ids = _SliceProbeList(old_prefix + [202, 103])

        result = self.renderer._remove_stop_word_ids(output_ids, [202, 103])

        self.assertEqual(result, [100] * 4094)
        comparison_starts = [
            start
            for start, stop, step in output_ids.slice_ranges
            if step == 1 and stop - start == 3
        ]
        self.assertEqual(comparison_starts, [4094])

    def test_non_suffix_delta_falls_back_to_full_scan_for_earlier_eos(self):
        output_ids = [100, 2, 101, 102]

        result = self.renderer._remove_stop_word_ids(output_ids, [999])

        self.assertEqual(result, [100])

    def test_non_suffix_delta_falls_back_to_full_scan_for_earlier_stop_word(self):
        self.renderer.stop_words_id_list = [[200, 201]]
        output_ids = [100, 200, 201, 102]

        result = self.renderer._remove_stop_word_ids(output_ids, [999])

        self.assertEqual(result, [100])

    def test_multi_token_stop_completing_at_output_limit_wins_tie(self):
        self.renderer.stop_words_id_list = [[200, 201]]
        output_ids = [100, 200, 201, 102]

        result = self.renderer._remove_stop_word_ids(
            output_ids, output_ids, output_token_limit=3
        )

        self.assertEqual(result, [100])

    def test_multi_token_stop_completing_after_output_limit_is_ignored(self):
        self.renderer.stop_words_id_list = [[200, 201]]
        output_ids = [100, 101, 200, 201, 102]

        result = self.renderer._remove_stop_word_ids(
            output_ids, output_ids, output_token_limit=3
        )

        self.assertEqual(result, [100, 101, 200])


class ProcessStopWordsTest(TestCase):
    """Test _process_stop_words method which handles string-level stop word processing."""

    def setUp(self):
        # Create a minimal mock renderer
        self.renderer = Mock(spec=CustomChatRenderer)
        self.renderer._process_stop_words = (
            CustomChatRenderer._process_stop_words.__get__(self.renderer)
        )
        self.status = StreamStatus(Mock())
        self.status.finish_reason = None

    def test_truncate_at_complete_stop_word(self):
        # Complete stop word found - should truncate and set finish_reason
        delta_string = "Hello<|observation|>world"
        stop_words_str = ["<|observation|>"]
        stop_word_slice_list = []

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        self.assertEqual(truncated, "Hello")
        self.assertEqual(self.status.finish_reason, FinisheReason.stop)
        self.assertFalse(should_buffer)

    def test_partial_stop_word_streaming(self):
        # Partial stop word at end in streaming mode - should buffer
        delta_string = "Hello<|obs"
        stop_words_str = ["<|observation|>"]
        stop_word_slice_list = get_stop_word_slices(["<|observation|>"])

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        # stop_words_str lookup doesn't find complete match, so no truncation
        # But stop_word_slice_list detects partial match at end, so should buffer
        self.assertEqual(truncated, "Hello<|obs")  # No truncation from stop_words_str
        self.assertIsNone(self.status.finish_reason)  # No complete stop word found
        self.assertTrue(should_buffer)  # Should buffer because partial match detected

    def test_no_stop_word(self):
        # No stop word - should pass through unchanged
        delta_string = "Hello world"
        stop_words_str = ["<|observation|>"]
        stop_word_slice_list = ["<|observation|>"]

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        self.assertEqual(truncated, "Hello world")
        self.assertIsNone(self.status.finish_reason)
        self.assertFalse(should_buffer)

    def test_empty_string(self):
        # Empty string - should return empty, no buffering
        delta_string = ""
        stop_words_str = ["<|observation|>"]
        stop_word_slice_list = []

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        self.assertEqual(truncated, "")
        self.assertFalse(should_buffer)

    def test_multiple_stop_words_truncate_at_first(self):
        # Multiple stop words - should truncate at earliest
        delta_string = "Start<|user|>middle<|observation|>end"
        stop_words_str = ["<|observation|>", "<|user|>"]
        stop_word_slice_list = []

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        self.assertEqual(truncated, "Start")
        self.assertEqual(self.status.finish_reason, FinisheReason.stop)

    def test_complete_before_partial(self):
        # Complete stop word found - should NOT buffer even if partial match exists
        delta_string = "Hello<|observation|>"
        stop_words_str = ["<|observation|>"]
        stop_word_slice_list = ["<|observation|>"]

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        self.assertEqual(truncated, "Hello")
        self.assertEqual(self.status.finish_reason, FinisheReason.stop)
        self.assertFalse(should_buffer)  # Complete match takes precedence

    def test_non_streaming_mode(self):
        # Non-streaming mode - same truncation behavior
        delta_string = "Hello<|observation|>world"
        stop_words_str = ["<|observation|>"]
        stop_word_slice_list = []

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, False, self.status
        )

        self.assertEqual(truncated, "Hello")
        self.assertEqual(self.status.finish_reason, FinisheReason.stop)
        self.assertFalse(should_buffer)

    def test_unicode_stop_words(self):
        # Unicode stop words
        delta_string = "文本<|结束|>后续"
        stop_words_str = ["<|结束|>"]
        stop_word_slice_list = []

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        self.assertEqual(truncated, "文本")
        self.assertEqual(self.status.finish_reason, FinisheReason.stop)


class _RendererTestBase(IsolatedAsyncioTestCase):
    """Shared helpers for ReasoningToolBaseRenderer stop-word tests."""

    @staticmethod
    def _make_tokenizer(token_map: dict):
        class DummyTokenizer:
            chat_template = ""
            path = None

            def __init__(self):
                self._map = token_map

            def decode(self, token_ids):
                if token_ids is None:
                    return ""
                if isinstance(token_ids, int):
                    token_ids = [token_ids]
                return "".join(self._map.get(t, "") for t in token_ids)

            def encode(self, text: str, add_special_tokens: bool = False):
                return []

            def convert_tokens_to_ids(self, word):
                return None

        return DummyTokenizer()

    @staticmethod
    def _make_renderer(
        tokenizer, eos_token_id=0, stop_word_ids_list=None, max_seq_len=2048
    ):
        class TestRenderer(ReasoningToolBaseRenderer):
            def _setup_chat_template(self):
                self.chat_template = "test"

            def in_think_mode(self, request: ChatCompletionRequest):
                return False

        return TestRenderer(
            tokenizer=tokenizer,
            renderer_params=RendererParams(
                model_type="test",
                max_seq_len=max_seq_len,
                eos_token_id=eos_token_id,
                stop_word_ids_list=stop_word_ids_list or [],
            ),
            generate_env_config=GenerateEnvConfig(),
        )

    @staticmethod
    def _create_output(tokens, input_len=0):
        aux_info = AuxInfo()
        aux_info.input_len = input_len
        aux_info.output_len = len(tokens)
        aux_info.reuse_len = 0
        output = GenerateOutput()
        output.output_ids = torch.tensor([tokens])
        output.aux_info = aux_info
        return output

    async def _make_status(self, renderer):
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")], tools=[]
        )
        status_list = await renderer._create_status_list(1, request)
        return status_list[0]

    @staticmethod
    def _update_sync(
        renderer, status, tokens, max_new_tokens, stop_words_str, input_len=0
    ):
        return renderer._update_single_status_sync(
            status,
            input_len=input_len,
            output_len=len(status.output_ids_list) + len(tokens),
            reuse_len=0,
            all_probs=None,
            output_ids=torch.tensor([tokens]),
            max_new_tokens=max_new_tokens,
            stop_words_str=stop_words_str,
            stop_word_slice_list=get_stop_word_slices(stop_words_str),
            is_streaming=True,
        )


class TestStopWordTruncation(_RendererTestBase):
    """Tests for multi-token stop word handling in _update_single_status."""

    async def test_buffered_stop_word_prefix_not_leaked_when_token_stop_truncates(self):
        """MTP: trailing tokens after stop word. _check_finish_reason misses (suffix ≠ stop word),
        _remove_stop_word_ids truncates output_ids backward. Without the rewind guard,
        delta_output_string would retain the buffered "ST" prefix and _flush_buffer()
        (called after the streaming loop ends, custom_renderer.py:955) would emit it."""
        tokenizer = self._make_tokenizer(
            {100: "Hello ", 200: "S", 201: "T", 202: "OP", 103: "after"}
        )
        renderer = self._make_renderer(tokenizer, stop_word_ids_list=[[200, 201, 202]])
        status = await self._make_status(renderer)

        stop_words_str = ["STOP"]
        stop_word_slice_list = get_stop_word_slices(stop_words_str)

        # Chunk 1: emits "Hello ", buffers "ST" (partial stop-word prefix)
        delta1 = await renderer._update_single_status(
            status,
            self._create_output([100, 200, 201]),
            max_new_tokens=100,
            stop_words_str=stop_words_str,
            stop_word_slice_list=stop_word_slice_list,
            is_streaming=True,
        )
        self.assertEqual(delta1.output_str, "Hello ")
        self.assertEqual(status.delta_output_string, "ST")
        self.assertIsNone(status.finish_reason)

        # Chunk 2: completes stop-word [200,201,202] with trailing token 103.
        # Rewind guard must: (1) clear "ST" from delta_output_string so _flush_buffer
        # won't emit it, (2) set finish_reason=stop so _check_all_finished breaks the
        # loop and no further chunks are processed.
        delta2 = await renderer._update_single_status(
            status,
            self._create_output([202, 103]),
            max_new_tokens=100,
            stop_words_str=stop_words_str,
            stop_word_slice_list=stop_word_slice_list,
            is_streaming=True,
        )

        self.assertEqual(delta2.output_str, "")
        self.assertEqual(status.finish_reason, FinisheReason.stop)
        # Critical: delta_output_string must be empty, otherwise _flush_buffer leaks "ST"
        self.assertEqual(status.delta_output_string, "")

    def test_sync_cross_chunk_stop_does_not_leak_prefix_or_trailing_token(self):
        tokenizer = self._make_tokenizer(
            {100: "Hello ", 200: "S", 201: "T", 202: "OP", 103: "after"}
        )
        renderer = self._make_renderer(tokenizer, stop_word_ids_list=[[200, 201, 202]])
        status = StreamStatusSync(_make_chat_request())

        first = self._update_sync(renderer, status, [100, 200, 201], 100, ["STOP"])
        second = self._update_sync(renderer, status, [202, 103], 100, ["STOP"])

        self.assertEqual(first.output_str, "")
        self.assertEqual(second.output_str, "Hello ")
        self.assertEqual(status.responded_string, "Hello ")
        self.assertEqual(status.finish_reason, FinisheReason.stop)

    def test_sync_token_stop_wins_when_same_mtp_chunk_crosses_length(self):
        tokenizer = self._make_tokenizer({100: "A", 999: "X", 101: "B", 102: "C"})
        renderer = self._make_renderer(tokenizer, stop_word_ids_list=[[999]])
        status = StreamStatusSync(_make_chat_request())

        delta = self._update_sync(
            renderer, status, [100, 999, 101, 102], 3, ["X"]
        )

        self.assertEqual(delta.output_str, "A")
        self.assertEqual(status.responded_string, "A")
        self.assertEqual(status.finish_reason, FinisheReason.stop)
        self.assertEqual(delta.output_length, 2)
        stream_response = renderer._generate_stream_response_sync([delta])
        self.assertEqual(stream_response.usage.completion_tokens, 2)
        final_response = renderer._generate_final_sync([status], [0], [4], [0])
        self.assertEqual(final_response.usage.completion_tokens, 2)

    def test_sync_stop_at_cutoff_wins_but_stop_after_cutoff_is_length(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 101: "B", 102: "C", 999: "X", 103: "after"}
        )

        for tokens, expected_text, expected_reason in (
            ([100, 101, 999, 103], "AB", FinisheReason.stop),
            ([100, 101, 102, 999, 103], "ABC", FinisheReason.length),
        ):
            with self.subTest(tokens=tokens):
                renderer = self._make_renderer(
                    tokenizer, stop_word_ids_list=[[999]]
                )
                status = StreamStatusSync(_make_chat_request())

                delta = self._update_sync(renderer, status, tokens, 3, ["X"])

                self.assertEqual(delta.output_str, expected_text)
                self.assertEqual(status.output_ids, tokens[: len(expected_text)])
                self.assertEqual(status.finish_reason, expected_reason)

    def test_sync_max_seq_cutoff_ignores_later_stop(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 101: "B", 102: "C", 999: "X", 103: "after"}
        )
        renderer = self._make_renderer(
            tokenizer, stop_word_ids_list=[[999]], max_seq_len=5
        )
        status = StreamStatusSync(_make_chat_request())

        delta = self._update_sync(
            renderer,
            status,
            [100, 101, 102, 999, 103],
            100,
            ["X"],
            input_len=2,
        )

        self.assertEqual(delta.output_str, "ABC")
        self.assertEqual(status.output_ids, [100, 101, 102])
        self.assertEqual(status.finish_reason, FinisheReason.length)

    async def test_multi_token_stop_word_completes_at_chunk_boundary(self):
        """Standard generation: stop word completes exactly at the end of output_ids_list.
        _check_finish_reason catches it via suffix check; truncation guard is NOT triggered.
        """
        tokenizer = self._make_tokenizer({100: "Hello ", 200: "S", 201: "T", 202: "OP"})
        renderer = self._make_renderer(tokenizer, stop_word_ids_list=[[200, 201, 202]])
        status = await self._make_status(renderer)

        stop_words_str = ["STOP"]
        stop_word_slice_list = get_stop_word_slices(stop_words_str)

        # Chunk 1: partial stop word, buffers "ST"
        delta1 = await renderer._update_single_status(
            status,
            self._create_output([100, 200, 201]),
            max_new_tokens=100,
            stop_words_str=stop_words_str,
            stop_word_slice_list=stop_word_slice_list,
            is_streaming=True,
        )
        self.assertEqual(delta1.output_str, "Hello ")
        self.assertIsNone(status.finish_reason)

        # Chunk 2: only the completing token, no trailing tokens.
        # _check_finish_reason sees output_ids_list ending with [200,201,202] → finish_reason=stop.
        # _remove_stop_word_ids truncates to [100]. last_output_ids was [100,200,201].
        delta2 = await renderer._update_single_status(
            status,
            self._create_output([202]),
            max_new_tokens=100,
            stop_words_str=stop_words_str,
            stop_word_slice_list=stop_word_slice_list,
            is_streaming=True,
        )
        self.assertEqual(delta2.output_str, "")
        self.assertEqual(status.finish_reason, FinisheReason.stop)

    async def test_single_token_stop_word_in_mtp_chunk(self):
        """MTP: single-token stop word appears mid-chunk with trailing tokens.
        _check_finish_reason only checks the suffix of output_ids_list, so it
        misses a stop word that isn't at the end. Token truncation must also set
        the renderer terminal state without relying on the engine."""
        tokenizer = self._make_tokenizer({100: "A", 101: "B", 999: "X", 102: "C"})
        renderer = self._make_renderer(tokenizer, stop_word_ids_list=[[999]])
        status = await self._make_status(renderer)

        # Single MTP chunk: [100, 101, 999, 102]. Stop word 999 in middle.
        delta = await renderer._update_single_status(
            status,
            self._create_output([100, 101, 999, 102]),
            max_new_tokens=100,
            stop_words_str=["X"],
            stop_word_slice_list=get_stop_word_slices(["X"]),
            is_streaming=True,
        )
        # Content is correctly truncated — "X" and "C" are not emitted
        self.assertEqual(delta.output_str, "AB")
        self.assertEqual(status.finish_reason, FinisheReason.stop)

    async def test_eos_in_mtp_chunk_with_trailing_tokens(self):
        """MTP: EOS token appears mid-chunk with trailing tokens.
        _check_finish_reason only checks the last token, so token truncation must
        set the renderer terminal state for mid-chunk EOS.
        """
        eos = 2
        tokenizer = self._make_tokenizer({100: "Hello", eos: "", 103: "extra"})
        renderer = self._make_renderer(
            tokenizer, eos_token_id=eos, stop_word_ids_list=[]
        )
        status = await self._make_status(renderer)

        # MTP chunk with EOS mid-stream: [100, 2, 103]
        delta = await renderer._update_single_status(
            status,
            self._create_output([100, eos, 103]),
            max_new_tokens=100,
            stop_words_str=[],
            stop_word_slice_list=[],
            is_streaming=True,
        )
        # Content correctly truncated — tokens after EOS are not emitted
        self.assertEqual(delta.output_str, "Hello")
        self.assertEqual(status.finish_reason, FinisheReason.stop)

    async def test_token_stop_wins_when_same_mtp_chunk_crosses_length(self):
        tokenizer = self._make_tokenizer({100: "A", 999: "X", 101: "B", 102: "C"})
        renderer = self._make_renderer(tokenizer, stop_word_ids_list=[[999]])
        status = await self._make_status(renderer)

        delta = await renderer._update_single_status(
            status,
            self._create_output([100, 999, 101, 102]),
            max_new_tokens=3,
            stop_words_str=["X"],
            stop_word_slice_list=get_stop_word_slices(["X"]),
            is_streaming=True,
        )

        self.assertEqual(delta.output_str, "A")
        self.assertEqual(status.finish_reason, FinisheReason.stop)

    async def test_token_stop_at_cutoff_wins_over_length(self):
        tokenizer = self._make_tokenizer({100: "A", 101: "B", 999: "X", 102: "C"})
        renderer = self._make_renderer(tokenizer, stop_word_ids_list=[[999]])
        status = await self._make_status(renderer)

        delta = await renderer._update_single_status(
            status,
            self._create_output([100, 101, 999, 102]),
            max_new_tokens=3,
            stop_words_str=["X"],
            stop_word_slice_list=get_stop_word_slices(["X"]),
            is_streaming=True,
        )

        self.assertEqual(delta.output_str, "AB")
        self.assertEqual(status.output_ids, [100, 101])
        self.assertEqual(status.finish_reason, FinisheReason.stop)

    async def test_length_wins_over_token_stop_after_mtp_cutoff(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 101: "B", 102: "C", 999: "X", 103: "after"}
        )
        renderer = self._make_renderer(tokenizer, stop_word_ids_list=[[999]])
        status = await self._make_status(renderer)

        delta = await renderer._update_single_status(
            status,
            self._create_output([100, 101, 102, 999, 103]),
            max_new_tokens=3,
            stop_words_str=["X"],
            stop_word_slice_list=get_stop_word_slices(["X"]),
            is_streaming=True,
        )

        self.assertEqual(delta.output_str, "ABC")
        self.assertEqual(status.output_ids, [100, 101, 102])
        self.assertEqual(status.finish_reason, FinisheReason.length)

    async def test_max_seq_cutoff_ignores_later_token_stop(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 101: "B", 102: "C", 999: "X", 103: "after"}
        )
        renderer = self._make_renderer(
            tokenizer, stop_word_ids_list=[[999]], max_seq_len=5
        )
        status = await self._make_status(renderer)

        delta = await renderer._update_single_status(
            status,
            self._create_output([100, 101, 102, 999, 103], input_len=2),
            max_new_tokens=100,
            stop_words_str=["X"],
            stop_word_slice_list=get_stop_word_slices(["X"]),
            is_streaming=True,
        )

        self.assertEqual(delta.output_str, "ABC")
        self.assertEqual(status.output_ids, [100, 101, 102])
        self.assertEqual(status.finish_reason, FinisheReason.length)

    async def test_mtp_cutoff_clamps_usage_aux_info_and_return_output_ids(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 101: "B", 102: "C", 103: "D", 104: "E"}
        )
        renderer = self._make_renderer(tokenizer)
        request = _make_chat_request()
        output = self._create_output([100, 101, 102, 103, 104], input_len=7)
        output.aux_info.step_output_len = 5
        output.aux_info.softmax_probs = [0.1, 0.2, 0.3, 0.4, 0.5]
        output.aux_info.cum_log_probs = [-1.5]
        output.aux_info.beam_responses = ["ABCDE"]

        async def output_generator():
            yield GenerateOutputs(generate_outputs=[output])

        responses = [
            response
            async for response in renderer.render_response_stream(
                output_generator(),
                request,
                GenerateConfig(
                    is_streaming=True,
                    max_new_tokens=3,
                    return_output_ids=True,
                ),
            )
        ]

        content_response = responses[1]
        self.assertEqual(content_response.choices[0].delta.content, "ABC")
        self.assertEqual(content_response.usage.completion_tokens, 3)
        self.assertEqual(content_response.usage.total_tokens, 10)
        self.assertEqual(content_response.extra_outputs.output_ids, [[100, 101, 102]])

        flush_response = responses[-2]
        self.assertEqual(flush_response.usage.completion_tokens, 3)
        self.assertEqual(flush_response.usage.total_tokens, 10)

        final_response = responses[-1]
        self.assertEqual(final_response.usage.completion_tokens, 3)
        self.assertEqual(final_response.usage.total_tokens, 10)
        self.assertEqual(final_response.aux_info.output_len, 3)
        self.assertEqual(final_response.aux_info.step_output_len, 3)
        self.assertEqual(final_response.aux_info.softmax_probs, [0.1, 0.2, 0.3])
        self.assertEqual(final_response.aux_info.cum_log_probs, [])
        self.assertEqual(final_response.aux_info.beam_responses, [])
        self.assertEqual(output.aux_info.output_len, 5)
        self.assertEqual(output.aux_info.step_output_len, 5)
        self.assertEqual(output.aux_info.softmax_probs, [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(output.aux_info.cum_log_probs, [-1.5])
        self.assertEqual(output.aux_info.beam_responses, ["ABCDE"])

    async def test_token_stop_and_eos_project_public_metadata(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 2: "", 999: "X", 102: "trailing"}
        )

        for terminal_token, eos_token_id, stop_word_ids_list, stop_words_str in (
            (999, 0, [[999]], ["X"]),
            (2, 2, [], []),
        ):
            with self.subTest(terminal_token=terminal_token):
                renderer = self._make_renderer(
                    tokenizer,
                    eos_token_id=eos_token_id,
                    stop_word_ids_list=stop_word_ids_list,
                )
                output = self._create_output(
                    [100, terminal_token, 102], input_len=7
                )
                output.aux_info.step_output_len = 3
                output.aux_info.softmax_probs = [0.1, 0.2, 0.3]

                async def output_generator():
                    yield GenerateOutputs(generate_outputs=[output])

                responses = [
                    response
                    async for response in renderer.render_response_stream(
                        output_generator(),
                        _make_chat_request(),
                        GenerateConfig(
                            is_streaming=True,
                            max_new_tokens=100,
                            stop_words_str=stop_words_str,
                            return_output_ids=True,
                        ),
                    )
                ]

                self.assertEqual(responses[1].choices[0].delta.content, "A")
                self.assertEqual(responses[1].usage.completion_tokens, 2)
                self.assertEqual(
                    responses[1].extra_outputs.output_ids,
                    [[100, terminal_token]],
                )
                self.assertEqual(responses[-1].usage.completion_tokens, 2)
                self.assertEqual(responses[-1].aux_info.output_len, 2)
                self.assertEqual(responses[-1].aux_info.step_output_len, 2)
                self.assertEqual(
                    responses[-1].aux_info.softmax_probs, [0.1, 0.2]
                )
                self.assertEqual(output.aux_info.output_len, 3)
                self.assertEqual(output.aux_info.step_output_len, 3)

    async def test_string_stop_projects_public_metadata(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 101: "<END>hidden", 102: "trailing"}
        )
        renderer = self._make_renderer(tokenizer)
        output = self._create_output([100, 101, 102], input_len=7)
        output.aux_info.step_output_len = 3

        async def output_generator():
            yield GenerateOutputs(generate_outputs=[output])

        responses = [
            response
            async for response in renderer.render_response_stream(
                output_generator(),
                _make_chat_request(),
                GenerateConfig(
                    is_streaming=True,
                    max_new_tokens=100,
                    stop_words_str=["<END>"],
                    return_output_ids=True,
                ),
            )
        ]

        self.assertEqual(responses[1].choices[0].delta.content, "A")
        self.assertEqual(responses[1].usage.completion_tokens, 2)
        self.assertEqual(responses[1].extra_outputs.output_ids, [[100, 101]])
        self.assertEqual(responses[-1].usage.completion_tokens, 2)
        self.assertEqual(responses[-1].aux_info.output_len, 2)
        self.assertEqual(responses[-1].aux_info.step_output_len, 2)

    async def test_cross_chunk_stop_projects_only_terminal_chunk_prefix(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 200: "S", 201: "TOP", 102: "trailing"}
        )
        renderer = self._make_renderer(
            tokenizer, stop_word_ids_list=[[200, 201]]
        )
        first_output = self._create_output([100, 200], input_len=7)
        first_output.aux_info.step_output_len = 2
        second_output = self._create_output([201, 102], input_len=7)
        second_output.aux_info.output_len = 4
        second_output.aux_info.step_output_len = 2

        async def output_generator():
            yield GenerateOutputs(generate_outputs=[first_output])
            yield GenerateOutputs(generate_outputs=[second_output])

        responses = [
            response
            async for response in renderer.render_response_stream(
                output_generator(),
                _make_chat_request(),
                GenerateConfig(
                    is_streaming=True,
                    max_new_tokens=100,
                    stop_words_str=["STOP"],
                    return_output_ids=True,
                ),
            )
        ]

        self.assertEqual(responses[1].extra_outputs.output_ids, [[100, 200]])
        self.assertEqual(responses[2].extra_outputs.output_ids, [[201]])
        self.assertEqual(responses[2].usage.completion_tokens, 3)
        self.assertEqual(responses[-1].usage.completion_tokens, 3)
        self.assertEqual(responses[-1].aux_info.output_len, 3)
        self.assertEqual(responses[-1].aux_info.step_output_len, 1)

    async def test_cross_chunk_string_stop_projects_terminal_token_boundary(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 200: "<EN", 201: "D>", 102: "trailing"}
        )
        renderer = self._make_renderer(tokenizer)
        first_output = self._create_output([100, 200], input_len=7)
        first_output.aux_info.step_output_len = 2
        second_output = self._create_output([201, 102], input_len=7)
        second_output.aux_info.output_len = 4
        second_output.aux_info.step_output_len = 2

        async def output_generator():
            yield GenerateOutputs(generate_outputs=[first_output])
            yield GenerateOutputs(generate_outputs=[second_output])

        responses = [
            response
            async for response in renderer.render_response_stream(
                output_generator(),
                _make_chat_request(),
                GenerateConfig(
                    is_streaming=True,
                    max_new_tokens=100,
                    stop_words_str=["<END>"],
                    return_output_ids=True,
                ),
            )
        ]

        self.assertEqual(responses[1].choices[0].delta.content, "A")
        self.assertEqual(responses[1].extra_outputs.output_ids, [[100, 200]])
        self.assertEqual(responses[2].extra_outputs.output_ids, [[201]])
        self.assertEqual(responses[2].usage.completion_tokens, 3)
        self.assertEqual(responses[-1].usage.completion_tokens, 3)
        self.assertEqual(responses[-1].aux_info.output_len, 3)
        self.assertEqual(responses[-1].aux_info.step_output_len, 1)

    async def test_qwen_async_stop_usage_counts_terminal_token(self):
        tokenizer = self._make_tokenizer(
            {100: "abcdefghij", 999: "X", 102: "trailing"}
        )
        renderer = QwenRenderer(
            tokenizer=tokenizer,
            renderer_params=RendererParams(
                model_type="test",
                max_seq_len=2048,
                eos_token_id=0,
                stop_word_ids_list=[[999]],
            ),
            generate_env_config=GenerateEnvConfig(),
        )
        status = QwenStreamStatus(_make_chat_request())
        output = self._create_output([100, 999, 102])

        delta = await renderer._update_single_status(
            status,
            output,
            max_new_tokens=100,
            stop_words_str=["X"],
            stop_word_slice_list=get_stop_word_slices(["X"]),
            is_streaming=True,
        )

        self.assertEqual(delta.output_length, 2)
        self.assertEqual(status.reported_output_length(output.aux_info.output_len), 2)

    async def test_two_chunk_mtp_cutoff_projects_only_current_visible_ids(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 101: "B", 102: "C", 103: "D", 104: "E"}
        )
        renderer = self._make_renderer(tokenizer)
        first_output = self._create_output([100, 101], input_len=7)
        first_output.aux_info.step_output_len = 2
        second_output = self._create_output([102, 103, 104], input_len=7)
        second_output.aux_info.output_len = 5
        second_output.aux_info.step_output_len = 3

        async def output_generator():
            yield GenerateOutputs(generate_outputs=[first_output])
            yield GenerateOutputs(generate_outputs=[second_output])

        responses = [
            response
            async for response in renderer.render_response_stream(
                output_generator(),
                _make_chat_request(),
                GenerateConfig(
                    is_streaming=True,
                    max_new_tokens=3,
                    return_output_ids=True,
                ),
            )
        ]

        self.assertEqual(responses[1].choices[0].delta.content, "AB")
        self.assertEqual(responses[1].extra_outputs.output_ids, [[100, 101]])
        self.assertEqual(responses[2].choices[0].delta.content, "C")
        self.assertEqual(responses[2].extra_outputs.output_ids, [[102]])
        self.assertEqual(responses[2].usage.completion_tokens, 3)

        final_response = responses[-1]
        self.assertEqual(final_response.usage.completion_tokens, 3)
        self.assertEqual(final_response.aux_info.output_len, 3)
        self.assertEqual(final_response.aux_info.step_output_len, 1)
        self.assertEqual(second_output.aux_info.output_len, 5)
        self.assertEqual(second_output.aux_info.step_output_len, 3)

    async def test_max_seq_cutoff_clamps_public_metadata(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 101: "B", 102: "C", 103: "D", 104: "E"}
        )
        renderer = self._make_renderer(tokenizer, max_seq_len=10)
        output = self._create_output([100, 101, 102, 103, 104], input_len=7)
        output.aux_info.step_output_len = 5

        async def output_generator():
            yield GenerateOutputs(generate_outputs=[output])

        responses = [
            response
            async for response in renderer.render_response_stream(
                output_generator(),
                _make_chat_request(),
                GenerateConfig(
                    is_streaming=True,
                    max_new_tokens=100,
                    return_output_ids=True,
                ),
            )
        ]

        self.assertEqual(responses[1].choices[0].delta.content, "ABC")
        self.assertEqual(responses[1].usage.completion_tokens, 3)
        self.assertEqual(responses[1].usage.total_tokens, 10)
        self.assertEqual(responses[1].extra_outputs.output_ids, [[100, 101, 102]])
        self.assertEqual(responses[-1].usage.completion_tokens, 3)
        self.assertEqual(responses[-1].aux_info.output_len, 3)
        self.assertEqual(responses[-1].aux_info.step_output_len, 3)

    async def test_clipped_chunk_rejects_unprojectable_output_tensors(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 101: "B", 102: "C", 103: "D", 104: "E"}
        )

        for config_field, output_field in (
            ("return_hidden_states", "hidden_states"),
            ("return_all_hidden_states", "all_hidden_states"),
            ("return_logits", "logits"),
        ):
            with self.subTest(config_field=config_field):
                renderer = self._make_renderer(tokenizer)
                output = self._create_output([100, 101, 102, 103, 104])
                setattr(output, output_field, torch.tensor([[1.0, 2.0]]))
                generate_config = GenerateConfig(
                    is_streaming=True,
                    max_new_tokens=3,
                    **{config_field: True},
                )

                async def output_generator():
                    yield GenerateOutputs(generate_outputs=[output])

                with self.assertRaisesRegex(
                    RuntimeError,
                    "cannot project hidden states or logits from a clipped output chunk",
                ):
                    async for _ in renderer.render_response_stream(
                        output_generator(), _make_chat_request(), generate_config
                    ):
                        pass

    async def test_clipped_chunk_rejects_cumulative_log_probs(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 101: "B", 102: "C", 103: "D", 104: "E"}
        )
        renderer = self._make_renderer(tokenizer)
        output = self._create_output([100, 101, 102, 103, 104])
        output.aux_info.cum_log_probs = [-1.5]

        async def output_generator():
            yield GenerateOutputs(generate_outputs=[output])

        with self.assertRaisesRegex(
            RuntimeError,
            "cannot project cumulative log probabilities from a clipped output chunk",
        ):
            async for _ in renderer.render_response_stream(
                output_generator(),
                _make_chat_request(),
                GenerateConfig(
                    is_streaming=True,
                    max_new_tokens=3,
                    return_cum_log_probs=True,
                ),
            ):
                pass

    def test_sync_mtp_cutoff_clamps_stream_and_final_usage(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 101: "B", 102: "C", 103: "D", 104: "E"}
        )
        renderer = self._make_renderer(tokenizer)
        status = StreamStatusSync(_make_chat_request())

        delta = self._update_sync(
            renderer,
            status,
            [100, 101, 102, 103, 104],
            max_new_tokens=3,
            stop_words_str=[],
        )

        self.assertEqual(delta.output_str, "ABC")
        self.assertEqual(delta.output_length, 3)
        stream_response = renderer._generate_stream_response_sync([delta])
        self.assertEqual(stream_response.usage.completion_tokens, 3)
        self.assertEqual(stream_response.usage.total_tokens, 3)

        final_response = renderer._generate_final_sync(
            [status], [0], [5], [0]
        )
        self.assertEqual(final_response.usage.completion_tokens, 3)
        self.assertEqual(final_response.usage.total_tokens, 3)

    async def test_mtp_logprobs_are_explicitly_unsupported(self):
        tokenizer = self._make_tokenizer(
            {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
        )
        renderer = self._make_renderer(tokenizer)
        request = _make_chat_request(logprobs=True, top_logprobs=1)
        status = (await renderer._create_status_list(1, request))[0]
        output = self._create_output([1, 2, 3, 4, 5])
        output.all_probs = torch.tensor([0.01, 0.10, 0.20, 0.40, 0.19, 0.10])

        with self.assertRaisesRegex(
            RuntimeError, "logprobs are not supported for multi-token output chunks"
        ):
            await renderer._update_single_status(
                status,
                output,
                max_new_tokens=3,
                stop_words_str=[],
                stop_word_slice_list=[],
                is_streaming=True,
            )

    async def test_string_level_stop_word_without_token_truncation(self):
        """String-level stop word that doesn't correspond to a token boundary.
        Token-level truncation doesn't fire; _process_stop_words handles it.
        Tokens after the stop word in the same MTP chunk must not be emitted."""
        tokenizer = self._make_tokenizer({100: "Hello", 101: "<|end|>", 102: "world"})
        renderer = self._make_renderer(tokenizer, stop_word_ids_list=[])
        status = await self._make_status(renderer)

        stop_words_str = ["<|end|>"]
        stop_word_slice_list = get_stop_word_slices(stop_words_str)

        delta = await renderer._update_single_status(
            status,
            self._create_output([100, 101, 102]),
            max_new_tokens=100,
            stop_words_str=stop_words_str,
            stop_word_slice_list=stop_word_slice_list,
            is_streaming=True,
        )
        self.assertEqual(delta.output_str, "Hello")
        self.assertEqual(status.finish_reason, FinisheReason.stop)

    async def test_string_stop_wins_when_same_mtp_chunk_crosses_length(self):
        tokenizer = self._make_tokenizer(
            {100: "Hello", 101: "<|end|>", 102: "world", 103: "after"}
        )
        renderer = self._make_renderer(tokenizer, stop_word_ids_list=[])
        status = await self._make_status(renderer)

        stop_words_str = ["<|end|>"]
        delta = await renderer._update_single_status(
            status,
            self._create_output([100, 101, 102, 103]),
            max_new_tokens=3,
            stop_words_str=stop_words_str,
            stop_word_slice_list=get_stop_word_slices(stop_words_str),
            is_streaming=True,
        )

        self.assertEqual(delta.output_str, "Hello")
        self.assertEqual(status.finish_reason, FinisheReason.stop)

    async def test_string_stop_at_cutoff_wins_but_after_cutoff_is_length(self):
        tokenizer = self._make_tokenizer(
            {100: "A", 101: "B", 102: "C", 103: "<|end|>", 104: "after"}
        )
        stop_words_str = ["<|end|>"]

        for tokens, expected_text, expected_reason in (
            ([100, 101, 103, 104], "AB", FinisheReason.stop),
            ([100, 101, 102, 103, 104], "ABC", FinisheReason.length),
        ):
            with self.subTest(tokens=tokens):
                renderer = self._make_renderer(tokenizer, stop_word_ids_list=[])
                status = await self._make_status(renderer)

                delta = await renderer._update_single_status(
                    status,
                    self._create_output(tokens),
                    max_new_tokens=3,
                    stop_words_str=stop_words_str,
                    stop_word_slice_list=get_stop_word_slices(stop_words_str),
                    is_streaming=True,
                )

                self.assertEqual(delta.output_str, expected_text)
                self.assertEqual(status.finish_reason, expected_reason)

    async def test_string_level_stop_word_single_token_per_chunk(self):
        """String-level stop word — standard (non-MTP) case: one token per chunk.
        After the stop-word token, no more chunks arrive."""
        tokenizer = self._make_tokenizer({100: "Hello", 101: "<|end|>"})
        renderer = self._make_renderer(tokenizer, stop_word_ids_list=[])
        status = await self._make_status(renderer)

        stop_words_str = ["<|end|>"]
        stop_word_slice_list = get_stop_word_slices(stop_words_str)

        delta1 = await renderer._update_single_status(
            status,
            self._create_output([100]),
            max_new_tokens=100,
            stop_words_str=stop_words_str,
            stop_word_slice_list=stop_word_slice_list,
            is_streaming=True,
        )
        self.assertEqual(delta1.output_str, "Hello")
        self.assertIsNone(status.finish_reason)

        delta2 = await renderer._update_single_status(
            status,
            self._create_output([101]),
            max_new_tokens=100,
            stop_words_str=stop_words_str,
            stop_word_slice_list=stop_word_slice_list,
            is_streaming=True,
        )
        # Stop word consumed, nothing emitted, finish_reason set
        self.assertEqual(delta2.output_str, "")
        self.assertEqual(status.finish_reason, FinisheReason.stop)

    async def test_no_stop_word_normal_streaming(self):
        """Baseline: normal streaming with no stop words. All content emitted."""
        tokenizer = self._make_tokenizer({100: "Hello", 101: " world"})
        renderer = self._make_renderer(tokenizer)
        status = await self._make_status(renderer)

        delta1 = await renderer._update_single_status(
            status,
            self._create_output([100]),
            max_new_tokens=100,
            stop_words_str=[],
            stop_word_slice_list=[],
            is_streaming=True,
        )
        self.assertEqual(delta1.output_str, "Hello")
        self.assertIsNone(status.finish_reason)

        delta2 = await renderer._update_single_status(
            status,
            self._create_output([101]),
            max_new_tokens=100,
            stop_words_str=[],
            stop_word_slice_list=[],
            is_streaming=True,
        )
        self.assertEqual(delta2.output_str, " world")
        self.assertIsNone(status.finish_reason)

    async def test_subsequent_calls_after_finish_return_empty(self):
        """After finish_reason is set, subsequent calls must return empty."""
        eos = 2
        tokenizer = self._make_tokenizer({100: "A", eos: ""})
        renderer = self._make_renderer(tokenizer, eos_token_id=eos)
        status = await self._make_status(renderer)

        delta1 = await renderer._update_single_status(
            status,
            self._create_output([100, eos]),
            max_new_tokens=100,
            stop_words_str=[],
            stop_word_slice_list=[],
            is_streaming=True,
        )
        self.assertEqual(delta1.output_str, "A")
        self.assertEqual(status.finish_reason, FinisheReason.stop)

        # Subsequent call must be a no-op
        delta2 = await renderer._update_single_status(
            status,
            self._create_output([]),
            max_new_tokens=100,
            stop_words_str=[],
            stop_word_slice_list=[],
            is_streaming=True,
        )
        self.assertEqual(delta2.output_str, "")


if __name__ == "__main__":
    main()
