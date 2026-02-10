from typing import List
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, Mock

import torch

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
)
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput
from rtp_llm.utils.word_util import get_stop_word_slices


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
    def _make_renderer(tokenizer, eos_token_id=0, stop_word_ids_list=None):
        class TestRenderer(ReasoningToolBaseRenderer):
            def _setup_chat_template(self):
                self.chat_template = "test"

            def in_think_mode(self, request: ChatCompletionRequest):
                return False

        return TestRenderer(
            tokenizer=tokenizer,
            renderer_params=RendererParams(
                model_type="test",
                max_seq_len=2048,
                eos_token_id=eos_token_id,
                stop_word_ids_list=stop_word_ids_list or [],
            ),
            generate_env_config=GenerateEnvConfig(),
        )

    @staticmethod
    def _create_output(tokens):
        aux_info = AuxInfo()
        aux_info.input_len = 0
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
        misses a stop word that isn't at the end. _remove_stop_word_ids truncates
        the content correctly, but finish_reason is not set by the renderer
        (the engine is expected to set it)."""
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
        # NOTE: finish_reason is None because _check_finish_reason only checks
        # the suffix of output_ids_list ([102] ≠ [999]). In production the engine
        # sets finish_reason; the renderer relies on that.
        self.assertIsNone(status.finish_reason)

    async def test_eos_in_mtp_chunk_with_trailing_tokens(self):
        """MTP: EOS token appears mid-chunk with trailing tokens.
        _check_finish_reason only checks the last token — does NOT catch mid-chunk EOS.
        _remove_stop_word_ids truncates content at EOS position. Engine sets finish_reason.
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
        # finish_reason is None for the same reason as the stop-word case:
        # _check_finish_reason checks token_ids[-1] == eos_token_id, but the
        # last token is 103, not EOS. Engine handles this.
        self.assertIsNone(status.finish_reason)

    async def test_string_level_stop_word_without_token_truncation(self):
        """String-level stop word that doesn't correspond to a token boundary.
        Token-level truncation doesn't fire; _process_stop_words handles it.

        Known limitation: _process_streaming_tokens doesn't break on
        finish_reason, so tokens after the string-level stop word still get
        processed and emitted in the same chunk. In production this is masked
        because the engine stops generating when it hits stop words at the
        token level.  Here we test the actual (imperfect) renderer behavior."""
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
        # "world" leaks because the per-token loop doesn't break on finish_reason.
        # In production the engine wouldn't generate token 102 after stop word.
        self.assertEqual(delta.output_str, "Helloworld")
        self.assertEqual(status.finish_reason, FinisheReason.stop)

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
