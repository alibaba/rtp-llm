import json
import os
import tempfile
from pathlib import Path
from typing import Any, List
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, Mock

import torch

from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    FinisheReason,
    GPTFunctionDefinition,
    GPTToolDefinition,
    RoleEnum,
)
from rtp_llm.openai.renderers.chatglm45_renderer import ChatGlm45Renderer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RendererParams,
    StreamStatus,
)
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
    ReasoningToolStreamStatus,
)
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


class EncodeExtraStopWordsTest(TestCase):
    """Test encode_extra_stop_words, which resolves stop words against the live tokenizer."""

    def setUp(self):
        self.renderer = Mock(spec=CustomChatRenderer)
        self.renderer.tokenizer = Mock()
        self.renderer.encode_extra_stop_words = (
            CustomChatRenderer.encode_extra_stop_words.__get__(self.renderer)
        )

    def test_ids_come_from_tokenizer_not_hardcoded(self):
        # A 248K-vocab checkpoint maps the legacy 151K ids onto unrelated tokens,
        # so the ids must be derived per tokenizer instead of being written down.
        encoded = {"Observation:": [1, 2, 3], "<|endoftext|>": [248044]}
        self.renderer.tokenizer.encode = Mock(
            side_effect=lambda word, add_special_tokens: encoded[word]
        )

        result = self.renderer.encode_extra_stop_words(
            ["Observation:", "<|endoftext|>"]
        )

        self.assertEqual(result, [[1, 2, 3], [248044]])
        for call in self.renderer.tokenizer.encode.call_args_list:
            self.assertFalse(call.kwargs["add_special_tokens"])

    def test_skips_words_the_tokenizer_cannot_encode(self):
        self.renderer.tokenizer.encode = Mock(return_value=[])

        self.assertEqual(self.renderer.encode_extra_stop_words(["<|absent|>"]), [])

    def test_falls_back_for_tokenizers_without_add_special_tokens(self):
        # Legacy tokenizers expose encode(text) only. The repo already tolerates
        # this for request stop words; extra stop words must not be stricter.
        def encode(word, **kwargs):
            if kwargs:
                raise TypeError("encode() got an unexpected keyword argument")
            return [9, 9]

        self.renderer.tokenizer.encode = Mock(side_effect=encode)

        self.assertEqual(self.renderer.encode_extra_stop_words(["x"]), [[9, 9]])

    def _legacy_tokenizer(self, encode_result, special_token_map):
        tokenizer = Mock()
        tokenizer.all_special_tokens = list(special_token_map)
        tokenizer.convert_tokens_to_ids = Mock(
            side_effect=lambda tokens: (
                special_token_map[tokens]
                if isinstance(tokens, str)
                else [special_token_map[token] for token in tokens]
            )
        )

        def encode(word, **kwargs):
            if kwargs:
                raise TypeError("encode() got an unexpected keyword argument")
            return list(encode_result)

        tokenizer.encode = Mock(side_effect=encode)
        return tokenizer

    def test_fallback_trims_bos_eos_wrapped_around_the_word(self):
        # 旧式 encode() 默认会追加 BOS/EOS：包着特殊 token 的序列在生成输出中
        # 永不出现，注册了也匹配不到，等于静默失效。
        self.renderer.tokenizer = self._legacy_tokenizer(
            [1, 42, 43, 2], {"<s>": 1, "</s>": 2}
        )

        self.assertEqual(
            self.renderer.encode_extra_stop_words(["Observation:"]), [[42, 43]]
        )

    def test_fallback_keeps_a_word_that_is_itself_a_special_token(self):
        # <|endoftext|> 解析出来就是特殊 token：可以去掉包裹的 BOS，但不能把
        # 词自己一起裁掉，否则停止词静默消失。
        self.renderer.tokenizer = self._legacy_tokenizer(
            [1, 2], {"<s>": 1, "<|endoftext|>": 2}
        )

        self.assertEqual(
            self.renderer.encode_extra_stop_words(["<|endoftext|>"]), [[2]]
        )

    def test_fallback_keeps_special_word_between_bos_and_eos(self):
        self.renderer.tokenizer = self._legacy_tokenizer(
            [1, 7, 2], {"<s>": 1, "<|endoftext|>": 7, "</s>": 2}
        )

        self.assertEqual(
            self.renderer.encode_extra_stop_words(["<|endoftext|>"]), [[7]]
        )

    def test_fallback_without_special_token_metadata_is_unchanged(self):
        # 拿不到 all_special_tokens 时不猜：保持原样返回，行为与旧实现相同。
        def encode(word, **kwargs):
            if kwargs:
                raise TypeError("encode() got an unexpected keyword argument")
            return [9, 9]

        self.renderer.tokenizer.encode = Mock(side_effect=encode)

        self.assertEqual(self.renderer.encode_extra_stop_words(["x"]), [[9, 9]])

    def test_returns_copy_not_tokenizer_buffer(self):
        shared = [7, 8]
        self.renderer.tokenizer.encode = Mock(return_value=shared)

        result = self.renderer.encode_extra_stop_words(["x"])
        result[0].append(9)

        self.assertEqual(shared, [7, 8])


class RealTokenizerStopWordTest(TestCase):
    """用真实 tokenizer 验证停止词按字符串反查。

    EncodeExtraStopWordsTest 只覆盖了拼接与降级分支，未验证真实词表上的解析结果。
    回归背景：旧实现写死 151K 词表的 id（[37763, 367, 25] / [151643]），换词表后会
    把无关 token 注册成停止序列；这里锁定解析结果必须可回解为原字符串。
    """

    TOKENIZER_RELATIVE_PATH = (
        "rtp_llm/test/model_test/fake_test/testdata/qwen3_30b/tokenizer"
    )

    def setUp(self):
        self.tokenizer = BaseTokenizer(
            os.path.join(os.getcwd(), self.TOKENIZER_RELATIVE_PATH)
        )
        self.renderer = Mock(spec=CustomChatRenderer)
        self.renderer.tokenizer = self.tokenizer
        self.renderer.encode_extra_stop_words = (
            CustomChatRenderer.encode_extra_stop_words.__get__(self.renderer)
        )

    def test_words_round_trip_through_the_live_tokenizer(self):
        words = ["Observation:", "<|endoftext|>"]
        result = self.renderer.encode_extra_stop_words(words)

        self.assertEqual(len(result), len(words))
        for ids, word in zip(result, words):
            self.assertEqual(self.tokenizer.decode(ids), word)

    def test_legacy_ids_are_reproduced_on_the_151k_vocab(self):
        """反查结果必须与旧的硬编码 id 逐位相同，即 151K 词表上零行为变化。

        这是唯一能钉住「换成了哪个字符串」的断言：round-trip 与「解析为该词表自己
        的 id」对任何字符串都成立，改错词也照样通过。改动过程中确实一度把 151643
        写成了 `<|fim_middle|>`（实际是 151660），只有这条断言能拦住。
        """
        self.assertEqual(
            self.renderer.encode_extra_stop_words(["Observation:", "<|endoftext|>"]),
            [[37763, 367, 25], [151643]],
        )

    def test_fim_middle_is_not_endoftext(self):
        # 钉住那次改错的具体事实，避免再被"151643 是 <|fim_middle|>"的说法带偏。
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("<|fim_middle|>"), 151660)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("<|endoftext|>"), 151643)

    def test_special_token_resolves_to_the_tokenizer_own_id(self):
        # <|endoftext|> 必须解析为该词表自己的特殊 token id（151K 词表上是
        # 151643），而不能退化成对字面量的逐字切分。
        (ids,) = self.renderer.encode_extra_stop_words(["<|endoftext|>"])
        self.assertEqual(ids, [self.tokenizer.convert_tokens_to_ids("<|endoftext|>")])

    def test_empty_word_is_skipped(self):
        self.assertEqual(self.renderer.encode_extra_stop_words([""]), [])


class RealRendererStopWordRegistrationTest(TestCase):
    """真实构造 renderer，验证构造函数注册的额外停止 id。

    上面的用例只覆盖 encode_extra_stop_words 这个 helper，而缺陷原址是
    QwenReasoningToolRenderer._setup_stop_words / QwenRenderer 的构造函数。
    helper 正确但构造函数传错字符串同样会注册错 token，所以这里从构造结果断言。
    """

    TOKENIZER_RELATIVE_PATH = (
        "rtp_llm/test/model_test/fake_test/testdata/qwen3_30b/tokenizer"
    )

    def setUp(self):
        from rtp_llm.openai.renderers.qwen_reasoning_tool_renderer import (
            QwenReasoningToolRenderer,
        )

        self.tokenizer = BaseTokenizer(
            os.path.join(os.getcwd(), self.TOKENIZER_RELATIVE_PATH)
        )
        self.renderer = QwenReasoningToolRenderer(
            tokenizer=self.tokenizer,
            renderer_params=RendererParams(
                model_type="qwen_3",
                max_seq_len=2048,
                eos_token_id=151645,
                stop_word_ids_list=[],
            ),
            generate_env_config=GenerateEnvConfig(),
        )

    def test_registers_endoftext_only(self):
        # 对照 A 的形态：这条继承链只注册一组 id，且必须是 <|endoftext|>。
        self.assertEqual(self.renderer.extra_stop_word_ids_list, [[151643]])

    def test_registered_ids_decode_back_to_the_intended_word(self):
        for ids in self.renderer.extra_stop_word_ids_list:
            self.assertEqual(self.tokenizer.decode(ids), "<|endoftext|>")


class Qwen35RealTokenizerStopWordRegistrationTest(TestCase):
    @staticmethod
    def _special_token(content: str, token_id: int) -> dict[str, Any]:
        return {
            "id": token_id,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }

    def _create_tokenizer(self, root: Path) -> Path:
        tokenizer_path = root / "qwen35_tokenizer"
        tokenizer_path.mkdir()
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                self._special_token("<|endoftext|>", 248044),
                self._special_token("<|im_start|>", 248045),
                self._special_token("<|im_end|>", 248046),
            ],
            "normalizer": None,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<unk>": 0,
                    "Observation:": 7,
                    "<|endoftext|>": 248044,
                    "<|im_start|>": 248045,
                    "<|im_end|>": 248046,
                },
                "unk_token": "<unk>",
            },
        }
        (tokenizer_path / "tokenizer.json").write_text(json.dumps(tokenizer_json))
        tokenizer_config = {
            "added_tokens_decoder": {
                str(token_id): {
                    "content": content,
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True,
                }
                for content, token_id in (
                    ("<|endoftext|>", 248044),
                    ("<|im_start|>", 248045),
                    ("<|im_end|>", 248046),
                )
            },
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            "eos_token": "<|im_end|>",
            "model_max_length": 32768,
            "pad_token": "<|endoftext|>",
            "tokenizer_class": "TokenizersBackend",
            "unk_token": "<unk>",
        }
        (tokenizer_path / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config)
        )
        (tokenizer_path / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5"})
        )
        return tokenizer_path

    def test_qwen35_registers_248k_stop_ids_from_live_tokenizer(self):
        from rtp_llm.openai.renderers.qwen35_renderer import Qwen35Renderer

        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer_path = self._create_tokenizer(Path(temp_dir))
            tokenizer = TokenizerFactory.create(
                str(tokenizer_path), str(tokenizer_path), "qwen35_dense"
            )
            renderer = Qwen35Renderer(
                tokenizer=tokenizer,
                renderer_params=RendererParams(
                    model_type="qwen35_dense",
                    max_seq_len=32768,
                    eos_token_id=248046,
                    stop_word_ids_list=[],
                ),
                generate_env_config=GenerateEnvConfig(),
            )

            self.assertIn([7], renderer.extra_stop_word_ids_list)
            self.assertIn([248044], renderer.extra_stop_word_ids_list)
            self.assertNotIn([151643], renderer.extra_stop_word_ids_list)
            self.assertEqual(tokenizer.decode([7]), "Observation:")
            self.assertEqual(tokenizer.decode([248044]), "<|endoftext|>")


class CreateReasoningParserTest(TestCase):
    """覆盖各 renderer 的 _create_reasoning_parser。

    核心不变量：渲染后的 prompt 以 think 锚点结尾（anchored）时，即便请求侧
    thinking_mode 为 DISABLED 也必须创建解析器，否则思考块会泄漏进可见回复。
    """

    THINK_START_TAG = "<think>\n"
    TAGLESS_REPLY = "plain answer"

    def _make_request(self, recorded_anchor=None, tools=None):
        """recorded_anchor 模拟 endpoint 在渲染时记下的锚点状态。

        None 表示没有走过 endpoint（例如 dash_sc / raw 链路），此时 renderer
        必须自己回退到渲染探测。
        """
        request = Mock()
        request.tools = tools
        request.logprobs = None
        request.prompt_has_think_anchor = Mock(return_value=recorded_anchor)
        return request

    def _make_renderer(
        self,
        renderer_cls,
        anchored,
        in_think_mode,
        render_raises=False,
        prompt_tail=None,
    ):
        renderer = Mock(spec=renderer_cls)
        renderer.think_start_tag = self.THINK_START_TAG
        renderer.in_think_mode = Mock(return_value=in_think_mode)
        if render_raises:
            renderer.render_chat = Mock(side_effect=RuntimeError("render failed"))
        else:
            if prompt_tail is None:
                prompt = (
                    f"user hello\n{self.THINK_START_TAG}" if anchored else "user hello"
                )
            else:
                prompt = f"user hello\n{prompt_tail}"
            rendered = Mock()
            rendered.rendered_prompt = prompt
            renderer.render_chat = Mock(return_value=rendered)
        renderer._create_reasoning_parser = (
            renderer_cls._create_reasoning_parser.__get__(renderer)
        )
        renderer._prompt_ends_with_think_anchor = (
            renderer_cls._prompt_ends_with_think_anchor.__get__(renderer)
        )
        renderer._resolve_think_anchor = renderer_cls._resolve_think_anchor.__get__(
            renderer
        )
        return renderer

    def _assert_forces_reasoning(self, parser, expected):
        """force_reasoning 的可观测效果：无标签文本是否被整体当成思考内容。"""
        reasoning_text, normal_text = parser.parse_non_stream(self.TAGLESS_REPLY)
        if expected:
            self.assertEqual(reasoning_text, self.TAGLESS_REPLY)
            self.assertEqual(normal_text, "")
        else:
            self.assertEqual(reasoning_text, "")
            self.assertEqual(normal_text, self.TAGLESS_REPLY)

    def _check(self, renderer_cls, expected_forced, render_raises=False):
        # render_chat 抛异常时锚点无从探测，anchored 恒为 False。
        if render_raises:
            cases = [(False, True, expected_forced)]
        else:
            cases = [
                (True, False, True),
                (True, True, True),
                (False, True, expected_forced),
            ]
        for anchored, in_think_mode, forced in cases:
            with self.subTest(anchored=anchored, in_think_mode=in_think_mode):
                renderer = self._make_renderer(
                    renderer_cls,
                    anchored=anchored,
                    in_think_mode=in_think_mode,
                    render_raises=render_raises,
                )
                parser = renderer._create_reasoning_parser(self._make_request())
                self.assertIsNotNone(parser)
                self._assert_forces_reasoning(parser, forced)

    def _check_returns_none(self, renderer_cls, render_raises=False):
        """既未锚定又未开启 thinking_mode 时不应创建解析器。"""
        renderer = self._make_renderer(
            renderer_cls,
            anchored=False,
            in_think_mode=False,
            render_raises=render_raises,
        )
        self.assertIsNone(renderer._create_reasoning_parser(self._make_request()))

    def _all_renderers(self):
        from rtp_llm.openai.renderers.chatglm45_renderer import ChatGlm45Renderer
        from rtp_llm.openai.renderers.deepseekv4_renderer import DeepseekV4Renderer
        from rtp_llm.openai.renderers.deepseekv31_renderer import DeepseekV31Renderer
        from rtp_llm.openai.renderers.deepseekv32_renderer import DeepseekV32Renderer
        from rtp_llm.openai.renderers.kimik2_renderer import KimiK2Renderer
        from rtp_llm.openai.renderers.qwen3_code_renderer import Qwen3CoderRenderer
        from rtp_llm.openai.renderers.qwen_reasoning_tool_renderer import (
            QwenReasoningToolRenderer,
        )

        # 第二项：在「已开启 thinking_mode 但未锚定」时是否仍强制解析。
        return [
            (Qwen3CoderRenderer, False),
            (QwenReasoningToolRenderer, False),
            (DeepseekV31Renderer, False),
            (DeepseekV32Renderer, False),
            (DeepseekV4Renderer, False),
            (ChatGlm45Renderer, False),
            (KimiK2Renderer, True),
        ]

    def test_anchored_creates_parser_even_when_thinking_disabled(self):
        for renderer_cls, expected_forced in self._all_renderers():
            with self.subTest(renderer=renderer_cls.__name__):
                self._check(renderer_cls, expected_forced)
                self._check_returns_none(renderer_cls)

    def test_render_failure_falls_back_to_thinking_mode(self):
        """render_chat 抛异常时应退化为仅按 thinking_mode 判断，而非整体失败。"""
        for renderer_cls, expected_forced in self._all_renderers():
            with self.subTest(renderer=renderer_cls.__name__):
                self._check(renderer_cls, expected_forced, render_raises=True)
                self._check_returns_none(renderer_cls, render_raises=True)

    def test_bare_think_anchor_without_trailing_newline_is_detected(self):
        """DeepSeek encoding 只追加裸 `<think>`，而默认 think_start_tag 是 `<think>\\n`。
        严格比较会失配：ENABLED 路径 force_reasoning 由 True 翻成 False，思考内容
        泄漏进可见回复（openai_response_test 的 deepseek_v31 用例即为此回归）。"""
        for renderer_cls, _ in self._all_renderers():
            for in_think_mode in (False, True):
                with self.subTest(
                    renderer=renderer_cls.__name__, in_think_mode=in_think_mode
                ):
                    renderer = self._make_renderer(
                        renderer_cls,
                        anchored=True,
                        in_think_mode=in_think_mode,
                        prompt_tail="<think>",
                    )
                    parser = renderer._create_reasoning_parser(self._make_request())
                    self.assertIsNotNone(parser)
                    self._assert_forces_reasoning(parser, True)

    def test_tag_and_prompt_tail_may_differ_in_newline(self):
        """配置锚点带换行、模板不带（DeepSeek），或反之（配置裸锚点、Qwen 模板带
        换行），两个方向都必须判定为 anchored。"""
        for tag, tail in (("<think>\n", "<think>"), ("<think>", "<think>\n")):
            for renderer_cls, _ in self._all_renderers():
                with self.subTest(renderer=renderer_cls.__name__, tag=tag, tail=tail):
                    renderer = self._make_renderer(
                        renderer_cls,
                        anchored=True,
                        in_think_mode=False,
                        prompt_tail=tail,
                    )
                    renderer.think_start_tag = tag
                    parser = renderer._create_reasoning_parser(self._make_request())
                    self.assertIsNotNone(parser)
                    self._assert_forces_reasoning(parser, True)

    def test_closed_empty_think_block_is_not_an_anchor(self):
        """模板关闭 think 时注入的是 `<think></think>` 空块（对照 B），不能判定为
        anchored，否则 DISABLED 请求会平白多出一个解析器。"""
        for renderer_cls, _ in self._all_renderers():
            with self.subTest(renderer=renderer_cls.__name__):
                renderer = self._make_renderer(
                    renderer_cls,
                    anchored=False,
                    in_think_mode=False,
                    prompt_tail="<think></think>",
                )
                self.assertIsNone(
                    renderer._create_reasoning_parser(self._make_request())
                )

    def test_recorded_anchor_is_used_without_rendering_again(self):
        """endpoint 渲染时已记下锚点状态，renderer 不得再渲染一次。

        回归背景：早先的实现在每次 _create_reasoning_parser 里重渲染一遍 prompt，
        给带 tools 的请求平白加了一次完整 jinja 渲染加编码。
        """
        for renderer_cls, expected_forced in self._all_renderers():
            for recorded in (True, False):
                with self.subTest(renderer=renderer_cls.__name__, recorded=recorded):
                    renderer = self._make_renderer(
                        renderer_cls,
                        anchored=not recorded,  # 与记录值相反，证明用的是记录值
                        in_think_mode=True,
                    )
                    parser = renderer._create_reasoning_parser(
                        self._make_request(recorded_anchor=recorded)
                    )
                    renderer.render_chat.assert_not_called()
                    self.assertIsNotNone(parser)
                    self._assert_forces_reasoning(
                        parser, True if recorded else expected_forced
                    )

    def test_recorded_anchor_builds_parser_when_thinking_disabled(self):
        """案例一的一般形态：模板注入了锚点但请求侧 thinking_mode 为 DISABLED。

        此时 in_think_mode 为假，只有锚点这一项能触发建解析器；建不出来思考块就
        会整段泄漏进可见回复。
        """
        for renderer_cls, _ in self._all_renderers():
            with self.subTest(renderer=renderer_cls.__name__):
                renderer = self._make_renderer(
                    renderer_cls, anchored=False, in_think_mode=False
                )
                parser = renderer._create_reasoning_parser(
                    self._make_request(recorded_anchor=True)
                )
                renderer.render_chat.assert_not_called()
                self.assertIsNotNone(parser)
                self._assert_forces_reasoning(parser, True)


class NeedsReasoningToolStatusTest(TestCase):
    """状态列表门控：解析器只在门控放行时才会被创建。

    案例一的失效链有两道门：门控与工厂方法。早先只修了工厂方法，门控仍然是
    `tools or in_think_mode`，于是「模板有锚点 + DISABLED + 无 tools」的请求走
    普通状态对象，解析器根本不会被创建，思考块照旧泄漏。
    """

    def _make_renderer(self, in_think_mode):
        renderer = Mock(spec=CustomChatRenderer)
        renderer.in_think_mode = Mock(return_value=in_think_mode)
        renderer._effective_tools = CustomChatRenderer._effective_tools.__get__(
            renderer
        )
        renderer.needs_reasoning_tool_status = (
            CustomChatRenderer.needs_reasoning_tool_status.__get__(renderer)
        )
        return renderer

    def _make_request(self, recorded_anchor=None, tools=None, tool_choice=None):
        request = Mock()
        request.tools = tools
        request.tool_choice = tool_choice
        request.prompt_has_think_anchor = Mock(return_value=recorded_anchor)
        return request

    def test_anchor_alone_opens_the_gate(self):
        renderer = self._make_renderer(in_think_mode=False)
        self.assertTrue(
            renderer.needs_reasoning_tool_status(
                self._make_request(recorded_anchor=True)
            )
        )

    def test_tools_or_think_mode_still_open_the_gate(self):
        self.assertTrue(
            self._make_renderer(in_think_mode=True).needs_reasoning_tool_status(
                self._make_request()
            )
        )
        self.assertTrue(
            self._make_renderer(in_think_mode=False).needs_reasoning_tool_status(
                self._make_request(tools=["a tool"])
            )
        )

    def test_disabled_tools_do_not_open_the_gate(self):
        # tool_choice=none 时工具不可用，门控只应看 think 与锚点。
        renderer = self._make_renderer(in_think_mode=False)
        self.assertFalse(
            renderer.needs_reasoning_tool_status(
                self._make_request(tools=["a tool"], tool_choice="none")
            )
        )

    def test_plain_request_keeps_the_gate_shut(self):
        renderer = self._make_renderer(in_think_mode=False)
        for recorded in (None, False):
            with self.subTest(recorded=recorded):
                self.assertFalse(
                    renderer.needs_reasoning_tool_status(
                        self._make_request(recorded_anchor=recorded)
                    )
                )

    def test_unknown_anchor_does_not_open_the_gate(self):
        """未探测过（None）不能当成有锚点：那会让门控在无 tools 无 think 的常见
        路径上放行，等于把渲染成本加回来。"""
        renderer = self._make_renderer(in_think_mode=False)
        request = self._make_request(recorded_anchor=None)
        self.assertFalse(renderer.needs_reasoning_tool_status(request))


class ReasoningStatusWithLogprobsTest(IsolatedAsyncioTestCase):
    async def test_logprobs_keeps_reasoning_aware_status(self):
        renderer = Mock(spec=ReasoningToolBaseRenderer)
        renderer.needs_reasoning_tool_status = Mock(return_value=True)
        renderer._create_detector = Mock(return_value=None)
        parser = Mock()
        renderer._create_reasoning_parser = Mock(return_value=parser)
        renderer._create_status_list = (
            ReasoningToolBaseRenderer._create_status_list.__get__(renderer)
        )
        request = Mock(logprobs=True)

        statuses = await renderer._create_status_list(1, request)

        self.assertEqual(len(statuses), 1)
        self.assertIsInstance(statuses[0], ReasoningToolStreamStatus)
        self.assertIs(statuses[0].reasoning_parser, parser)


def _weather_tool() -> GPTToolDefinition:
    return GPTToolDefinition(
        type="function",
        function=GPTFunctionDefinition(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {}},
        ),
    )


class RenderChatRecordsThinkAnchorTest(TestCase):
    """渲染即记录锚点：门控只认渲染时写下的标记，非 endpoint 链路不能漏。"""

    THINK_START_TAG = "<think>\n"

    def _make_renderer(self, prompt):
        renderer = Mock(spec=ReasoningToolBaseRenderer)
        renderer.think_start_tag = self.THINK_START_TAG
        for name in (
            "_prompt_ends_with_think_anchor",
            "_record_prompt_think_anchor",
            "_effective_tools",
            "needs_reasoning_tool_status",
        ):
            setattr(renderer, name, getattr(CustomChatRenderer, name).__get__(renderer))
        renderer._build_prompt = Mock(return_value=prompt)
        renderer.tokenizer = Mock()
        renderer.tokenizer.encode = Mock(return_value=[1, 2, 3])
        renderer.render_chat = ReasoningToolBaseRenderer.render_chat.__get__(renderer)
        return renderer

    def _make_request(self):
        return ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hi")]
        )

    def test_render_chat_records_anchored_prompt(self):
        request = self._make_request()
        self._make_renderer(f"user hi\n{self.THINK_START_TAG}").render_chat(request)

        self.assertIs(request.prompt_has_think_anchor(), True)

    def test_render_chat_records_unanchored_prompt(self):
        request = self._make_request()
        self._make_renderer("user hi\nassistant").render_chat(request)

        self.assertIs(request.prompt_has_think_anchor(), False)

    def test_render_chat_does_not_overwrite_recorded_anchor(self):
        # endpoint 在 prepopulation 之后记录的值是权威，renderer 不得覆盖。
        request = self._make_request()
        request.set_prompt_has_think_anchor(False)
        self._make_renderer(f"user hi\n{self.THINK_START_TAG}").render_chat(request)

        self.assertIs(request.prompt_has_think_anchor(), False)

    def test_recorded_anchor_opens_the_gate_without_a_second_render(self):
        renderer = self._make_renderer(f"user hi\n{self.THINK_START_TAG}")
        renderer.in_think_mode = Mock(return_value=False)
        request = self._make_request()

        self.assertFalse(renderer.needs_reasoning_tool_status(request))
        renderer.render_chat(request)
        self.assertTrue(renderer.needs_reasoning_tool_status(request))
        renderer._build_prompt.assert_called_once()


class ToolChoiceNoneTest(TestCase):
    """tool_choice=none 表示模型不得调用任何工具：提示词、检测器与门控都不应再
    看到被禁用的工具列表。"""

    def _make_request(self, tool_choice):
        return ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hi")],
            tools=[_weather_tool()],
            tool_choice=tool_choice,
        )

    def _make_prompt_renderer(self):
        renderer = Mock(spec=ReasoningToolBaseRenderer)
        renderer.chat_template = "{% if tools %}HAS_TOOLS{% else %}NO_TOOLS{% endif %}"
        for name in (
            "_effective_tools",
            "_normalize_tools_context",
            "_preprocess_messages",
            "_customize_jinja_env",
            "_build_prompt",
        ):
            setattr(
                renderer,
                name,
                getattr(ReasoningToolBaseRenderer, name).__get__(renderer),
            )
        return renderer

    def test_effective_tools_hide_disabled_tools(self):
        renderer = Mock(spec=CustomChatRenderer)
        renderer._effective_tools = CustomChatRenderer._effective_tools.__get__(
            renderer
        )

        self.assertIsNone(renderer._effective_tools(self._make_request("none")))
        for tool_choice in (None, "auto", "required"):
            with self.subTest(tool_choice=tool_choice):
                tools = renderer._effective_tools(self._make_request(tool_choice))
                self.assertEqual(len(tools), 1)

    def test_prompt_context_hides_disabled_tools(self):
        renderer = self._make_prompt_renderer()

        self.assertEqual(renderer._build_prompt(self._make_request("none")), "NO_TOOLS")
        self.assertEqual(
            renderer._build_prompt(self._make_request("auto")), "HAS_TOOLS"
        )

    def test_glm_detector_is_absent_when_tools_are_disabled(self):
        renderer = Mock(spec=ChatGlm45Renderer)
        renderer._effective_tools = CustomChatRenderer._effective_tools.__get__(
            renderer
        )
        renderer._create_detector = ChatGlm45Renderer._create_detector.__get__(renderer)

        self.assertIsNotNone(renderer._create_detector(self._make_request("auto")))
        self.assertIsNone(renderer._create_detector(self._make_request("none")))

    def test_gate_ignores_disabled_tools(self):
        renderer = Mock(spec=CustomChatRenderer)
        renderer.in_think_mode = Mock(return_value=False)
        for name in ("_effective_tools", "needs_reasoning_tool_status"):
            setattr(renderer, name, getattr(CustomChatRenderer, name).__get__(renderer))

        self.assertTrue(
            renderer.needs_reasoning_tool_status(self._make_request("auto"))
        )
        self.assertFalse(
            renderer.needs_reasoning_tool_status(self._make_request("none"))
        )

    def test_deepseek_v31_hides_tools_and_disables_detector(self):
        from rtp_llm.openai.renderers.deepseekv31_renderer import DeepseekV31Renderer

        renderer = DeepseekV31Renderer.__new__(DeepseekV31Renderer)
        renderer.tokenizer = Mock()
        renderer.tokenizer.bos_token = "<bos>"
        renderer._setup_chat_template()
        request = self._make_request("none")
        request.chat_template_kwargs = {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "injected_tool"},
                }
            ]
        }

        prompt = renderer._build_prompt(request)

        self.assertNotIn("get_weather", prompt)
        self.assertNotIn("injected_tool", prompt)
        self.assertIsNone(renderer._create_detector(request))

    def test_qwen_vl_template_kwargs_cannot_reintroduce_disabled_tools(self):
        from rtp_llm.openai.renderers.qwen_vl_renderer import Qwen2VLRenderer

        renderer = Qwen2VLRenderer.__new__(Qwen2VLRenderer)
        renderer.tokenizer = Mock()
        renderer.tokenizer.apply_chat_template = Mock(return_value="prompt")
        request = self._make_request("none")
        request.chat_template_kwargs = {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "injected_tool"},
                }
            ]
        }

        renderer._render_messages(request, add_vision_id=False)

        kwargs = renderer.tokenizer.apply_chat_template.call_args.kwargs
        self.assertEqual(kwargs["tools"], [])


class GlmTojsonFilterTest(TestCase):
    """GLM4.5 renderer 不再覆盖 _customize_jinja_env：基类提供的 tojson 必须覆盖
    被删除的实现——字符串原样返回（不加引号）、ensure_ascii 语义一致，并且把
    json.dumps 的 kwargs 透传出去（删掉的那份会忽略 extra kwargs）。"""

    def test_renderer_relies_on_the_base_filter(self):
        self.assertIs(
            ChatGlm45Renderer._customize_jinja_env,
            ReasoningToolBaseRenderer._customize_jinja_env,
        )

    def _render_value(self, value, template="{{ value | tojson }}"):
        renderer = Mock(spec=ReasoningToolBaseRenderer)
        renderer.chat_template = template
        for name in (
            "_effective_tools",
            "_normalize_tools_context",
            "_preprocess_messages",
            "_customize_jinja_env",
            "_build_prompt",
        ):
            setattr(
                renderer,
                name,
                getattr(ReasoningToolBaseRenderer, name).__get__(renderer),
            )
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hi")],
            chat_template_kwargs={"value": value},
        )
        return renderer._build_prompt(request)

    def test_string_value_is_rendered_unquoted(self):
        self.assertEqual(self._render_value("plain"), "plain")

    def test_non_string_value_is_serialized_without_sorted_keys(self):
        self.assertEqual(self._render_value({"b": 1, "a": 2}), '{"b": 1, "a": 2}')

    def test_ensure_ascii_arg_matches_the_deleted_override(self):
        # GLM4.5 模板只用 tojson(ensure_ascii=False)（testdata/glm45/tokenizer/
        # chat_template.jinja 第 11、77 行）；被删的覆盖同样走
        # kwargs.get("ensure_ascii", False)，两者一致：非 ASCII 不转义、字符串
        # 不加引号。
        template = "{{ value | tojson(ensure_ascii=False) }}"

        self.assertEqual(
            self._render_value({"名字": "张三"}, template), '{"名字": "张三"}'
        )

    def test_json_dumps_kwargs_are_passed_through(self):
        # 被删的覆盖会忽略 extra kwargs，基类透传（Kimi-K2 模板依赖 separators）。
        # GLM4.5 模板未使用这些 kwargs，故无行为变化；这里锁定基类的透传语义。
        template = "{{ value | tojson(sort_keys=True, separators=(',', ':')) }}"

        self.assertEqual(
            self._render_value({"b": 1, "a": 2}, template), '{"a":2,"b":1}'
        )


if __name__ == "__main__":
    main()
