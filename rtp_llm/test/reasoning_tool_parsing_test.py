from typing import Optional
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import torch

from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionCall,
    RoleEnum,
    ToolCall,
)
from rtp_llm.openai.renderers.custom_renderer import RendererParams
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
    ReasoningToolStreamStatus,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import (
    Qwen3Detector,
    ReasoningParser,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput


class ProcessReasoningAndToolCallsTest(IsolatedAsyncioTestCase):
    """Test _process_reasoning_and_tool_calls method."""

    def setUp(self):
        # Create a minimal mock renderer
        self.renderer = Mock(spec=ReasoningToolBaseRenderer)
        self.renderer._process_reasoning_and_tool_calls = (
            ReasoningToolBaseRenderer._process_reasoning_and_tool_calls.__get__(
                self.renderer
            )
        )
        self.renderer._extract_reasoning_content = Mock()
        self.renderer._extract_tool_calls_content = AsyncMock()
        self.renderer._generate_log_probs = AsyncMock(return_value=None)

        # Create mock request and status
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")], tools=[]
        )
        detector = Mock(spec=BaseFormatDetector)
        reasoning_parser = Mock(spec=ReasoningParser)
        self.status = ReasoningToolStreamStatus(request, detector, reasoning_parser)
        self.status.generating_tool_call = False

        # Create mock output
        aux_info = AuxInfo()
        aux_info.input_len = 10
        aux_info.output_len = 20
        aux_info.reuse_len = 0
        self.output = Mock(spec=GenerateOutput)
        self.output.aux_info = aux_info

    async def test_only_reasoning_streaming(self):
        # Only reasoning content, no tool calls, streaming mode
        self.status.delta_output_string = "<think>思考内容</think>正常文本"

        # Mock extraction methods
        self.renderer._extract_reasoning_content.return_value = ("思考内容", "正常文本")
        self.renderer._extract_tool_calls_content.return_value = (None, "正常文本")

        delta = await self.renderer._process_reasoning_and_tool_calls(
            self.status, self.output, is_streaming=True
        )

        # Should return delta with reasoning_content and content
        self.assertIsNotNone(delta)
        self.assertEqual(delta.output_str.reasoning_content, "思考内容")
        self.assertEqual(delta.output_str.content, "正常文本")
        self.assertIsNone(delta.output_str.tool_calls)
        # Status should be updated
        self.assertEqual(self.status.delta_output_string, "正常文本")

    async def test_only_reasoning_non_streaming(self):
        # Only reasoning content, no tool calls, non-streaming mode
        self.status.delta_output_string = "<think>思考内容</think>正常文本"

        self.renderer._extract_reasoning_content.return_value = ("思考内容", "正常文本")
        self.renderer._extract_tool_calls_content.return_value = (None, "正常文本")

        delta = await self.renderer._process_reasoning_and_tool_calls(
            self.status, self.output, is_streaming=False
        )

        # Should return delta with reasoning_content but NO content (non-streaming)
        self.assertIsNotNone(delta)
        self.assertEqual(delta.output_str.reasoning_content, "思考内容")
        self.assertIsNone(delta.output_str.content)  # Not included in non-streaming
        self.assertIsNone(delta.output_str.tool_calls)
        self.assertEqual(self.status.delta_output_string, "正常文本")

    async def test_reasoning_and_tool_calls_streaming(self):
        # Both reasoning and tool calls, streaming mode
        self.status.delta_output_string = (
            "<think>思考</think><tool_call>...</tool_call>文本"
        )

        self.renderer._extract_reasoning_content.return_value = (
            "思考",
            "<tool_call>...</tool_call>文本",
        )
        tool_call = ToolCall(
            index=0,
            type="function",
            function=FunctionCall(name="test_func", arguments="{}"),
        )
        self.renderer._extract_tool_calls_content.return_value = (
            [tool_call],
            "文本",
        )

        delta = await self.renderer._process_reasoning_and_tool_calls(
            self.status, self.output, is_streaming=True
        )

        # Should return delta with all three fields
        self.assertIsNotNone(delta)
        self.assertEqual(delta.output_str.reasoning_content, "思考")
        self.assertEqual(len(delta.output_str.tool_calls), 1)
        self.assertEqual(delta.output_str.content, "文本")
        # Status should be updated
        self.assertTrue(self.status.generating_tool_call)
        self.assertEqual(self.status.delta_output_string, "文本")

    async def test_reasoning_and_tool_calls_non_streaming(self):
        # Both reasoning and tool calls, non-streaming mode
        self.status.delta_output_string = (
            "<think>思考</think><tool_call>...</tool_call>文本"
        )

        self.renderer._extract_reasoning_content.return_value = (
            "思考",
            "<tool_call>...</tool_call>文本",
        )
        tool_call = ToolCall(
            index=0,
            type="function",
            function=FunctionCall(name="test_func", arguments="{}"),
        )
        self.renderer._extract_tool_calls_content.return_value = (
            [tool_call],
            "文本",
        )

        delta = await self.renderer._process_reasoning_and_tool_calls(
            self.status, self.output, is_streaming=False
        )

        # Should return delta with reasoning and tool calls, but no content
        self.assertIsNotNone(delta)
        self.assertEqual(delta.output_str.reasoning_content, "思考")
        self.assertEqual(len(delta.output_str.tool_calls), 1)
        self.assertIsNone(delta.output_str.content)  # Not included in non-streaming
        self.assertTrue(self.status.generating_tool_call)

    async def test_only_tool_calls(self):
        # Only tool calls, no reasoning
        self.status.delta_output_string = "<tool_call>...</tool_call>文本"

        self.renderer._extract_reasoning_content.return_value = (
            "",
            "<tool_call>...</tool_call>文本",
        )
        tool_call = ToolCall(
            index=0,
            type="function",
            function=FunctionCall(name="test_func", arguments="{}"),
        )
        self.renderer._extract_tool_calls_content.return_value = (
            [tool_call],
            "文本",
        )

        delta = await self.renderer._process_reasoning_and_tool_calls(
            self.status, self.output, is_streaming=True
        )

        # Should return delta with tool calls and content, no reasoning
        self.assertIsNotNone(delta)
        self.assertFalse(delta.output_str.reasoning_content)
        self.assertEqual(len(delta.output_str.tool_calls), 1)
        self.assertEqual(delta.output_str.content, "文本")

    async def test_no_reasoning_or_tools(self):
        # No reasoning or tool calls found - should return None
        self.status.delta_output_string = "只是普通文本"

        self.renderer._extract_reasoning_content.return_value = ("", "只是普通文本")
        self.renderer._extract_tool_calls_content.return_value = (None, "只是普通文本")

        delta = await self.renderer._process_reasoning_and_tool_calls(
            self.status, self.output, is_streaming=True
        )

        # Should return None to let default handler process normal text
        self.assertIsNone(delta)
        # Status should still be updated with remaining text
        self.assertEqual(self.status.delta_output_string, "只是普通文本")

    async def test_empty_input(self):
        # Empty delta_output_string
        self.status.delta_output_string = ""

        self.renderer._extract_reasoning_content.return_value = ("", "")
        self.renderer._extract_tool_calls_content.return_value = (None, "")

        delta = await self.renderer._process_reasoning_and_tool_calls(
            self.status, self.output, is_streaming=True
        )

        # Should return None
        self.assertIsNone(delta)

    async def test_no_remaining_text_after_parsing(self):
        # All text consumed by reasoning and tool calls
        self.status.delta_output_string = (
            "<think>思考</think><tool_call>...</tool_call>"
        )

        self.renderer._extract_reasoning_content.return_value = (
            "思考",
            "<tool_call>...</tool_call>",
        )
        tool_call = ToolCall(
            index=0,
            type="function",
            function=FunctionCall(name="test_func", arguments="{}"),
        )
        self.renderer._extract_tool_calls_content.return_value = ([tool_call], "")

        delta = await self.renderer._process_reasoning_and_tool_calls(
            self.status, self.output, is_streaming=True
        )

        # Should return delta with reasoning and tools, no content
        self.assertIsNotNone(delta)
        self.assertEqual(delta.output_str.reasoning_content, "思考")
        self.assertEqual(len(delta.output_str.tool_calls), 1)
        self.assertIsNone(delta.output_str.content)  # No remaining text
        self.assertEqual(self.status.delta_output_string, "")

    async def test_unicode_content(self):
        # Unicode characters in reasoning and remaining text
        self.status.delta_output_string = "<think>用户询问天气</think>我来帮您查询"

        self.renderer._extract_reasoning_content.return_value = (
            "用户询问天气",
            "我来帮您查询",
        )
        self.renderer._extract_tool_calls_content.return_value = (None, "我来帮您查询")

        delta = await self.renderer._process_reasoning_and_tool_calls(
            self.status, self.output, is_streaming=True
        )

        # Should handle unicode correctly
        self.assertIsNotNone(delta)
        self.assertEqual(delta.output_str.reasoning_content, "用户询问天气")
        self.assertEqual(delta.output_str.content, "我来帮您查询")

    async def test_no_parser_configured(self):
        # No reasoning parser or detector configured - should return None
        self.status.reasoning_parser = None
        self.status.detector = None
        self.status.delta_output_string = "文本"

        self.renderer._extract_reasoning_content.return_value = ("", "文本")
        self.renderer._extract_tool_calls_content.return_value = (None, "文本")

        delta = await self.renderer._process_reasoning_and_tool_calls(
            self.status, self.output, is_streaming=True
        )

        # Should return None
        self.assertIsNone(delta)

    async def test_mtp_edge_case_streaming(self):
        # MTP edge case: reasoning ends mid-chunk with normal content
        # In streaming mode, normal content should be included in delta
        self.status.delta_output_string = "</think>我来帮您"

        self.renderer._extract_reasoning_content.return_value = ("", "</think>我来帮您")
        self.renderer._extract_tool_calls_content.return_value = (None, "我来帮您")

        delta = await self.renderer._process_reasoning_and_tool_calls(
            self.status, self.output, is_streaming=True
        )

        # No reasoning/tools found, should return None
        self.assertIsNone(delta)


class StreamingReasoningParserXmlTagTest(IsolatedAsyncioTestCase):
    """Test for XML tag boundary issue in streaming reasoning parser.

    Regression test for: When reasoning content contains `<` followed by other text
    (like `<wait.user`), the `<` should not be lost even when it's buffered as a
    potential prefix of `</think>`.
    """

    async def test_streaming_xml_tag_not_lost_when_buffered(self):
        """Test that opening XML tag `<` is not lost when buffered as think token prefix.

        Bug scenario from curl.log:
        1. Parser receives `"<"` alone - buffers it (prefix of `</think>`)
        2. Parser receives `"wait.user"` - should output `"<wait.user"`
        3. If dual buffering occurs, the `"<"` is lost
        """
        # Create detector in streaming mode, force_reasoning=True (model always starts in reasoning)
        detector = Qwen3Detector(stream_reasoning=True, force_reasoning=True)

        # Simulate the problematic sequence:
        # Chunk 1: Just the `<` character (gets buffered as prefix of </think>)
        result1 = detector.parse_streaming_increment("<")
        # Should buffer and return empty (since `<` is prefix of `</think>`)
        self.assertEqual(result1.reasoning_text, "")
        self.assertEqual(result1.normal_text, "")

        # Chunk 2: The rest of the content
        result2 = detector.parse_streaming_increment("wait.user tool")
        # Should output the buffered `<` plus new content
        self.assertEqual(result2.reasoning_text, "<wait.user tool")
        self.assertEqual(result2.normal_text, "")

    async def test_streaming_xml_tag_with_think_end_transition(self):
        """Test XML tag handling across reasoning-to-content transition.

        Scenario: reasoning contains `<tag>text</think>content`
        The `<` should be correctly attributed to reasoning, not lost.
        """
        detector = Qwen3Detector(stream_reasoning=True, force_reasoning=True)

        # Chunk 1: Start with text ending in `<`
        result1 = detector.parse_streaming_increment("reasoning text <")
        # The `<` might be buffered, but "reasoning text " should be output
        # or the whole thing might be buffered depending on implementation

        # Chunk 2: Complete with think end and normal content
        result2 = detector.parse_streaming_increment("more</think>normal content")
        # Combined result should have:
        # reasoning: everything before </think>
        # normal: "normal content"
        combined_reasoning = result1.reasoning_text + result2.reasoning_text
        self.assertIn("<more", combined_reasoning)  # The `<` should NOT be lost
        self.assertEqual(result2.normal_text, "normal content")

    async def test_streaming_multiple_xml_tags_in_reasoning(self):
        """Test multiple XML-like tags within reasoning content.

        Ensures that XML tags like `<wait.user>`, `<on>` etc. within reasoning
        are preserved correctly even when they start with `<`.
        """
        detector = Qwen3Detector(stream_reasoning=True, force_reasoning=True)

        # Full content with multiple XML-like tags
        chunks = ["I should use ", "<", "wait.user", "> tool to wait"]

        all_reasoning = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk)
            all_reasoning += result.reasoning_text

        # End reasoning
        result_end = detector.parse_streaming_increment("</think>done")
        all_reasoning += result_end.reasoning_text

        # Verify the full content is preserved
        self.assertIn("<wait.user>", all_reasoning)
        self.assertEqual(result_end.normal_text, "done")

    async def test_streaming_xml_tags_after_reasoning(self):
        """Test multiple XML-like tags within reasoning content.

        Ensures that XML tags like `<wait.user>`, `<on>` etc. within reasoning
        are preserved correctly even when they start with `<`.
        """
        detector = Qwen3Detector(stream_reasoning=True, force_reasoning=True)

        # Full content with multiple XML-like tags
        chunks = ["think", "</think>", "<", "wait.user", "> tool to wait"]

        all_reasoning = ""
        all_content = ""
        in_reasoning = True
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk)
            if in_reasoning:
                all_reasoning += result.reasoning_text
            else:
                all_content += result.normal_text
            if "</think>" in chunk:
                in_reasoning = False
                all_content += result.normal_text

        print(f"all_reasoning: {all_reasoning}, all_content: {all_content}")
        self.assertIn("<wait.user>", all_content)
        self.assertFalse(in_reasoning)

    async def test_streaming_xml_tags_after_reasoning2(self):
        """Test multiple XML-like tags within reasoning content.

        Ensures that XML tags like `<wait.user>`, `<on>` etc. within reasoning
        are preserved correctly even when they start with `<`.
        """
        detector = Qwen3Detector(stream_reasoning=True, force_reasoning=True)

        # Full content with multiple XML-like tags
        chunks = ["think", "</think><", "wait.user", "> tool to wait"]

        all_reasoning = ""
        all_content = ""
        in_reasoning = True
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk)
            if in_reasoning:
                all_reasoning += result.reasoning_text
            else:
                all_content += result.normal_text
            if "</think>" in chunk:
                in_reasoning = False
                all_content += result.normal_text

        print(f"all_reasoning: {all_reasoning}, all_content: {all_content}")
        self.assertIn("<wait.user>", all_content)
        self.assertFalse(in_reasoning)

    async def test_streaming_xml_tags_after_reasoning3(self):
        """Test multiple XML-like tags within reasoning content.

        Ensures that XML tags like `<wait.user>`, `<on>` etc. within reasoning
        are preserved correctly even when they start with `<`.
        """
        detector = Qwen3Detector(stream_reasoning=True, force_reasoning=True)

        # Full content with multiple XML-like tags
        chunks = ["think", "</think>", "<wait.user", "> tool to wait"]

        all_reasoning = ""
        all_content = ""
        in_reasoning = True
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk)
            if in_reasoning:
                all_reasoning += result.reasoning_text
            else:
                all_content += result.normal_text
            if "</think>" in chunk:
                in_reasoning = False
                all_content += result.normal_text

        print(f"all_reasoning: {all_reasoning}, all_content: {all_content}")
        self.assertIn("<wait.user>", all_content)
        self.assertFalse(in_reasoning)


class RendererIntegration3ChunkXmlTagTest(IsolatedAsyncioTestCase):
    """Integration test with 3 chunks: reasoning ends and '<' starts in same chunk.

    This reproduces the exact production bug scenario from model.log where:
    - Chunk 1: Reasoning content "用户的回复。"
    - Chunk 2: "</think>" + "<" arrive together (tokens [151351, 27])
    - Chunk 3: Rest of content "wait.user>\n<on"

    The bug: In chunk 2, parser returns normal_text='<' but _process_reasoning_and_tool_calls
    returns None. Lines 249-261 detect delta_output_string changed and call update_result(),
    marking token 27 as processed. Then chunk 3 re-decodes from prev_token_id=[151351, 27],
    gets decoded_prev='</think><', so delta is 'wait.user>\n<on' (missing the '<').
    """

    def setUp(self):
        """Set up a real ReasoningToolBaseRenderer with REAL GLM45 tokenizer."""
        import os

        # Create a concrete subclass of ReasoningToolBaseRenderer for testing
        class TestRenderer(ReasoningToolBaseRenderer):
            def _setup_chat_template(self):
                self.chat_template = "test"

            def in_think_mode(self, request: ChatCompletionRequest):
                # Always return True to enable reasoning mode
                return True

            def _create_reasoning_parser(
                self, request: ChatCompletionRequest
            ) -> Optional[ReasoningParser]:
                return ReasoningParser(
                    model_type="glm45", stream_reasoning=True, force_reasoning=True
                )

        # Load REAL GLM45 tokenizer
        tokenizer_path = os.path.join(
            os.path.dirname(__file__), "model_test/fake_test/testdata/glm45/tokenizer"
        )
        from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer

        self.tokenizer = BaseTokenizer(tokenizer_path)

        # Real token IDs from production log analysis:
        # "用户的回复。" = [107961, 104125, 1773]
        # "</think>" = [151351] (single token!)
        # "<" = [27]
        # "wait.user>\n<on" = [11484, 3324, 397, 27, 263]

        # Configure stop words (these all start with '<' which is important for the bug!)
        # Token IDs from GLM4.5 tokenizer:
        self.stop_token_ids = [
            151336,  # <|user|>
            151338,  # <|observation|>
        ]
        # Decode stop words to get their text forms
        self.stop_words_str = [
            self.tokenizer.decode([token_id]) for token_id in self.stop_token_ids
        ]

        # Create stop word slices - each stop word as a single-element list
        self.stop_word_slice_list = [
            "<|user|>",
            "<|user|",
            "<|user",
            "<|use",
            "<|us",
            "<|u",
            "<|",
            "<",
            "<|observation|>",
            "<|observation|",
            "<|observation",
            "<|observatio",
            "<|observati",
            "<|observat",
            "<|observa",
            "<|observ",
            "<|obser",
            "<|obse",
            "<|obs",
            "<|ob",
            "<|o",
            "<|",
            "<",
        ]

        # Create renderer
        renderer_params = RendererParams(
            model_type="qwen3",
            max_seq_len=2048,
            eos_token_id=0,
            stop_word_ids_list=[
                [151336],  # <|user|>
                [151338],  # <|observation|>
            ],
        )
        self.renderer = TestRenderer(
            tokenizer=self.tokenizer,
            renderer_params=renderer_params,
        )

        # Create request
        self.request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")], tools=[]
        )

    async def test_3chunk_missing_opening_bracket(self):
        """Test 3-chunk scenario: "</think>" and "<" arrive together.

        This reproduces the exact production bug:
        1. Chunk 1: "用户的回复。" (reasoning content)
        2. Chunk 2: "</think>" + "<" together (tokens [151351, 27])
        3. Chunk 3: "wait.user>\n<on" (rest of content)

        Expected: Chunk 3 output should be "<wait.user>\n<on" (including the '<' from chunk 2)
        Bug: The '<' is lost because update_result() is called in chunk 2 even though
             '<' wasn't output yet.
        """

        # Create status list
        status_list = await self.renderer._create_status_list(1, self.request)
        status = status_list[0]

        # Helper to create GenerateOutput
        def create_output(tokens):
            aux_info = AuxInfo()
            aux_info.input_len = 10
            aux_info.output_len = len(tokens)
            aux_info.reuse_len = 0
            output = GenerateOutput()
            output.output_ids = torch.tensor([tokens])  # Shape: [1, seq_len]
            output.aux_info = aux_info
            return output

        # Simulate 3-chunk streaming
        all_outputs = []

        # Chunk 1: Reasoning content "用户的回复。"
        tokens1 = [107961, 104125, 1773]
        output1 = create_output(tokens1)

        delta1 = await self.renderer._update_single_status(
            status,
            output1,
            max_new_tokens=100,
            stop_words_str=self.stop_words_str,
            stop_word_slice_list=self.stop_word_slice_list,
            is_streaming=True,
        )
        all_outputs.append(("chunk1", delta1))
        print(
            f"\nChunk 1 - Reasoning: '{delta1.output_str.reasoning_content if hasattr(delta1.output_str, 'reasoning_content') else ''}'"
        )

        # Chunk 2: "</think>" + "<" together - THE CRITICAL BUG SCENARIO!
        # Token 151351 = "</think>", Token 27 = "<"
        tokens2 = [151351, 27]  # Both arrive together
        output2 = create_output(tokens2)

        print(f"\nBefore Chunk 2:")
        print(
            f"  Reasoning parser buffer: '{status.reasoning_parser.detector._buffer}'"
        )
        print(f"  In reasoning mode: {status.reasoning_parser.detector._in_reasoning}")
        print(f"  last_output_ids: {status.last_output_ids}")
        print(f"  last_token_length: {status.last_token_length}")

        delta2 = await self.renderer._update_single_status(
            status,
            output2,
            max_new_tokens=100,
            stop_words_str=self.stop_words_str,
            stop_word_slice_list=self.stop_word_slice_list,
            is_streaming=True,
        )
        all_outputs.append(("chunk2", delta2))

        print(f"\nAfter Chunk 2:")
        print(
            f"  Reasoning parser buffer: '{status.reasoning_parser.detector._buffer}'"
        )
        print(f"  Status delta_output_string: '{status.delta_output_string}'")
        print(f"  last_output_ids: {status.last_output_ids}")
        print(f"  last_token_length: {status.last_token_length}")
        chunk2_content = ""
        if isinstance(delta2.output_str, str):
            chunk2_content = delta2.output_str
        elif hasattr(delta2.output_str, "content"):
            chunk2_content = delta2.output_str.content
        print(f"Chunk 2 - Content: '{chunk2_content}'")

        # Chunk 3: Rest of the content "wait.user>\n<on"
        tokens3 = [11484, 3324, 397, 27, 263]
        output3 = create_output(tokens3)

        print(f"\nBefore Chunk 3:")
        print(
            f"  Reasoning parser buffer: '{status.reasoning_parser.detector._buffer}'"
        )
        print(f"  In reasoning mode: {status.reasoning_parser.detector._in_reasoning}")
        print(f"  last_output_ids: {status.last_output_ids}")
        print(f"  last_token_length: {status.last_token_length}")

        delta3 = await self.renderer._update_single_status(
            status,
            output3,
            max_new_tokens=100,
            stop_words_str=self.stop_words_str,
            stop_word_slice_list=self.stop_word_slice_list,
            is_streaming=True,
        )
        all_outputs.append(("chunk3", delta3))

        print(f"\nAfter Chunk 3:")
        print(
            f"  Reasoning parser buffer: '{status.reasoning_parser.detector._buffer}'"
        )
        print(f"  Status delta_output_string: '{status.delta_output_string}'")
        chunk3_content = ""
        if isinstance(delta3.output_str, str):
            chunk3_content = delta3.output_str
        elif hasattr(delta3.output_str, "content"):
            chunk3_content = delta3.output_str.content
        print(f"Chunk 3 - Content: '{chunk3_content}'")

        # Extract the actual content from chunk 2 and chunk 3
        chunk2_content = ""
        if isinstance(delta2.output_str, str):
            chunk2_content = delta2.output_str
        elif hasattr(delta2.output_str, "content") and delta2.output_str.content:
            chunk2_content = delta2.output_str.content

        chunk3_content = ""
        if isinstance(delta3.output_str, str):
            chunk3_content = delta3.output_str
        elif hasattr(delta3.output_str, "content") and delta3.output_str.content:
            chunk3_content = delta3.output_str.content

        print(f"Chunk 3 - Content: '{chunk3_content}'")
        print(f"\nAll outputs:")
        for chunk_name, delta in all_outputs:
            if isinstance(delta.output_str, str):
                print(f"  {chunk_name}: '{delta.output_str}'")
            elif hasattr(delta.output_str, "reasoning_content"):
                print(
                    f"  {chunk_name}: reasoning='{delta.output_str.reasoning_content}', content='{delta.output_str.content}'"
                )
            else:
                print(f"  {chunk_name}: {delta.output_str}")

        # THE BUG CHECK: The '<' should be output (either in chunk 2 or chunk 3)
        # In this 3-chunk scenario with "</think><" together, the '<' is parsed
        # as normal content in chunk 2 since reasoning already ended
        combined_content = chunk2_content + chunk3_content
        self.assertEqual(
            combined_content,
            "<wait.user>\n<on",
            f"BUG: The '<' character was lost! Expected combined '<wait.user>\\n<on' but got '{combined_content}'",
        )


if __name__ == "__main__":
    main()
