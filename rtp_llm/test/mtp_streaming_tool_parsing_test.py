"""
MTP-Safe Streaming Tool Call Parsing Tests

Tests for tool call parsing under MTP (Speculative Decoding) conditions where
multiple tokens may arrive in a single chunk, including scenarios where:
1. Complete tool call blocks arrive in single chunk
2. Think-end tag and tool-start tag arrive in same chunk
3. Multiple complete tool calls arrive in single chunk
"""

import unittest

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv31_detector import (
    DeepSeekV31Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv32_detector import (
    DeepSeekV32Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.glm4_moe_detector import (
    Glm4MoeDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.kimik2_detector import (
    KimiK2Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen25_detector import (
    Qwen25Detector,
)


def create_tools():
    """Create test tool definitions."""
    return [
        Tool(
            type="function",
            function=Function(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                    },
                    "required": ["location"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="get_time",
                description="Get current time",
                parameters={"type": "object", "properties": {}},
            ),
        ),
    ]


def create_glm4_tools():
    """Create GLM-4 test tool definitions."""
    return [
        Tool(
            type="function",
            function=Function(
                name="ask_user_question",
                description="Ask the user questions",
                parameters={
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "description": "Questions to ask",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "header": {"type": "string"},
                                    "multiSelect": {"type": "boolean"},
                                    "options": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "label": {"type": "string"},
                                                "description": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "required": ["questions"],
                },
            ),
        ),
    ]


class TestQwen25DetectorMTP(unittest.TestCase):
    """Test Qwen25Detector MTP compatibility."""

    def setUp(self):
        self.detector = Qwen25Detector()
        self.tools = create_tools()

    def test_mtp_complete_tool_call_single_chunk(self):
        """
        MTP scenario: Complete tool call block arrives in single chunk.
        This simulates MTP returning the entire tool call at once instead of
        token-by-token.
        """
        # Complete tool call in one chunk
        chunk = '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "杭州"}}\n</tool_call>'
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result.calls[0].name}'. Calls: {result.calls}",
        )
        self.assertIn(
            '"location"',
            result.calls[0].parameters,
            f"Expected '\"location\"' in parameters. Calls: {result.calls}",
        )
        self.assertIn(
            "杭州",
            result.calls[0].parameters,
            f"Expected '杭州' in parameters. Calls: {result.calls}",
        )

    def test_mtp_think_end_and_tool_start_same_chunk(self):
        """
        MTP scenario: Think-end tag and tool-start tag arrive in same chunk.
        This is the most common MTP failure case.
        """
        self.detector = Qwen25Detector()

        # First chunk: reasoning content
        chunk1 = "I need to check the weather"
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertEqual(
            result1.normal_text,
            "I need to check the weather",
            f"Expected normal_text 'I need to check the weather', got '{result1.normal_text}'. Calls: {result1.calls}",
        )
        self.assertEqual(
            len(result1.calls),
            0,
            f"Expected 0 calls, got {len(result1.calls)}. Calls: {result1.calls}",
        )

        # MTP chunk: newlines followed by complete tool call
        # Simulates </think>\n\n<tool_call>... in one chunk
        chunk2 = '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "杭州"}}\n</tool_call>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result2.calls[0].name}'. Calls: {result2.calls}",
        )

    def test_mtp_multiple_tool_calls_single_chunk(self):
        """
        MTP scenario: Multiple complete tool calls arrive in single chunk.
        """
        # Two complete tool calls in one chunk
        chunk = (
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "杭州"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京"}}\n</tool_call>'
        )
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            2,
            f"Expected 2 calls, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "get_current_weather",
            f"Expected calls[0].name 'get_current_weather', got '{result.calls[0].name}'. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[1].name,
            "get_current_weather",
            f"Expected calls[1].name 'get_current_weather', got '{result.calls[1].name}'. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].tool_index,
            0,
            f"Expected calls[0].tool_index 0, got {result.calls[0].tool_index}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[1].tool_index,
            1,
            f"Expected calls[1].tool_index 1, got {result.calls[1].tool_index}. Calls: {result.calls}",
        )

    def test_mtp_partial_then_complete(self):
        """
        MTP scenario: Partial tool call followed by completion in next chunk.
        """
        # First chunk: start of tool call
        chunk1 = '<tool_call>\n{"name": "get_current_weather"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        # Should not have complete call yet
        print(f"result1.calls: {result1.calls}")

        # Second chunk: completion of tool call (MTP style - multiple tokens)
        chunk2 = ', "arguments": {"location": "杭州"}}\n</tool_call>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # The complete call should be returned
        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result2.calls[0].name}'. Calls: {result2.calls}",
        )

    def test_incremental_still_works(self):
        """
        Verify that traditional single-token incremental streaming still works.
        """
        detector = Qwen25Detector()

        # Simulate token-by-token streaming
        chunks = [
            "<tool_call>",
            "\n",
            "{",
            '"name": "get_current_weather"',
            ', "arguments": {"location": "杭州"}',
            "}",
            "\n</tool_call>",
        ]

        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        # Should have parsed the tool call
        print(f"all_calls: {all_calls}")
        self.assertGreaterEqual(
            len(all_calls),
            1,
            f"Should have at least 1 call, got {len(all_calls)}. Calls: {all_calls}",
        )
        # Find the call with name set (first chunk sends name)
        named_calls = [c for c in all_calls if c.name]
        self.assertTrue(
            len(named_calls) > 0,
            f"Should have a call with name. All calls: {all_calls}",
        )
        self.assertEqual(
            named_calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{named_calls[0].name}'. Named calls: {named_calls}",
        )


class TestKimiK2DetectorMTP(unittest.TestCase):
    """Test KimiK2Detector MTP compatibility."""

    def setUp(self):
        self.detector = KimiK2Detector()
        self.tools = create_tools()

    def test_mtp_complete_tool_call_single_chunk(self):
        """
        MTP scenario: Complete KimiK2 tool call block arrives in single chunk.
        """
        chunk = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location": "杭州"}<|tool_call_end|><|tool_calls_section_end|>'
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result.calls[0].name}'. Calls: {result.calls}",
        )
        self.assertIn(
            "杭州",
            result.calls[0].parameters,
            f"Expected '杭州' in parameters. Calls: {result.calls}",
        )

    def test_mtp_multiple_tool_calls_single_chunk(self):
        """
        MTP scenario: Multiple complete KimiK2 tool calls in single chunk.
        """
        chunk = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location": "杭州"}<|tool_call_end|>'
            '<|tool_call_begin|>functions.get_current_weather:1 <|tool_call_argument_begin|>{"location": "北京"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            2,
            f"Expected 2 calls, got {len(result.calls)}. Calls: {result.calls}",
        )

    def test_mtp_partial_then_complete(self):
        """
        MTP scenario: Partial tool call followed by completion.
        """
        # First chunk: start of tool call
        chunk1 = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        print(f"result1.calls: {result1.calls}")

        # Second chunk: completion (MTP style)
        chunk2 = ': "杭州"}<|tool_call_end|><|tool_calls_section_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )

    def test_mtp_thinking_tag_closing_gt_not_swallowed(self):
        """
        Stream scenario: closing '>' of '</thinking>' and tool call split across chunks.

        When stream=true, ">" and tool block may arrive in same chunk. The prefix
        before bot_token must be returned as normal_text so content is not missing.
        """
        # Chunk 1: thinking content + partial closing tag (no ">")
        chunk1 = "<thinking>\n用户想买行李箱，我需要帮他搜索。\n</thinking"
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertEqual(result1.normal_text, chunk1, "Chunk1 should be normal_text")
        self.assertEqual(len(result1.calls), 0)

        # Chunk 2: ">" + newlines + complete tool call (MTP: all in one chunk)
        chunk2 = (
            ">\n\n<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location": "杭州"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result2.calls[0].name}'",
        )
        self.assertEqual(
            result2.normal_text,
            ">\n\n",
            f"Expected normal_text '>\\n\\n', got {repr(result2.normal_text)}. "
            "The '>' in '</thinking>' was swallowed when tool block parsed.",
        )


class TestDeepSeekV31DetectorMTP(unittest.TestCase):
    """Test DeepSeekV31Detector MTP compatibility."""

    def setUp(self):
        self.detector = DeepSeekV31Detector()
        self.tools = create_tools()

    def test_mtp_complete_tool_call_single_chunk(self):
        """
        MTP scenario: Complete DeepSeek tool call block arrives in single chunk.
        """
        chunk = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "杭州"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        print(f"result.calls: {result.calls}")
        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result.calls[0].name}'. Calls: {result.calls}",
        )
        self.assertIn(
            "杭州",
            result.calls[0].parameters,
            f"Expected '杭州' in parameters. Calls: {result.calls}",
        )

    def test_mtp_multiple_tool_calls_single_chunk(self):
        """
        MTP scenario: Multiple complete DeepSeek tool calls in single chunk.
        """
        chunk = (
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "杭州"}<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "北京"}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            2,
            f"Expected 2 calls, got {len(result.calls)}. Calls: {result.calls}",
        )

    def test_mtp_partial_then_complete(self):
        """
        MTP scenario: Partial tool call followed by completion.
        """
        # First chunk: start of tool call
        chunk1 = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        print(f"result1.calls: {result1.calls}")

        # Second chunk: completion (MTP style)
        chunk2 = ': "杭州"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )

    def test_mtp_thinking_tag_closing_gt_not_swallowed(self):
        """
        Stream scenario: closing '>' of '</thinking>' and tool call split across chunks.

        When stream=true, tokenizer may output:
        - Chunk 1: "</thinking" (content sent)
        - Chunk 2: ">\\n\\n" + bot_token + tool_call (">" and tool in same chunk)

        Bug: Chunk 2's ">" was swallowed because parse_streaming_increment returned
        normal_text="" when parsing complete tool block, dropping prefix before bot_token.
        """
        # Chunk 1: thinking content + partial closing tag (no ">")
        chunk1 = "<thinking>\n用户想买行李箱，我需要帮他搜索。\n</thinking"
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertEqual(result1.normal_text, chunk1, "Chunk1 should be normal_text")
        self.assertEqual(len(result1.calls), 0)

        # Chunk 2: ">" + newlines + complete tool call (MTP: all in one chunk)
        chunk2 = '>\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "杭州"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result2.calls[0].name}'",
        )
        # The fix: ">" must be in normal_text so streamed content is not missing it.
        # normal_text = prefix before bot_token, i.e. ">\n\n" (bot_token excluded).
        self.assertEqual(
            result2.normal_text,
            ">\n\n",
            f"Expected normal_text '>\\n\\n', got {repr(result2.normal_text)}. "
            "The '>' in '</thinking>' was swallowed when tool block parsed.",
        )


class TestDeepSeekV32DetectorMTP(unittest.TestCase):
    """Test DeepSeekV32Detector MTP compatibility."""

    def setUp(self):
        self.detector = DeepSeekV32Detector()
        self.tools = create_tools()

    def test_mtp_thinking_tag_closing_gt_not_swallowed(self):
        """
        Stream scenario: closing '>' of '</thinking>' and tool call split across chunks.

        When stream=true, ">" and invoke block may arrive in same chunk. The prefix
        before bot_token must be returned as normal_text so content is not missing.
        """
        # Chunk 1: thinking content + partial closing tag (no ">")
        chunk1 = "<thinking>\n用户想买行李箱，我需要帮他搜索。\n</thinking"
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertEqual(result1.normal_text, chunk1, "Chunk1 should be normal_text")
        self.assertEqual(len(result1.calls), 0)

        # Chunk 2: ">" + newlines + complete invoke block (MTP: all in one chunk)
        chunk2 = (
            ">\n\n<｜DSML｜function_calls>\n"
            '<｜DSML｜invoke name="get_current_weather">{"location": "杭州"}</｜DSML｜invoke>\n'
            "</｜DSML｜function_calls>"
        )
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            2,
            f"Expected 2 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result2.calls[0].name}'",
        )
        self.assertEqual(
            result2.normal_text,
            ">\n\n",
            f"Expected normal_text '>\\n\\n', got {repr(result2.normal_text)}. "
            "The '>' in '</thinking>' was swallowed when invoke block parsed.",
        )


class TestGlm4MoeDetectorMTP(unittest.TestCase):
    """Test Glm4MoeDetector MTP compatibility with GLM-4.7 format."""

    def setUp(self):
        self.detector = Glm4MoeDetector()
        self.tools = create_glm4_tools()

    def test_glm47_with_reasoning_and_tool_call(self):
        """
        Test GLM-4.7 format with <think> tags and tool calls.
        This reproduces the issue reported in commit 91fc0bc536fd1176e711349cdd81a8ddd1b5d1ba.

        Raw output format:
        <think>reasoning content</think>normal text<tool_call>...</tool_call><|observation|>

        Expected behavior:
        - finish_reason: "tool_calls" (not "stop")
        - reasoning_content: "reasoning content"
        - content: "normal text"
        - tool_calls: parsed tool call
        """
        # Note: The raw output provided by the user shows the complete response including <think> tags
        # However, the Glm4MoeDetector only parses <tool_call> tags, not <think> tags.
        # The <think> tags are handled by the ReasoningParser in the renderer layer.
        # For this unit test, we test the detector's ability to parse tool calls
        # from text that may have normal text before the tool call.

        raw_output = (
            "帮助用户做出明确的选择。我来调用 ask_user_question 工具，构造一些示例参数："
            "<tool_call>ask_user_question<arg_key>questions</arg_key>"
            '<arg_value>[{"question": "您希望使用哪种编程语言来开发这个功能？", '
            '"header": "编程语言", "multiSelect": false, '
            '"options": [{"label": "TypeScript", "description": "类型安全的 JavaScript 超集，适合大型项目"}, '
            '{"label": "Python", "description": "简洁易读，适合快速开发和数据处理"}, '
            '{"label": "Go", "description": "高性能并发，适合后端服务和微服务"}]}, '
            '{"question": "您希望启用哪些功能特性？", "header": "功能特性", "multiSelect": true, '
            '"options": [{"label": "实时更新", "description": "数据变更时自动同步更新界面"}, '
            '{"label": "离线缓存", "description": "支持离线访问和数据缓存"}, '
            '{"label": "主题切换", "description": "支持明暗主题切换"}]}]</arg_value>'
            "</tool_call>"
        )

        result = self.detector.detect_and_parse(raw_output, self.tools)

        # Should have normal text before the tool call
        self.assertIn(
            "ask_user_question",
            result.normal_text,
            f"Expected normal text to contain intro text, got '{result.normal_text}'",
        )

        # Should have 1 tool call
        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 tool call, got {len(result.calls)}. Calls: {result.calls}",
        )

        # Verify tool call name
        self.assertEqual(
            result.calls[0].name,
            "ask_user_question",
            f"Expected tool name 'ask_user_question', got '{result.calls[0].name}'",
        )

        # Verify parameters contain questions
        self.assertIn(
            '"questions"',
            result.calls[0].parameters,
            f"Expected 'questions' in parameters. Got: {result.calls[0].parameters}",
        )

    def test_glm47_mtp_streaming_with_normal_text(self):
        """
        Test GLM-4.7 streaming scenario where tool call arrives with normal text in one chunk.
        This simulates the MTP scenario where multiple tokens arrive together.
        """
        # Simulate streaming: first chunk has normal text, second chunk has tool call
        chunk1 = "我来调用工具："
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        self.assertEqual(
            result1.normal_text,
            "我来调用工具：",
            f"Expected normal text, got '{result1.normal_text}'",
        )
        self.assertEqual(
            len(result1.calls),
            0,
            f"Expected 0 calls in first chunk, got {len(result1.calls)}",
        )

        # Second chunk: complete tool call
        chunk2 = (
            "<tool_call>ask_user_question<arg_key>questions</arg_key>"
            '<arg_value>[{"question": "test", "header": "test", "multiSelect": false, '
            '"options": [{"label": "A", "description": "Option A"}]}]</arg_value>'
            "</tool_call>"
        )
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call in second chunk, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "ask_user_question",
            f"Expected 'ask_user_question', got '{result2.calls[0].name}'",
        )

    def test_glm47_mtp_complete_tool_call_single_chunk(self):
        """
        Test GLM-4.7 MTP scenario: complete tool call with normal text arrives in single chunk.
        """
        # Complete response in one chunk (MTP style)
        chunk = (
            "让我帮您创建问题："
            "<tool_call>ask_user_question<arg_key>questions</arg_key>"
            '<arg_value>[{"question": "选择语言？", "header": "语言", "multiSelect": false, '
            '"options": [{"label": "Python", "description": "简单"}, '
            '{"label": "Go", "description": "快速"}]}]</arg_value>'
            "</tool_call>"
        )

        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "ask_user_question",
            f"Expected 'ask_user_question', got '{result.calls[0].name}'",
        )
        self.assertIn(
            "questions",
            result.calls[0].parameters,
            f"Expected 'questions' in parameters: {result.calls[0].parameters}",
        )

    def test_glm47_stop_word_handling(self):
        """
        Test that <|observation|> stop word is properly handled (should be truncated).
        Note: The detector itself doesn't handle stop words - that's done by the renderer.
        This test verifies the detector works correctly with text that may have had
        stop words removed.
        """
        # Text with stop word already removed (as it would be by renderer)
        text_without_stop = (
            "<tool_call>ask_user_question<arg_key>questions</arg_key>"
            '<arg_value>[{"question": "test?", "header": "T", "multiSelect": false, '
            '"options": [{"label": "A", "description": "Opt A"}]}]</arg_value>'
            "</tool_call>"
        )

        result = self.detector.detect_and_parse(text_without_stop, self.tools)

        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}",
        )


if __name__ == "__main__":
    unittest.main()
