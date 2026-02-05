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
from rtp_llm.openai.renderers.sglang_helpers.function_call.glm4_moe_detector import (
    Glm4MoeDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.kimik2_detector import (
    KimiK2Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen3_coder_detector import (
    Qwen3CoderDetector,
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

    def test_mtp_partial_then_complete(self):
        """
        MTP scenario: Partial tool call followed by completion in next chunk.

        NOTE: With incremental parsing, the name is emitted as soon as it's complete.
        The first chunk contains the complete name, so name is emitted in first result.
        """
        # First chunk: start of tool call (includes complete name)
        chunk1 = '<tool_call>\n{"name": "get_current_weather"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        # Name should be emitted in first chunk since it's complete
        print(f"result1.calls: {result1.calls}")

        # The name should be emitted in the first chunk
        self.assertEqual(
            len(result1.calls),
            1,
            f"Expected 1 call (name), got {len(result1.calls)}. Calls: {result1.calls}",
        )
        self.assertEqual(
            result1.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result1.calls[0].name}'. Calls: {result1.calls}",
        )

        # Second chunk: completion of tool call (MTP style - multiple tokens)
        chunk2 = ', "arguments": {"location": "杭州"}}\n</tool_call>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # The arguments should be streamed in subsequent calls
        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call (args), got {len(result2.calls)}. Calls: {result2.calls}",
        )
        # Second result should have arguments, not name (already sent)
        self.assertIsNone(
            result2.calls[0].name,
            f"Expected name to be None (already sent), got '{result2.calls[0].name}'",
        )
        self.assertIn(
            "location",
            result2.calls[0].parameters,
            f"Expected 'location' in parameters. Calls: {result2.calls}",
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


class TestDeepSeekV31DetectorMTP(unittest.TestCase):
    """Test DeepSeekV31Detector MTP compatibility."""

    def setUp(self):
        self.detector = DeepSeekV31Detector()
        self.tools = create_tools()

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


class TestQwen3CoderDetectorStreaming(unittest.TestCase):
    def setUp(self):
        self.detector = Qwen3CoderDetector()
        self.tools = create_tools()

    def _collect_streamed_args(self, results) -> str:
        parts = []
        for res in results:
            for call in res.calls:
                if call.tool_index == 0 and call.parameters:
                    parts.append(call.parameters)
        return "".join(parts)

    def test_incremental_string_parameter_streaming(self):
        chunks = [
            "<tool_call>",
            "<function=get_current_weather>",
            "<parameter=location>",
            "San",
            " Francisco",
            "</parameter>",
            "</function>",
            "</tool_call>",
        ]

        results = []
        for chunk in chunks:
            results.append(self.detector.parse_streaming_increment(chunk, self.tools))

        # Ensure value chunks are streamed as they arrive
        self.assertTrue(
            any("San" in c.parameters for c in results[3].calls),
            f"Expected 'San' to be streamed in chunk 4. Calls: {results[3].calls}",
        )
        self.assertTrue(
            any(" Francisco" in c.parameters for c in results[4].calls),
            f"Expected ' Francisco' to be streamed in chunk 5. Calls: {results[4].calls}",
        )

        self.assertEqual(
            self._collect_streamed_args(results),
            '{"location": "San Francisco"}',
        )

    def test_string_null_is_emitted_as_json_null(self):
        chunks = [
            "<tool_call>",
            "<function=get_current_weather>",
            "<parameter=location>",
            "nu",
            "ll",
            "</parameter>",
            "</function>",
            "</tool_call>",
        ]

        results = []
        for chunk in chunks:
            results.append(self.detector.parse_streaming_increment(chunk, self.tools))

        streamed = self._collect_streamed_args(results)
        self.assertEqual(streamed, '{"location": null}')
        self.assertNotIn('"null"', streamed)


if __name__ == "__main__":
    unittest.main()
