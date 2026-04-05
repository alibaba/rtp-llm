import json
import unittest

import pytest

pytestmark = [pytest.mark.gpu(type="A10")]

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


class TestQwen25DetectorStreaming(unittest.TestCase):
    """Test Qwen25Detector streaming tool parsing."""

    def setUp(self):
        self.detector = Qwen25Detector()
        self.tools = create_tools()

    def test_think_end_and_tool_start_same_chunk(self):
        """Normal text and tool-start tag arrive in same chunk.
        Verifies buffer-scan handles prefix text before <tool_call>."""

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

        # Second chunk: newlines followed by complete tool call
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

    def test_thinking_tag_closing_gt_not_swallowed(self):
        """The '>' closing '</think>' must be returned as normal_text, not swallowed."""
        r = self.detector.parse_streaming_increment("</thinking", self.tools)
        self.assertEqual(
            r.normal_text,
            "</thinking",
            f"Expected '</thinking' as normal_text, got '{r.normal_text}'. Calls: {r.calls}",
        )
        self.assertEqual(
            len(r.calls),
            0,
            f"Expected 0 calls, got {len(r.calls)}. Calls: {r.calls}",
        )

        # ">" arrives as separate token after normalizer split
        r = self.detector.parse_streaming_increment(">", self.tools)
        self.assertEqual(
            r.normal_text,
            ">",
            f"Expected '>' as normal_text, got '{r.normal_text}'. Calls: {r.calls}",
        )
        self.assertEqual(
            len(r.calls),
            0,
            f"Expected 0 calls after '>', got {len(r.calls)}. Calls: {r.calls}",
        )

        # Tool call tokens follow — verify per-chunk
        tool_chunks = [
            "<tool_call>",
            "\n",
            '{"name": "get_current_weather"',
            ', "arguments": {"location": "杭州"}}',
            "\n</tool_call>",
        ]
        results = []
        for t in tool_chunks:
            results.append(self.detector.parse_streaming_increment(t, self.tools))

        # Chunks 0-1: no calls yet (bot_token accumulating)
        self.assertEqual(
            len(results[0].calls),
            0,
            f"Expected 0 calls at chunk 0, got {len(results[0].calls)}. Calls: {results[0].calls}",
        )
        self.assertEqual(
            len(results[1].calls),
            0,
            f"Expected 0 calls at chunk 1, got {len(results[1].calls)}. Calls: {results[1].calls}",
        )

        # Chunk 2: name emitted with empty parameters
        self.assertEqual(
            len(results[2].calls),
            1,
            f"Expected 1 call at chunk 2 (name), got {len(results[2].calls)}. Calls: {results[2].calls}",
        )
        self.assertEqual(
            results[2].calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{results[2].calls[0].name}'. Calls: {results[2].calls}",
        )
        self.assertEqual(
            results[2].calls[0].parameters,
            "",
            f"Expected empty parameters on name emission, got '{results[2].calls[0].parameters}'",
        )

        # Chunk 3: arguments emitted
        self.assertEqual(
            len(results[3].calls),
            1,
            f"Expected 1 call at chunk 3 (args), got {len(results[3].calls)}. Calls: {results[3].calls}",
        )
        self.assertIsNone(
            results[3].calls[0].name,
            f"Expected name to be None on arg chunk, got '{results[3].calls[0].name}'",
        )
        self.assertEqual(
            json.loads(results[3].calls[0].parameters),
            {"location": "杭州"},
            f'Expected arguments {{"location": "杭州"}}, got \'{results[3].calls[0].parameters}\'',
        )

    def test_partial_then_complete(self):
        """Partial tool call followed by completion in next chunk.

        The first chunk contains the complete name, so name is emitted in first result.
        """
        # First chunk: start of tool call (includes complete name)
        chunk1 = '<tool_call>\n{"name": "get_current_weather"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

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

        # Second chunk: completion
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


class TestKimiK2DetectorStreaming(unittest.TestCase):
    """Test KimiK2Detector streaming tool parsing."""

    def setUp(self):
        self.detector = KimiK2Detector()
        self.tools = create_tools()

    def test_partial_then_complete(self):
        """Partial tool call followed by completion."""
        # First chunk: start of tool call
        chunk1 = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        # Second chunk: completion
        chunk2 = ': "杭州"}<|tool_call_end|><|tool_calls_section_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )

    def test_thinking_tag_closing_gt_not_swallowed(self):
        """The '>' closing '</thinking>' must be returned as normal_text, not swallowed."""
        r = self.detector.parse_streaming_increment("</thinking", self.tools)
        self.assertEqual(
            r.normal_text,
            "</thinking",
            f"Expected '</thinking' as normal_text, got '{r.normal_text}'. Calls: {r.calls}",
        )
        self.assertEqual(
            len(r.calls),
            0,
            f"Expected 0 calls, got {len(r.calls)}. Calls: {r.calls}",
        )

        # ">" arrives as separate token after normalizer split
        r = self.detector.parse_streaming_increment(">", self.tools)
        self.assertEqual(
            r.normal_text,
            ">",
            f"Expected '>' as normal_text, got '{r.normal_text}'. Calls: {r.calls}",
        )
        self.assertEqual(
            len(r.calls),
            0,
            f"Expected 0 calls after '>', got {len(r.calls)}. Calls: {r.calls}",
        )

        # Tool call tokens follow — verify per-chunk
        tool_chunks = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>",
            '{"location": "杭州"}',
            "<|tool_call_end|><|tool_calls_section_end|>",
        ]
        results = []
        for t in tool_chunks:
            results.append(self.detector.parse_streaming_increment(t, self.tools))

        # Chunks 0-1: no calls yet
        self.assertEqual(
            len(results[0].calls),
            0,
            f"Expected 0 calls at chunk 0, got {len(results[0].calls)}. Calls: {results[0].calls}",
        )
        self.assertEqual(
            len(results[1].calls),
            0,
            f"Expected 0 calls at chunk 1, got {len(results[1].calls)}. Calls: {results[1].calls}",
        )

        # Chunk 2: name emitted with empty parameters
        self.assertEqual(
            len(results[2].calls),
            1,
            f"Expected 1 call at chunk 2 (name), got {len(results[2].calls)}. Calls: {results[2].calls}",
        )
        self.assertEqual(
            results[2].calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{results[2].calls[0].name}'. Calls: {results[2].calls}",
        )
        self.assertEqual(
            results[2].calls[0].parameters,
            "",
            f"Expected empty parameters on name emission, got '{results[2].calls[0].parameters}'",
        )

        # Chunk 3: arguments emitted
        self.assertEqual(
            len(results[3].calls),
            1,
            f"Expected 1 call at chunk 3 (args), got {len(results[3].calls)}. Calls: {results[3].calls}",
        )
        self.assertIsNone(
            results[3].calls[0].name,
            f"Expected name to be None on arg chunk, got '{results[3].calls[0].name}'",
        )
        self.assertEqual(
            json.loads(results[3].calls[0].parameters),
            {"location": "杭州"},
            f'Expected arguments {{"location": "杭州"}}, got \'{results[3].calls[0].parameters}\'',
        )


class TestDeepSeekV31DetectorStreaming(unittest.TestCase):
    """Test DeepSeekV31Detector streaming tool parsing."""

    def setUp(self):
        self.detector = DeepSeekV31Detector()
        self.tools = create_tools()

    def test_partial_then_complete(self):
        """Partial tool call followed by completion."""
        # First chunk: start of tool call
        chunk1 = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        # Second chunk: completion
        chunk2 = ': "杭州"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )

    def test_thinking_tag_closing_gt_not_swallowed(self):
        """The '>' closing '</thinking>' must be returned as normal_text, not swallowed."""
        r = self.detector.parse_streaming_increment("</thinking", self.tools)
        self.assertEqual(
            r.normal_text,
            "</thinking",
            f"Expected '</thinking' as normal_text, got '{r.normal_text}'. Calls: {r.calls}",
        )
        self.assertEqual(
            len(r.calls),
            0,
            f"Expected 0 calls, got {len(r.calls)}. Calls: {r.calls}",
        )

        r = self.detector.parse_streaming_increment(">", self.tools)
        self.assertEqual(
            r.normal_text,
            ">",
            f"Expected '>' as normal_text, got '{r.normal_text}'. Calls: {r.calls}",
        )
        self.assertEqual(
            len(r.calls),
            0,
            f"Expected 0 calls after '>', got {len(r.calls)}. Calls: {r.calls}",
        )

        # Tool call tokens — verify per-chunk
        tool_chunks = [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>",
            "get_current_weather",
            "<｜tool▁sep｜>",
            '{"location": "杭州"}',
            "<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ]
        results = []
        for t in tool_chunks:
            results.append(self.detector.parse_streaming_increment(t, self.tools))

        # Chunks 0-2: no calls yet (markers + name accumulating)
        for i in range(3):
            self.assertEqual(
                len(results[i].calls),
                0,
                f"Expected 0 calls at chunk {i}, got {len(results[i].calls)}. Calls: {results[i].calls}",
            )

        # Chunk 3 (<tool_sep>): name emitted with empty parameters
        self.assertEqual(
            len(results[3].calls),
            1,
            f"Expected 1 call at chunk 3 (name), got {len(results[3].calls)}. Calls: {results[3].calls}",
        )
        self.assertEqual(
            results[3].calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{results[3].calls[0].name}'. Calls: {results[3].calls}",
        )
        self.assertEqual(
            results[3].calls[0].parameters,
            "",
            f"Expected empty parameters on name emission, got '{results[3].calls[0].parameters}'",
        )

        # Chunk 4: arguments emitted
        self.assertEqual(
            len(results[4].calls),
            1,
            f"Expected 1 call at chunk 4 (args), got {len(results[4].calls)}. Calls: {results[4].calls}",
        )
        self.assertIsNone(
            results[4].calls[0].name,
            f"Expected name to be None on arg chunk, got '{results[4].calls[0].name}'",
        )
        self.assertEqual(
            json.loads(results[4].calls[0].parameters),
            {"location": "杭州"},
            f'Expected arguments {{"location": "杭州"}}, got \'{results[4].calls[0].parameters}\'',
        )


class TestDeepSeekV32DetectorStreaming(unittest.TestCase):
    """Test DeepSeekV32Detector streaming tool parsing."""

    def setUp(self):
        self.detector = DeepSeekV32Detector()
        self.tools = create_tools()

    def test_thinking_tag_closing_gt_not_swallowed(self):
        """The '>' closing '</thinking>' must be returned as normal_text, not swallowed."""
        r = self.detector.parse_streaming_increment("</thinking", self.tools)
        self.assertEqual(
            r.normal_text,
            "</thinking",
            f"Expected '</thinking' as normal_text, got '{r.normal_text}'. Calls: {r.calls}",
        )
        self.assertEqual(
            len(r.calls),
            0,
            f"Expected 0 calls, got {len(r.calls)}. Calls: {r.calls}",
        )

        r = self.detector.parse_streaming_increment(">", self.tools)
        self.assertEqual(
            r.normal_text,
            ">",
            f"Expected '>' as normal_text, got '{r.normal_text}'. Calls: {r.calls}",
        )
        self.assertEqual(
            len(r.calls),
            0,
            f"Expected 0 calls after '>', got {len(r.calls)}. Calls: {r.calls}",
        )

        # Tool call tokens — verify per-chunk
        tool_chunks = [
            "<｜DSML｜function_calls>",
            "\n",
            '<｜DSML｜invoke name="get_current_weather">',
            '{"location": "杭州"}',
            "</｜DSML｜invoke>",
            "\n",
            "</｜DSML｜function_calls>",
        ]
        results = []
        for t in tool_chunks:
            results.append(self.detector.parse_streaming_increment(t, self.tools))

        # Chunks 0-1: no calls yet
        self.assertEqual(
            len(results[0].calls),
            0,
            f"Expected 0 calls at chunk 0, got {len(results[0].calls)}. Calls: {results[0].calls}",
        )
        self.assertEqual(
            len(results[1].calls),
            0,
            f"Expected 0 calls at chunk 1, got {len(results[1].calls)}. Calls: {results[1].calls}",
        )

        # Chunk 2 (invoke tag): name emitted with empty parameters
        self.assertEqual(
            len(results[2].calls),
            1,
            f"Expected 1 call at chunk 2 (name), got {len(results[2].calls)}. Calls: {results[2].calls}",
        )
        self.assertEqual(
            results[2].calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{results[2].calls[0].name}'. Calls: {results[2].calls}",
        )
        self.assertEqual(
            results[2].calls[0].parameters,
            "",
            f"Expected empty parameters on name emission, got '{results[2].calls[0].parameters}'",
        )

        # Chunks 3-4: arguments streamed incrementally
        arg_calls = [c for r in results[3:5] for c in r.calls if c.parameters]
        self.assertTrue(
            len(arg_calls) > 0,
            f"Expected argument increment calls in chunks 3-4. Results: {[r.calls for r in results[3:5]]}",
        )

        # Verify full arguments
        all_params = "".join(
            c.parameters
            for r in results
            for c in r.calls
            if c.parameters and c.tool_index == 0
        )
        self.assertEqual(
            json.loads(all_params),
            {"location": "杭州"},
            f'Expected arguments {{"location": "杭州"}}, got \'{all_params}\'',
        )


class TestGlm4MoeDetectorStreaming(unittest.TestCase):
    """Test Glm4MoeDetector streaming tool parsing with GLM-4.7 format."""

    def setUp(self):
        self.detector = Glm4MoeDetector()
        self.tools = create_glm4_tools()

    def test_glm47_with_reasoning_and_tool_call(self):
        """Test GLM-4.7 format: normal text followed by tool call (detect_and_parse)."""

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

    def test_glm47_streaming_with_normal_text(self):
        """Test GLM-4.7 streaming: normal text chunk then tool call chunk."""
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

    def test_glm47_complete_tool_call_single_chunk(self):
        """Test GLM-4.7: complete tool call with normal text in single chunk."""
        # Complete response in one chunk
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
