"""
GLM-4.7 MOE Detector Tests

Tests for the Glm47MoeDetector which supports incremental streaming returns.
This detector uses XML-like format:
  <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

Tests cover:
1. detect_and_parse (one-time parsing)
2. parse_streaming_increment (incremental streaming)
3. MTP scenarios (multiple tokens per step)
4. Argument type handling
5. Edge cases specific to XML format
"""

import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import List

from rtp_llm.openai.renderer_factory_register import _renderer_factory
from rtp_llm.openai.renderers.chatglm45_renderer import ChatGlm45Renderer
from rtp_llm.openai.renderers.chatglm47_renderer import ChatGlm47Renderer
from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    ToolParseError,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.glm47_moe_detector import (
    Glm47MoeDetector,
    get_argument_type,
    parse_arguments,
)


def create_basic_tools() -> List[Tool]:
    """Create basic test tool definitions."""
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "date": {"type": "string", "description": "Date"},
                    },
                    "required": ["city"],
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


def create_complex_tools() -> List[Tool]:
    """Create tool definitions with complex argument types."""
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
        Tool(
            type="function",
            function=Function(
                name="calculate",
                description="Calculate expression",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "precision": {"type": "number"},
                    },
                },
            ),
        ),
    ]


def collect_streaming_results(
    detector: Glm47MoeDetector, chunks: List[str], tools: List[Tool]
) -> List[StreamingParseResult]:
    """Feed chunks through detector and collect all results."""
    results = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        results.append(result)
    return results


class TestGlm5RendererRegistration(unittest.TestCase):
    def test_glm5_uses_incremental_tool_renderer(self):
        self.assertIs(_renderer_factory.get("glm_5"), ChatGlm47Renderer)
        self.assertIs(_renderer_factory.get("glm47_moe"), ChatGlm47Renderer)
        self.assertIs(_renderer_factory.get("glm4_moe"), ChatGlm45Renderer)


class TestGlm47MoeDetectorHelpers(unittest.TestCase):
    """Test helper functions."""

    def setUp(self):
        self.tools = create_basic_tools()

    def test_get_argument_type_existing(self):
        """Test get_argument_type for existing parameter."""
        result = get_argument_type("get_weather", "city", self.tools)
        self.assertEqual(result, "string")

    def test_get_argument_type_missing_function(self):
        """Test get_argument_type for non-existent function."""
        result = get_argument_type("non_existent", "city", self.tools)
        self.assertIsNone(result)

    def test_get_argument_type_missing_param(self):
        """Test get_argument_type for non-existent parameter."""
        result = get_argument_type("get_weather", "non_existent", self.tools)
        self.assertIsNone(result)

    def test_parse_arguments_string(self):
        """Test parse_arguments with string value."""
        value, is_valid = parse_arguments('"hello"', "string")
        self.assertEqual(value, "hello")
        self.assertTrue(is_valid)

    def test_parse_arguments_number_int(self):
        """Test parse_arguments with integer."""
        value, is_valid = parse_arguments("42", "number")
        self.assertEqual(value, 42)
        self.assertTrue(is_valid)

    def test_parse_arguments_number_float(self):
        """Test parse_arguments with float."""
        value, is_valid = parse_arguments("3.14", "number")
        self.assertEqual(value, 3.14)
        self.assertTrue(is_valid)

    def test_parse_arguments_object(self):
        """Test parse_arguments with object."""
        value, is_valid = parse_arguments('{"key": "value"}', "object")
        self.assertEqual(value, {"key": "value"})
        self.assertTrue(is_valid)

    def test_parse_arguments_array(self):
        """Test parse_arguments with array."""
        value, is_valid = parse_arguments("[1, 2, 3]", "array")
        self.assertEqual(value, [1, 2, 3])
        self.assertTrue(is_valid)


class TestGlm47MoeDetectorDetectAndParse(unittest.TestCase):
    """Test Glm47MoeDetector.detect_and_parse (one-time parsing)."""

    def setUp(self):
        self.tools = create_basic_tools()

    def test_single_tool_call_simple(self):
        """Test parsing a single simple tool call."""
        detector = Glm47MoeDetector()
        text = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>"

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "北京")

    def test_single_tool_call_multiple_args(self):
        """Test parsing tool call with multiple arguments."""
        detector = Glm47MoeDetector()
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>北京</arg_value>"
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>"
            "</tool_call>"
        )

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "北京")
        self.assertEqual(params["date"], "2024-06-27")

    def test_no_arg_function(self):
        """Test parsing a no-argument function call."""
        detector = Glm47MoeDetector()
        text = "<tool_call>get_time</tool_call>"

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_time")
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params, {})

    def test_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        detector = Glm47MoeDetector()
        text = (
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>上海</arg_value></tool_call>"
        )

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "get_weather")
        params0 = json.loads(result.calls[0].parameters)
        params1 = json.loads(result.calls[1].parameters)
        self.assertEqual(params0["city"], "北京")
        self.assertEqual(params1["city"], "上海")

    def test_normal_text_before_tool(self):
        """Test that normal text before tool call is captured."""
        detector = Glm47MoeDetector()
        text = (
            "Let me check the weather for you. "
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>"
        )

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Let me check the weather for you.")
        self.assertEqual(len(result.calls), 1)

    def test_normal_text_between_tools(self):
        """Test normal text extraction between multiple tool calls."""
        detector = Glm47MoeDetector()
        text = (
            "First query: "
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>"
            " Second query: "
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>上海</arg_value></tool_call>"
        )

        result = detector.detect_and_parse(text, self.tools)

        # Normal text should include text before, between, and after
        self.assertIn("First query:", result.normal_text)
        self.assertEqual(len(result.calls), 2)

    def test_no_tool_call(self):
        """Test when there's no tool call in text."""
        detector = Glm47MoeDetector()
        text = "This is just regular text without any tool calls."

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(len(result.calls), 0)


class TestGlm47MoeDetectorComplexArgs(unittest.TestCase):
    """Test complex argument handling."""

    def setUp(self):
        self.tools = create_complex_tools()

    def test_array_argument(self):
        """Test parsing array-type argument."""
        detector = Glm47MoeDetector()
        text = (
            "<tool_call>ask_user_question"
            "<arg_key>questions</arg_key>"
            '<arg_value>[{"question": "What language?", "header": "Lang", '
            '"multiSelect": false, "options": [{"label": "Python", "description": "Simple"}]}]</arg_value>'
            "</tool_call>"
        )

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "ask_user_question")
        params = json.loads(result.calls[0].parameters)
        self.assertIn("questions", params)
        self.assertIsInstance(params["questions"], list)
        self.assertEqual(len(params["questions"]), 1)
        self.assertEqual(params["questions"][0]["question"], "What language?")

    def test_number_argument(self):
        """Test parsing number-type argument."""
        detector = Glm47MoeDetector()
        text = (
            "<tool_call>calculate"
            "<arg_key>expression</arg_key><arg_value>2+2</arg_value>"
            "<arg_key>precision</arg_key><arg_value>3</arg_value>"
            "</tool_call>"
        )

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["expression"], "2+2")
        self.assertEqual(params["precision"], 3)


class TestGlm47MoeDetectorStreaming(unittest.TestCase):
    """Test incremental streaming behavior."""

    def setUp(self):
        self.tools = create_basic_tools()

    def test_streaming_tool_name_commits_after_complete_block(self):
        """The tool name is not public until the whole block is valid."""
        detector = Glm47MoeDetector()

        chunks = [
            "<tool_call>",
            "get_weather",
            "<arg_key>",
            "city</arg_key><arg_value>Beijing</arg_value>",
            "</tool_call>",
        ]

        results = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            results.append(result)

        self.assertTrue(all(not result.calls for result in results[:-1]))
        self.assertEqual(len(results[-1].calls), 1)
        self.assertEqual(results[-1].calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(results[-1].calls[0].parameters), {"city": "Beijing"}
        )

    def test_streaming_complete_single_chunk(self):
        """Test complete tool call in single chunk (MTP scenario)."""
        detector = Glm47MoeDetector()

        chunk = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>北京</arg_value>"
            "</tool_call>"
        )

        result = detector.parse_streaming_increment(chunk, self.tools)

        self.assertGreaterEqual(
            len(result.calls), 1, f"Expected at least 1 call. Calls: {result.calls}"
        )
        # Find name call
        name_calls = [c for c in result.calls if c.name]
        self.assertEqual(len(name_calls), 1)
        self.assertEqual(name_calls[0].name, "get_weather")

    def test_streaming_incremental_arguments(self):
        """Test that arguments are streamed incrementally."""
        detector = Glm47MoeDetector()

        chunks = [
            "<tool_call>get_weather<arg_key>",  # Start
            "city</arg_key>",  # Key complete
            "<arg_value>",  # Value start
            "北",  # Partial value
            "京",  # Complete value
            "</arg_value>",  # Value end
            "</tool_call>",  # Tool end
        ]

        results = collect_streaming_results(detector, chunks, self.tools)
        all_calls = []
        for r in results:
            all_calls.extend(r.calls)

        # Should have name and argument increments
        name_calls = [c for c in all_calls if c.name]
        self.assertGreaterEqual(len(name_calls), 1)
        self.assertEqual(name_calls[0].name, "get_weather")

        # Collect argument increments
        arg_calls = [c for c in all_calls if c.parameters]
        # Should have at least one argument call
        self.assertGreaterEqual(
            len(arg_calls), 1, f"Expected argument increments. All calls: {all_calls}"
        )

    def test_streaming_multiple_tools(self):
        """Test streaming with multiple tool calls."""
        detector = Glm47MoeDetector()

        chunks = [
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>上海</arg_value></tool_call>",
        ]

        results = collect_streaming_results(detector, chunks, self.tools)
        all_calls = []
        for r in results:
            all_calls.extend(r.calls)

        # Should have 2 name emissions with different tool indices
        name_calls = [c for c in all_calls if c.name]
        self.assertEqual(
            len(name_calls), 2, f"Expected 2 name emissions. All calls: {all_calls}"
        )
        self.assertEqual(name_calls[0].tool_index, 0)
        self.assertEqual(name_calls[1].tool_index, 1)

    def test_streaming_no_arg_function(self):
        """Test streaming no-argument function."""
        detector = Glm47MoeDetector()

        chunks = [
            "<tool_call>get_time",
            "</tool_call>",
        ]

        results = collect_streaming_results(detector, chunks, self.tools)
        all_calls = []
        for r in results:
            all_calls.extend(r.calls)

        name_calls = [c for c in all_calls if c.name]
        self.assertGreaterEqual(len(name_calls), 1)
        self.assertEqual(name_calls[0].name, "get_time")

        # Should emit {} for no-arg function
        arg_calls = [c for c in all_calls if c.parameters]
        full_args = "".join(c.parameters for c in arg_calls)
        self.assertEqual(full_args, "{}")

    def test_streaming_normal_text_before_tool(self):
        """Test normal text is returned before tool call starts."""
        detector = Glm47MoeDetector()

        chunks = [
            "Let me check: ",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>",
        ]

        results = collect_streaming_results(detector, chunks, self.tools)

        # First chunk should return normal text
        self.assertEqual(results[0].normal_text, "Let me check: ")
        self.assertEqual(len(results[0].calls), 0)

    def test_streaming_todowrite_array_argument(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="todowrite",
                    description="Update the todo list",
                    parameters={
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "status": {"type": "string"},
                                        "priority": {"type": "string"},
                                    },
                                },
                            }
                        },
                        "required": ["todos"],
                    },
                ),
            )
        ]
        chunks = [
            "<tool_call>todowrite<arg_key>todos</arg_key><arg_value>",
            '[{"content":"inspect","status":"in_progress","priority":"high"}]',
            "</arg_value></tool_call>",
        ]

        results = collect_streaming_results(Glm47MoeDetector(), chunks, tools)
        calls = [call for result in results for call in result.calls]
        self.assertEqual([call.name for call in calls if call.name], ["todowrite"])
        parameters = "".join(call.parameters for call in calls if call.parameters)
        self.assertEqual(
            json.loads(parameters),
            {
                "todos": [
                    {
                        "content": "inspect",
                        "status": "in_progress",
                        "priority": "high",
                    }
                ]
            },
        )


class TestGlm47MoeDetectorMTP(unittest.TestCase):
    """Test MTP (Multi-Token-Per-step) scenarios."""

    def setUp(self):
        self.tools = create_basic_tools()

    def test_mtp_complete_tool_single_chunk(self):
        """MTP scenario: complete tool call arrives in single chunk."""
        detector = Glm47MoeDetector()

        chunk = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>杭州</arg_value>"
            "</tool_call>"
        )

        result = detector.parse_streaming_increment(chunk, self.tools)

        name_calls = [c for c in result.calls if c.name]
        self.assertEqual(len(name_calls), 1)
        self.assertEqual(name_calls[0].name, "get_weather")

    def test_mtp_text_and_tool_same_chunk(self):
        """MTP scenario: normal text and tool start in same chunk."""
        detector = Glm47MoeDetector()

        chunk = (
            "让我查一下天气："
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>北京</arg_value>"
            "</tool_call>"
        )

        result = detector.parse_streaming_increment(chunk, self.tools)

        # Should have normal text
        self.assertIn("让我查一下天气", result.normal_text)

        # Should have tool call
        name_calls = [c for c in result.calls if c.name]
        self.assertEqual(len(name_calls), 1)
        self.assertEqual(name_calls[0].name, "get_weather")

    def test_mtp_partial_then_complete(self):
        """MTP scenario: partial tool call followed by completion."""
        detector = Glm47MoeDetector()

        # First chunk: partial tool call
        chunk1 = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北"
        result1 = detector.parse_streaming_increment(chunk1, self.tools)

        # Second chunk: completion
        chunk2 = "京</arg_value></tool_call>"
        result2 = detector.parse_streaming_increment(chunk2, self.tools)

        # Collect all calls
        all_calls = list(result1.calls) + list(result2.calls)

        # Should have name emission
        name_calls = [c for c in all_calls if c.name]
        self.assertGreaterEqual(
            len(name_calls), 1, f"Expected name emission. All calls: {all_calls}"
        )
        self.assertEqual(name_calls[0].name, "get_weather")

    def test_mtp_multiple_tools_single_chunk(self):
        """MTP scenario: multiple complete tool calls in single chunk."""
        detector = Glm47MoeDetector()

        chunk = (
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>上海</arg_value></tool_call>"
        )

        result = detector.parse_streaming_increment(chunk, self.tools)

        # The serving stream can finish after this MTP chunk, so both calls must be
        # emitted immediately without a synthetic empty follow-up chunk.
        all_name_calls = [c for c in result.calls if c.name]
        self.assertEqual(
            len(all_name_calls),
            2,
            f"Expected 2 name emissions. All calls: {result.calls}",
        )
        self.assertEqual([c.tool_index for c in all_name_calls], [0, 1])
        merged = merge_tool_call_deltas(result.calls)
        self.assertEqual(json.loads(merged[0]["parameters"]), {"city": "北京"})
        self.assertEqual(json.loads(merged[1]["parameters"]), {"city": "上海"})
        self.assertEqual(detector._buffer, "")

    def test_mtp_multiple_tools_and_trailing_text_single_chunk(self):
        detector = Glm47MoeDetector()
        chunk = (
            "准备查询"
            "<tool_call>get_time</tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>深圳</arg_value></tool_call>"
            "已提交"
        )

        result = detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(result.normal_text, "准备查询已提交")
        name_calls = [c for c in result.calls if c.name]
        self.assertEqual([c.name for c in name_calls], ["get_time", "get_weather"])
        merged = merge_tool_call_deltas(result.calls)
        self.assertEqual(json.loads(merged[0]["parameters"]), {})
        self.assertEqual(json.loads(merged[1]["parameters"]), {"city": "深圳"})
        self.assertEqual(detector._buffer, "")

    def test_mtp_tool_boundary_fusion(self):
        """MTP scenario: tool end and next tool start in same chunk."""
        detector = Glm47MoeDetector()

        # First chunk: complete first tool
        chunk1 = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>"
        result1 = detector.parse_streaming_increment(chunk1, self.tools)

        # Second chunk: immediately starts next tool (no separation)
        chunk2 = "<tool_call>get_time</tool_call>"
        result2 = detector.parse_streaming_increment(chunk2, self.tools)

        all_calls = list(result1.calls) + list(result2.calls)
        name_calls = [c for c in all_calls if c.name]

        self.assertEqual(
            len(name_calls), 2, f"Expected 2 name emissions. All calls: {all_calls}"
        )
        self.assertEqual(name_calls[0].name, "get_weather")
        self.assertEqual(name_calls[1].name, "get_time")


class TestGlm47MoeDetectorEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def setUp(self):
        self.tools = create_basic_tools()

    def test_potential_tool_start_buffering(self):
        """Test that potential tool start (<) is buffered."""
        detector = Glm47MoeDetector()

        # '<' alone could be start of <tool_call>
        result1 = detector.parse_streaming_increment("<", self.tools)
        self.assertEqual(result1.normal_text, "")  # Should buffer, not emit

        # Continue with non-tool content
        result2 = detector.parse_streaming_increment("some text", self.tools)
        # Should now release buffered content since it's not a tool call
        self.assertIn("<some text", result2.normal_text)

    def test_unicode_in_arguments(self):
        """Test Unicode characters in argument values."""
        detector = Glm47MoeDetector()
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>东京 🌸</arg_value>"
            "</tool_call>"
        )

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "东京 🌸")

    def test_special_characters_in_arguments(self):
        """Test special JSON characters in arguments."""
        detector = Glm47MoeDetector()
        text = (
            "<tool_call>get_weather"
            '<arg_key>city</arg_key><arg_value>New York "City"</arg_value>'
            "</tool_call>"
        )

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertIn("New York", params["city"])

    def test_empty_buffer_after_complete_tool(self):
        """Test buffer is cleared after complete tool call."""
        detector = Glm47MoeDetector()

        chunk = "<tool_call>get_time</tool_call>"
        result1 = detector.parse_streaming_increment(chunk, self.tools)

        # Verify tool was parsed
        name_calls = [c for c in result1.calls if c.name]
        self.assertGreaterEqual(len(name_calls), 1)

        # Buffer should be empty, next normal text should work
        result2 = detector.parse_streaming_increment("Normal text", self.tools)
        # Note: After tool calls, normal text might be filtered
        # This is expected behavior as we're in "tool call mode"

    def test_has_tool_call(self):
        """Test has_tool_call method."""
        detector = Glm47MoeDetector()

        self.assertTrue(detector.has_tool_call("<tool_call>something"))
        self.assertTrue(detector.has_tool_call("prefix<tool_call>suffix"))
        self.assertFalse(detector.has_tool_call("no tool call here"))
        self.assertFalse(detector.has_tool_call(""))


class TestGlm47MoeDetectorStateReset(unittest.TestCase):
    """Test state management and reset behavior."""

    def setUp(self):
        self.tools = create_basic_tools()

    def test_fresh_detector_state(self):
        """Test initial state of new detector."""
        detector = Glm47MoeDetector()

        self.assertEqual(detector.current_tool_id, -1)
        self.assertEqual(detector._buffer, "")

    def test_state_after_tool_call(self):
        """Test state is properly updated after tool call."""
        detector = Glm47MoeDetector()

        chunk = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>北京</arg_value>"
            "</tool_call>"
        )
        detector.parse_streaming_increment(chunk, self.tools)

        # After first tool, current_tool_id should be incremented
        self.assertGreaterEqual(detector.current_tool_id, 0)

    def test_streaming_state_reset_between_tools(self):
        """Test streaming state is reset between tool calls."""
        detector = Glm47MoeDetector()

        # First tool
        chunk1 = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>北京</arg_value>"
            "</tool_call>"
        )
        result1 = detector.parse_streaming_increment(chunk1, self.tools)

        # Second tool
        chunk2 = "<tool_call>get_time" "</tool_call>"
        result2 = detector.parse_streaming_increment(chunk2, self.tools)

        # Both should parse correctly
        all_calls = list(result1.calls) + list(result2.calls)
        name_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(name_calls), 2)


class TestGlm47MoeDetectorArgumentStreaming(unittest.TestCase):
    """Test incremental argument value streaming."""

    def setUp(self):
        self.tools = create_basic_tools()

    def test_string_value_incremental_streaming(self):
        """Test string value is streamed incrementally with proper JSON escaping."""
        detector = Glm47MoeDetector()

        chunks = [
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>",
            "San ",
            "Francisco",
            "</arg_value></tool_call>",
        ]

        results = collect_streaming_results(detector, chunks, self.tools)
        all_calls = []
        for r in results:
            all_calls.extend(r.calls)

        # Collect argument increments (excluding name emission)
        arg_calls = [c for c in all_calls if c.parameters]
        full_args = "".join(c.parameters for c in arg_calls)

        # Should form valid JSON
        try:
            parsed = json.loads(full_args)
            self.assertEqual(parsed["city"], "San Francisco")
        except json.JSONDecodeError:
            self.fail(f"Arguments should form valid JSON. Got: {full_args}")

    def test_number_value_streaming(self):
        """Test number value streaming."""
        tools = create_complex_tools()
        detector = Glm47MoeDetector()

        chunks = [
            "<tool_call>calculate<arg_key>expression</arg_key><arg_value>1+1</arg_value>",
            "<arg_key>precision</arg_key><arg_value>",
            "5",
            "</arg_value></tool_call>",
        ]

        results = collect_streaming_results(detector, chunks, tools)
        all_calls = []
        for r in results:
            all_calls.extend(r.calls)

        # Get all argument increments
        arg_calls = [c for c in all_calls if c.parameters]
        full_args = "".join(c.parameters for c in arg_calls)

        # Should have valid JSON with number
        try:
            parsed = json.loads(full_args)
            self.assertEqual(parsed["precision"], 5)
        except json.JSONDecodeError:
            self.fail(f"Arguments should form valid JSON. Got: {full_args}")

    def test_invalid_number_value_is_forwarded_as_valid_json_string(self):
        tools = create_complex_tools()
        text = (
            "<tool_call>calculate"
            "<arg_key>precision</arg_key><arg_value>abc</arg_value>"
            "</tool_call>"
        )

        for chunks in ([text], list(text)):
            with self.subTest(chunk_count=len(chunks)):
                calls = [
                    call
                    for result in collect_streaming_results(
                        Glm47MoeDetector(), chunks, tools
                    )
                    for call in result.calls
                ]
                parameters = "".join(c.parameters for c in calls if c.parameters)
                self.assertEqual(json.loads(parameters), {"precision": "abc"})

    def test_object_valued_final_argument_closes_outer_arguments(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="configure",
                    description="Configure a task",
                    parameters={
                        "type": "object",
                        "properties": {"payload": {"type": "object"}},
                    },
                ),
            )
        ]
        text = (
            "<tool_call>configure"
            '<arg_key>payload</arg_key><arg_value>{"mode":"fast"}</arg_value>'
            "</tool_call>"
        )

        for chunks in ([text], list(text)):
            with self.subTest(chunk_count=len(chunks)):
                calls = [
                    call
                    for result in collect_streaming_results(
                        Glm47MoeDetector(), chunks, tools
                    )
                    for call in result.calls
                ]
                parameters = "".join(c.parameters for c in calls if c.parameters)
                self.assertEqual(
                    json.loads(parameters), {"payload": {"mode": "fast"}}
                )


class TestGlm47MoeDetectorHighConcurrency(unittest.TestCase):
    def setUp(self):
        self.tools = create_basic_tools()

    def _parse_request(self, request_index: int) -> tuple[str, dict]:
        city = f"city-{request_index}"
        text = (
            "<tool_call>get_weather<arg_key>city</arg_key>"
            f"<arg_value>{city}</arg_value></tool_call>"
        )
        chunk_size = request_index % 11 + 1
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        detector = Glm47MoeDetector()
        calls = [
            call
            for result in collect_streaming_results(detector, chunks, self.tools)
            for call in result.calls
        ]
        names = [call.name for call in calls if call.name]
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(detector._buffer, "")
        parameters = "".join(call.parameters for call in calls if call.parameters)
        return city, json.loads(parameters)

    def test_256_concurrent_detectors_do_not_cross_request_state(self):
        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(self._parse_request, range(256)))

        self.assertEqual(
            results,
            [(f"city-{i}", {"city": f"city-{i}"}) for i in range(256)],
        )

    def test_every_two_chunk_boundary_matches_complete_parse(self):
        text = (
            "前缀"
            "<tool_call>get_time</tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>杭州 🌸</arg_value></tool_call>"
            "后缀"
        )

        for split in range(len(text) + 1):
            with self.subTest(split=split):
                detector = Glm47MoeDetector()
                results = collect_streaming_results(
                    detector, [text[:split], text[split:]], self.tools
                )
                calls = [call for result in results for call in result.calls]
                normal_text = "".join(result.normal_text for result in results)
                merged = merge_tool_call_deltas(calls)
                self.assertEqual(normal_text, "前缀后缀")
                self.assertEqual(
                    [call.name for call in calls if call.name],
                    ["get_time", "get_weather"],
                )
                self.assertEqual(json.loads(merged[0]["parameters"]), {})
                self.assertEqual(
                    json.loads(merged[1]["parameters"]), {"city": "杭州 🌸"}
                )
                self.assertEqual(detector._buffer, "")


class TestGlm47MoeDetectorAtomicToolBlocks(unittest.TestCase):
    def setUp(self):
        self.tools = create_basic_tools()

    def test_missing_arg_value_is_not_partially_emitted(self):
        detector = Glm47MoeDetector()
        prefix = "<tool_call>get_weather<arg_key>city</arg_key>"

        partial = detector.parse_streaming_increment(prefix, self.tools)
        self.assertEqual(partial.calls, [])
        self.assertEqual(partial.normal_text, "")
        self.assertEqual(detector.current_tool_id, -1)
        self.assertEqual(detector.prev_tool_call_arr, [])
        self.assertEqual(detector.streamed_args_for_tool, [])

        with self.assertRaisesRegex(ToolParseError, "malformed tool arguments"):
            detector.parse_streaming_increment("</tool_call>", self.tools)

        self.assertEqual(detector._buffer, "")
        self.assertEqual(detector.current_tool_id, -1)
        self.assertEqual(detector.prev_tool_call_arr, [])
        self.assertEqual(detector.streamed_args_for_tool, [])

    def test_deeply_nested_value_is_not_partially_emitted(self):
        detector = Glm47MoeDetector()
        nested_value = "[" * 1100 + "0" + "]" * 1100
        prefix = (
            "<tool_call>ask_user_question"
            "<arg_key>questions</arg_key><arg_value>"
            f"{nested_value}</arg_value>"
        )

        partial = detector.parse_streaming_increment(prefix, create_complex_tools())
        self.assertEqual(partial.calls, [])
        self.assertEqual(partial.normal_text, "")

        with self.assertRaisesRegex(ToolParseError, "failed to parse tool arguments"):
            detector.parse_streaming_increment("</tool_call>", create_complex_tools())

        self.assertEqual(detector._buffer, "")
        self.assertEqual(detector.current_tool_id, -1)
        self.assertEqual(detector.prev_tool_call_arr, [])
        self.assertEqual(detector.streamed_args_for_tool, [])

    def test_non_streaming_missing_arg_value_is_terminal(self):
        text = "<tool_call>get_weather<arg_key>city</arg_key></tool_call>"

        with self.assertRaisesRegex(ToolParseError, "malformed tool arguments"):
            Glm47MoeDetector().detect_and_parse(text, self.tools)

    def test_non_streaming_unclosed_tool_is_terminal(self):
        text = (
            "visible prefix<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Hangzhou</arg_value>"
        )

        with self.assertRaisesRegex(ToolParseError, "incomplete tool call"):
            Glm47MoeDetector().detect_and_parse(text, self.tools)

    def test_non_streaming_unclosed_tool_after_valid_tool_is_terminal(self):
        text = (
            "<tool_call>get_time</tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key>"
        )

        with self.assertRaisesRegex(ToolParseError, "incomplete tool call"):
            Glm47MoeDetector().detect_and_parse(text, self.tools)

    def test_nested_tool_start_in_function_name_is_terminal(self):
        text = "<tool_call>broken<tool_call>get_time</tool_call>"

        with self.assertRaisesRegex(ToolParseError, "malformed tool call"):
            Glm47MoeDetector().detect_and_parse(text, self.tools)

    def test_tool_start_literal_is_allowed_inside_argument_value(self):
        text = (
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>literal <tool_call> marker</arg_value></tool_call>"
        )

        non_stream = Glm47MoeDetector().detect_and_parse(text, self.tools)
        stream = Glm47MoeDetector().parse_streaming_increment(text, self.tools)

        self.assertEqual(
            json.loads(non_stream.calls[0].parameters),
            {"city": "literal <tool_call> marker"},
        )
        self.assertEqual(
            json.loads(stream.calls[0].parameters),
            {"city": "literal <tool_call> marker"},
        )

    def test_malformed_argument_scan_uses_monotonic_cursor(self):
        class CountingText(str):
            def __new__(cls, value):
                instance = super().__new__(cls, value)
                instance.find_calls = 0
                return instance

            def find(self, *args):
                self.find_calls += 1
                return super().find(*args)

        raw_arguments = CountingText("<arg_key>" * 16384)

        with self.assertRaisesRegex(ToolParseError, "malformed tool arguments"):
            Glm47MoeDetector()._parse_argument_text_strict(
                raw_arguments, "get_weather", self.tools
            )

        self.assertEqual(raw_arguments.find_calls, 1)

    def test_non_streaming_unclosed_scan_is_single_pass(self):
        class CountingText(str):
            def __new__(cls, value):
                instance = super().__new__(cls, value)
                instance.find_calls = 0
                return instance

            def find(self, *args):
                self.find_calls += 1
                return super().find(*args)

        text = CountingText("<tool_call>" * 16384)

        with self.assertRaisesRegex(ToolParseError, "incomplete tool call"):
            Glm47MoeDetector().detect_and_parse(text, self.tools)

        self.assertEqual(text.find_calls, 2)

    def test_valid_tool_commits_only_after_closing_tag(self):
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Hangzhou</arg_value>"
            "</tool_call>"
        )

        for split in range(1, len(text)):
            with self.subTest(split=split):
                detector = Glm47MoeDetector()
                partial = detector.parse_streaming_increment(text[:split], self.tools)
                self.assertEqual(partial.calls, [])
                self.assertEqual(partial.normal_text, "")
                self.assertEqual(detector.current_tool_id, -1)

                completed = detector.parse_streaming_increment(
                    text[split:], self.tools
                )
                self.assertEqual(len(completed.calls), 1)
                self.assertEqual(completed.calls[0].tool_index, 0)
                self.assertEqual(completed.calls[0].name, "get_weather")
                self.assertEqual(
                    json.loads(completed.calls[0].parameters), {"city": "Hangzhou"}
                )
                self.assertEqual(detector._buffer, "")

    def test_long_argument_is_parsed_once_at_commit(self):
        class CountingDetector(Glm47MoeDetector):
            def __init__(self):
                super().__init__()
                self.complete_parse_count = 0

            def _parse_tool_action(self, func_name, raw_arguments, tools):
                self.complete_parse_count += 1
                return super()._parse_tool_action(
                    func_name, raw_arguments, tools
                )

        detector = CountingDetector()
        prefix = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>"
        detector.parse_streaming_increment(prefix, self.tools)

        for _ in range(4096):
            result = detector.parse_streaming_increment("abcdefgh", self.tools)
            self.assertEqual(result.calls, [])
        self.assertEqual(detector.complete_parse_count, 0)
        self.assertEqual(detector._buffer, "")
        self.assertLessEqual(
            len(detector._pending_tool_tail), len(detector.eot_token) - 1
        )

        completed = detector.parse_streaming_increment(
            "</arg_value></tool_call>", self.tools
        )
        self.assertEqual(detector.complete_parse_count, 1)
        self.assertEqual(len(completed.calls), 1)
        self.assertEqual(
            json.loads(completed.calls[0].parameters),
            {"city": "abcdefgh" * 4096},
        )

    def test_natural_eof_rejects_and_clears_incomplete_tool(self):
        detector = Glm47MoeDetector()
        detector.parse_streaming_increment(
            "<tool_call>get_weather<arg_key>city</arg_key>", self.tools
        )

        with self.assertRaisesRegex(ToolParseError, "incomplete tool call"):
            detector.finalize_streaming()

        self.assertIsNone(detector._pending_tool_buffer)
        self.assertEqual(detector._pending_tool_tail, "")
        self.assertEqual(detector.prev_tool_call_arr, [])

    def test_length_finish_discards_only_incomplete_transaction(self):
        detector = Glm47MoeDetector()
        first = detector.parse_streaming_increment(
            "<tool_call>get_time</tool_call>", self.tools
        )
        self.assertEqual([call.name for call in first.calls], ["get_time"])
        detector.parse_streaming_increment(
            "<tool_call>get_weather<arg_key>city</arg_key>", self.tools
        )

        finalized = detector.finalize_streaming(truncated=True)

        self.assertEqual(finalized.calls, [])
        self.assertEqual(finalized.normal_text, "")
        self.assertIsNone(detector._pending_tool_buffer)
        self.assertEqual(detector.current_tool_id, 1)
        self.assertEqual(len(detector.prev_tool_call_arr), 1)

    def test_eof_releases_only_unconfirmed_tool_prefix(self):
        detector = Glm47MoeDetector()
        partial = detector.parse_streaming_increment("visible<tool_ca", self.tools)
        self.assertEqual(partial.normal_text, "visible")

        finalized = detector.finalize_streaming()

        self.assertEqual(finalized.normal_text, "<tool_ca")
        self.assertEqual(finalized.calls, [])
        self.assertEqual(detector._buffer, "")

    def test_many_tools_in_one_chunk_keep_one_input_object(self):
        class CountingDetector(Glm47MoeDetector):
            def __init__(self):
                super().__init__()
                self.append_inputs = []

            def _append_pending_tool(self, text, cursor):
                self.append_inputs.append(text)
                return super()._append_pending_tool(text, cursor)

        detector = CountingDetector()
        text = "<tool_call>get_time</tool_call>" * 1024

        result = detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(len(result.calls), 1024)
        self.assertEqual(
            [call.tool_index for call in result.calls], list(range(1024))
        )
        self.assertTrue(
            all(item is detector.append_inputs[0] for item in detector.append_inputs)
        )
        self.assertEqual(detector._buffer, "")


class TestGlm47MoeDetectorUnknownTools(unittest.TestCase):
    def setUp(self):
        self.tools = create_basic_tools()
        self.original_forward_unknown = os.environ.get(
            "RTP_LLM_FORWARD_UNKNOWN_TOOLS"
        )

    def tearDown(self):
        if self.original_forward_unknown is None:
            os.environ.pop("RTP_LLM_FORWARD_UNKNOWN_TOOLS", None)
        else:
            os.environ["RTP_LLM_FORWARD_UNKNOWN_TOOLS"] = (
                self.original_forward_unknown
            )

    def test_streaming_unknown_tool_is_dropped_by_default(self):
        os.environ.pop("RTP_LLM_FORWARD_UNKNOWN_TOOLS", None)
        text = (
            "<tool_call>delete_everything<arg_key>scope</arg_key>"
            "<arg_value>all</arg_value></tool_call>"
            "<tool_call>get_time</tool_call>"
        )
        calls = []
        detector = Glm47MoeDetector()
        for chunk in list(text):
            calls.extend(detector.parse_streaming_increment(chunk, self.tools).calls)

        self.assertEqual([call.name for call in calls if call.name], ["get_time"])
        self.assertEqual(
            json.loads("".join(call.parameters for call in calls if call.parameters)),
            {},
        )
        self.assertEqual(detector._buffer, "")

    def test_malformed_unknown_does_not_block_following_valid_tool(self):
        os.environ.pop("RTP_LLM_FORWARD_UNKNOWN_TOOLS", None)
        text = (
            "<tool_call>unknown<arg_key>missing_value</arg_key></tool_call>"
            "<tool_call>get_time</tool_call>"
        )

        stream = Glm47MoeDetector().parse_streaming_increment(text, self.tools)
        non_stream = Glm47MoeDetector().detect_and_parse(text, self.tools)

        self.assertEqual([call.name for call in stream.calls], ["get_time"])
        self.assertEqual([call.name for call in non_stream.calls], ["get_time"])

    def test_forwarded_malformed_unknown_is_terminal(self):
        os.environ["RTP_LLM_FORWARD_UNKNOWN_TOOLS"] = "true"
        text = "<tool_call>unknown<arg_key>missing_value</arg_key></tool_call>"

        with self.assertRaisesRegex(ToolParseError, "malformed tool arguments"):
            Glm47MoeDetector().parse_streaming_increment(text, self.tools)
        with self.assertRaisesRegex(ToolParseError, "malformed tool arguments"):
            Glm47MoeDetector().detect_and_parse(text, self.tools)



def merge_tool_call_deltas(calls: list) -> dict:
    """
    Merge ToolCallItem deltas by tool_index (simulates client-side merge).

    This is what clients typically do - merge incremental deltas into complete tool calls.

    Args:
        calls: List of ToolCallItem from detector

    Returns:
        Dict mapping tool_index -> {"name": str, "parameters": str}
    """
    merged = {}
    for call in calls:
        idx = call.tool_index
        if idx not in merged:
            merged[idx] = {"name": None, "parameters": ""}

        if call.name:
            merged[idx]["name"] = call.name
        if call.parameters:
            merged[idx]["parameters"] += call.parameters

    return merged


class TestGlm47MoeDetectorExactIncrementalValues(unittest.TestCase):
    """
    Rigorous per-chunk tests with exact value assertions.

    These tests examine the exact return value of each parse_streaming_increment call
    to verify correct incremental behavior, including:
    1. Per-chunk: exact normal_text, number of calls, and call contents
    2. Merged: final tool call state after merging deltas by tool_index
    """

    def setUp(self):
        self.tools = create_basic_tools()

    def test_exact_values_simple_tool_call(self):
        """
        Verify that a simple tool call is committed atomically.

        Input chunks and expected outputs:
        | Chunk                          | normal_text | calls                                           |
        |--------------------------------|-------------|-------------------------------------------------|
        | block before closing tag       | ""          | []                                              |
        | </tool_call>                   | ""          | [name + complete canonical parameters]          |

        Merged: tool_index=0 -> {"name": "get_weather", "parameters": '{"city":"北京"}'}
        """
        detector = Glm47MoeDetector()

        # Define chunks with expected per-chunk results
        # Note: Some chunks may buffer without emitting (e.g., <arg_value> waits for value content)
        # Format: (chunk, expected_normal_text, min_calls, check_fn_or_none, description)
        test_cases = [
            (
                "<tool_call>get_weather<arg_key>",
                "",
                0,
                None,
                "Chunk 0: transaction remains private",
            ),
            (
                "city</arg_key>",
                "",
                0,
                None,
                "Chunk 1: transaction remains private",
            ),
            (
                "<arg_value>",
                "",
                0,  # May buffer (no emission required)
                None,
                "Chunk 2: <arg_value> tag may be buffered",
            ),
            (
                "北",
                "",
                0,  # May buffer or emit
                None,
                "Chunk 3: first value char",
            ),
            (
                "京",
                "",
                0,  # May buffer or emit
                None,
                "Chunk 4: second value char",
            ),
            (
                "</arg_value>",
                "",
                0,  # May buffer or emit
                None,
                "Chunk 5: value end tag",
            ),
            (
                "</tool_call>",
                "",
                1,
                lambda calls: (
                    len(calls) == 1
                    and calls[0].name == "get_weather"
                    and json.loads(calls[0].parameters) == {"city": "北京"}
                ),
                "Chunk 6: commit one complete tool call",
            ),
        ]

        all_calls = []
        for i, (chunk, exp_normal, min_calls, check_fn, msg) in enumerate(test_cases):
            result = detector.parse_streaming_increment(chunk, self.tools)

            # Assert exact normal_text
            self.assertEqual(
                result.normal_text,
                exp_normal,
                f"Chunk {i} '{chunk}': expected normal_text='{exp_normal}', got '{result.normal_text}'",
            )

            # Assert minimum calls
            self.assertGreaterEqual(
                len(result.calls),
                min_calls,
                f"Chunk {i} '{chunk}': expected at least {min_calls} calls, got {len(result.calls)}. {msg}",
            )

            # Assert check function if provided
            if check_fn is not None and result.calls:
                self.assertTrue(
                    check_fn(result.calls),
                    f"Chunk {i} '{chunk}': {msg}. Got calls: {result.calls}",
                )

            all_calls.extend(result.calls)

        # Verify merged result (client-side view)
        merged = merge_tool_call_deltas(all_calls)
        self.assertEqual(len(merged), 1, f"Expected 1 merged tool. Got: {merged}")
        self.assertIn(0, merged)
        self.assertEqual(merged[0]["name"], "get_weather")

        # Verify merged parameters form valid JSON
        parsed = json.loads(merged[0]["parameters"])
        self.assertEqual(parsed, {"city": "北京"})

    def test_exact_values_no_arg_function(self):
        """
        Verify exact values for no-argument function.

        Input chunks and expected outputs:
        | Chunk                | normal_text | calls                              |
        |----------------------|-------------|-----------------------------------|
        | <tool_call>get_time  | ""          | [] (buffering)                    |
        | </tool_call>         | ""          | [name='get_time'], [params='{}']  |

        Merged: tool_index=0 -> {"name": "get_time", "parameters": "{}"}
        """
        detector = Glm47MoeDetector()

        # Chunk 0: start + function name (buffering, no emission)
        result0 = detector.parse_streaming_increment("<tool_call>get_time", self.tools)
        self.assertEqual(
            result0.normal_text, "", "Chunk 0: normal_text should be empty"
        )
        self.assertEqual(len(result0.calls), 0, "Chunk 0: should buffer, no calls yet")

        # Chunk 1: end tag confirms no args, emits name and {}
        result1 = detector.parse_streaming_increment("</tool_call>", self.tools)
        self.assertEqual(
            result1.normal_text, "", "Chunk 1: normal_text should be empty"
        )
        self.assertGreaterEqual(
            len(result1.calls), 1, f"Chunk 1: should have calls. Got: {result1.calls}"
        )

        # Verify chunk 1 has name emission
        self.assertTrue(
            any(c.name == "get_time" for c in result1.calls),
            f"Chunk 1: should emit name='get_time'. Got: {result1.calls}",
        )

        # Verify merged result
        all_calls = list(result0.calls) + list(result1.calls)
        merged = merge_tool_call_deltas(all_calls)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["name"], "get_time")
        self.assertEqual(merged[0]["parameters"], "{}")

    def test_exact_values_multiple_args(self):
        """
        Verify atomic commit for a function with multiple arguments.

        Input chunks and expected outputs:
        | Chunk                                                          | calls contain          |
        |----------------------------------------------------------------|------------------------|
        | <tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo  | name, {, "city": "Tokyo|
        | </arg_value><arg_key>date</arg_key><arg_value>2024-01-01       | ", "date": "2024-01-01 |
        | </arg_value></tool_call>                                       | "}                     |

        Merged: {"name": "get_weather", "parameters": '{"city":"Tokyo","date":"2024-01-01"}'}
        """
        detector = Glm47MoeDetector()

        # Define chunks and per-chunk checks
        test_cases = [
            (
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo",
                "",
                lambda calls: not calls,
                "Chunk 0: transaction remains private",
            ),
            (
                "</arg_value><arg_key>date</arg_key><arg_value>2024-01-01",
                "",
                lambda calls: not calls,
                "Chunk 1: transaction remains private",
            ),
            (
                "</arg_value></tool_call>",
                "",
                lambda calls: (
                    len(calls) == 1
                    and calls[0].name == "get_weather"
                    and json.loads(calls[0].parameters)
                    == {"city": "Tokyo", "date": "2024-01-01"}
                ),
                "Chunk 2: should commit once",
            ),
        ]

        all_calls = []
        for i, (chunk, exp_normal, check_fn, msg) in enumerate(test_cases):
            result = detector.parse_streaming_increment(chunk, self.tools)
            self.assertEqual(
                result.normal_text, exp_normal, f"Chunk {i}: normal_text mismatch"
            )
            self.assertTrue(
                check_fn(result.calls), f"Chunk {i}: {msg}. Got: {result.calls}"
            )
            all_calls.extend(result.calls)

        # Verify merged result
        merged = merge_tool_call_deltas(all_calls)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["name"], "get_weather")

        parsed = json.loads(merged[0]["parameters"])
        self.assertEqual(parsed["city"], "Tokyo")
        self.assertEqual(parsed["date"], "2024-01-01")

    def test_exact_values_normal_text_handling(self):
        """
        Verify exact normal text handling before, during, and after tool calls.
        """
        detector = Glm47MoeDetector()

        # Chunk 0: Normal text only
        result0 = detector.parse_streaming_increment("Let me help: ", self.tools)
        self.assertEqual(result0.normal_text, "Let me help: ")
        self.assertEqual(len(result0.calls), 0)

        # Chunk 1: Tool call
        result1 = detector.parse_streaming_increment(
            "<tool_call>get_time</tool_call>", self.tools
        )
        # Normal text should be empty since we're inside tool call processing
        self.assertEqual(result1.normal_text, "")
        self.assertGreaterEqual(len(result1.calls), 1)

    def test_exact_values_mtp_complete_chunk(self):
        """
        Verify exact values when MTP delivers complete tool call in one chunk.
        """
        detector = Glm47MoeDetector()

        # Complete tool call in single chunk (MTP scenario)
        chunk = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Shanghai</arg_value>"
            "</tool_call>"
        )

        result = detector.parse_streaming_increment(chunk, self.tools)

        # Should have all components in calls
        self.assertGreaterEqual(
            len(result.calls), 1, f"Expected calls for MTP chunk. Got: {result.calls}"
        )

        # Verify name
        name_calls = [c for c in result.calls if c.name]
        self.assertEqual(len(name_calls), 1)
        self.assertEqual(name_calls[0].name, "get_weather")
        self.assertEqual(name_calls[0].tool_index, 0)

        # Verify arguments form valid JSON
        full_args = "".join(c.parameters for c in result.calls if c.parameters)
        parsed = json.loads(full_args)
        self.assertEqual(parsed, {"city": "Shanghai"})

    def test_exact_values_sequential_tools(self):
        """
        Verify exact tool_index values for sequential tool calls.

        Input chunks:
        | Chunk                                                              | tool_index in calls |
        |--------------------------------------------------------------------|---------------------|
        | <tool_call>get_weather<arg_key>city</arg_key><arg_value>A</arg_... | 0                   |
        | <tool_call>get_time</tool_call>                                    | 1                   |

        Merged:
        - tool_index=0 -> {"name": "get_weather", "parameters": '{"city":"A"}'}
        - tool_index=1 -> {"name": "get_time", "parameters": "{}"}
        """
        detector = Glm47MoeDetector()

        # First tool call - verify exact per-chunk values
        result1 = detector.parse_streaming_increment(
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>A</arg_value></tool_call>",
            self.tools,
        )
        self.assertEqual(
            result1.normal_text, "", "Chunk 0: normal_text should be empty"
        )
        self.assertGreaterEqual(
            len(result1.calls), 1, f"Chunk 0: should have calls. Got: {result1.calls}"
        )

        # Verify tool_index=0 for all calls in first chunk
        for call in result1.calls:
            self.assertEqual(
                call.tool_index,
                0,
                f"Chunk 0: all calls should have tool_index=0. Got: {call}",
            )

        # Verify name emission in first chunk
        self.assertTrue(
            any(c.name == "get_weather" for c in result1.calls),
            f"Chunk 0: should emit name='get_weather'. Got: {result1.calls}",
        )

        # Second tool call
        result2 = detector.parse_streaming_increment(
            "<tool_call>get_time</tool_call>",
            self.tools,
        )
        self.assertEqual(
            result2.normal_text, "", "Chunk 1: normal_text should be empty"
        )
        self.assertGreaterEqual(
            len(result2.calls), 1, f"Chunk 1: should have calls. Got: {result2.calls}"
        )

        # Verify tool_index=1 for all calls in second chunk
        for call in result2.calls:
            self.assertEqual(
                call.tool_index,
                1,
                f"Chunk 1: all calls should have tool_index=1. Got: {call}",
            )

        # Verify name emission in second chunk
        self.assertTrue(
            any(c.name == "get_time" for c in result2.calls),
            f"Chunk 1: should emit name='get_time'. Got: {result2.calls}",
        )

        # Verify merged result - 2 distinct tools
        all_calls = list(result1.calls) + list(result2.calls)
        merged = merge_tool_call_deltas(all_calls)

        self.assertEqual(len(merged), 2, f"Should have 2 merged tools. Got: {merged}")
        self.assertEqual(merged[0]["name"], "get_weather")
        self.assertEqual(merged[1]["name"], "get_time")

        # Verify merged parameters
        parsed0 = json.loads(merged[0]["parameters"])
        self.assertEqual(parsed0, {"city": "A"})
        self.assertEqual(merged[1]["parameters"], "{}")

    def test_exact_values_partial_tag_buffering(self):
        """
        Verify partial tag buffering behavior.

        When we receive '<' alone, it could be start of <tool_call>.
        It should be buffered, not emitted as normal text.
        """
        detector = Glm47MoeDetector()

        # Single '<' should be buffered
        result0 = detector.parse_streaming_increment("<", self.tools)
        self.assertEqual(result0.normal_text, "")
        self.assertEqual(len(result0.calls), 0)

        # 't' continues potential tag
        result1 = detector.parse_streaming_increment("t", self.tools)
        self.assertEqual(result1.normal_text, "")
        self.assertEqual(len(result1.calls), 0)

        # 'x' breaks the pattern - should release buffered content
        result2 = detector.parse_streaming_increment("x", self.tools)
        self.assertIn("<tx", result2.normal_text)
        self.assertEqual(len(result2.calls), 0)

    def test_exact_values_character_by_character_value(self):
        """
        Verify character-by-character input remains private until commit.
        """
        detector = Glm47MoeDetector()

        # Set up an incomplete transaction.
        result0 = detector.parse_streaming_increment(
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>", self.tools
        )

        self.assertEqual(result0.calls, [])

        # Now stream value character by character
        chars = ["A", "B", "C"]
        char_results = []
        for char in chars:
            result = detector.parse_streaming_increment(char, self.tools)
            char_results.append(result)

            self.assertEqual(result.calls, [])

        # Finalize
        result_end = detector.parse_streaming_increment(
            "</arg_value></tool_call>", self.tools
        )
        self.assertEqual(len(result_end.calls), 1)

        # Aggregate all arguments
        all_calls = list(result0.calls)
        for r in char_results:
            all_calls.extend(r.calls)
        all_calls.extend(result_end.calls)

        full_args = "".join(c.parameters for c in all_calls if c.parameters)
        parsed = json.loads(full_args)
        self.assertEqual(parsed["city"], "ABC")


if __name__ == "__main__":
    unittest.main()
