import json
import os
import unittest
from unittest.mock import patch

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.glm4_moe_detector import (
    Glm4MoeDetector,
)


def create_tools():
    """Create test tool definitions."""
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get the weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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


class TestGlm4MoeDetector(unittest.TestCase):
    """Test Glm4MoeDetector with various GLM-4 and GLM-4.7 formats."""

    def setUp(self):
        self.detector = Glm4MoeDetector()
        self.tools = create_tools()

    # ========== With Args Tests ==========

    def test_with_args_newline_separator(self):
        """GLM-4 style: function name and args separated by actual newline."""
        text = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>杭州</arg_value>\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)

        expected_name = "get_weather"
        expected_params = '"city": "杭州"'
        self.assertEqual(
            len(result.calls), 1, f"expected 1 call, actual {len(result.calls)}"
        )
        self.assertEqual(
            result.calls[0].name,
            expected_name,
            f"expected {expected_name}, actual {result.calls[0].name}",
        )
        self.assertIn(
            expected_params,
            result.calls[0].parameters,
            f"expected {expected_params} in actual {result.calls[0].parameters}",
        )

    def test_with_args_literal_newline_separator(self):
        """GLM-4 style: function name and args separated by literal \\n."""
        text = "<tool_call>get_weather\\n<arg_key>city</arg_key>\\n<arg_value>杭州</arg_value>\\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)

        expected_name = "get_weather"
        expected_params = '"city": "杭州"'
        self.assertEqual(
            len(result.calls), 1, f"expected 1 call, actual {len(result.calls)}"
        )
        self.assertEqual(
            result.calls[0].name,
            expected_name,
            f"expected {expected_name}, actual {result.calls[0].name}",
        )
        self.assertIn(
            expected_params,
            result.calls[0].parameters,
            f"expected {expected_params} in actual {result.calls[0].parameters}",
        )

    def test_with_args_no_separator(self):
        """GLM-4.7 style: no separator between function name and args."""
        text = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>杭州</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)

        expected_name = "get_weather"
        expected_params = '"city": "杭州"'
        self.assertEqual(
            len(result.calls), 1, f"expected 1 call, actual {len(result.calls)}"
        )
        self.assertEqual(
            result.calls[0].name,
            expected_name,
            f"expected {expected_name}, actual {result.calls[0].name}",
        )
        self.assertIn(
            expected_params,
            result.calls[0].parameters,
            f"expected {expected_params} in actual {result.calls[0].parameters}",
        )

    # ========== Without Args Tests ==========

    def test_no_args_newline_separator(self):
        """GLM-4 style: no args, with trailing newline."""
        text = "<tool_call>get_time\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            len(result.calls), 1, f"Expected 1 call, got {len(result.calls)}"
        )
        self.assertEqual(
            result.calls[0].name,
            "get_time",
            f"Expected get_time, got {result.calls[0].name}",
        )
        self.assertEqual(
            result.calls[0].parameters,
            "{}",
            f"Expected {{}}, got {result.calls[0].parameters}",
        )

    def test_no_args_literal_newline_separator(self):
        """GLM-4 style: no args, with literal \\n."""
        text = "<tool_call>get_time\\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            len(result.calls), 1, f"Expected 1 call, got {len(result.calls)}"
        )
        self.assertEqual(
            result.calls[0].name,
            "get_time",
            f"Expected get_time, got {result.calls[0].name}",
        )
        self.assertEqual(
            result.calls[0].parameters,
            "{}",
            f"Expected {{}}, got {result.calls[0].parameters}",
        )

    def test_no_args_no_separator(self):
        """GLM-4.7 style: no args, no separator."""
        text = "<tool_call>get_time</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            len(result.calls), 1, f"Expected 1 call, got {len(result.calls)}"
        )
        self.assertEqual(
            result.calls[0].name,
            "get_time",
            f"Expected get_time, got {result.calls[0].name}",
        )
        self.assertEqual(
            result.calls[0].parameters,
            "{}",
            f"Expected {{}}, got {result.calls[0].parameters}",
        )

    # ========== Multiple Args Tests ==========

    def test_multiple_args_newline_separator(self):
        """GLM-4 style: multiple args with newlines."""
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>杭州</arg_value>\n"
            "<arg_key>unit</arg_key>\n<arg_value>celsius</arg_value>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = result.calls[0].parameters
        self.assertIn('"city": "杭州"', params, f"Expected city in {params}")
        self.assertIn('"unit": "celsius"', params, f"Expected unit in {params}")

    def test_multiple_args_no_separator(self):
        """GLM-4.7 style: multiple args without separators."""
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>杭州</arg_value>"
            "<arg_key>unit</arg_key><arg_value>celsius</arg_value>"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = result.calls[0].parameters
        self.assertIn('"city": "杭州"', params, f"Expected city in {params}")
        self.assertIn('"unit": "celsius"', params, f"Expected unit in {params}")

    # ========== Multiple Tool Calls Tests ==========

    def test_multiple_tool_calls(self):
        """Test multiple tool calls in one text."""
        text = (
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>杭州</arg_value></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            len(result.calls), 2, f"Expected 2 calls, got {len(result.calls)}"
        )
        self.assertIn('"city": "杭州"', result.calls[0].parameters)
        self.assertIn('"city": "北京"', result.calls[1].parameters)

    def test_streaming_final_chunk_drains_all_tool_calls_and_trailing_text(self):
        text = (
            "准备查询"
            "<tool_call>get_time</tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>杭州</arg_value></tool_call>"
            "查询已提交"
        )

        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "准备查询查询已提交")
        self.assertEqual(
            [call.name for call in result.calls], ["get_time", "get_weather"]
        )
        self.assertEqual([call.tool_index for call in result.calls], [0, 1])
        self.assertEqual(json.loads(result.calls[0].parameters), {})
        self.assertEqual(json.loads(result.calls[1].parameters), {"city": "杭州"})
        self.assertEqual(self.detector._buffer, "")

    def test_streaming_multiple_tool_calls_are_independent_of_chunk_boundary(self):
        text = (
            "前缀"
            "<tool_call>get_time</tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>杭州</arg_value></tool_call>"
            "后缀"
        )

        for split in range(len(text) + 1):
            with self.subTest(split=split):
                detector = Glm4MoeDetector()
                results = [
                    detector.parse_streaming_increment(text[:split], self.tools),
                    detector.parse_streaming_increment(text[split:], self.tools),
                ]
                calls = [call for result in results for call in result.calls]

                self.assertEqual(
                    "".join(result.normal_text for result in results), "前缀后缀"
                )
                self.assertEqual(
                    [call.name for call in calls], ["get_time", "get_weather"]
                )
                self.assertEqual([call.tool_index for call in calls], [0, 1])
                self.assertEqual(json.loads(calls[0].parameters), {})
                self.assertEqual(json.loads(calls[1].parameters), {"city": "杭州"})
                self.assertEqual(detector._buffer, "")

    def test_streaming_preserves_whitespace_around_multiple_tool_calls(self):
        text = (
            "  前文\n"
            "<tool_call>get_time</tool_call>"
            "\n  中间正文  \n"
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>杭州</arg_value></tool_call>"
            "\n尾文  "
        )

        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "  前文\n\n  中间正文  \n\n尾文  ")
        self.assertEqual(
            [call.name for call in result.calls], ["get_time", "get_weather"]
        )
        self.assertEqual(self.detector._buffer, "")

    def test_streaming_releases_partial_tool_prefix_that_diverges(self):
        first = self.detector.parse_streaming_increment("abc<tool_", self.tools)
        second = self.detector.parse_streaming_increment("x", self.tools)

        self.assertEqual(first.normal_text, "abc")
        self.assertEqual(second.normal_text, "<tool_x")
        self.assertEqual(self.detector._buffer, "")

    def test_streaming_invalid_closed_blocks_do_not_consume_tool_indices(self):
        text = (
            "<tool_call>get_time</tool_call>"
            "<tool_call>missing_tool</tool_call>"
            "<tool_call></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>杭州</arg_value></tool_call>"
        )

        with patch.dict(os.environ, {"RTP_LLM_FORWARD_UNKNOWN_TOOLS": ""}):
            result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(
            [call.name for call in result.calls], ["get_time", "get_weather"]
        )
        self.assertEqual([call.tool_index for call in result.calls], [0, 1])
        self.assertEqual(self.detector.current_tool_id, 2)
        self.assertEqual(
            self.detector.prev_tool_call_arr,
            [
                {"name": "get_time", "arguments": {}},
                {"name": "get_weather", "arguments": {"city": "杭州"}},
            ],
        )
        self.assertEqual(
            [json.loads(value) for value in self.detector.streamed_args_for_tool],
            [{}, {"city": "杭州"}],
        )
        self.assertEqual(self.detector._buffer, "")

    def test_streaming_forwarded_unknown_tool_uses_a_contiguous_index(self):
        text = (
            "<tool_call>get_time</tool_call>"
            "<tool_call>custom_tool</tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>杭州</arg_value></tool_call>"
        )

        with patch.dict(os.environ, {"RTP_LLM_FORWARD_UNKNOWN_TOOLS": "true"}):
            result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(
            [call.name for call in result.calls],
            ["get_time", "custom_tool", "get_weather"],
        )
        self.assertEqual([call.tool_index for call in result.calls], [0, 1, 2])
        self.assertEqual(self.detector.current_tool_id, 3)
        self.assertEqual(len(self.detector.prev_tool_call_arr), 3)
        self.assertEqual(len(self.detector.streamed_args_for_tool), 3)
        self.assertEqual(self.detector._buffer, "")

    def test_streaming_isolated_end_tag_before_tool_call_is_normal_text(self):
        text = "literal</tool_call>text<tool_call>get_time</tool_call>"

        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "literal</tool_call>text")
        self.assertEqual([call.name for call in result.calls], ["get_time"])
        self.assertEqual([call.tool_index for call in result.calls], [0])
        self.assertEqual(self.detector._buffer, "")

    # ========== Normal Text Tests ==========

    def test_normal_text_before_tool_call(self):
        """Test that normal text before tool call is preserved."""
        text = "让我查询天气<tool_call>get_weather<arg_key>city</arg_key><arg_value>杭州</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            result.normal_text,
            "让我查询天气",
            f"Expected '让我查询天气', got '{result.normal_text}'",
        )
        self.assertEqual(len(result.calls), 1)

    def test_no_tool_call(self):
        """Test text without any tool call."""
        text = "这是普通文本，没有工具调用"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            result.normal_text, text, f"Expected '{text}', got '{result.normal_text}'"
        )
        self.assertEqual(
            len(result.calls), 0, f"Expected 0 calls, got {len(result.calls)}"
        )

    # ========== has_tool_call Tests ==========

    def test_has_tool_call_true(self):
        """Test has_tool_call returns True when tool call exists."""
        text = "<tool_call>get_weather</tool_call>"
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        """Test has_tool_call returns False when no tool call."""
        text = "这是普通文本"
        self.assertFalse(self.detector.has_tool_call(text))


if __name__ == "__main__":
    unittest.main()
