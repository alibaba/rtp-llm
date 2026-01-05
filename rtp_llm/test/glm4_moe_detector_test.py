import unittest

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
