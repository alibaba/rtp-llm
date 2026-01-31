"""
Comprehensive test suite for Qwen3CoderDetector with MTP (Multiple Tokens Per chunk) support.

Tests verify:
1. Lookbehind mechanism for stripping newlines between tool calls
2. MTP compatibility (content and tool_call in same chunk)
3. Various chunk boundary scenarios
"""

import json
import unittest
from typing import List

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen3_coder_detector import (
    Qwen3CoderDetector,
)


class TestQwen3CoderDetectorMTP(unittest.TestCase):
    """Test MTP (Multiple Tokens Per chunk) support and lookbehind mechanism."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = Qwen3CoderDetector()

        # Define sample tools for testing
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search for information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                ),
            ),
        ]

    def _parse_chunks(self, chunks: List[str]) -> tuple:
        """
        Helper to parse a sequence of chunks and return all results.
        Returns (normal_texts, tool_calls) where each is a list.
        """
        detector = Qwen3CoderDetector()
        normal_texts = []
        tool_calls = []

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            if result.normal_text:
                normal_texts.append(result.normal_text)
            if result.calls:
                tool_calls.extend(result.calls)

        return normal_texts, tool_calls

    def test_case1_newline_at_boundaries(self):
        """
        Test Case 1: "xxx\n<tool_call>", ..., "</tool_call>\n"
        Newline before tool call is part of normal content and should be preserved.
        """
        chunks = [
            "Here is the weather info\n<tool_call>",
            "<function=get_weather>",
            "<parameter=location>",
            "\nSan Francisco",
            "\n</parameter>",
            "<parameter=unit>",
            "\ncelsius",
            "\n</parameter>",
            "\n</function>",
            "</tool_call>\n",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # Should have normal text with the newline before tool call preserved
        self.assertEqual("".join(normal_texts), "Here is the weather info\n")

        # Should have exactly 1 tool call with complete parameters
        # Count unique tool indices
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(len(tool_indices), 1)

        # Reconstruct the tool call
        name_calls = [tc for tc in tool_calls if tc.name]
        param_calls = [tc for tc in tool_calls if tc.parameters]

        self.assertEqual(len(name_calls), 1)
        self.assertEqual(name_calls[0].name, "get_weather")

        # Reconstruct parameters
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)
        self.assertEqual(params["location"], "San Francisco")
        self.assertEqual(params["unit"], "celsius")

    def test_case2_newline_as_separate_chunks(self):
        """
        Test Case 2: "xxx", "\n", "<tool_call>", ..., "</tool_call>", "\n"
        Newlines arrive as separate chunks between content and tool call.
        The newline before the first tool call is part of content formatting.
        """
        chunks = [
            "The answer is: ",
            "\n",
            "<tool_call>",
            "<function=search>",
            "<parameter=query>",
            "\nPython tutorial",
            "\n</parameter>",
            "<parameter=limit>",
            "\n5",
            "\n</parameter>",
            "\n</function>",
            "</tool_call>",
            "\n",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # Normal text preserves the newline before tool call (part of content formatting)
        self.assertEqual("".join(normal_texts), "The answer is: \n")

        # Verify tool call
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(len(tool_indices), 1)

        name_calls = [tc for tc in tool_calls if tc.name]
        self.assertEqual(name_calls[0].name, "search")

        param_calls = [tc for tc in tool_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)
        self.assertEqual(params["query"], "Python tutorial")
        self.assertEqual(params["limit"], 5)

    def test_case3_single_chunk_with_newlines(self):
        """
        Test Case 3: "xxx\n<tool_call>...</tool_call>\n"
        Complete tool call with newlines in a single or few chunks.
        """
        chunks = [
            "Response:\n<tool_call><function=get_weather><parameter=location>\nTokyo\n</parameter><parameter=unit>\nfahrenheit\n</parameter>\n</function></tool_call>\n",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # Normal text should preserve the newline before tool call
        self.assertEqual("".join(normal_texts), "Response:\n")

        # Verify tool call
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(len(tool_indices), 1)

        name_calls = [tc for tc in tool_calls if tc.name]
        self.assertEqual(name_calls[0].name, "get_weather")

        param_calls = [tc for tc in tool_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)
        self.assertEqual(params["location"], "Tokyo")
        self.assertEqual(params["unit"], "fahrenheit")

    def test_case4_long_arguments_multiple_chunks(self):
        """
        Test Case 4: "xxx", "\n", "<tool_call>", ... multiple chunks with long arguments, "</tool_call>", "\n"
        Long parameter values split across multiple chunks.
        """
        long_query = "How to implement machine learning algorithms in Python with detailed examples"

        chunks = [
            "Search results:\n",
            "<tool_call>",
            "<function=search>",
            "<parameter=query>\n",
            long_query[:20],  # Split long query into chunks
            long_query[20:40],
            long_query[40:],
            "\n</parameter>",
            "<parameter=limit>\n",
            "10",
            "\n</parameter>",
            "\n</function>",
            "</tool_call>",
            "\n",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # Normal text should preserve the newline before tool call
        self.assertEqual("".join(normal_texts), "Search results:\n")

        # Verify tool call
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(len(tool_indices), 1, "Should have exactly 1 unique tool call")

        name_calls = [tc for tc in tool_calls if tc.name]
        self.assertEqual(len(name_calls), 1, "Should have exactly 1 name call")
        self.assertEqual(name_calls[0].name, "search")

        param_calls = [tc for tc in tool_calls if tc.parameters]
        self.assertGreater(len(param_calls), 0, "Should have parameter calls")
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)
        self.assertEqual(params["query"], long_query)
        self.assertEqual(params["limit"], 10)
        self.assertIsInstance(params["limit"], int, "limit should be integer type")

    def test_mtp_content_and_tool_call_in_same_chunk(self):
        """
        Test MTP: "content<tool_call>"
        Content and tool call start in the same chunk.
        """
        chunks = [
            "Here's what I found:<tool_call><function=get_weather>",
            "<parameter=location>\nLondon\n</parameter>",
            "</function></tool_call>",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # Normal text should be "Here's what I found:"
        self.assertEqual("".join(normal_texts), "Here's what I found:")

        # Verify tool call
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(len(tool_indices), 1)

        name_calls = [tc for tc in tool_calls if tc.name]
        self.assertEqual(name_calls[0].name, "get_weather")

        param_calls = [tc for tc in tool_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)
        self.assertEqual(params["location"], "London")

    def test_mtp_tool_call_end_and_content_in_same_chunk(self):
        """
        Test MTP: "tool_arguments</tool_call>content"
        Tool call end and content in the same chunk.
        """
        chunks = [
            "<tool_call><function=search>",
            "<parameter=query>\ntest\n</parameter>",
            "</function></tool_call> and here's more text",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # Normal text should be " and here's more text"
        self.assertEqual("".join(normal_texts), " and here's more text")

        # Verify tool call
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(len(tool_indices), 1)

        name_calls = [tc for tc in tool_calls if tc.name]
        self.assertEqual(name_calls[0].name, "search")

        param_calls = [tc for tc in tool_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)
        self.assertEqual(params["query"], "test")

    def test_multiple_serial_tool_calls_with_newline_stripping(self):
        """
        Test newline stripping between serial tool calls.
        Ensures newlines between </tool_call> and <tool_call> are stripped properly.
        This is the core requirement from the user.
        """
        # Test scenario 1: "first tool call</tool_call>", "\n", "<tool_call>second tool call"
        chunks = [
            "Results:\n",
            "<tool_call><function=get_weather><parameter=location>\nParis\n</parameter></function></tool_call>",
            "\n",  # This newline should be stripped
            "<tool_call><function=search><parameter=query>\nEiffel Tower\n</parameter></function></tool_call>",
            "\n",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # Normal text should be "Results:\n" (preserve newline before first tool call,
        # but NO newlines between tool calls)
        self.assertEqual("".join(normal_texts), "Results:\n")

        # Verify we have 2 tool calls
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(
            len(tool_indices), 2, "Should have exactly 2 unique tool calls"
        )

        name_calls = [tc for tc in tool_calls if tc.name]
        self.assertEqual(len(name_calls), 2, "Should have exactly 2 name calls")
        self.assertEqual(name_calls[0].name, "get_weather")
        self.assertEqual(name_calls[1].name, "search")

        # Verify parameters for both tool calls
        tool0_params = [tc for tc in tool_calls if tc.tool_index == 0 and tc.parameters]
        tool1_params = [tc for tc in tool_calls if tc.tool_index == 1 and tc.parameters]

        self.assertGreater(len(tool0_params), 0, "First tool should have parameters")
        self.assertGreater(len(tool1_params), 0, "Second tool should have parameters")

        params0 = json.loads("".join(tc.parameters for tc in tool0_params))
        params1 = json.loads("".join(tc.parameters for tc in tool1_params))

        self.assertEqual(params0["location"], "Paris")
        self.assertEqual(params1["query"], "Eiffel Tower")

    def test_mtp_complex_interleaving(self):
        """
        Test complex MTP scenario with content and tool calls interleaved.
        """
        chunks = [
            "Let me help you with that.\n<tool_call>",
            "<function=get_weather><parameter=location>\nBerlin\n</parameter>",
            "</function></tool_call>\nBased on the weather, ",
            "<tool_call><function=search><parameter=query>\nBerlin attractions\n</parameter>",
            "</function></tool_call> you should visit these places.",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # Normal text should include all the text parts
        combined_text = "".join(normal_texts)
        self.assertIn("Let me help you with that.", combined_text)
        self.assertIn("Based on the weather,", combined_text)
        self.assertIn("you should visit these places.", combined_text)

        # Verify we have 2 tool calls
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(
            len(tool_indices), 2, "Should have exactly 2 unique tool calls"
        )

        name_calls = [tc for tc in tool_calls if tc.name]
        self.assertEqual(len(name_calls), 2, "Should have exactly 2 name calls")
        self.assertEqual(name_calls[0].name, "get_weather")
        self.assertEqual(name_calls[1].name, "search")

        # Verify parameters for both tool calls
        tool0_params = [tc for tc in tool_calls if tc.tool_index == 0 and tc.parameters]
        tool1_params = [tc for tc in tool_calls if tc.tool_index == 1 and tc.parameters]

        params0 = json.loads("".join(tc.parameters for tc in tool0_params))
        params1 = json.loads("".join(tc.parameters for tc in tool1_params))

        self.assertEqual(params0["location"], "Berlin")
        self.assertEqual(params1["query"], "Berlin attractions")

    def test_empty_and_whitespace_chunks(self):
        """
        Test handling of empty chunks and whitespace-only chunks.
        """
        chunks = [
            "Start",
            "",  # empty chunk
            " ",  # whitespace chunk
            "\n",
            "<tool_call><function=search><parameter=query>\ntest\n</parameter></function></tool_call>",
            "",  # empty chunk
            "\n",
            "End",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # Should handle empty chunks gracefully
        combined_text = "".join(normal_texts)
        self.assertIn("Start", combined_text)
        self.assertIn("End", combined_text)

        # Verify tool call
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(len(tool_indices), 1)

    def test_parameter_without_newlines(self):
        """
        Test parameters without surrounding newlines (edge case).
        """
        chunks = [
            "<tool_call><function=get_weather>",
            "<parameter=location>Tokyo</parameter>",
            "<parameter=unit>celsius</parameter>",
            "</function></tool_call>",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # No normal text expected
        self.assertEqual("".join(normal_texts), "")

        # Verify tool call
        name_calls = [tc for tc in tool_calls if tc.name]
        self.assertEqual(name_calls[0].name, "get_weather")

        param_calls = [tc for tc in tool_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)
        self.assertEqual(params["location"], "Tokyo")
        self.assertEqual(params["unit"], "celsius")

    def test_lookbehind_only_strips_after_tool_call(self):
        """
        Test that lookbehind only strips newlines after tool calls, not regular content.
        """
        chunks = [
            "Line 1\n",
            "Line 2\n",
            "Line 3",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # All newlines should be preserved in regular content
        combined_text = "".join(normal_texts)
        self.assertEqual(combined_text, "Line 1\nLine 2\nLine 3")

        # No tool calls
        self.assertEqual(len(tool_calls), 0)

    def test_scenario1_newline_in_separate_chunk(self):
        """
        User scenario 1: "first tool call</tool_call>", "\n", "<tool_call>second tool call"
        The newline arrives as a separate chunk between two tool calls.
        """
        chunks = [
            "<tool_call><function=get_weather><parameter=location>\nParis\n</parameter></function></tool_call>",
            "\n",  # Separate newline chunk - should be stripped
            "<tool_call><function=search><parameter=query>\ntest\n</parameter></function></tool_call>",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # No normal text - the newline between tool calls should be stripped
        self.assertEqual("".join(normal_texts), "", "No normal text should be present")

        # Verify 2 tool calls
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(
            len(tool_indices), 2, "Should have exactly 2 unique tool calls"
        )

        # Verify both tool calls have valid parameters
        tool0_params = [tc for tc in tool_calls if tc.tool_index == 0 and tc.parameters]
        tool1_params = [tc for tc in tool_calls if tc.tool_index == 1 and tc.parameters]

        params0 = json.loads("".join(tc.parameters for tc in tool0_params))
        params1 = json.loads("".join(tc.parameters for tc in tool1_params))

        self.assertEqual(params0["location"], "Paris")
        self.assertEqual(params1["query"], "test")

    def test_scenario2_newline_with_first_tool_call(self):
        """
        User scenario 2: "first tool call</tool_call>\n", "<tool_call>second tool call"
        The newline is appended to the first tool call chunk.
        """
        chunks = [
            "<tool_call><function=get_weather><parameter=location>\nParis\n</parameter></function></tool_call>\n",
            "<tool_call><function=search><parameter=query>\ntest\n</parameter></function></tool_call>",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # No normal text - the newline after first tool call should be stripped
        self.assertEqual("".join(normal_texts), "", "No normal text should be present")

        # Verify 2 tool calls
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(
            len(tool_indices), 2, "Should have exactly 2 unique tool calls"
        )

        # Verify both tool calls have valid parameters
        tool0_params = [tc for tc in tool_calls if tc.tool_index == 0 and tc.parameters]
        tool1_params = [tc for tc in tool_calls if tc.tool_index == 1 and tc.parameters]

        params0 = json.loads("".join(tc.parameters for tc in tool0_params))
        params1 = json.loads("".join(tc.parameters for tc in tool1_params))

        self.assertEqual(params0["location"], "Paris")
        self.assertEqual(params1["query"], "test")

    def test_scenario3_newline_with_second_tool_call(self):
        """
        User scenario 3: "first tool call</tool_call>", "\n<tool_call>second tool call"
        The newline is prepended to the second tool call chunk.
        """
        chunks = [
            "<tool_call><function=get_weather><parameter=location>\nParis\n</parameter></function></tool_call>",
            "\n<tool_call><function=search><parameter=query>\ntest\n</parameter></function></tool_call>",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # No normal text - the newline before second tool call should be stripped
        self.assertEqual("".join(normal_texts), "", "No normal text should be present")

        # Verify 2 tool calls
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(
            len(tool_indices), 2, "Should have exactly 2 unique tool calls"
        )

        # Verify both tool calls have valid parameters
        tool0_params = [tc for tc in tool_calls if tc.tool_index == 0 and tc.parameters]
        tool1_params = [tc for tc in tool_calls if tc.tool_index == 1 and tc.parameters]

        params0 = json.loads("".join(tc.parameters for tc in tool0_params))
        params1 = json.loads("".join(tc.parameters for tc in tool1_params))

        self.assertEqual(params0["location"], "Paris")
        self.assertEqual(params1["query"], "test")

    def test_tool_call_end_with_real_content(self):
        """
        Test that </tool_call>\n with real content after preserves the content.
        The user said they're OK either way with the newline, so we just verify content is kept.
        """
        chunks = [
            "<tool_call><function=get_weather><parameter=location>\nParis\n</parameter></function></tool_call>\n",
            "Here is the weather forecast.",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # The real content should be preserved
        combined_text = "".join(normal_texts)
        self.assertIn("Here is the weather forecast.", combined_text)

        # Verify 1 tool call
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(len(tool_indices), 1)

    def test_incremental_string_parameter_streaming(self):
        """
        Test incremental streaming of string parameter values.
        Long string values should be streamed incrementally as chunks arrive.
        """
        # Simulate: <parameter=location>\n/Users/dingyang/Develop\n</parameter>
        chunks = [
            "<tool_call><function=get_weather>",
            "<parameter=location>\n/Users",
            "/dingyang/",
            "Develop",
            "\n</parameter>",
            "</function></tool_call>",
        ]

        detector = Qwen3CoderDetector()
        all_calls = []

        # Chunk 0: <tool_call><function=get_weather>
        result = detector.parse_streaming_increment(chunks[0], self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].parameters, "")
        all_calls.extend(result.calls)

        # Chunk 1: <parameter=location>\n/Users
        result = detector.parse_streaming_increment(chunks[1], self.tools)
        self.assertEqual(len(result.calls), 3)
        self.assertEqual(result.calls[0].parameters, "{")  # Open object
        self.assertEqual(
            result.calls[1].parameters, '"location": "'
        )  # Key with opening quote
        self.assertEqual(result.calls[2].parameters, "/Users")  # First content chunk
        all_calls.extend(result.calls)

        # Chunk 2: /dingyang/
        result = detector.parse_streaming_increment(chunks[2], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            result.calls[0].parameters, "/dingyang/"
        )  # Second content chunk
        all_calls.extend(result.calls)

        # Chunk 3: Develop
        result = detector.parse_streaming_increment(chunks[3], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "Develop")  # Third content chunk
        all_calls.extend(result.calls)

        # Chunk 4: \n</parameter>
        result = detector.parse_streaming_increment(chunks[4], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            result.calls[0].parameters, '"'
        )  # Closing quote (no more content)
        all_calls.extend(result.calls)

        # Chunk 5: </function></tool_call>
        result = detector.parse_streaming_increment(chunks[5], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "}")  # Close object
        all_calls.extend(result.calls)

        # Verify final structure
        param_calls = [tc for tc in all_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        self.assertEqual(params_str, '{"location": "/Users/dingyang/Develop"}')
        params = json.loads(params_str)
        self.assertEqual(params["location"], "/Users/dingyang/Develop")

    def test_incremental_string_parameter_with_special_chars(self):
        """
        Test incremental streaming handles JSON escaping correctly.
        """
        chunks = [
            "<tool_call><function=search>",
            '<parameter=query>\n"Hello',
            ' World"\n\n',
            "\nLine2\n",
            "</parameter>",
            "</function></tool_call>",
        ]

        detector = Qwen3CoderDetector()
        all_calls = []

        # Chunk 0
        result = detector.parse_streaming_increment(chunks[0], self.tools)
        self.assertEqual(result.calls[0].name, "search")
        all_calls.extend(result.calls)

        # Chunk 1: <parameter=query>\n"Hello
        result = detector.parse_streaming_increment(chunks[1], self.tools)
        self.assertEqual(len(result.calls), 3)
        self.assertEqual(result.calls[0].parameters, "{")
        self.assertEqual(result.calls[1].parameters, '"query": "')
        self.assertEqual(
            result.calls[2].parameters, '\\"Hello'
        )  # Escaped quote + Hello
        all_calls.extend(result.calls)

        # Chunk 2:  World"\n
        result = detector.parse_streaming_increment(chunks[2], self.tools)
        self.assertEqual(len(result.calls), 1)
        # Should have space, World, escaped quote, and newline
        self.assertIn('\\"', result.calls[0].parameters)
        self.assertIn("World", result.calls[0].parameters)
        all_calls.extend(result.calls)

        # Chunk 3: Line2
        result = detector.parse_streaming_increment(chunks[3], self.tools)
        self.assertEqual(len(result.calls), 2, f"result: {result}")
        self.assertEqual(result.calls[0].parameters, "\\n\\n", f"result: {result}")
        self.assertEqual(result.calls[1].parameters, "\\nLine2", f"result: {result}")
        all_calls.extend(result.calls)

        # Chunk 4: \n</parameter>
        result = detector.parse_streaming_increment(chunks[4], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, '"')  # Just closing quote
        all_calls.extend(result.calls)

        # Chunk 5: </function></tool_call>
        result = detector.parse_streaming_increment(chunks[5], self.tools)
        self.assertEqual(result.calls[0].parameters, "}")
        all_calls.extend(result.calls)

        # Verify final structure and correct unescaping
        param_calls = [tc for tc in all_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)
        self.assertEqual(params["query"], '"Hello World"\n\n\nLine2')

    def test_non_string_parameter_no_streaming(self):
        """
        Test that non-string parameters (integer, etc.) are NOT streamed incrementally.
        They should be emitted as complete values.
        """
        chunks = [
            "<tool_call><function=search>",
            "<parameter=limit>\n",
            "10",
            "\n</parameter>",
            "</function></tool_call>",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # Verify no incremental streaming for integer
        param_calls = [tc for tc in tool_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)

        # Integer should be emitted as number, not streamed string
        self.assertEqual(params["limit"], 10)
        self.assertIsInstance(params["limit"], int)

    def test_mixed_streaming_and_non_streaming_parameters(self):
        """
        Test tool call with both string (streamed) and non-string (not streamed) parameters.
        """
        chunks = [
            "<tool_call><function=search>",
            "<parameter=query>\nlong ",
            "search ",
            "query",
            "\n</parameter>",
            "<parameter=limit>\n5\n</parameter>",
            "</function></tool_call>",
        ]

        detector = Qwen3CoderDetector()
        all_calls = []

        # Chunk 0
        result = detector.parse_streaming_increment(chunks[0], self.tools)
        all_calls.extend(result.calls)

        # Chunk 1: String param starts streaming
        result = detector.parse_streaming_increment(chunks[1], self.tools)
        self.assertEqual(len(result.calls), 3)
        self.assertEqual(result.calls[0].parameters, "{")
        self.assertEqual(result.calls[1].parameters, '"query": "')
        self.assertEqual(result.calls[2].parameters, "long ")  # First content
        all_calls.extend(result.calls)

        # Chunk 2: String param continues streaming
        result = detector.parse_streaming_increment(chunks[2], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "search ")
        all_calls.extend(result.calls)

        # Chunk 3: String param continues streaming
        result = detector.parse_streaming_increment(chunks[3], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "query")
        all_calls.extend(result.calls)

        # Chunk 4: String param completes
        result = detector.parse_streaming_increment(chunks[4], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, '"')  # Just closing quote
        all_calls.extend(result.calls)

        # Chunk 5: Integer param - NOT streamed, emitted as complete value
        result = detector.parse_streaming_increment(chunks[5], self.tools)
        self.assertEqual(len(result.calls), 1)
        # Should be complete key-value pair with comma, not streamed
        self.assertIn('"limit"', result.calls[0].parameters)
        self.assertIn("5", result.calls[0].parameters)
        self.assertNotIn(
            '"', result.calls[0].parameters.replace('"limit"', "").replace(": ", "")
        )  # Value not quoted
        all_calls.extend(result.calls)

        # Chunk 6
        result = detector.parse_streaming_increment(chunks[6], self.tools)
        self.assertEqual(result.calls[0].parameters, "}")
        all_calls.extend(result.calls)

        # Verify final structure
        param_calls = [tc for tc in all_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)

        self.assertEqual(params["query"], "long search query")
        self.assertEqual(params["limit"], 5)
        self.assertIsInstance(params["limit"], int)

    def test_mixed_streaming_and_non_streaming_parameters_opening(self):
        """
        Test tool call with both string (streamed) and non-string (not streamed) parameters.
        """
        chunks = [
            "<tool_call><function=search>",
            "<parameter=query>",
            "\n",
            "long ",
            "search ",
            "query",
            "\n</parameter>",
            "<parameter=limit>\n5\n</parameter>",
            "</function></tool_call>",
        ]

        detector = Qwen3CoderDetector()
        all_calls = []

        # Chunk 0
        result = detector.parse_streaming_increment(chunks[0], self.tools)
        all_calls.extend(result.calls)

        # Chunk 1: String param starts streaming
        result = detector.parse_streaming_increment(chunks[1], self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].parameters, "{")
        self.assertEqual(result.calls[1].parameters, '"query": "')
        all_calls.extend(result.calls)

        result = detector.parse_streaming_increment(chunks[2], self.tools)
        self.assertEqual(len(result.calls), 0)
        all_calls.extend(result.calls)

        result = detector.parse_streaming_increment(chunks[3], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "long ")  # First content
        all_calls.extend(result.calls)
        # Chunk 2: String param continues streaming
        result = detector.parse_streaming_increment(chunks[4], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "search ")
        all_calls.extend(result.calls)

        # Chunk 3: String param continues streaming
        result = detector.parse_streaming_increment(chunks[5], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "query")
        all_calls.extend(result.calls)

        # Chunk 4: String param completes
        result = detector.parse_streaming_increment(chunks[6], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, '"')  # Just closing quote
        all_calls.extend(result.calls)

        # Chunk 5: Integer param - NOT streamed, emitted as complete value
        result = detector.parse_streaming_increment(chunks[7], self.tools)
        self.assertEqual(len(result.calls), 1)
        # Should be complete key-value pair with comma, not streamed
        self.assertIn('"limit"', result.calls[0].parameters)
        self.assertIn("5", result.calls[0].parameters)
        self.assertNotIn(
            '"', result.calls[0].parameters.replace('"limit"', "").replace(": ", "")
        )  # Value not quoted
        all_calls.extend(result.calls)

        # Chunk 6
        result = detector.parse_streaming_increment(chunks[8], self.tools)
        self.assertEqual(result.calls[0].parameters, "}")
        all_calls.extend(result.calls)

        # Verify final structure
        param_calls = [tc for tc in all_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)

        self.assertEqual(params["query"], "long search query")
        self.assertEqual(params["limit"], 5)
        self.assertIsInstance(params["limit"], int)

    def test_incremental_streaming_empty_parameter(self):
        """
        Test incremental streaming with empty string parameter.
        """
        chunks = [
            "<tool_call><function=search>",
            "<parameter=query>\n\n</parameter>",
            "</function></tool_call>",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        param_calls = [tc for tc in tool_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)

        self.assertEqual(params["query"], "")

    def test_incremental_streaming_single_chunk_complete(self):
        """
        Test that short string parameters in single chunk work correctly.
        Should not trigger streaming (complete in one go).
        """
        chunks = [
            "<tool_call><function=get_weather>",
            "<parameter=location>\nTokyo\n</parameter>",
            "</function></tool_call>",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        param_calls = [tc for tc in tool_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)

        self.assertEqual(params["location"], "Tokyo")

    def test_incremental_streaming_very_long_value(self):
        """
        Test incremental streaming with very long parameter value.
        Simulates real scenario like writing a large file.
        """
        # Create a long content string
        long_content = "def hello():\n    print('Hello, World!')\n" * 12
        chunk_size = 30

        chunks = ["<tool_call><function=search>", "<parameter=query>\n"]

        # Split long content into small chunks
        for i in range(0, len(long_content), chunk_size):
            chunks.append(long_content[i : i + chunk_size])

        chunks.extend(["\n</parameter>", "</function></tool_call>"])

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # Should have many incremental chunks
        param_calls = [tc for tc in tool_calls if tc.parameters]
        self.assertGreater(
            len(param_calls), 10, "Should have many incremental chunks for long content"
        )

        # Verify final result is correct
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)
        self.assertEqual(params["query"], long_content)

    def test_incremental_streaming_unicode_content(self):
        """
        Test incremental streaming with Unicode characters.
        """
        chunks = [
            "<tool_call><function=search>",
            "<parameter=query>\n你好",
            "世界",
            "🌍",
            "\n</parameter>",
            "</function></tool_call>",
        ]

        detector = Qwen3CoderDetector()
        all_calls = []

        # Chunk 0
        result = detector.parse_streaming_increment(chunks[0], self.tools)
        all_calls.extend(result.calls)

        # Chunk 1: Start streaming with Chinese chars
        result = detector.parse_streaming_increment(chunks[1], self.tools)
        self.assertEqual(len(result.calls), 3)
        self.assertEqual(result.calls[0].parameters, "{")
        self.assertEqual(result.calls[1].parameters, '"query": "')
        self.assertEqual(result.calls[2].parameters, "你好")  # First content
        all_calls.extend(result.calls)

        # Chunk 2: Continue with Chinese
        result = detector.parse_streaming_increment(chunks[2], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "世界")
        all_calls.extend(result.calls)

        # Chunk 3: Emoji
        result = detector.parse_streaming_increment(chunks[3], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "🌍")
        all_calls.extend(result.calls)

        # Chunk 4: Complete
        result = detector.parse_streaming_increment(chunks[4], self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, '"')  # Just closing quote
        all_calls.extend(result.calls)

        # Chunk 5
        result = detector.parse_streaming_increment(chunks[5], self.tools)
        all_calls.extend(result.calls)

        # Verify final structure
        param_calls = [tc for tc in all_calls if tc.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        params = json.loads(params_str)
        self.assertEqual(params["query"], "你好世界🌍")

    def test_incremental_streaming_state_reset_between_parameters(self):
        """
        Test that streaming state properly resets between different parameters.
        """
        chunks = [
            "<tool_call><function=search>",
            "<parameter=query>\nfirst ",
            "param",
            "\n</parameter>",
            "<parameter=limit>\n5\n</parameter>",
            "</function></tool_call>",
            "<tool_call><function=get_weather>",
            "<parameter=location>\nsecond ",
            "call",
            "\n</parameter>",
            "</function></tool_call>",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)

        # Verify both tool calls have correct parameters
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(len(tool_indices), 2)

        # Get parameters for each tool call
        tool0_params = [tc for tc in tool_calls if tc.tool_index == 0 and tc.parameters]
        tool1_params = [tc for tc in tool_calls if tc.tool_index == 1 and tc.parameters]

        params0_str = "".join(tc.parameters for tc in tool0_params)
        params1_str = "".join(tc.parameters for tc in tool1_params)

        params0 = json.loads(params0_str)
        params1 = json.loads(params1_str)

        self.assertEqual(params0["query"], "first param")
        self.assertEqual(params0["limit"], 5)
        self.assertEqual(params1["location"], "second call")

    def test_real_world_streaming_pattern(self):
        """
        Test based on real-world streaming pattern from user's example.
        Simulates: search_codebase(query="AI artificial intelligence machine learning NLP features",
                                   key_words="AI,ML,NLP")
        followed by list_dir call.
        """
        # Add search_codebase and list_dir to tools
        tools_extended = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="search_codebase",
                    description="Search the codebase",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "key_words": {"type": "string"},
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="list_dir",
                    description="List directory contents",
                    parameters={
                        "type": "object",
                        "properties": {
                            "relative_workspace_path": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

        # Simulate the exact streaming pattern from the example
        chunks = [
            "<tool_call><function=search_codebase>",
            "<parameter=query>\nAI artificial",
            " intelligence machine learning N",
            "LP features",
            "\n</parameter>",
            "<parameter=key_words>\nAI",
            ",ML,NLP",
            "\n</parameter>",
            "</function></tool_call>\n",
            "<tool_call><function=list_dir>",
            "<parameter=relative_workspace_path>\n",
        ]

        detector = Qwen3CoderDetector()
        all_calls = []

        # Chunk 0: Tool call 0 starts with name
        result = detector.parse_streaming_increment(chunks[0], tools_extended)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(result.calls[0].name, "search_codebase")
        self.assertEqual(result.calls[0].parameters, "")
        all_calls.extend(result.calls)

        # Chunk 1: First parameter starts streaming
        result = detector.parse_streaming_increment(chunks[1], tools_extended)
        self.assertEqual(len(result.calls), 3)
        self.assertEqual(result.calls[0].parameters, "{")
        self.assertEqual(result.calls[1].parameters, '"query": "')
        self.assertEqual(result.calls[2].parameters, "AI artificial")
        all_calls.extend(result.calls)

        # Chunk 2: Continue streaming query value
        result = detector.parse_streaming_increment(chunks[2], tools_extended)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, " intelligence machine learning N")
        all_calls.extend(result.calls)

        # Chunk 3: Continue streaming query value
        result = detector.parse_streaming_increment(chunks[3], tools_extended)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "LP features")
        all_calls.extend(result.calls)

        # Chunk 4: Complete query parameter
        result = detector.parse_streaming_increment(chunks[4], tools_extended)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, '"')  # Close quote
        all_calls.extend(result.calls)

        # Chunk 5: Second parameter starts streaming
        result = detector.parse_streaming_increment(chunks[5], tools_extended)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].parameters, ', "key_words": "')
        self.assertEqual(result.calls[1].parameters, "AI")
        all_calls.extend(result.calls)

        # Chunk 6: Continue streaming key_words value
        result = detector.parse_streaming_increment(chunks[6], tools_extended)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, ",ML,NLP")
        all_calls.extend(result.calls)

        # Chunk 7: Complete key_words parameter
        result = detector.parse_streaming_increment(chunks[7], tools_extended)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, '"')  # Close quote
        all_calls.extend(result.calls)

        # Chunk 8: Complete function and tool call
        result = detector.parse_streaming_increment(chunks[8], tools_extended)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "}")
        all_calls.extend(result.calls)

        # Chunk 9: Second tool call starts
        result = detector.parse_streaming_increment(chunks[9], tools_extended)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].tool_index, 1)
        self.assertEqual(result.calls[0].name, "list_dir")
        all_calls.extend(result.calls)

        # Chunk 10: Second tool call parameter starts
        result = detector.parse_streaming_increment(chunks[10], tools_extended)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].parameters, "{")
        self.assertEqual(result.calls[1].parameters, '"relative_workspace_path": "')
        all_calls.extend(result.calls)

        # Verify final structure for first tool call
        tool0_params = [tc for tc in all_calls if tc.tool_index == 0 and tc.parameters]
        params0_str = "".join(tc.parameters for tc in tool0_params)
        self.assertEqual(
            params0_str,
            '{"query": "AI artificial intelligence machine learning NLP features", "key_words": "AI,ML,NLP"}',
        )
        params0 = json.loads(params0_str)
        self.assertEqual(
            params0["query"], "AI artificial intelligence machine learning NLP features"
        )
        self.assertEqual(params0["key_words"], "AI,ML,NLP")

        # Verify incremental emission pattern matches example
        # When serialized to JSON wire format, these would become:
        # "{\"query\": \"AI artificial" (first chunk)
        # " intelligence machine learning N" (second chunk)
        # "LP features" (third chunk)
        # etc.

    def test_partial_end_tag_buffering(self):
        """
        Test that partial end tags like "</para" are buffered correctly.
        When streaming "/Users/path</para", we can't know if "</para" is:
        1. Start of closing tag "</parameter>" - should stop and close
        2. Just content that continues - should emit and continue
        We buffer "</para" and wait for next chunk to decide.
        """
        # Scenario: Content ends with "</para" which completes to "</parameter>" (the closing tag)
        chunks = [
            "<tool_call><function=read_file>",
            "<parameter=file_path>\n/Users/dingyang/Develop\n</para",  # Partial end tag
            "meter>",  # Completes the closing tag
            "</function></tool_call>",
        ]

        tools = [
            Tool(
                type="function",
                function=Function(
                    name="read_file",
                    parameters={"properties": {"file_path": {"type": "string"}}},
                ),
            )
        ]

        detector = Qwen3CoderDetector()
        all_calls = []

        # Chunk 0: Function name
        result = detector.parse_streaming_increment(chunks[0], tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "read_file")
        all_calls.extend(result.calls)

        # Chunk 1: Parameter starts with path, ends with partial end tag "</para"
        # Should emit: "{", "file_path": ", "/Users/dingyang/Develop"
        # Should buffer: "</para" (not emitted)
        result = detector.parse_streaming_increment(chunks[1], tools)
        self.assertEqual(len(result.calls), 3)
        self.assertEqual(result.calls[0].parameters, "{")
        self.assertEqual(result.calls[1].parameters, '"file_path": "')
        self.assertEqual(result.calls[2].parameters, "/Users/dingyang/Develop")
        all_calls.extend(result.calls)

        # Chunk 2: Continues with "meter>" which completes "</parameter>"
        # Buffered "</para" + "meter>" = "</parameter>" which is the closing tag
        # Should emit: closing quote only (brace comes in a separate call)
        result = detector.parse_streaming_increment(chunks[2], tools)
        self.assertEqual(len(result.calls), 1, f"calls: {result.calls}")
        self.assertEqual(result.calls[0].parameters, '"')
        all_calls.extend(result.calls)

        # Chunk 3: End of tool call
        result = detector.parse_streaming_increment(chunks[3], tools)
        # Should emit closing brace
        param_calls_chunk3 = [call for call in result.calls if call.parameters]
        if param_calls_chunk3:
            self.assertEqual(param_calls_chunk3[0].parameters, "}")
        all_calls.extend(result.calls)

        # Verify final result - should be just "/Users/dingyang/Develop"
        param_calls = [call for call in all_calls if call.parameters]
        params_str = "".join(tc.parameters for tc in param_calls)
        self.assertEqual(params_str, '{"file_path": "/Users/dingyang/Develop"}')
        params = json.loads(params_str)
        self.assertEqual(params["file_path"], "/Users/dingyang/Develop")

    def test_case1_four_serial_tool_calls(self):
        """
        ============================================================
        Tool Calls:
        ============================================================
        [0] name: read_file
            arguments: {"file_path": "/Users/dingyang/Develop/github/memos/plugin/markdown/markdown.go"}
        [1] name: search_codebase
            arguments: {"query": "natural language processing NLP AI features", "key_words": "NLP,AI,processing"}
        [2] name: search_file
            arguments: {"query": "*.go", "path": "/Users/dingyang/Develop/github/memos/plugin"}
        """
        chunks = [
            "<tool_call>",
            "\n<",
            "function=read_file>\n<",
            "parameter=file",
            "_path>\n/Users/ding",
            "yang/Develop/github/m",
            "emos/plugin",
            "/mark",
            "down",
            "/mark",
            "down.go\n</parameter",
            ">\n</function>\n</tool_call>\n<tool_call>",
            "\n<function=search",
            "_code",
            "base>\n<parameter=query",
            ">\nnatural",
            " language",
            " processing N",
            "LP",
            " AI",
            " features",
            "\n</parameter>\n<",
            "parameter=key",
            "_words>\nN",
            "LP,A",
            "I,processing",
            "\n</parameter>\n</",
            "function>\n</tool_call>\n",
            "<tool_call>\n<function=search",
            "_file>\n<parameter=query",
            ">\n*.",
            "go\n</parameter>\n",
            "<parameter=path",
            ">\n/Users/dingyang",
            "/Develop/github/memos",
            "/plugin",
            "\n</parameter>\n</",
            "function>\n</tool_call>",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)
        self.assertEqual(len(normal_texts), 0)
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(len(tool_indices), 3)

        # check tool name
        name_calls = [tc for tc in tool_calls if tc.name]
        self.assertEqual(len(name_calls), 3, "Should have exactly 3 tool names")
        self.assertEqual(name_calls[0].name, "read_file")
        self.assertEqual(name_calls[1].name, "search_codebase")
        self.assertEqual(name_calls[2].name, "search_file")

        # check tool parameters
        tool0_params_list = [
            tc for tc in tool_calls if tc.tool_index == 0 and tc.parameters
        ]
        tool1_params_list = [
            tc for tc in tool_calls if tc.tool_index == 1 and tc.parameters
        ]
        tool2_params_list = [
            tc for tc in tool_calls if tc.tool_index == 2 and tc.parameters
        ]

        tool0_params = json.loads("".join(tc.parameters for tc in tool0_params_list))
        tool1_params = json.loads("".join(tc.parameters for tc in tool1_params_list))
        tool2_params = json.loads("".join(tc.parameters for tc in tool2_params_list))

        self.assertEqual(
            tool0_params["file_path"],
            "/Users/dingyang/Develop/github/memos/plugin/markdown/markdown.go",
        )
        self.assertEqual(
            tool1_params["query"], "natural language processing NLP AI features"
        )
        self.assertEqual(tool1_params["key_words"], "NLP,AI,processing")
        self.assertEqual(tool2_params["query"], "*.go")
        self.assertEqual(
            tool2_params["path"], "/Users/dingyang/Develop/github/memos/plugin"
        )

    def test_case2_four_serial_tool_calls(self):
        """
        ============================================================
        Tool Calls:
        ============================================================
        [0] name: read_file
            arguments: {"file_path": "/Users/dingyang/Develop/github/memos/web/src/components/MemoEditor.tsx"}
        [1] name: read_file
            arguments: {"file_path": "/Users/dingyang/Develop/github/memos/web/src/components/MemoView.tsx"}
        [2] name: search_codebase
            arguments: {"query": "natural language processing AI features", "key_words": "NLP,AI,processing"}
        [3] name: list_dir
            arguments: {"relative_workspace_path": "/Users/dingyang/Develop/github/memos/plugin"}
        """
        chunks = [
            "<tool_call>",
            "\n<",
            "function=read_file>\n<",
            "parameter=file",
            "_path>\n/Users/ding",
            "yang/Develop/github/m",
            "emos/web",
            "/src/components/MemoEditor",
            ".ts",
            "x\n</parameter>\n",
            "</function>\n</tool_call>\n",
            "<tool_call>\n<function=read",
            "_file>\n<parameter=file",
            "_path>\n/Users/ding",
            "yang/Develop/github/m",
            "emos/web/src/components/M",
            "emoView",
            ".ts",
            "x\n</parameter>\n",
            "</function>\n</tool_call>",
            "\n" "<tool_call>\n<function=search",
            "_code",
            "base>\n<parameter=query",
            ">\nnatural",
            " language",
            " processing AI",
            " features",
            "\n</parameter>\n<",
            "parameter=key",
            "_words>\nN",
            "LP,A",
            "I,processing",
            "\n</parameter>\n</",
            "function>\n</tool_call>",
            "\n<tool_call>\n<function=list",
            "_dir>\n<parameter=",
            "relative_workspace_path>\n/Users",
            "/dingyang/Develop",
            "/github/memos/plugin",
            "\n</parameter>\n</",
            "function>\n</tool_call>",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)
        self.assertEqual(len(normal_texts), 0, f"data: {normal_texts}")
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(len(tool_indices), 4)

        # check tool name
        name_calls = [tc for tc in tool_calls if tc.name]
        self.assertEqual(len(name_calls), 4, "Should have exactly 4 tool names")
        self.assertEqual(name_calls[0].name, "read_file")
        self.assertEqual(name_calls[1].name, "read_file")
        self.assertEqual(name_calls[2].name, "search_codebase")
        self.assertEqual(name_calls[3].name, "list_dir")

        # check tool parameters
        tool0_params_list = [
            tc for tc in tool_calls if tc.tool_index == 0 and tc.parameters
        ]
        tool1_params_list = [
            tc for tc in tool_calls if tc.tool_index == 1 and tc.parameters
        ]
        tool2_params_list = [
            tc for tc in tool_calls if tc.tool_index == 2 and tc.parameters
        ]
        tool3_params_list = [
            tc for tc in tool_calls if tc.tool_index == 3 and tc.parameters
        ]

        tool0_params = json.loads("".join(tc.parameters for tc in tool0_params_list))
        tool1_params = json.loads("".join(tc.parameters for tc in tool1_params_list))
        tool2_params = json.loads("".join(tc.parameters for tc in tool2_params_list))
        tool3_params = json.loads("".join(tc.parameters for tc in tool3_params_list))

        self.assertEqual(
            tool0_params["file_path"],
            "/Users/dingyang/Develop/github/memos/web/src/components/MemoEditor.tsx",
        )
        self.assertEqual(
            tool1_params["file_path"],
            "/Users/dingyang/Develop/github/memos/web/src/components/MemoView.tsx",
        )
        self.assertEqual(
            tool2_params["query"], "natural language processing AI features"
        )
        self.assertEqual(tool2_params["key_words"], "NLP,AI,processing")
        self.assertEqual(
            tool3_params["relative_workspace_path"],
            "/Users/dingyang/Develop/github/memos/plugin",
        )

    def test_case3_for_five_params(self):
        """
        ============================================================
        Tool Calls:
        ============================================================
        <tool_call>
        <function=mcp__daily-business-monitor-daignose-mcp__query_business_monitor>
        <parameter=appName>
        tblive-prod
        </parameter>
        <parameter=serviceName>
        com.alibaba.tbliveprod.facade.hsf.ShopWindowHsfService:1.0.0
        </parameter>
        <parameter=methodName>
        queryMateItemList~S
        </parameter>
        <parameter=startTime>
        2026-01-28 14:55:00
        </parameter>
        <parameter=endTime>
        2026-01-28 15:05:00
        </parameter>
        </function>
        </tool_call>
        """
        chunks = [
            "<tool_call>",
            "\n<",
            "function=mcp__daily",
            "-business-monitor-daignose",
            "-mcp__query_business",
            "_monitor>\n<",
            "parameter=appName",
            ">\ntbl",
            "ive-pro",
            "d\n</parameter>\n",
            "<parameter=serviceName",
            ">\ncom",
            ".alibaba.tbl",
            "iveprod.facade.h",
            "sf.ShopWindowH",
            "sfService:1.",
            "0.0\n</",
            "parameter>\n<parameter=",
            "methodName",
            ">\nqueryMateItemList~",
            "S\n</parameter>\n",
            "<parameter=startTime>\n",
            "2026-",
            "01-2",
            "8 14",
            ":5",
            "5",
            ":00\n",
            "</parameter>\n<",
            "parameter=endTime>\n2",
            "026-0",
            "1-28 ",
            "15:0",
            "5:00\n",
            "</parameter>\n</function",
            ">\n</tool_call>",
        ]

        normal_texts, tool_calls = self._parse_chunks(chunks)
        self.assertEqual(len(normal_texts), 0)
        tool_indices = set(tc.tool_index for tc in tool_calls)
        self.assertEqual(len(tool_indices), 1)

        # check tool name
        name_calls = [tc for tc in tool_calls if tc.name]
        self.assertEqual(len(name_calls), 1, "Should have exactly 1 tool name")
        self.assertEqual(
            name_calls[0].name,
            "mcp__daily-business-monitor-daignose-mcp__query_business_monitor",
        )

        # check tool parameters
        tool0_params_list = [
            tc for tc in tool_calls if tc.tool_index == 0 and tc.parameters
        ]
        tool0_params = json.loads("".join(tc.parameters for tc in tool0_params_list))

        self.assertEqual(tool0_params["appName"], "tblive-prod")
        self.assertEqual(
            tool0_params["serviceName"],
            "com.alibaba.tbliveprod.facade.hsf.ShopWindowHsfService:1.0.0",
        )
        self.assertEqual(tool0_params["methodName"], "queryMateItemList~S")
        self.assertEqual(tool0_params["startTime"], "2026-01-28 14:55:00")
        self.assertEqual(tool0_params["endTime"], "2026-01-28 15:05:00")


if __name__ == "__main__":
    unittest.main()
