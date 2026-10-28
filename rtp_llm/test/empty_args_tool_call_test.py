"""
Tests for tool calls with empty arguments ({}).

Reproduces a bug where empty dict arguments are falsy in Python,
causing the detector to skip argument processing and tool call finalization.
This results in duplicate tool_call emissions with different IDs in the
OpenAI streaming response.

The root cause is in BaseFormatDetector.parse_streaming_increment:
    if cur_arguments:  # {} is falsy!
        ...
Should be:
    if cur_arguments is not None:
        ...
"""

import json
import unittest
from typing import List

import pytest

pytestmark = [pytest.mark.gpu(type="A10")]

from rtp_llm.openai.api_datatype import FunctionCall, ToolCall
from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.format_convert_helper import (
    streaming_parse_result_to_tool_calls,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen25_detector import (
    Qwen25Detector,
)


def create_search_order_tools() -> List[Tool]:
    """Create tool definitions matching the user's scenario."""
    return [
        Tool(
            type="function",
            function=Function(
                name="search_order",
                description="Search for orders",
                parameters={"type": "object", "properties": {}},
            ),
        ),
    ]


class TestEmptyArgumentsToolCall(unittest.TestCase):
    """
    Test that tool calls with empty arguments ({}) are handled correctly.

    Reproduces the scenario where the model outputs:
        正在帮您查询订单\n\n<tool_call>
        {"name": "search_order", "arguments": {}}
        </tool_call>

    The bug: empty dict {} is falsy in Python, so `if cur_arguments:` skips
    argument processing, causing the tool call to never finalize properly.
    """

    def setUp(self):
        self.tools = create_search_order_tools()

    def test_single_tool_call_empty_args_token_by_token(self):
        """
        Simulate token-by-token streaming of a single tool call with empty arguments.
        Verifies that exactly ONE tool_call name is emitted and arguments are '{}'.
        """
        detector = Qwen25Detector()

        # Token-by-token chunks simulating the user's raw output
        chunks = [
            "正在",
            "帮",
            "您",
            "查询",
            "订单",
            "\n\n",
            "<tool_call>",
            "\n",
            '{"',
            "name",
            '":',
            ' "',
            "search",
            "_order",
            '",',
            ' "',
            "arguments",
            '":',
            " {}",  # arguments value is empty dict
            "}",  # closing brace of outer JSON
            "\n",
            "</tool_call>",
        ]

        all_calls = []
        all_normal_text = []

        for i, chunk in enumerate(chunks):
            result = detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                all_calls.extend(result.calls)
            if result.normal_text:
                all_normal_text.append(result.normal_text)

        # Verify normal text was captured
        normal_text = "".join(all_normal_text)
        self.assertIn("正在", normal_text)
        self.assertIn("查询订单", normal_text)

        # Verify exactly ONE name emission
        name_calls = [c for c in all_calls if c.name]
        self.assertEqual(
            len(name_calls),
            1,
            f"Expected exactly 1 name emission, got {len(name_calls)}. "
            f"All calls: {all_calls}",
        )
        self.assertEqual(name_calls[0].name, "search_order")
        self.assertEqual(name_calls[0].tool_index, 0)
        self.assertEqual(name_calls[0].parameters, "")

        # Verify arguments are emitted
        arg_calls = [c for c in all_calls if c.parameters]
        self.assertTrue(
            len(arg_calls) > 0,
            f"Expected at least 1 argument emission, got 0. All calls: {all_calls}",
        )

        # Verify concatenated arguments form '{}'
        full_args = "".join(c.parameters for c in all_calls if c.parameters)
        self.assertEqual(
            full_args,
            "{}",
            f"Expected arguments '{{}}', got '{full_args}'. All calls: {all_calls}",
        )

    def test_single_tool_call_empty_args_coarse_chunks(self):
        """
        Test with coarser chunks (fewer, larger pieces).
        This simulates a scenario closer to MTP or batch processing.
        """
        detector = Qwen25Detector()

        chunks = [
            "正在帮您查询订单\n\n",
            "<tool_call>\n",
            '{"name": "search_order", "arguments": {}}',
            "\n</tool_call>",
        ]

        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        name_calls = [c for c in all_calls if c.name]
        self.assertEqual(
            len(name_calls),
            1,
            f"Expected 1 name emission, got {len(name_calls)}. All calls: {all_calls}",
        )
        self.assertEqual(name_calls[0].name, "search_order")

        full_args = "".join(c.parameters for c in all_calls if c.parameters)
        self.assertEqual(
            full_args,
            "{}",
            f"Expected arguments '{{}}', got '{full_args}'. All calls: {all_calls}",
        )

    def test_empty_args_tool_call_then_end_tag(self):
        """
        Specifically test the scenario where arguments {} arrive and then
        the end tag arrives. The detector must finalize the tool call
        when the JSON is complete, even with empty arguments.
        """
        detector = Qwen25Detector()

        # Step 1: Start tool call and provide name
        r1 = detector.parse_streaming_increment("<tool_call>\n", self.tools)
        self.assertEqual(len(r1.calls), 0, "Should buffer on start tag")

        r2 = detector.parse_streaming_increment('{"name": "search_order"', self.tools)
        self.assertEqual(len(r2.calls), 1, "Should emit name")
        self.assertEqual(r2.calls[0].name, "search_order")

        # Step 2: Provide empty arguments and closing brace
        r3 = detector.parse_streaming_increment(', "arguments": {}}', self.tools)

        # The arguments should be processed even though {} is falsy
        self.assertEqual(
            len(r3.calls),
            1,
            f"Expected 1 call with arguments, got {len(r3.calls)}. "
            f"Result: {r3}. "
            f"Detector state: current_tool_id={detector.current_tool_id}, "
            f"current_tool_name_sent={detector.current_tool_name_sent}, "
            f"buffer='{detector._buffer}'",
        )
        if r3.calls:
            self.assertEqual(r3.calls[0].parameters, "{}")
            self.assertIsNone(r3.calls[0].name)

        # Step 3: End tag should be clean
        r4 = detector.parse_streaming_increment("\n</tool_call>", self.tools)
        # No additional tool calls should be emitted
        name_calls_in_r4 = [c for c in r4.calls if c.name]
        self.assertEqual(
            len(name_calls_in_r4),
            0,
            f"End tag should not emit new name. Got: {r4.calls}",
        )

    def test_empty_args_vs_nonempty_args(self):
        """
        Compare behavior of empty args {} vs non-empty args {"key": "val"}.
        Both should produce correct tool calls.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="func_a",
                    description="Function A",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="func_b",
                    description="Function B",
                    parameters={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

        # Test 1: Non-empty arguments (should always work)
        detector1 = Qwen25Detector()
        chunks1 = [
            '<tool_call>\n{"name": "func_b", "arguments": {"key": "val"}}',
            "\n</tool_call>",
        ]
        calls1 = []
        for chunk in chunks1:
            r = detector1.parse_streaming_increment(chunk, tools)
            calls1.extend(r.calls)

        name_calls1 = [c for c in calls1 if c.name]
        self.assertEqual(len(name_calls1), 1)
        self.assertEqual(name_calls1[0].name, "func_b")
        full_args1 = "".join(c.parameters for c in calls1 if c.parameters)
        parsed1 = json.loads(full_args1)
        self.assertEqual(parsed1, {"key": "val"})

        # Test 2: Empty arguments (the bug case)
        detector2 = Qwen25Detector()
        chunks2 = [
            '<tool_call>\n{"name": "func_a", "arguments": {}}',
            "\n</tool_call>",
        ]
        calls2 = []
        for chunk in chunks2:
            r = detector2.parse_streaming_increment(chunk, tools)
            calls2.extend(r.calls)

        name_calls2 = [c for c in calls2 if c.name]
        self.assertEqual(
            len(name_calls2),
            1,
            f"Expected 1 name emission for empty args, got {len(name_calls2)}. "
            f"All calls: {calls2}",
        )
        self.assertEqual(name_calls2[0].name, "func_a")
        full_args2 = "".join(c.parameters for c in calls2 if c.parameters)
        self.assertEqual(
            full_args2,
            "{}",
            f"Expected '{{}}' for empty args, got '{full_args2}'. All calls: {calls2}",
        )

    def test_streaming_parse_result_to_tool_calls_no_duplicate_ids(self):
        """
        Test the full flow: detector → streaming_parse_result_to_tool_calls.
        Verifies that across all streaming chunks, there is only ONE ToolCall
        with a name and id (no duplicate emissions).
        """
        detector = Qwen25Detector()

        chunks = [
            "正在帮您查询订单",
            "\n\n",
            "<tool_call>",
            "\n",
            '{"name": "search_order", "arguments": {}}',
            "\n",
            "</tool_call>",
        ]

        all_tool_calls_with_name = []
        all_tool_calls_with_args = []

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            tool_calls, remaining = streaming_parse_result_to_tool_calls(result)
            for tc in tool_calls:
                if tc.function and tc.function.name:
                    all_tool_calls_with_name.append(tc)
                if tc.function and tc.function.arguments:
                    all_tool_calls_with_args.append(tc)

        # Should have exactly ONE ToolCall with a name
        self.assertEqual(
            len(all_tool_calls_with_name),
            1,
            f"Expected 1 ToolCall with name, got {len(all_tool_calls_with_name)}. "
            f"ToolCalls: {[(tc.id, tc.function.name, tc.function.arguments) for tc in all_tool_calls_with_name]}",
        )

        # The ToolCall with name should have an id
        self.assertIsNotNone(
            all_tool_calls_with_name[0].id,
            "ToolCall with name should have an id",
        )

        # Should have at least ONE ToolCall with arguments
        self.assertTrue(
            len(all_tool_calls_with_args) > 0,
            f"Expected at least 1 ToolCall with arguments. "
            f"All tool_calls_with_name: {all_tool_calls_with_name}",
        )

    def test_detect_and_parse_empty_args(self):
        """Test non-streaming parsing of tool call with empty arguments."""
        detector = Qwen25Detector()
        text = '<tool_call>\n{"name": "search_order", "arguments": {}}\n</tool_call>'

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}. Result: {result}",
        )
        self.assertEqual(result.calls[0].name, "search_order")
        self.assertEqual(
            result.calls[0].parameters,
            "{}",
            f"Expected '{{}}', got '{result.calls[0].parameters}'",
        )

    def test_detect_and_parse_empty_args_with_prefix_text(self):
        """Test non-streaming parsing with prefix text and empty arguments."""
        detector = Qwen25Detector()
        text = '正在帮您查询订单\n\n<tool_call>\n{"name": "search_order", "arguments": {}}\n</tool_call>'

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "正在帮您查询订单")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search_order")
        self.assertEqual(result.calls[0].parameters, "{}")


class TestEmptyArgumentsDetectorState(unittest.TestCase):
    """
    Tests that verify the detector's internal state is correct
    after processing a tool call with empty arguments.
    """

    def setUp(self):
        self.tools = create_search_order_tools()

    def test_detector_state_after_empty_args_complete(self):
        """
        After a complete tool call with empty args, the detector should:
        - Increment current_tool_id
        - Reset current_tool_name_sent to False
        - Clear the buffer
        """
        detector = Qwen25Detector()

        chunks = [
            "<tool_call>\n",
            '{"name": "search_order", "arguments": {}}',
            "\n</tool_call>",
        ]

        for chunk in chunks:
            detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            detector.current_tool_id,
            1,
            f"Expected current_tool_id=1 after completing tool call, "
            f"got {detector.current_tool_id}. "
            f"Buffer: '{detector._buffer}'",
        )
        self.assertFalse(
            detector.current_tool_name_sent,
            "current_tool_name_sent should be False after completing tool call",
        )

    def test_two_tool_calls_second_has_empty_args(self):
        """
        Test two sequential tool calls where the second has empty arguments.
        Both should be properly finalized.
        Uses token-level chunk boundaries matching real TokenNormalizer output.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search_order",
                    description="Search orders",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        detector = Qwen25Detector()

        # Token-level chunks (each element = one token's decoded text)
        chunks = [
            "<tool_call>",
            "\n",
            '{"name": "get_weather", "arguments": {"location": "杭州"}}',
            "\n",
            "</tool_call>",
            "\n",
            "<tool_call>",
            "\n",
            '{"name": "search_order", "arguments": {}}',
            "\n",
            "</tool_call>",
        ]

        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            all_calls.extend(result.calls)

        name_calls = [c for c in all_calls if c.name]
        self.assertEqual(
            len(name_calls),
            2,
            f"Expected 2 name emissions, got {len(name_calls)}. All calls: {all_calls}",
        )
        self.assertEqual(name_calls[0].name, "get_weather")
        self.assertEqual(name_calls[0].tool_index, 0)
        self.assertEqual(name_calls[1].name, "search_order")
        self.assertEqual(name_calls[1].tool_index, 1)

        # Verify arguments for both
        tool0_args = "".join(
            c.parameters for c in all_calls if c.tool_index == 0 and c.parameters
        )
        tool1_args = "".join(
            c.parameters for c in all_calls if c.tool_index == 1 and c.parameters
        )

        parsed0 = json.loads(tool0_args)
        self.assertEqual(parsed0, {"location": "杭州"})
        self.assertEqual(
            tool1_args,
            "{}",
            f"Expected '{{}}' for second tool, got '{tool1_args}'",
        )


if __name__ == "__main__":
    unittest.main()
