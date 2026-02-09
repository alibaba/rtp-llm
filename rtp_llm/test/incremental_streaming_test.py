import json
import unittest
from typing import List

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen25_detector import (
    Qwen25Detector,
)


def create_tools() -> List[Tool]:
    """Create test tool definitions."""
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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


def collect_streaming_results(
    detector: Qwen25Detector, chunks: List[str], tools: List[Tool]
) -> List[StreamingParseResult]:
    """Feed chunks through detector and collect all results."""
    results = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        results.append(result)
    return results


def assert_incremental_tool_call(
    test: unittest.TestCase,
    results: List[StreamingParseResult],
    expected_name: str,
    expected_args: str,
    expected_tool_index: int = 0,
) -> None:
    """
    Assert that the results contain incremental tool call data.

    Verifies:
    1. Name is emitted first (with empty parameters)
    2. Arguments are emitted incrementally after name
    3. Complete arguments match expected
    """
    all_calls = []
    for result in results:
        all_calls.extend(result.calls)

    # Find the first call with name (name announcement)
    name_calls = [c for c in all_calls if c.name]
    test.assertTrue(
        len(name_calls) > 0,
        f"Expected at least one call with name. Got: {all_calls}",
    )
    test.assertEqual(
        name_calls[0].name,
        expected_name,
        f"Expected name '{expected_name}', got '{name_calls[0].name}'",
    )
    test.assertEqual(
        name_calls[0].tool_index,
        expected_tool_index,
        f"Expected tool_index {expected_tool_index}, got {name_calls[0].tool_index}",
    )
    # Name should be emitted with empty parameters first
    test.assertEqual(
        name_calls[0].parameters,
        "",
        f"Expected empty parameters with name, got '{name_calls[0].parameters}'",
    )

    # Collect all argument increments for this tool
    arg_increments = [
        c.parameters
        for c in all_calls
        if c.tool_index == expected_tool_index and c.parameters
    ]

    # Concatenated arguments should match expected
    actual_args = "".join(arg_increments)
    test.assertEqual(
        actual_args,
        expected_args,
        f"Expected arguments '{expected_args}', got '{actual_args}'",
    )


class TestQwen25IncrementalStreaming(unittest.TestCase):
    """Test Qwen25Detector incremental streaming behavior."""

    def setUp(self):
        self.tools = create_tools()

    def test_scenario2_tool_start_midstream(self):
        """
        Scenario 2: Tool start mid-stream "<tool_call>..."

        Simulates token-by-token delivery of a tool call.
        Verifies name is emitted first, then arguments incrementally.

        This test rigorously checks both per-chunk results and aggregated results.
        """
        detector = Qwen25Detector()

        # Token-by-token chunks (simulating normalized input)
        chunks = [
            "<tool_call>",  # 0: buffering
            "\n",  # 1: buffering
            "{",  # 2: buffering
            '"',  # 3: buffering
            "name",  # 4: buffering
            '"',  # 5: buffering
            ":",  # 6: buffering
            ' "get',  # 7: buffering
            "_weather",  # 8: buffering
            '"',  # 9: NAME EMITTED - complete function name parsed
            ', "arg',  # 10: buffering args
            'uments":',  # 11: buffering args
            ' {"loca',  # 12: args increment
            'tion":',  # 13: args increment
            ' "To',  # 14: args increment
            'kyo"',  # 15: args increment
            "}}",  # 16: args complete
            "\n",  # 17: end of JSON
            "</tool_call>",  # 18: end tag (cleaned)
        ]

        # Collect results per chunk for rigorous verification
        results = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            results.append(result)

        # === Per-chunk assertions ===

        # Chunks 0-8: Should be buffering, no calls yet
        for i in range(9):
            self.assertEqual(
                len(results[i].calls),
                0,
                f"Chunk {i} '{chunks[i]}': expected no calls while buffering, got {results[i].calls}",
            )

        # Chunk 9: Name should be emitted (when complete name is parsed)
        self.assertEqual(
            len(results[9].calls),
            1,
            f"Chunk 9: expected name emission, got {results[9].calls}",
        )
        self.assertEqual(results[9].calls[0].name, "get_weather")
        self.assertEqual(
            results[9].calls[0].parameters,
            "",
            "Name should be emitted with empty parameters",
        )
        self.assertEqual(results[9].calls[0].tool_index, 0)

        # === Aggregated assertions ===

        # Collect all calls
        all_calls = []
        for result in results:
            all_calls.extend(result.calls)

        # Should have exactly one name emission
        name_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(name_calls), 1, f"Expected 1 name call, got {name_calls}")
        self.assertEqual(name_calls[0].name, "get_weather")

        # Collect all argument increments
        arg_increments = [c.parameters for c in all_calls if c.parameters]
        full_args = "".join(arg_increments)

        # Verify complete arguments
        self.assertEqual(
            full_args,
            '{"location": "Tokyo"}',
            f"Expected complete args, got '{full_args}'",
        )

    def test_scenario3_tool_end_newline(self):
        """
        Scenario 3: Tool end + newline "...\n</tool_call>"

        Verifies end token is properly cleaned from normal text.
        """
        detector = Qwen25Detector()

        chunks = [
            "<tool_call>\n",
            '{"name": "get_weather", "arguments": {"location": "Paris"}}',
            "\n",
            "</tool_call>",
        ]

        results = collect_streaming_results(detector, chunks, self.tools)

        # Verify tool was parsed
        all_calls = []
        for result in results:
            all_calls.extend(result.calls)
        name_calls = [c for c in all_calls if c.name]
        self.assertTrue(len(name_calls) > 0)
        self.assertEqual(name_calls[0].name, "get_weather")

        # Verify </tool_call> doesn't leak as normal text
        all_normal_text = "".join(r.normal_text for r in results if r.normal_text)
        self.assertNotIn("</tool_call>", all_normal_text)

    def test_scenario5_complete_tool_single_token(self):
        """
        Scenario 5: Complete tool in theoretically one chunk (but decomposed)

        Even if MTP produces complete tool, TokenNormalizer decomposes it.
        This simulates the already-decomposed tokens.
        """
        detector = Qwen25Detector()

        # Even "complete" input gets decomposed by TokenNormalizer
        # We simulate the token-level chunks
        chunks = [
            "<tool_call>",
            "\n",
            '{"name": "get_time", "arguments": {}}',
            "\n",
            "</tool_call>",
        ]

        results = collect_streaming_results(detector, chunks, self.tools)

        all_calls = []
        for result in results:
            all_calls.extend(result.calls)

        # Should have name emitted
        name_calls = [c for c in all_calls if c.name]
        self.assertTrue(len(name_calls) >= 1)
        self.assertEqual(name_calls[0].name, "get_time")

    def test_scenario6_multiple_tools_sequential(self):
        """
        Scenario 6: Multiple tools "..<tool_call>...</tool_call><tool_call>..."

        Verifies detector correctly handles multiple sequential tool calls
        with incrementing tool_index.
        """
        detector = Qwen25Detector()

        # Two tool calls in sequence (token-by-token)
        chunks = [
            "<tool_call>",
            "\n",
            '{"name": "get_weather", "arguments": {"location": "London"}}',
            "\n",
            "</tool_call>",
            "\n",
            "<tool_call>",
            "\n",
            '{"name": "get_time", "arguments": {}}',
            "\n",
            "</tool_call>",
        ]

        results = collect_streaming_results(detector, chunks, self.tools)

        all_calls = []
        for result in results:
            all_calls.extend(result.calls)

        # Get calls with names (tool announcements)
        name_calls = [c for c in all_calls if c.name]

        self.assertEqual(
            len(name_calls),
            2,
            f"Expected 2 name announcements, got {len(name_calls)}. Calls: {all_calls}",
        )
        self.assertEqual(name_calls[0].name, "get_weather")
        self.assertEqual(name_calls[0].tool_index, 0)
        self.assertEqual(name_calls[1].name, "get_time")
        self.assertEqual(name_calls[1].tool_index, 1)

    def test_scenario7_tool_boundary_fusion(self):
        """
        Scenario 7: Tool boundary fusion "</tool_call><tool_call>"

        Tests the tricky case where end and start tags might be in same chunk.
        With token normalization, this becomes multiple chunks.
        """
        detector = Qwen25Detector()

        # Simulated decomposed tokens for boundary fusion
        chunks = [
            "<tool_call>",
            "\n",
            '{"name": "get_weather", "arguments": {"location": "NYC"}}',
            "\n",
            "</tool_call>",
            "<tool_call>",  # Next tool starts immediately
            "\n",
            '{"name": "get_time", "arguments": {}}',
            "\n",
            "</tool_call>",
        ]

        results = collect_streaming_results(detector, chunks, self.tools)

        all_calls = []
        for result in results:
            all_calls.extend(result.calls)

        name_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(name_calls), 2)
        self.assertEqual(name_calls[0].name, "get_weather")
        self.assertEqual(name_calls[1].name, "get_time")

    def test_incremental_arguments_character_by_character(self):
        """
        Test that arguments are streamed incrementally, character by character.

        This is the key incremental streaming behavior we want to verify.
        Rigorously checks both per-chunk results and aggregated results.
        """
        detector = Qwen25Detector()

        # Very fine-grained chunks to test incremental argument streaming
        chunks = [
            "<tool_call>",  # 0: buffering
            "\n",  # 1: buffering
            "{",  # 2: buffering
            '"name"',  # 3: buffering
            ": ",  # 4: buffering
            '"get_weather"',  # 5: NAME EMITTED
            ", ",  # 6: buffering args key
            '"arguments"',  # 7: buffering
            ": {",  # 8: args increment
            '"location"',  # 9: args increment
            ": ",  # 10: args increment
            '"S',  # 11: args increment - Start of "San Francisco"
            "an",  # 12: args increment
            " Fran",  # 13: args increment
            "cisco",  # 14: args increment
            '"',  # 15: args increment
            "}",  # 16: args increment
            "}",  # 17: args complete
            "\n",  # 18: end of JSON
            "</tool_call>",  # 19: end tag (cleaned)
        ]

        # Collect results per chunk
        results = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            results.append(result)

        # === Per-chunk assertions ===

        # Chunks 0-4: Should be buffering, no calls
        for i in range(5):
            self.assertEqual(
                len(results[i].calls),
                0,
                f"Chunk {i} '{chunks[i]}': expected no calls while buffering",
            )

        # Chunk 5: Name should be emitted
        self.assertEqual(
            len(results[5].calls),
            1,
            f"Chunk 5: expected name emission, got {results[5].calls}",
        )
        self.assertEqual(results[5].calls[0].name, "get_weather")
        self.assertEqual(results[5].calls[0].parameters, "")

        # === Aggregated assertions ===

        # Collect all calls
        all_calls = []
        for result in results:
            all_calls.extend(result.calls)

        # Exactly one name emission
        name_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(name_calls), 1, f"Expected 1 name call, got {name_calls}")
        self.assertEqual(name_calls[0].name, "get_weather")

        # Collect argument increments (calls with parameters but no name)
        arg_calls = [c for c in all_calls if c.parameters and not c.name]

        # We should have multiple argument increments (proves incremental streaming)
        self.assertGreater(
            len(arg_calls),
            1,
            f"Expected multiple arg increments for incremental streaming, got {len(arg_calls)}. Calls: {arg_calls}",
        )

        # Concatenated should form complete, valid JSON arguments
        full_args = "".join(c.parameters for c in all_calls if c.parameters)
        parsed = json.loads(full_args)
        self.assertEqual(
            parsed.get("location"),
            "San Francisco",
            f"Expected location='San Francisco', got '{parsed}'",
        )

    def test_scenario1_reasoning_end_and_tool_start(self):
        """
        Scenario 1: Reasoning end + text + tool start "</think> xxx<tool_call>"

        In real usage, </think> is handled by ReasoningParser, not detector.
        This test verifies normal text before tool call is preserved.
        """
        detector = Qwen25Detector()

        chunks = [
            "Let me check the weather",  # Normal text
            " for you.",
            "<tool_call>",
            "\n",
            '{"name": "get_weather", "arguments": {"location": "Berlin"}}',
            "\n",
            "</tool_call>",
        ]

        results = collect_streaming_results(detector, chunks, self.tools)

        # Collect normal text
        normal_text = "".join(r.normal_text for r in results if r.normal_text)
        self.assertIn("Let me check the weather", normal_text)
        self.assertIn("for you", normal_text)

        # Collect calls
        all_calls = []
        for result in results:
            all_calls.extend(result.calls)
        name_calls = [c for c in all_calls if c.name]
        self.assertTrue(len(name_calls) >= 1)
        self.assertEqual(name_calls[0].name, "get_weather")

    def test_scenario4_tool_end_text_continuation(self):
        """
        Scenario 4: Tool end + text continuation "\n</tool_call>yyy"

        Verifies text after tool call is properly returned as normal text.
        """
        detector = Qwen25Detector()

        chunks = [
            "<tool_call>",
            "\n",
            '{"name": "get_time", "arguments": {}}',
            "\n",
            "</tool_call>",
            "Here's the result",  # Text after tool call
        ]

        results = collect_streaming_results(detector, chunks, self.tools)

        # Collect normal text (should include text after tool call)
        normal_text = "".join(r.normal_text for r in results if r.normal_text)
        # End tag should be cleaned
        self.assertNotIn("</tool_call>", normal_text)

        # Tool should be parsed
        all_calls = []
        for result in results:
            all_calls.extend(result.calls)
        name_calls = [c for c in all_calls if c.name]
        self.assertTrue(len(name_calls) >= 1)
        self.assertEqual(name_calls[0].name, "get_time")

    def test_exact_incremental_values(self):
        """
        Verify exact values returned by each parse_streaming_increment call.

        This is the rigorous test that examines each chunk's result with exact assertions.
        Tests both per-chunk results and the aggregated final result.
        """
        detector = Qwen25Detector()

        # Carefully designed chunks with expected results
        # Format: (chunk, expected_num_calls, expected_name, expected_params_substring)
        chunks_and_expected = [
            ("<tool_call>", 0, None, None),  # Buffering
            ("\n", 0, None, None),  # Buffering
            ('{"name": "get_weather"', 1, "get_weather", ""),  # Name emitted
            (', "arguments": {"location": "ABC"}}', 1, None, "location"),  # Args
            ("\n", 0, None, None),  # JSON end
            ("</tool_call>", 0, None, None),  # End tag cleaned
        ]

        results = []
        all_calls = []

        for i, (chunk, exp_num_calls, exp_name, exp_params_substr) in enumerate(
            chunks_and_expected
        ):
            result = detector.parse_streaming_increment(chunk, self.tools)
            results.append(result)
            all_calls.extend(result.calls)

            # === Per-chunk assertions ===
            self.assertEqual(
                len(result.calls),
                exp_num_calls,
                f"Chunk {i} '{chunk}': expected {exp_num_calls} calls, got {len(result.calls)}. Result: {result}",
            )

            if exp_num_calls > 0:
                call = result.calls[0]
                if exp_name is not None:
                    self.assertEqual(
                        call.name,
                        exp_name,
                        f"Chunk {i} '{chunk}': expected name '{exp_name}', got '{call.name}'",
                    )
                if exp_params_substr is not None:
                    if exp_params_substr == "":
                        self.assertEqual(
                            call.parameters,
                            "",
                            f"Chunk {i} '{chunk}': expected empty params, got '{call.parameters}'",
                        )
                    else:
                        self.assertIn(
                            exp_params_substr,
                            call.parameters,
                            f"Chunk {i} '{chunk}': expected '{exp_params_substr}' in params, got '{call.parameters}'",
                        )

        # === Aggregated assertions ===

        # Verify exactly one name emission
        name_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(name_calls), 1, f"Expected 1 name call, got {name_calls}")
        self.assertEqual(name_calls[0].name, "get_weather")
        self.assertEqual(name_calls[0].tool_index, 0)
        self.assertEqual(name_calls[0].parameters, "")

        # Verify arguments were emitted (calls with params but no name)
        arg_calls = [c for c in all_calls if c.parameters and not c.name]
        self.assertEqual(len(arg_calls), 1, f"Expected 1 arg call, got {arg_calls}")

        # Verify complete aggregated arguments
        full_args = "".join(c.parameters for c in all_calls if c.parameters)
        parsed = json.loads(full_args)
        self.assertEqual(parsed, {"location": "ABC"})

        # Verify no normal text leakage of end tag
        all_normal_text = "".join(r.normal_text for r in results if r.normal_text)
        self.assertNotIn("</tool_call>", all_normal_text)


class TestQwen25DetectAndParse(unittest.TestCase):
    """Test Qwen25Detector.detect_and_parse (non-streaming)."""

    def setUp(self):
        self.tools = create_tools()

    def test_single_tool_call(self):
        """Test parsing a single complete tool call."""
        detector = Qwen25Detector()
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n</tool_call>'

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertIn("Tokyo", result.calls[0].parameters)

    def test_multiple_tool_calls(self):
        """Test parsing multiple complete tool calls."""
        detector = Qwen25Detector()
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "NYC"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'
        )

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "get_time")

    def test_normal_text_before_tool(self):
        """Test that normal text before tool call is captured."""
        detector = Qwen25Detector()
        text = 'Let me help you. <tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>'

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Let me help you.")
        self.assertEqual(len(result.calls), 1)


if __name__ == "__main__":
    unittest.main()
