import json
import os
import unittest
from typing import List, Optional, Tuple

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen25_detector import (
    Qwen25Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.token_normalizer import (
    TokenNormalizer,
    expand_prev_window,
)

# Delta token IDs per MTP step, captured from a production SSE stream
# (Qwen3-8B fine-tune emitting a search_item tool call). The boundary
# between steps 7→8 and 13→14 splits ' 免' across tokens, leaving an
# orphan UTF-8 tail byte in the next step's prev-token window.
_PRODUCTION_MTP_STEPS: List[List[int]] = [
    [151657],
    [198, 4913, 606, 788],
    [330, 1836, 5634, 497],
    [330, 16370, 788, 5212],
    [56431, 788, 330, 113335],
    [99364],
    [39165, 34369],
    [235, 105596, 80268, 497],
    [330, 52473],
    [2859],
    [788],
    [330, 113335, 99364],
    [39165, 34369],
    [235, 105596, 80268, 95642],
    [151658],
]
_PRODUCTION_EXPECTED_ARGS = json.dumps(
    {"intent": "快餐便当 免配送费", "rewriteQuery": "快餐便当 免配送费"},
    ensure_ascii=False,
)

# Exact raw model output from the user's bug report. Different request
# from _PRODUCTION_MTP_STEPS (different intent value); both exercise the
# same UTF-8 prev-window bug class.
_USER_REPORTED_RAW_OUTPUT = (
    "<tool_call>\n"
    '{"name": "search_item", '
    '"arguments": {"intent": "快餐便当", '
    '"rewriteQuery": "快餐便当 免配送费"}}'
    "\n</tool_call>"
)
_USER_REPORTED_EXPECTED_ARGS = (
    '{"intent": "快餐便当", "rewriteQuery": "快餐便当 免配送费"}'
)


def _get_qwen_tokenizer():
    """Load a Qwen tokenizer from repo testdata. Returns None if unavailable."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    path = "rtp_llm/test/model_test/fake_test/testdata/qwen3_30b/tokenizer"
    if os.path.exists(path):
        return AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, verbose=False
        )
    return None


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


class TestStreamingStringWithSpaces(unittest.TestCase):
    """Test that string values containing spaces are not truncated during streaming.

    This guards against a bug where partial_json_parser auto-adds closing quotes
    to incomplete strings, and the streaming logic failed to handle the first
    argument streaming after tool name was sent (prev_arguments was None).

    Bug: When streaming "可乐 薯片 饼干", the content after the first space was lost.
    """

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search_item",
                    description="Search for items",
                    parameters={
                        "type": "object",
                        "properties": {
                            "intent": {"type": "string"},
                            "isDeduplicate": {"type": "integer"},
                        },
                    },
                ),
            )
        ]

    def test_string_with_spaces_not_truncated(self):
        """
        Test that string values containing spaces are NOT truncated during streaming.

        This is the exact bug scenario:
        - Chunk 1: tool name + partial string "可乐"
        - Chunk 2: more string content " 薯片" (space + content)
        - Chunk 3: remaining string " 饼干..." (space + content)

        Previously, chunks 2 and 3 were lost because:
        1. After chunk 1, prev_tool_call_arr was not updated
        2. When chunk 2 arrived, prev_arguments was None
        3. The elif branch was skipped, argument_diff stayed None
        """
        detector = Qwen25Detector()

        chunks = [
            detector.bot_token,
            '{"name": "search_item", "arguments": {"intent": "可乐',
            " ",
            "薯片",  # This was being lost
            " ",
            "饼干",
            '"',
            ",",
            " ",
            '"',
            "isDeduplicate",
            '"',
            ":",
            "0}}",
            detector.eot_token,
        ]

        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        # Collect all parameters
        final_params = "".join([c.parameters for c in all_calls if c.parameters])
        expected = json.dumps(
            {"intent": "可乐 薯片 饼干", "isDeduplicate": 0}, ensure_ascii=False
        )

        self.assertEqual(
            final_params,
            expected,
            f"Expected '{expected}' but got '{final_params}'. "
            f"String with spaces was truncated!",
        )

    def test_string_with_spaces_character_by_character(self):
        """
        Test string with spaces streamed character by character.

        This is an even more extreme case where each character arrives separately.
        """
        detector = Qwen25Detector()

        # Character-by-character streaming after the tool name is sent
        chunks = [
            detector.bot_token,
            '{"name": "search_item", "arguments": {"intent": "',
            "a",  # First char
            " ",  # Space - critical test
            "b",  # Char after space
            " ",
            "c",
            '"}',
            "}",
            detector.eot_token,
        ]

        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        final_params = "".join([c.parameters for c in all_calls if c.parameters])
        expected = json.dumps({"intent": "a b c"}, ensure_ascii=False)

        self.assertEqual(final_params, expected)

    def test_multiple_string_parameters_with_spaces(self):
        """Test multiple string parameters, each with spaces."""
        detector = Qwen25Detector()

        tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "filter": {"type": "string"},
                        },
                    },
                ),
            )
        ]

        chunks = [
            detector.bot_token,
            '{"name": "search", "arguments": {"query": "hello',
            " world",
            '", "filter": "foo',
            " bar",
            '"}}',
            detector.eot_token,
        ]

        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            all_calls.extend(result.calls)

        final_params = "".join([c.parameters for c in all_calls if c.parameters])
        expected = json.dumps({"query": "hello world", "filter": "foo bar"})

        self.assertEqual(final_params, expected)


class TestAdjacentValueBleedThrough(unittest.TestCase):
    """Regression test for adjacent-value bleed-through.

    Real-world bug shape: a tool call whose second string argument re-uses
    content from the first string argument. If any boundary between the two
    values gets dropped from the buffer, partial_json_parser reads the
    combined content as a single key's value (because the closing quote /
    comma / next-key prefix are missing), and the streamed args end up
    looking like ``{"intent": "<v1> <v2-prefix>...`` -- the second value
    bleeds into the first.

    This class drives the detector through every chunk-boundary that could
    plausibly arise in MTP / token-by-token streaming, plus several
    pathological boundaries that exercise the diff-with-baseline,
    is_complete=True, and partial_json_parser trailing-whitespace-strip
    interactions.
    """

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search_item",
                    parameters={
                        "type": "object",
                        "properties": {
                            "intent": {"type": "string"},
                            "rewriteQuery": {"type": "string"},
                        },
                    },
                ),
            )
        ]
        self.target_args = {
            "intent": "快餐便当",
            "rewriteQuery": "快餐便当 免配送费",
        }
        self.expected = json.dumps(self.target_args, ensure_ascii=False)

    def _drive(self, detector: Qwen25Detector, chunks: List[str]) -> str:
        for chunk in chunks:
            detector.parse_streaming_increment(chunk, self.tools)
        return "".join(detector.streamed_args_for_tool)

    def test_token_by_token_natural_qwen_split(self):
        """The Qwen tokenizer splits this target into ~35 tokens. Token-by-token
        streaming must produce the full args."""
        # These chunks mirror the actual Qwen tokenization of:
        # <tool_call>\n{"name": "search_item", "arguments":
        #   {"intent": "快餐便当", "rewriteQuery": "快餐便当 免配送费"}}\n</tool_call>
        chunks = [
            "<tool_call>",
            "\n",
            '{"',
            "name",
            '":',
            ' "',
            "search",
            "_item",
            '",',
            ' "',
            "arguments",
            '":',
            ' {"',
            "intent",
            '":',
            ' "',
            "快餐",
            "便",
            "当",
            '",',
            ' "',
            "rewrite",
            "Query",
            '":',
            ' "',
            "快餐",
            "便",
            "当",
            " ",
            "免",
            "配送",
            "费",
            '"}}',
            "\n",
            "</tool_call>",
        ]
        self.assertEqual(self._drive(Qwen25Detector(), chunks), self.expected)

    def test_chunk_boundary_at_value_separator(self):
        """Boundary lands EXACTLY between intent's closing quote and
        rewriteQuery's opening quote -- the most likely place for
        bleed-through if a structural char gets lost."""
        chunks = [
            '<tool_call>\n{"name": "search_item", "arguments": {"intent": "快餐便当',
            '"',
            ", ",
            '"rewriteQuery": "',
            "快餐便当 ",
            "免配送费",
            '"}}\n</tool_call>',
        ]
        self.assertEqual(self._drive(Qwen25Detector(), chunks), self.expected)

    def test_trailing_space_at_chunk_boundary(self):
        """Buffer ends with ``"快餐便当 `` (trailing space inside open string).
        partial_json_parser strips the trailing space, producing
        ``{"intent": "快餐便当", "rewriteQuery": "快餐便当"}``. The next
        chunk extends the value past the space; the diff machinery must
        not have already committed a ``"`` after the stripped whitespace."""
        chunks = [
            '<tool_call>\n{"name": "search_item", "arguments": {"intent": "快餐便当", "rewriteQuery": "快餐便当 ',
            "免",
            "配送",
            "费",
            '"}}\n</tool_call>',
        ]
        self.assertEqual(self._drive(Qwen25Detector(), chunks), self.expected)

    def test_name_and_first_value_complete_in_same_chunk(self):
        """A single chunk delivers the tool name AND the complete first
        argument value. This exercises the Case-1 prev_tool_call_arr
        seeding fix: without it, the next chunk's diff calculation has a
        baseline of ``{}`` and must reconstruct the intent value via the
        common-prefix machinery on the next iteration."""
        chunks = [
            '<tool_call>\n{"name": "search_item", "arguments": {"intent": "快餐便当", "rewriteQuery": "',
            "快餐便当 免配送费",
            '"}}\n</tool_call>',
        ]
        self.assertEqual(self._drive(Qwen25Detector(), chunks), self.expected)

    def test_one_shot_complete_tool_call(self):
        """The entire tool call arrives in one chunk (degenerate MTP=many).
        is_current_complete fires immediately; argument_diff must equal
        the full args JSON."""
        chunks = [
            '<tool_call>\n{"name": "search_item", "arguments": '
            + self.expected
            + "}\n</tool_call>",
        ]
        self.assertEqual(self._drive(Qwen25Detector(), chunks), self.expected)

    def test_streamed_args_invariant_holds(self):
        """At every step of streaming, streamed_args_for_tool must remain a
        prefix of the eventually-correct args JSON. This is the core
        defensive invariant: any deviation signals corrupted state."""
        chunks = [
            "<tool_call>",
            "\n",
            '{"',
            "name",
            '":',
            ' "',
            "search",
            "_item",
            '",',
            ' "',
            "arguments",
            '":',
            ' {"',
            "intent",
            '":',
            ' "',
            "快餐",
            "便",
            "当",
            '",',
            ' "',
            "rewrite",
            "Query",
            '":',
            ' "',
            "快餐",
            "便",
            "当",
            " ",
            "免",
            "配送",
            "费",
            '"}}',
            "\n",
            "</tool_call>",
        ]
        detector = Qwen25Detector()
        for i, chunk in enumerate(chunks):
            detector.parse_streaming_increment(chunk, self.tools)
            so_far = "".join(detector.streamed_args_for_tool)
            self.assertTrue(
                self.expected.startswith(so_far),
                f"After chunk {i} ({chunk!r}), streamed args {so_far!r} "
                f"is not a prefix of expected {self.expected!r}",
            )
        self.assertEqual("".join(detector.streamed_args_for_tool), self.expected)

    def test_value_bleed_does_not_corrupt_intent(self):
        """The exact symptom shape from the user report: streamed args must
        NEVER contain ``"intent": "快餐便当 免配送`` -- which would mean the
        rewriteQuery value bled into intent's value."""
        chunks = [
            "<tool_call>",
            "\n",
            '{"',
            "name",
            '":',
            ' "',
            "search",
            "_item",
            '",',
            ' "',
            "arguments",
            '":',
            ' {"',
            "intent",
            '":',
            ' "',
            "快餐",
            "便",
            "当",
            '",',
            ' "',
            "rewrite",
            "Query",
            '":',
            ' "',
            "快餐",
            "便",
            "当",
            " ",
            "免",
            "配送",
            "费",
            '"}}',
            "\n",
            "</tool_call>",
        ]
        detector = Qwen25Detector()
        # Drive streaming and snapshot at every step
        snapshots = []
        for chunk in chunks:
            detector.parse_streaming_increment(chunk, self.tools)
            snapshots.append("".join(detector.streamed_args_for_tool))
        # Bleed-through marker: intent's value extending past 快餐便当 without
        # a closing quote first.
        bleed_marker = '"intent": "快餐便当 '
        for snap in snapshots:
            self.assertNotIn(
                bleed_marker,
                snap,
                f"Detected bleed-through: {snap!r} contains "
                f"intent value extending past its closing quote.",
            )


class TestUserReportedSearchItemRegression(unittest.TestCase):
    """Drives the user-reported raw output through the real Qwen tokenizer
    + TokenNormalizer + Qwen25Detector at multiple MTP sizes. Asserts
    the per-step prefix invariant and that the SSE-aggregated arguments
    equal the non-streaming canonical form."""

    RAW_OUTPUT = _USER_REPORTED_RAW_OUTPUT
    EXPECTED_ARGS = _USER_REPORTED_EXPECTED_ARGS
    # Bleed marker: intent's value extending past its first word without
    # a closing quote means a later value's content has leaked into it.
    BLEED_MARKER = '"intent": "快餐便当 '

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _get_qwen_tokenizer()
        cls.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search_item",
                    parameters={
                        "type": "object",
                        "properties": {
                            "intent": {"type": "string"},
                            "rewriteQuery": {"type": "string"},
                        },
                    },
                ),
            )
        ]

    def _drive_through_normalizer(
        self, mtp_size: int
    ) -> Tuple[List[ToolCallItem], Qwen25Detector]:
        """Tokenize RAW_OUTPUT, deliver in fixed-size MTP chunks through
        TokenNormalizer + Qwen25Detector, and return all emitted
        ToolCallItems alongside the detector.

        Mirrors the renderer's state model: ``output_ids`` accumulates,
        ``last_output_ids`` only advances when the normalizer yields.
        Skipped tokens (e.g. incomplete UTF-8)
        stay queued in ``output_ids`` so a later chunk can resolve them.
        """
        if self.tokenizer is None:
            self.skipTest("Qwen tokenizer not available")
        ids = self.tokenizer.encode(self.RAW_OUTPUT)
        detector = Qwen25Detector()
        output_ids: List[int] = []  # cumulative across all chunks
        last_output_ids: List[int] = []  # cursor; may lag if normalizer skips
        last_token_length = 0
        all_items = []
        i = 0
        while i < len(ids):
            chunk_ids = ids[i : i + mtp_size]
            i += mtp_size
            output_ids = output_ids + chunk_ids
            new_tokens = output_ids[len(last_output_ids) :]
            prev_token_id = (
                last_output_ids[-last_token_length:] if last_token_length > 0 else []
            )
            normalizer = TokenNormalizer(self.tokenizer)
            any_yield = False
            for delta in normalizer.normalize_tokens(prev_token_id, new_tokens):
                any_yield = True
                result = detector.parse_streaming_increment(delta, self.tools)
                all_items.extend(result.calls)
                # Per-step prefix invariant: streamed_args must remain a
                # strict prefix of the eventual EXPECTED_ARGS.
                streamed = "".join(detector.streamed_args_for_tool)
                self.assertTrue(
                    self.EXPECTED_ARGS.startswith(streamed),
                    f"Per-step prefix invariant violated at MTP={mtp_size}: "
                    f"streamed={streamed!r} is not a prefix of "
                    f"expected={self.EXPECTED_ARGS!r}",
                )
                # Bleed marker must NEVER appear in streamed content.
                self.assertNotIn(
                    self.BLEED_MARKER,
                    streamed,
                    f"Bleed-through detected at MTP={mtp_size}: {streamed!r}",
                )
            if any_yield and new_tokens:
                last_token_length = len(new_tokens)
                last_output_ids = list(output_ids)
                last_token_length = expand_prev_window(
                    self.tokenizer, last_output_ids, last_token_length
                )
        return all_items, detector

    def _assert_sse_aggregation(self, items, detector):
        """Mirror what an OpenAI streaming client does: concatenate
        ``parameters`` across ToolCallItems by tool_index."""
        # Exactly one name announcement
        names = [c for c in items if c.name]
        self.assertEqual(
            len(names), 1, f"Expected 1 name announcement, got {len(names)}: {items}"
        )
        self.assertEqual(names[0].name, "search_item")
        # Concatenated arguments must equal the non-streaming canonical form
        aggregated = "".join(c.parameters for c in items if c.parameters)
        self.assertEqual(
            aggregated,
            self.EXPECTED_ARGS,
            f"SSE aggregation mismatch: got {aggregated!r}, "
            f"expected {self.EXPECTED_ARGS!r}",
        )
        # And the detector's internal cursor must match (same source of truth)
        self.assertEqual("".join(detector.streamed_args_for_tool), self.EXPECTED_ARGS)

    def test_real_tokenizer_token_by_token(self):
        items, detector = self._drive_through_normalizer(mtp_size=1)
        self._assert_sse_aggregation(items, detector)

    def test_real_tokenizer_mtp_2(self):
        items, detector = self._drive_through_normalizer(mtp_size=2)
        self._assert_sse_aggregation(items, detector)

    def test_real_tokenizer_mtp_3(self):
        items, detector = self._drive_through_normalizer(mtp_size=3)
        self._assert_sse_aggregation(items, detector)

    def test_real_tokenizer_mtp_4(self):
        items, detector = self._drive_through_normalizer(mtp_size=4)
        self._assert_sse_aggregation(items, detector)

    def test_real_tokenizer_mtp_5(self):
        items, detector = self._drive_through_normalizer(mtp_size=5)
        self._assert_sse_aggregation(items, detector)

    def test_real_tokenizer_mtp_8(self):
        """Large MTP chunks where multiple tokens cross structural
        JSON boundaries within a single chunk."""
        items, detector = self._drive_through_normalizer(mtp_size=8)
        self._assert_sse_aggregation(items, detector)


class TestProductionMTPStepBoundaries(unittest.TestCase):
    """Detector-level reproduction: drives the production MTP token stream
    through the normalizer + detector while inlining the renderer's
    prev-window expansion. Asserts the streamed args equal the expected
    JSON byte-for-byte."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _get_qwen_tokenizer()
        cls.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search_item",
                    parameters={
                        "type": "object",
                        "properties": {
                            "intent": {"type": "string"},
                            "rewriteQuery": {"type": "string"},
                        },
                    },
                ),
            )
        ]

    def test_production_mtp_steps_with_prev_window_fix(self):
        if self.tokenizer is None:
            self.skipTest("Qwen tokenizer not available")

        detector = Qwen25Detector()
        output_ids: List[int] = []
        last_output_ids: List[int] = []
        last_token_length = 0

        for step in _PRODUCTION_MTP_STEPS:
            output_ids = output_ids + step
            new_tokens = output_ids[len(last_output_ids) :]

            prev_token_id = (
                last_output_ids[-last_token_length:] if last_token_length > 0 else []
            )
            normalizer = TokenNormalizer(self.tokenizer)
            any_yield = False
            for delta in normalizer.normalize_tokens(prev_token_id, new_tokens):
                any_yield = True
                detector.parse_streaming_increment(delta, self.tools)

            if any_yield and new_tokens:
                last_token_length = len(new_tokens)
                last_output_ids = list(output_ids)
                last_token_length = expand_prev_window(
                    self.tokenizer, last_output_ids, last_token_length
                )

        streamed = "".join(detector.streamed_args_for_tool)
        self.assertEqual(streamed, _PRODUCTION_EXPECTED_ARGS)


class TestRendererPrevWindowExpansion(unittest.IsolatedAsyncioTestCase):
    """End-to-end check that the renderer (not the detector + normalizer
    in isolation) streams the full args across MTP step boundaries with
    orphan UTF-8 tail bytes. Uses a production debug_info token capture."""

    async def _drive_renderer(self, tokenizer, token_chunks: List[List[int]]) -> str:
        # Imports are local: keep the file importable without the renderer's
        # transitive deps when the tokenizer (and so this test) is unavailable.
        import torch

        from rtp_llm.openai.api_datatype import (
            ChatCompletionRequest,
            ChatMessage,
            GPTFunctionDefinition,
            GPTToolDefinition,
            RoleEnum,
        )
        from rtp_llm.openai.renderers.custom_renderer import RendererParams
        from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
            ReasoningToolBaseRenderer,
        )
        from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
            BaseFormatDetector,
        )
        from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import (
            ReasoningParser,
        )
        from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput

        class _Renderer(ReasoningToolBaseRenderer):
            def _setup_chat_template(self):
                self.chat_template = "test"

            def in_think_mode(self, request: ChatCompletionRequest) -> bool:
                return False

            def _create_detector(
                self, request: ChatCompletionRequest
            ) -> Optional[BaseFormatDetector]:
                return Qwen25Detector() if request.tools else None

            def _create_reasoning_parser(
                self, request: ChatCompletionRequest
            ) -> Optional[ReasoningParser]:
                return None

        from rtp_llm.config.py_config_modules import GenerateEnvConfig

        renderer = _Renderer(
            tokenizer=tokenizer,
            renderer_params=RendererParams(
                model_type="test",
                max_seq_len=2048,
                eos_token_id=0,
                stop_word_ids_list=[],
            ),
            generate_env_config=GenerateEnvConfig(),
        )
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            tools=[
                GPTToolDefinition(
                    type="function",
                    function=GPTFunctionDefinition(
                        name="search_item",
                        description="search item",
                        parameters={
                            "type": "object",
                            "properties": {
                                "intent": {"type": "string"},
                                "rewriteQuery": {"type": "string"},
                            },
                        },
                    ),
                )
            ],
        )

        (status,) = await renderer._create_status_list(1, request)
        collected_args = ""
        # update_output() treats output.output_ids as per-iteration delta
        # (it appends to its own accumulator), so pass per-step deltas.
        for chunk in token_chunks:
            aux = AuxInfo()
            aux.input_len = 0
            aux.output_len = len(chunk)
            aux.reuse_len = 0
            output = GenerateOutput()
            output.output_ids = torch.tensor([chunk])
            output.aux_info = aux

            delta = await renderer._update_single_status(
                status,
                output,
                max_new_tokens=4096,
                stop_words_str=[],
                stop_word_slice_list=[],
                is_streaming=True,
            )
            if delta is None:
                continue
            msg = delta.output_str
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function and tc.function.arguments:
                        collected_args += tc.function.arguments
        return collected_args

    async def test_renderer_streams_production_mtp_capture(self):
        """Production debug_info MTP step capture. Different request from
        the user report; same UTF-8 prev-window bug class."""
        tokenizer = _get_qwen_tokenizer()
        if tokenizer is None:
            self.skipTest("Qwen tokenizer not available")
        collected = await self._drive_renderer(tokenizer, _PRODUCTION_MTP_STEPS)
        self.assertEqual(
            collected,
            _PRODUCTION_EXPECTED_ARGS,
            "Renderer dropped or corrupted args across MTP boundaries; "
            "the prev-window expansion in reasoning_tool_base_renderer.py "
            "is likely missing or broken.",
        )

    # No user-reported renderer test: the bug corrupts the field emitted
    # AFTER the orphan boundary, and the user-reported token stream only
    # has an orphan in the last value — so any corruption falls past the
    # tool call's end and is invisible to the client. Detector-level
    # user-reported coverage at MTP=1..8 lives in
    # TestUserReportedSearchItemRegression above.


class TestDivergenceGuard(unittest.TestCase):
    """Negative test for the prefix-invariant divergence guard in
    BaseFormatDetector.parse_streaming_increment (P1.3)."""

    def test_diverged_streamed_args_are_dropped(self):
        """When streamed_args_for_tool diverges from the current parse,
        the guard must drop the diff and log an error."""
        detector = Qwen25Detector()
        tools = create_tools()

        # Stream a partial tool call to seed streamed_args_for_tool
        chunks = [
            '<tool_call>\n{"name": "get_weather", "arguments": {"loca',
            'tion": "Tokyo"',
        ]
        for c in chunks:
            detector.parse_streaming_increment(c, tools)

        # Mutate streamed state to simulate divergence
        self.assertTrue(len(detector.streamed_args_for_tool) > 0)
        detector.streamed_args_for_tool[0] = '{"location": "CORRUPTED'

        # Next chunk should trigger the guard: diff is dropped, not emitted
        with self.assertLogs(
            "rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector",
            level="WARNING",
        ) as cm:
            result = detector.parse_streaming_increment("}}", tools)

        self.assertTrue(
            any("diverges" in msg for msg in cm.output),
            f"Expected divergence error log, got: {cm.output}",
        )
        # No argument diff should have been emitted
        arg_calls = [c for c in result.calls if c.parameters]
        self.assertEqual(arg_calls, [], "Diverged diff must be dropped")
        # streamed_args must not have grown
        self.assertEqual(detector.streamed_args_for_tool[0], '{"location": "CORRUPTED')


if __name__ == "__main__":
    unittest.main()
