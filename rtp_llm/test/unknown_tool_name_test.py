"""
Tests for unknown tool name handling with RTP_LLM_FORWARD_UNKNOWN_TOOLS env switch.

Covers both non-streaming (parse_base_json) and streaming (parse_streaming_increment)
paths via Qwen25Detector, which delegates streaming to BaseFormatDetector.
"""

import json
import os
import unittest
from typing import List

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen25_detector import (
    Qwen25Detector,
)


def _make_tools() -> List[Tool]:
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                parameters={"type": "object", "properties": {}},
            ),
        )
    ]


class TestUnknownToolNameNonStreaming(unittest.TestCase):
    """Tests for detect_and_parse (non-streaming path via parse_base_json)."""

    def setUp(self):
        self.tools = _make_tools()
        self._orig_env = os.environ.get("RTP_LLM_FORWARD_UNKNOWN_TOOLS")

    def tearDown(self):
        if self._orig_env is None:
            os.environ.pop("RTP_LLM_FORWARD_UNKNOWN_TOOLS", None)
        else:
            os.environ["RTP_LLM_FORWARD_UNKNOWN_TOOLS"] = self._orig_env

    def test_unknown_tool_dropped_by_default(self):
        """Unknown tools are dropped when env var is not set (default)."""
        os.environ.pop("RTP_LLM_FORWARD_UNKNOWN_TOOLS", None)
        detector = Qwen25Detector()
        text = '<tool_call>\n{"name":"unknown_func","arguments":{"city":"Paris"}}\n</tool_call>'
        result = detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_unknown_tool_dropped_when_false(self):
        """Unknown tools are dropped when env var is explicitly false."""
        os.environ["RTP_LLM_FORWARD_UNKNOWN_TOOLS"] = "false"
        detector = Qwen25Detector()
        text = '<tool_call>\n{"name":"unknown_func","arguments":{"city":"Paris"}}\n</tool_call>'
        result = detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_unknown_tool_forwarded_when_true(self):
        """Unknown tools are forwarded when env var is true."""
        os.environ["RTP_LLM_FORWARD_UNKNOWN_TOOLS"] = "true"
        detector = Qwen25Detector()
        text = '<tool_call>\n{"name":"unknown_func","arguments":{"city":"Paris"}}\n</tool_call>'
        result = detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "unknown_func")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Paris")

    def test_known_tool_always_works(self):
        """Known tools work regardless of env var setting."""
        os.environ.pop("RTP_LLM_FORWARD_UNKNOWN_TOOLS", None)
        detector = Qwen25Detector()
        text = '<tool_call>\n{"name":"get_weather","arguments":{"city":"Tokyo"}}\n</tool_call>'
        result = detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")


class TestUnknownToolNameStreaming(unittest.TestCase):
    """Tests for parse_streaming_increment (streaming path via BaseFormatDetector)."""

    def setUp(self):
        self.tools = _make_tools()
        self._orig_env = os.environ.get("RTP_LLM_FORWARD_UNKNOWN_TOOLS")

    def tearDown(self):
        if self._orig_env is None:
            os.environ.pop("RTP_LLM_FORWARD_UNKNOWN_TOOLS", None)
        else:
            os.environ["RTP_LLM_FORWARD_UNKNOWN_TOOLS"] = self._orig_env

    def _stream_chunks(self, chunks: List[str]) -> tuple:
        """Parse chunks and collect all calls and normal text."""
        detector = Qwen25Detector()
        all_calls = []
        all_normal = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            if result.normal_text:
                all_normal.append(result.normal_text)
        return all_calls, all_normal

    def test_streaming_unknown_tool_dropped_by_default(self):
        """Streaming: unknown tool call block is silently skipped, matching non-streaming."""
        os.environ.pop("RTP_LLM_FORWARD_UNKNOWN_TOOLS", None)
        chunks = [
            "<tool_call>\n",
            '{"name',
            '":"unknown_func',
            '","arguments',
            '":{"city',
            '":"Paris"',
            "}}\n</tool_call>",
        ]
        all_calls, all_normal = self._stream_chunks(chunks)
        # No tool calls should be emitted
        self.assertEqual(len(all_calls), 0)
        # No normal text either — entire block is discarded
        joined_normal = "".join(all_normal)
        self.assertEqual(joined_normal, "")

    def test_streaming_unknown_tool_forwarded_when_true(self):
        """Streaming: unknown tools are forwarded when env var is true."""
        os.environ["RTP_LLM_FORWARD_UNKNOWN_TOOLS"] = "true"
        chunks = [
            "<tool_call>\n",
            '{"name',
            '":"unknown_func',
            '","arguments',
            '":{"city',
            '":"Paris"',
            "}}\n</tool_call>",
        ]
        all_calls, _ = self._stream_chunks(chunks)
        # First call should be the tool name
        self.assertGreaterEqual(len(all_calls), 1)
        self.assertEqual(all_calls[0].name, "unknown_func")
        self.assertEqual(all_calls[0].parameters, "")
        # Remaining calls should contain argument fragments that form valid JSON
        arg_fragments = "".join(c.parameters for c in all_calls[1:])
        parsed_args = json.loads(arg_fragments)
        self.assertEqual(parsed_args["city"], "Paris")

    def test_streaming_known_tool_always_works(self):
        """Streaming: known tools work regardless of env var setting."""
        os.environ.pop("RTP_LLM_FORWARD_UNKNOWN_TOOLS", None)
        chunks = [
            "<tool_call>\n",
            '{"name',
            '":"get_weather',
            '","arguments',
            '":{"city',
            '":"Tokyo"',
            "}}\n</tool_call>",
        ]
        all_calls, _ = self._stream_chunks(chunks)
        self.assertGreaterEqual(len(all_calls), 1)
        self.assertEqual(all_calls[0].name, "get_weather")
        arg_fragments = "".join(c.parameters for c in all_calls[1:])
        parsed_args = json.loads(arg_fragments)
        self.assertEqual(parsed_args["city"], "Tokyo")


if __name__ == "__main__":
    unittest.main()
