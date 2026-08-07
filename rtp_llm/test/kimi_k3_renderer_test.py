#!/usr/bin/env python3

import json
import unittest

from rtp_llm.openai.api_datatype import ChatCompletionRequest, ChatMessage
from rtp_llm.openai.renderers.kimi_k3_renderer import (
    KimiK3Renderer,
    _KimiK3StreamStatus,
)


class KimiK3RendererTest(unittest.TestCase):
    def request(self, enable_thinking: bool) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            enable_thinking=enable_thinking,
        )

    def test_thinking_xtml_is_split_across_stream_chunks(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=True))
        chunks = [
            "reasoning<|close|>thi",
            "nk<|sep|><|open|>respo",
            "nse<|sep|>ANSWER: B<|close|>res",
            "ponse<|sep|><|close|>message<|sep|>",
        ]
        deltas = [KimiK3Renderer._parse_xtml_delta(status, chunk) for chunk in chunks]

        self.assertEqual(
            "".join(delta.reasoning_content or "" for delta in deltas), "reasoning"
        )
        self.assertEqual("".join(delta.content or "" for delta in deltas), "ANSWER: B")
        self.assertTrue(status.response_closed)
        self.assertEqual(status.xtml_pending, "")

    def test_non_thinking_terminal_envelope_is_removed(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        delta = KimiK3Renderer._parse_xtml_delta(
            status,
            "ANSWER: D<|close|>response<|sep|><|close|>message<|sep|>",
        )

        self.assertEqual(delta.reasoning_content, "")
        self.assertEqual(delta.content, "ANSWER: D")
        self.assertTrue(status.response_closed)

    def test_invalid_channel_marker_remains_visible(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        text = "answer<|close|>think<|sep|>still-visible"
        delta = KimiK3Renderer._parse_xtml_delta(status, text, flush=True)

        self.assertEqual(delta.content, text)
        self.assertFalse(status.response_closed)

    def _run_chunks(self, status, chunks):
        deltas = []
        for index, chunk in enumerate(chunks):
            deltas.append(
                KimiK3Renderer._parse_xtml_delta(
                    status, chunk, flush=index == len(chunks) - 1
                )
            )
        return deltas

    def _all_tool_calls(self, deltas):
        calls = []
        for delta in deltas:
            if delta.tool_calls:
                calls.extend(delta.tool_calls)
        return calls

    def test_tools_channel_json_block_is_parsed(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        text = (
            "<|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{"city": "北京", "days": 3}'
            "<|close|>json<|sep|>"
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
        )
        deltas = self._run_chunks(status, [text])

        calls = self._all_tool_calls(deltas)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].index, 0)
        self.assertEqual(calls[0].type, "function")
        self.assertEqual(calls[0].function.name, "get_weather")
        self.assertTrue(calls[0].id.startswith("call_"))
        self.assertEqual(
            json.loads(calls[0].function.arguments), {"city": "北京", "days": 3}
        )
        self.assertEqual(status.tool_calls_seen, 1)
        self.assertEqual("".join(d.content or "" for d in deltas), "")

    def test_tools_channel_typed_arguments_are_coerced(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        text = (
            "<|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            '<|open|>call tool="search" index="1"<|sep|>'
            '<|open|>argument key="query" type="string"<|sep|>天气<|close|>argument<|sep|>'
            '<|open|>argument key="limit" type="number"<|sep|>5<|close|>argument<|sep|>'
            '<|open|>argument key="fresh" type="boolean"<|sep|>true<|close|>argument<|sep|>'
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
        )
        deltas = self._run_chunks(status, [text])

        calls = self._all_tool_calls(deltas)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            json.loads(calls[0].function.arguments),
            {"query": "天气", "limit": 5, "fresh": True},
        )

    def test_tools_channel_split_across_stream_chunks(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        chunks = [
            "OK<|close|>respo",
            "nse<|sep|><|open|>to",
            "ols<|sep|><|open|>call tool=\"get_wea",
            'ther" index="1"<|sep|><|open|>json type="object"<|sep|>{"ci',
            'ty": "杭州"}<|close|>js',
            "on<|sep|><|close|>call<|sep|><|close|>too",
            "ls<|sep|>",
        ]
        deltas = self._run_chunks(status, chunks)

        self.assertEqual("".join(d.content or "" for d in deltas), "OK")
        calls = self._all_tool_calls(deltas)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "get_weather")
        self.assertEqual(json.loads(calls[0].function.arguments), {"city": "杭州"})

    def test_tools_channel_multiple_calls(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        text = (
            "<|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            '<|open|>call tool="a" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{}<|close|>json<|sep|>'
            "<|close|>call<|sep|>"
            '<|open|>call tool="b" index="2"<|sep|>'
            '<|open|>json type="object"<|sep|>{"k": 1}<|close|>json<|sep|>'
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
        )
        deltas = self._run_chunks(status, [text])

        calls = self._all_tool_calls(deltas)
        self.assertEqual([c.index for c in calls], [0, 1])
        self.assertEqual([c.function.name for c in calls], ["a", "b"])

    def test_tools_channel_unclosed_is_flushed(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        deltas = self._run_chunks(
            status,
            [
                "<|close|>response<|sep|><|open|>tools<|sep|>"
                '<|open|>call tool="a" index="1"<|sep|>'
                '<|open|>json type="object"<|sep|>{}<|close|>json<|sep|>'
                "<|close|>call<|sep|>"
            ],
        )

        calls = self._all_tool_calls(deltas)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "a")

    def test_tools_channel_escaped_attributes(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        text = (
            "<|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            '<|open|>call tool="a&amp;b&quot;c" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{}<|close|>json<|sep|>'
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
        )
        deltas = self._run_chunks(status, [text])

        calls = self._all_tool_calls(deltas)
        self.assertEqual(calls[0].function.name, 'a&b"c')

    def test_thinking_then_tools_channel(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=True))
        text = (
            "think<|close|>think<|sep|><|open|>response<|sep|>"
            "<|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{"city": "上海"}<|close|>json<|sep|>'
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
        )
        deltas = self._run_chunks(status, [text])

        self.assertEqual(deltas[0].reasoning_content, "think")
        calls = self._all_tool_calls(deltas)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].function.arguments), {"city": "上海"})

    def test_no_tools_channel_keeps_empty_tool_calls(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        deltas = self._run_chunks(
            status, ["ANSWER<|close|>response<|sep|><|end_of_msg|>"]
        )

        self.assertEqual(self._all_tool_calls(deltas), [])
        self.assertEqual(status.tool_calls_seen, 0)
        self.assertEqual("".join(d.content or "" for d in deltas), "ANSWER")


if __name__ == "__main__":
    unittest.main()
