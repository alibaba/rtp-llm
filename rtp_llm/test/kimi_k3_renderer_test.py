#!/usr/bin/env python3

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


if __name__ == "__main__":
    unittest.main()
