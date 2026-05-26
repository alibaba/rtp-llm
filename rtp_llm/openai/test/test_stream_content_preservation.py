"""Tests for empty-string content preservation in stream aggregators (Fix C).

When a chat completion is truncated mid-`<think>`, every `delta.content`
chunk is the empty string. The aggregators previously collapsed `''` to
`None`, and `model_dump(exclude_none=True)` then dropped the `content` key
entirely from the response payload. Downstream consumers received a
message with no `content` field at all, violating the Chat Completions spec.
"""

import asyncio
import json
import unittest

from rtp_llm.openai.api_datatype import (
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    FinisheReason,
    RoleEnum,
    UsageInfo,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.custom_renderer import StreamResponseObject


def _choice(content, finish_reason=None):
    return ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role=RoleEnum.assistant, content=content),
        finish_reason=finish_reason,
    )


async def _gen(items):
    for it in items:
        yield it


class TestContentPreservation(unittest.TestCase):
    def test_all_empty_string_chunks_yield_empty_string(self):
        usage = UsageInfo(prompt_tokens=2, completion_tokens=4, total_tokens=6)
        items = [
            StreamResponseObject(choices=[_choice("")]),
            StreamResponseObject(choices=[_choice("")]),
            StreamResponseObject(
                choices=[_choice("", finish_reason=FinisheReason.length)],
                usage=usage,
            ),
        ]
        resp = asyncio.run(
            OpenaiEndpoint._collect_complete_response(
                _gen(items), debug_info=None, tokenizer=None
            )
        )
        self.assertEqual(len(resp.choices), 1)
        self.assertEqual(resp.choices[0].message.content, "")
        self.assertIsNotNone(resp.choices[0].message.content)

        payload = json.loads(resp.model_dump_json(exclude_none=True))
        self.assertIn("content", payload["choices"][0]["message"])
        self.assertEqual(payload["choices"][0]["message"]["content"], "")

    def test_none_delta_content_does_not_collapse_to_missing_key(self):
        items = [
            StreamResponseObject(choices=[_choice(None)]),
            StreamResponseObject(
                choices=[_choice(None, finish_reason=FinisheReason.stop)],
                usage=UsageInfo(prompt_tokens=1, completion_tokens=0, total_tokens=1),
            ),
        ]
        resp = asyncio.run(
            OpenaiEndpoint._collect_complete_response(
                _gen(items), debug_info=None, tokenizer=None
            )
        )
        self.assertEqual(resp.choices[0].message.content, "")
        payload = json.loads(resp.model_dump_json(exclude_none=True))
        self.assertIn("content", payload["choices"][0]["message"])

    def test_normal_text_unaffected(self):
        items = [
            StreamResponseObject(choices=[_choice("hello ")]),
            StreamResponseObject(choices=[_choice("world")]),
            StreamResponseObject(
                choices=[_choice("", finish_reason=FinisheReason.stop)],
                usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            ),
        ]
        resp = asyncio.run(
            OpenaiEndpoint._collect_complete_response(
                _gen(items), debug_info=None, tokenizer=None
            )
        )
        self.assertEqual(resp.choices[0].message.content, "hello world")


if __name__ == "__main__":
    unittest.main()
