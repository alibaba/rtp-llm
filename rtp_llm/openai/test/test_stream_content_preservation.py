"""Tests for stream aggregator `content` semantics.

Two cases the aggregator must preserve correctly:

1. When every `delta.content` is the empty string (e.g. chat completion
   truncated mid-`<think>`), the aggregated message must keep
   `content=""`, not collapse to `None`. Previously the aggregator
   coerced `""` to `None` via `delta.content or ""` /
   `delta.content or None`; after `model_dump(exclude_none=True)` the
   `content` key was dropped entirely, violating the spec which
   requires the field to be present.

2. When every `delta.content` is `None` (e.g. a tool-call only
   message), the aggregated message must keep `content=None` so that
   `exclude_none=True` correctly drops the field. The previous defense
   -in-depth pass coerced `None` to `""`, which leaked an empty content
   string into tool-call payloads.

The fix is a simple pass-through: don't coerce `""` to `None`, and don't
coerce `None` to `""`. Pydantic + `exclude_none` does the right thing
when the aggregator preserves the source delta semantics exactly.
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

    def test_all_none_delta_content_preserves_none(self):
        # Tool-call style stream: every delta.content is None. The
        # aggregator must NOT coerce None to "", otherwise tool-call
        # messages leak an empty content string into the JSON payload.
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
        self.assertIsNone(resp.choices[0].message.content)
        payload = json.loads(resp.model_dump_json(exclude_none=True))
        # exclude_none should drop the content key entirely.
        self.assertNotIn("content", payload["choices"][0]["message"])

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

    def test_mixed_none_then_text_starts_clean(self):
        # If the first delta is None (tool-call hint) and later deltas
        # carry text, the aggregator should not concatenate "None" or
        # double-coerce. Once content becomes a string it stays a string.
        items = [
            StreamResponseObject(choices=[_choice(None)]),
            StreamResponseObject(choices=[_choice("hi")]),
            StreamResponseObject(
                choices=[_choice("", finish_reason=FinisheReason.stop)],
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        ]
        resp = asyncio.run(
            OpenaiEndpoint._collect_complete_response(
                _gen(items), debug_info=None, tokenizer=None
            )
        )
        # The first None initializes content to None; then "hi" replaces
        # it (since content is None, not a string we can += to).
        self.assertEqual(resp.choices[0].message.content, "hi")


class TestReasoningContentPreservation(unittest.TestCase):
    """reasoning_content must get the same pass-through treatment as content:
    empty string stays empty string, None stays None."""

    def _choice_rc(self, content=None, reasoning_content=None, finish_reason=None):
        return ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(
                role=RoleEnum.assistant,
                content=content,
                reasoning_content=reasoning_content,
            ),
            finish_reason=finish_reason,
        )

    def test_empty_string_reasoning_content_preserved(self):
        usage = UsageInfo(prompt_tokens=2, completion_tokens=4, total_tokens=6)
        items = [
            StreamResponseObject(choices=[self._choice_rc(content="hi", reasoning_content="")]),
            StreamResponseObject(
                choices=[self._choice_rc(content="", reasoning_content="", finish_reason=FinisheReason.length)],
                usage=usage,
            ),
        ]
        resp = asyncio.run(
            OpenaiEndpoint._collect_complete_response(
                _gen(items), debug_info=None, tokenizer=None
            )
        )
        # reasoning_content="" must NOT collapse to None
        self.assertEqual(resp.choices[0].message.reasoning_content, "")

    def test_none_reasoning_content_stays_none(self):
        usage = UsageInfo(prompt_tokens=2, completion_tokens=1, total_tokens=3)
        items = [
            StreamResponseObject(choices=[self._choice_rc(content="ok", reasoning_content=None)]),
            StreamResponseObject(
                choices=[self._choice_rc(content="", reasoning_content=None, finish_reason=FinisheReason.stop)],
                usage=usage,
            ),
        ]
        resp = asyncio.run(
            OpenaiEndpoint._collect_complete_response(
                _gen(items), debug_info=None, tokenizer=None
            )
        )
        self.assertIsNone(resp.choices[0].message.reasoning_content)


if __name__ == "__main__":
    unittest.main()
