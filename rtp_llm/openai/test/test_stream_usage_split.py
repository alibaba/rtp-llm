"""Tests for OpenAI streaming `stream_options.include_usage` gating.

`OpenaiEndpoint._complete_stream_response` follows the OpenAI Chat
Completions streaming protocol:

  * `include_usage=True`  -> emit the trailing `usage` payload in its own
    `choices=[]` chunk after the chunk that carried `finish_reason`.
    Spec-compliant clients (vllm bench, OpenAI Python SDK) only inspect
    `usage` on choices-empty chunks, so bundling drops the payload.
  * `include_usage=False` -> suppress `usage` from the stream entirely.
  * `include_usage=None`  -> rtp-llm legacy behavior: bundle `usage` on
    the chunk that carried `finish_reason`. Preserved for backward
    compatibility with internal consumers that predate the spec layout.

`OpenaiEndpoint._collect_complete_response` must also tolerate the
split layout so the non-streaming aggregator can still recover `usage`
from a `choices=[]` chunk.
"""

import asyncio
import unittest
from typing import AsyncGenerator, List

from rtp_llm.openai.api_datatype import (
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    FinisheReason,
    RoleEnum,
    UsageInfo,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.custom_renderer import StreamResponseObject


def _make_choice(
    index: int = 0,
    content: str = "",
    finish_reason=None,
) -> ChatCompletionResponseStreamChoice:
    return ChatCompletionResponseStreamChoice(
        index=index,
        delta=DeltaMessage(role=RoleEnum.assistant, content=content),
        finish_reason=finish_reason,
    )


async def _drain(agen: AsyncGenerator) -> List:
    out = []
    async for x in agen:
        out.append(x)
    return out


async def _gen_from(items):
    for it in items:
        yield it


class TestIncludeUsageTrue(unittest.TestCase):
    """include_usage=True -> split the finish chunk and usage chunk."""

    def test_finish_with_usage_emits_two_chunks(self):
        usage = UsageInfo(prompt_tokens=4, completion_tokens=8, total_tokens=12)
        items = [
            StreamResponseObject(choices=[_make_choice(content="hello")]),
            StreamResponseObject(
                choices=[_make_choice(content="", finish_reason=FinisheReason.stop)],
                usage=usage,
            ),
        ]
        gen = OpenaiEndpoint._complete_stream_response(
            _gen_from(items),
            debug_info=None,
            tokenizer=None,
            include_usage=True,
        )
        out = asyncio.run(_drain(gen))

        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[0].choices), 1)
        self.assertIsNone(out[0].usage)
        self.assertEqual(len(out[1].choices), 1)
        self.assertEqual(out[1].choices[0].finish_reason, FinisheReason.stop)
        self.assertIsNone(out[1].usage)
        self.assertEqual(out[2].choices, [])
        self.assertIsNotNone(out[2].usage)
        self.assertEqual(out[2].usage.prompt_tokens, 4)
        self.assertEqual(out[2].usage.completion_tokens, 8)
        self.assertEqual(out[2].usage.total_tokens, 12)

    def test_no_finish_no_split(self):
        items = [
            StreamResponseObject(choices=[_make_choice(content="a")]),
            StreamResponseObject(choices=[_make_choice(content="b")]),
        ]
        gen = OpenaiEndpoint._complete_stream_response(
            _gen_from(items),
            debug_info=None,
            tokenizer=None,
            include_usage=True,
        )
        out = asyncio.run(_drain(gen))
        self.assertEqual(len(out), 2)
        self.assertTrue(all(r.usage is None for r in out))


class TestIncludeUsageFalse(unittest.TestCase):
    """include_usage=False -> usage suppressed from the stream entirely."""

    def test_finish_chunk_has_no_usage(self):
        usage = UsageInfo(prompt_tokens=4, completion_tokens=8, total_tokens=12)
        items = [
            StreamResponseObject(choices=[_make_choice(content="hi")]),
            StreamResponseObject(
                choices=[_make_choice(content="", finish_reason=FinisheReason.stop)],
                usage=usage,
            ),
        ]
        gen = OpenaiEndpoint._complete_stream_response(
            _gen_from(items),
            debug_info=None,
            tokenizer=None,
            include_usage=False,
        )
        out = asyncio.run(_drain(gen))
        self.assertEqual(len(out), 2)
        self.assertTrue(all(r.usage is None for r in out))


class TestIncludeUsageDefault(unittest.TestCase):
    """include_usage=None (default) -> legacy bundled behavior."""

    def test_usage_bundled_on_finish_chunk(self):
        usage = UsageInfo(prompt_tokens=4, completion_tokens=8, total_tokens=12)
        items = [
            StreamResponseObject(choices=[_make_choice(content="hi")]),
            StreamResponseObject(
                choices=[_make_choice(content="", finish_reason=FinisheReason.stop)],
                usage=usage,
            ),
        ]
        gen = OpenaiEndpoint._complete_stream_response(
            _gen_from(items), debug_info=None, tokenizer=None
        )
        out = asyncio.run(_drain(gen))
        # Two chunks (no split), usage rides on the finish chunk.
        self.assertEqual(len(out), 2)
        self.assertIsNone(out[0].usage)
        self.assertIsNotNone(out[1].usage)
        self.assertEqual(out[1].choices[0].finish_reason, FinisheReason.stop)
        self.assertEqual(out[1].usage.completion_tokens, 8)


class TestCollectCompleteResponseHandlesEmptyChoices(unittest.TestCase):
    """`_collect_complete_response` must capture `usage` from a
    choices-empty chunk so that downstream callers see the split-layout
    output produced when `include_usage=True`."""

    def test_aggregates_split_tail(self):
        usage = UsageInfo(prompt_tokens=3, completion_tokens=5, total_tokens=8)
        items = [
            StreamResponseObject(choices=[_make_choice(content="hi")]),
            StreamResponseObject(
                choices=[_make_choice(content="", finish_reason=FinisheReason.stop)]
            ),
            StreamResponseObject(choices=[], usage=usage),
        ]
        resp = asyncio.run(
            OpenaiEndpoint._collect_complete_response(
                _gen_from(items), debug_info=None, tokenizer=None
            )
        )
        self.assertEqual(len(resp.choices), 1)
        self.assertEqual(resp.choices[0].message.content, "hi")
        self.assertEqual(resp.choices[0].finish_reason, FinisheReason.stop)
        self.assertEqual(resp.usage.prompt_tokens, 3)
        self.assertEqual(resp.usage.completion_tokens, 5)
        self.assertEqual(resp.usage.total_tokens, 8)

    def test_empty_choices_chunk_does_not_crash_when_first(self):
        usage = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        items = [
            StreamResponseObject(choices=[], usage=usage),
            StreamResponseObject(choices=[_make_choice(content="hi")]),
            StreamResponseObject(
                choices=[_make_choice(content="", finish_reason=FinisheReason.stop)]
            ),
        ]
        resp = asyncio.run(
            OpenaiEndpoint._collect_complete_response(
                _gen_from(items), debug_info=None, tokenizer=None
            )
        )
        self.assertEqual(len(resp.choices), 1)
        self.assertEqual(resp.choices[0].message.content, "hi")


if __name__ == "__main__":
    unittest.main()
