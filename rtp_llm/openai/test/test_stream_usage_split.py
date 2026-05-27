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

    def test_mid_stream_usage_suppressed(self):
        # The backend tags every chunk with running usage totals. When
        # include_usage=True, only the trailing choices=[] chunk should
        # carry `usage`; intermediate chunks must drop the field.
        usage_running = UsageInfo(prompt_tokens=4, completion_tokens=1, total_tokens=5)
        usage_final = UsageInfo(prompt_tokens=4, completion_tokens=8, total_tokens=12)
        items = [
            StreamResponseObject(
                choices=[_make_choice(content="a")], usage=usage_running
            ),
            StreamResponseObject(
                choices=[_make_choice(content="b", finish_reason=FinisheReason.stop)],
                usage=usage_final,
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
        # Mid-stream chunk has choices but no usage.
        self.assertEqual(len(out[0].choices), 1)
        self.assertIsNone(out[0].usage)
        # Finish chunk has choices but no usage.
        self.assertEqual(len(out[1].choices), 1)
        self.assertIsNone(out[1].usage)
        # Tail chunk has usage only.
        self.assertEqual(out[2].choices, [])
        self.assertEqual(out[2].usage.completion_tokens, 8)


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


class TestIncludeUsageTrueUsageOnSeparateChunk(unittest.TestCase):
    """include_usage=True with usage arriving on a different chunk than
    finish_reason. The trailing usage chunk must still be emitted."""

    def test_usage_after_finish_on_separate_chunk(self):
        """Backend sends finish_reason first, then a standalone usage-only
        chunk. The trailing choices=[] usage chunk must still appear."""
        usage = UsageInfo(prompt_tokens=4, completion_tokens=8, total_tokens=12)
        items = [
            StreamResponseObject(choices=[_make_choice(content="hello")]),
            StreamResponseObject(
                choices=[_make_choice(content="", finish_reason=FinisheReason.stop)],
            ),
            # Usage arrives on a standalone chunk with no choices.
            StreamResponseObject(choices=[], usage=usage),
        ]
        gen = OpenaiEndpoint._complete_stream_response(
            _gen_from(items),
            debug_info=None,
            tokenizer=None,
            include_usage=True,
        )
        out = asyncio.run(_drain(gen))

        # 2 content/finish chunks + 1 trailing usage chunk
        # (the standalone empty-choices backend chunk is skipped)
        self.assertEqual(len(out), 3)
        # Content chunk
        self.assertIsNone(out[0].usage)
        self.assertEqual(len(out[0].choices), 1)
        # Finish chunk (usage suppressed)
        self.assertIsNone(out[1].usage)
        self.assertEqual(out[1].choices[0].finish_reason, FinisheReason.stop)
        # Trailing usage chunk emitted by fallback
        self.assertEqual(out[2].choices, [])
        self.assertIsNotNone(out[2].usage)
        self.assertEqual(out[2].usage.completion_tokens, 8)

    def test_usage_before_finish_on_separate_chunk(self):
        """Usage arrives on an early chunk, finish_reason on a later one.
        The trailing usage chunk should carry the last-seen usage."""
        early_usage = UsageInfo(prompt_tokens=4, completion_tokens=3, total_tokens=7)
        final_usage = UsageInfo(prompt_tokens=4, completion_tokens=8, total_tokens=12)
        items = [
            StreamResponseObject(
                choices=[_make_choice(content="hello")], usage=early_usage
            ),
            # Final usage on a mid-stream chunk, finish comes later without usage.
            StreamResponseObject(
                choices=[_make_choice(content=" world")], usage=final_usage
            ),
            StreamResponseObject(
                choices=[_make_choice(content="", finish_reason=FinisheReason.stop)],
            ),
        ]
        gen = OpenaiEndpoint._complete_stream_response(
            _gen_from(items),
            debug_info=None,
            tokenizer=None,
            include_usage=True,
        )
        out = asyncio.run(_drain(gen))

        # 3 content/finish chunks + 1 trailing usage chunk
        self.assertEqual(len(out), 4)
        # All content chunks have usage=None
        for i in range(3):
            self.assertIsNone(out[i].usage)
        # Trailing chunk carries the last-seen usage
        self.assertEqual(out[3].choices, [])
        self.assertIsNotNone(out[3].usage)
        self.assertEqual(out[3].usage.completion_tokens, 8)

    def test_no_usage_at_all_no_trailing_chunk(self):
        """If no usage is ever seen, no trailing chunk is emitted."""
        items = [
            StreamResponseObject(choices=[_make_choice(content="hi")]),
            StreamResponseObject(
                choices=[_make_choice(content="", finish_reason=FinisheReason.stop)],
            ),
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


class TestIncludeUsageEmptyChoicesSkipped(unittest.TestCase):
    """P2: Backend empty-choices usage chunks must not be forwarded as
    redundant empty chunks when include_usage=True."""

    def test_standalone_empty_choices_usage_chunk_is_skipped(self):
        """A standalone choices=[] usage chunk from the backend should
        NOT appear in the output stream — only the deferred trailing
        usage chunk should carry the usage."""
        usage = UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        items = [
            StreamResponseObject(choices=[_make_choice(content="hello")]),
            # Backend sends usage on a standalone empty-choices chunk
            StreamResponseObject(choices=[], usage=usage),
            StreamResponseObject(
                choices=[_make_choice(content="", finish_reason=FinisheReason.stop)],
            ),
        ]
        gen = OpenaiEndpoint._complete_stream_response(
            _gen_from(items),
            debug_info=None,
            tokenizer=None,
            include_usage=True,
        )
        out = asyncio.run(_drain(gen))

        # 2 content/finish chunks + 1 trailing usage chunk (the empty-choices
        # backend chunk is NOT forwarded)
        self.assertEqual(len(out), 3)
        # Content chunk
        self.assertIsNone(out[0].usage)
        self.assertEqual(len(out[0].choices), 1)
        # Finish chunk
        self.assertIsNone(out[1].usage)
        # Trailing usage chunk
        self.assertEqual(out[2].choices, [])
        self.assertIsNotNone(out[2].usage)
        self.assertEqual(out[2].usage.completion_tokens, 10)

    def test_empty_choices_between_content_chunks_skipped(self):
        """Multiple empty-choices chunks interspersed in the stream
        should all be filtered out."""
        usage1 = UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        usage2 = UsageInfo(prompt_tokens=5, completion_tokens=7, total_tokens=12)
        items = [
            StreamResponseObject(choices=[_make_choice(content="a")]),
            StreamResponseObject(choices=[], usage=usage1),
            StreamResponseObject(choices=[_make_choice(content="b")]),
            StreamResponseObject(choices=[], usage=usage2),
            StreamResponseObject(
                choices=[_make_choice(content="", finish_reason=FinisheReason.stop)],
                usage=usage2,
            ),
        ]
        gen = OpenaiEndpoint._complete_stream_response(
            _gen_from(items),
            debug_info=None,
            tokenizer=None,
            include_usage=True,
        )
        out = asyncio.run(_drain(gen))

        # 3 content/finish + 1 trailing usage = 4 total (2 empty skipped)
        self.assertEqual(len(out), 4)
        for chunk in out[:3]:
            self.assertIsNone(chunk.usage)
        self.assertEqual(out[3].choices, [])
        self.assertEqual(out[3].usage.completion_tokens, 7)


class TestIncludeUsageMultiFinishNoDuplicate(unittest.TestCase):
    """Round-4 guard: when finish_reason arrives on more than one chunk
    (e.g. n>=2 with staggered finishes), the trailing usage chunk must
    be emitted exactly once."""

    def test_two_finish_chunks_emit_single_trailing_usage(self):
        usage = UsageInfo(prompt_tokens=4, completion_tokens=9, total_tokens=13)
        # Two consecutive chunks each carrying finish_reason; usage is
        # already known by the first finish chunk.
        items = [
            StreamResponseObject(choices=[_make_choice(content="hi")]),
            StreamResponseObject(
                choices=[_make_choice(content="", finish_reason=FinisheReason.stop)],
                usage=usage,
            ),
            StreamResponseObject(
                choices=[_make_choice(content="", finish_reason=FinisheReason.stop)],
            ),
        ]
        gen = OpenaiEndpoint._complete_stream_response(
            _gen_from(items),
            debug_info=None,
            tokenizer=None,
            include_usage=True,
        )
        out = asyncio.run(_drain(gen))

        # Exactly one trailing usage chunk (choices=[]) should be present.
        trailing = [c for c in out if c.choices == [] and c.usage is not None]
        self.assertEqual(len(trailing), 1)
        self.assertEqual(trailing[0].usage.completion_tokens, 9)
        # All other chunks must have usage=None.
        for c in out:
            if c is trailing[0]:
                continue
            self.assertIsNone(c.usage)


class TestIncludeUsageStaggeredMultiChoiceFinish(unittest.TestCase):
    """Multi-choice (n>=2) regression: when choices finish on different
    chunks, the trailing usage chunk must be emitted exactly once after
    the entire stream is drained, with the FINAL usage value. The emit
    must not be triggered mid-stream by a single choice's finish_reason
    while other choices are still generating content."""

    def test_staggered_finish_emits_usage_after_all_choices_done(self):
        usage_at_first_finish = UsageInfo(
            prompt_tokens=4, completion_tokens=5, total_tokens=9
        )
        usage_final = UsageInfo(
            prompt_tokens=4, completion_tokens=12, total_tokens=16
        )
        items = [
            # Both choices producing content.
            StreamResponseObject(
                choices=[
                    _make_choice(index=0, content="a0"),
                    _make_choice(index=1, content="b0"),
                ]
            ),
            # Choice 0 finishes; choice 1 still generating. Backend
            # tags this chunk with running usage covering tokens so far.
            StreamResponseObject(
                choices=[
                    _make_choice(
                        index=0, content="", finish_reason=FinisheReason.stop
                    ),
                    _make_choice(index=1, content="b1"),
                ],
                usage=usage_at_first_finish,
            ),
            # Choice 1 continues for more chunks while choice 0 is done.
            StreamResponseObject(
                choices=[_make_choice(index=1, content="b2")]
            ),
            # Choice 1 finishes; backend carries the final usage.
            StreamResponseObject(
                choices=[
                    _make_choice(
                        index=1, content="", finish_reason=FinisheReason.stop
                    )
                ],
                usage=usage_final,
            ),
        ]
        gen = OpenaiEndpoint._complete_stream_response(
            _gen_from(items),
            debug_info=None,
            tokenizer=None,
            include_usage=True,
        )
        out = asyncio.run(_drain(gen))

        # 4 content/finish chunks + 1 trailing usage chunk.
        self.assertEqual(len(out), 5)

        # Every non-trailing chunk must have usage=None.
        for chunk in out[:-1]:
            self.assertIsNone(chunk.usage)
            self.assertNotEqual(chunk.choices, [])

        # The trailing chunk is the FINAL chunk on the wire, choices=[],
        # carries the final usage (not the stale value at first finish).
        self.assertEqual(out[-1].choices, [])
        self.assertIsNotNone(out[-1].usage)
        self.assertEqual(out[-1].usage.completion_tokens, 12)
        self.assertEqual(out[-1].usage.total_tokens, 16)



if __name__ == "__main__":
    unittest.main()
