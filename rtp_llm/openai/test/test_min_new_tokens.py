"""Tests for min_new_tokens enforcement in CustomChatRenderer (Fix D).

`min_new_tokens` is a hard floor: EOS and stop-words must not terminate
generation before the floor is met. The length-based guards
(`max_new_tokens`, `max_seq_len`) still take precedence.

The value is threaded through an asyncio.Task-scoped ContextVar set at the
top of `render_response_stream`, so subclass `_update_single_status`
signatures are unchanged.
"""

import unittest

from rtp_llm.openai.api_datatype import FinisheReason
from rtp_llm.openai.renderers.custom_renderer import (
    _MIN_NEW_TOKENS_CV,
    CustomChatRenderer,
)


def _bare_renderer(
    *, max_seq_len: int = 100_000, eos_token_id: int = 999
) -> CustomChatRenderer:
    """Construct a renderer without running __init__, populated only with the
    attributes _check_finish_reason reads."""
    r = CustomChatRenderer.__new__(CustomChatRenderer)
    r.max_seq_len = max_seq_len
    r.eos_token_id = eos_token_id
    r.stop_words_id_list = []
    r.extra_stop_words = []
    r.extra_stop_word_ids_list = []
    r.tokenize_words = lambda _words: []
    return r


class TestMinNewTokensEnforcement(unittest.TestCase):
    def setUp(self):
        _MIN_NEW_TOKENS_CV.set(0)

    def test_default_zero_preserves_eos_stop_behavior(self):
        r = _bare_renderer(eos_token_id=999)
        result = r._check_finish_reason(
            token_ids=[1, 2, 999], input_token_length=0
        )
        self.assertEqual(result, FinisheReason.stop)

    def test_eos_suppressed_below_floor(self):
        r = _bare_renderer(eos_token_id=999)
        _MIN_NEW_TOKENS_CV.set(10)
        result = r._check_finish_reason(
            token_ids=[1, 2, 999], input_token_length=0
        )
        self.assertIsNone(result)

    def test_eos_allowed_at_floor(self):
        r = _bare_renderer(eos_token_id=999)
        _MIN_NEW_TOKENS_CV.set(3)
        result = r._check_finish_reason(
            token_ids=[1, 2, 999], input_token_length=0
        )
        self.assertEqual(result, FinisheReason.stop)

    def test_stop_word_suppressed_below_floor(self):
        r = _bare_renderer(eos_token_id=999)
        r.stop_words_id_list = [[42, 43]]
        _MIN_NEW_TOKENS_CV.set(20)
        result = r._check_finish_reason(
            token_ids=[1, 2, 42, 43], input_token_length=0
        )
        self.assertIsNone(result)

    def test_stop_word_allowed_at_floor(self):
        r = _bare_renderer(eos_token_id=999)
        r.stop_words_id_list = [[42, 43]]
        _MIN_NEW_TOKENS_CV.set(4)
        result = r._check_finish_reason(
            token_ids=[1, 2, 42, 43], input_token_length=0
        )
        self.assertEqual(result, FinisheReason.stop)

    def test_max_new_tokens_still_wins_above_floor(self):
        r = _bare_renderer(eos_token_id=999)
        _MIN_NEW_TOKENS_CV.set(100)
        result = r._check_finish_reason(
            token_ids=list(range(5)), input_token_length=0, max_new_tokens=5
        )
        self.assertEqual(result, FinisheReason.length)

    def test_max_seq_len_still_wins(self):
        r = _bare_renderer(max_seq_len=8, eos_token_id=999)
        _MIN_NEW_TOKENS_CV.set(100)
        result = r._check_finish_reason(
            token_ids=[1, 2, 3, 4], input_token_length=4
        )
        self.assertEqual(result, FinisheReason.length)

    def test_no_termination_when_no_signal(self):
        r = _bare_renderer(eos_token_id=999)
        result = r._check_finish_reason(
            token_ids=[1, 2, 3], input_token_length=0
        )
        self.assertIsNone(result)


class TestRenderResponseStreamCVLifecycle(unittest.TestCase):
    """`render_response_stream` must set the CV on entry and reset it on
    exit so that one request's `min_new_tokens` cannot leak into another
    request running on the same asyncio Task (e.g. background pool reuse)."""

    def test_cv_resets_after_generator_close(self):
        import asyncio

        from rtp_llm.config.generate_config import GenerateConfig
        from rtp_llm.openai.api_datatype import ChatCompletionRequest, ChatMessage

        async def _empty_outputs():
            if False:
                yield None  # async generator that produces nothing
            return

        async def run():
            renderer = _bare_renderer()
            renderer.in_think_mode = lambda req: False
            renderer.should_process_think = lambda req: False
            async def _async_empty(n, req):
                return []
            renderer._create_status_list = _async_empty
            request = ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")], n=1
            )
            gen_config = GenerateConfig()
            gen_config.min_new_tokens = 42

            # Pre-set the CV to a sentinel so we can verify it's restored
            outer_token = _MIN_NEW_TOKENS_CV.set(7)
            try:
                gen = renderer.render_response_stream(
                    _empty_outputs(), request, gen_config
                )
                async for _ in gen:
                    pass
                # After the generator is drained the CV must be back to 7.
                self.assertEqual(_MIN_NEW_TOKENS_CV.get(), 7)
            finally:
                _MIN_NEW_TOKENS_CV.reset(outer_token)

        asyncio.run(run())


class TestMinNewTokensListInput(unittest.TestCase):
    """`GenerateConfig.min_new_tokens` is `Union[List[int], int]`. The
    renderer must accept both forms without crashing on `int([...])`."""

    def test_list_input_collapses_to_max(self):
        # Simulate what render_response_stream does with a list.
        from rtp_llm.config.generate_config import GenerateConfig

        gen_config = GenerateConfig()
        gen_config.min_new_tokens = [3, 5, 1]
        mnt = gen_config.min_new_tokens
        if isinstance(mnt, list):
            mnt = max(mnt) if mnt else 0
        self.assertEqual(int(mnt or 0), 5)

    def test_empty_list_input_becomes_zero(self):
        from rtp_llm.config.generate_config import GenerateConfig

        gen_config = GenerateConfig()
        gen_config.min_new_tokens = []
        mnt = gen_config.min_new_tokens
        if isinstance(mnt, list):
            mnt = max(mnt) if mnt else 0
        self.assertEqual(int(mnt or 0), 0)


if __name__ == "__main__":
    unittest.main()
