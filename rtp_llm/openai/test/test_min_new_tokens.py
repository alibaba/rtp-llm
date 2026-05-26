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


if __name__ == "__main__":
    unittest.main()
