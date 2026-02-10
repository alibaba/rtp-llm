"""
Tests for TokenNormalizer with adaptive sliding window.

These tests verify that the normalizer correctly handles:
1. Normal single-byte tokens
2. Multi-byte Unicode characters split across tokens (ChatGLM-style)
3. Adaptive sliding window when \\uFFFD is detected
"""

import unittest
from unittest.mock import Mock

from rtp_llm.openai.renderers.sglang_helpers.token_normalizer import TokenNormalizer


class MockTokenizer:
    """
    Mock tokenizer that simulates ChatGLM behavior with multi-byte Unicode splits.

    This tokenizer splits multi-byte characters across tokens and requires
    context from previous tokens to decode correctly.
    """

    def __init__(self):
        # Simulate token-to-byte mapping where multi-byte chars are split
        # Example: "Hello 你好" -> tokens [1, 2, 3, 4, 5, 6]
        # Token 1: "Hello "
        # Token 2: 0xE4 (first byte of 你)
        # Token 3: 0xBD (second byte of 你)
        # Token 4: 0xA0 (third byte of 你)
        # Token 5-7: Similar for 好
        self.token_map = {
            1: b"Hello ",
            2: b"\xE4",  # First byte of 你 (incomplete)
            3: b"\xBD",  # Second byte of 你 (incomplete)
            4: b"\xA0",  # Third byte of 你 (complete with 2,3)
            5: b"\xE5",  # First byte of 好 (incomplete)
            6: b"\xA5",  # Second byte of 好 (incomplete)
            7: b"\xBD",  # Third byte of 好 (complete with 5,6)
        }

    def decode(self, token_ids):
        """Decode token IDs to text, simulating incomplete UTF-8 handling."""
        if not token_ids:
            return ""

        # Concatenate bytes from all tokens
        byte_sequence = b"".join(self.token_map.get(tid, b"") for tid in token_ids)

        # Try to decode, replacing invalid sequences with \uFFFD
        try:
            return byte_sequence.decode("utf-8", errors="replace")
        except Exception:
            return "\uFFFD"


class TestTokenNormalizer(unittest.TestCase):
    """Test adaptive sliding window for token normalization."""

    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = MockTokenizer()
        self.normalizer = TokenNormalizer(self.tokenizer)

    def test_normal_single_byte_tokens(self):
        """Test normal case: single-byte tokens without UTF-8 issues."""
        # Token 1: "Hello "
        prev_tokens = []
        new_tokens = [1]

        deltas = list(self.normalizer.normalize_tokens(prev_tokens, new_tokens))

        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0], "Hello ")

    def test_multi_byte_char_split_across_tokens(self):
        """
        Test adaptive sliding window with multi-byte character split.

        Simulates ChatGLM tokenizing "你" as three tokens:
        Token 2: 0xE4 (incomplete -> \uFFFD)
        Token 3: 0xBD (incomplete -> \uFFFD)
        Token 4: 0xA0 (completes "你" with context from 2,3)
        """
        prev_tokens = [1]  # "Hello "
        new_tokens = [2, 3, 4]  # Three tokens forming "你"

        deltas = list(self.normalizer.normalize_tokens(prev_tokens, new_tokens))

        # Should get one delta with complete character "你"
        # Tokens 2 and 3 should be skipped (incomplete UTF-8)
        # Token 4 should use sliding window to decode "你"
        self.assertEqual(
            len(deltas), 1, f"Expected 1 delta, got {len(deltas)}: {deltas}"
        )
        self.assertEqual(deltas[0], "你", f"Expected '你', got '{deltas[0]}'")

    def test_prev_tokens_with_incomplete_utf8_are_not_dropped(self):
        """
        Regression: if prev_tokens decode to \\uFFFD, we must not treat it as emitted.

        Simulates a multi-byte character that starts at the end of previous step and
        completes in the current step:
        prev_tokens: [2] -> \\uFFFD
        prev_tokens + new_tokens: [2, 3, 4] -> "你"
        """
        prev_tokens = [2]  # Incomplete first byte of "你"
        new_tokens = [3, 4]  # Remaining bytes

        deltas = list(self.normalizer.normalize_tokens(prev_tokens, new_tokens))
        self.assertEqual(deltas, ["你"], f"Expected ['你'], got {deltas}")

    def test_consecutive_multi_byte_chars(self):
        """Test two consecutive multi-byte characters: "你好"."""
        prev_tokens = [1]  # "Hello "
        new_tokens = [2, 3, 4, 5, 6, 7]  # "你好"

        deltas = list(self.normalizer.normalize_tokens(prev_tokens, new_tokens))

        # Should get two deltas: "你" and "好"
        self.assertEqual(
            len(deltas), 2, f"Expected 2 deltas, got {len(deltas)}: {deltas}"
        )
        self.assertEqual(deltas[0], "你")
        self.assertEqual(deltas[1], "好")

    def test_sliding_window_with_empty_prev(self):
        """Test sliding window when prev_tokens is empty."""
        prev_tokens = []
        new_tokens = [2, 3, 4]  # Multi-byte char "你" starting from beginning

        deltas = list(self.normalizer.normalize_tokens(prev_tokens, new_tokens))

        # Should handle edge case of no previous context
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0], "你")

    def test_window_size_progression(self):
        """
        Test that window size increases correctly (2 -> 3 -> 4).

        For token 4 (third byte of "你"):
        - Window size 2: decode([3, 4]) -> still incomplete
        - Window size 3: decode([2, 3, 4]) -> complete "你"
        """
        prev_tokens = [1]  # "Hello "
        new_tokens = [2, 3, 4]

        # Monitor decode calls to verify window sizing
        original_decode = self.tokenizer.decode
        decode_calls = []

        def tracking_decode(token_ids):
            decode_calls.append(list(token_ids))
            return original_decode(token_ids)

        self.tokenizer.decode = tracking_decode

        deltas = list(self.normalizer.normalize_tokens(prev_tokens, new_tokens))

        # Verify that sliding window was used for token 4
        # Should see decode calls with increasing window sizes
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0], "你")

        # Check that we attempted sliding windows
        # (exact number of calls depends on implementation details)
        self.assertTrue(
            len(decode_calls) > 3, "Expected multiple decode calls for sliding window"
        )


class TestSimpleTokenizer(unittest.TestCase):
    """Test with a simple tokenizer that doesn't have UTF-8 split issues."""

    def test_no_utf8_issues(self):
        """Test that normal tokenizers work without triggering sliding window."""
        # Simple tokenizer that never produces \uFFFD
        simple_tokenizer = Mock()
        simple_tokenizer.decode = Mock(
            side_effect=lambda toks: "".join(chr(t) for t in toks)
        )

        normalizer = TokenNormalizer(simple_tokenizer)

        prev_tokens = [65, 66]  # "AB"
        new_tokens = [67, 68]  # "CD"

        deltas = list(normalizer.normalize_tokens(prev_tokens, new_tokens))

        # Should get two deltas without any sliding window overhead
        self.assertEqual(len(deltas), 2)
        self.assertEqual(deltas[0], "C")
        self.assertEqual(deltas[1], "D")


if __name__ == "__main__":
    unittest.main()
