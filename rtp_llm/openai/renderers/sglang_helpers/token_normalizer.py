"""
Token-level input normalization for MTP (Multi-Token-Per-step) streaming.

This module implements token-level decomposition to make detectors MTP-safe by ensuring
they receive text decoded from individual tokens rather than multi-token chunks.

Design Note:
    This normalizer is designed to be stateless and created fresh each iteration.
    The caller (reasoning_tool_base_renderer.py) handles state by:
    1. Tracking last_output_ids and last_token_length
    2. Computing prev_token_ids = last_output_ids[-last_token_length:]
    3. This gives the normalizer the context it needs to resolve incomplete UTF-8

Theoretical Foundation:
    Extended Transition Function Decomposition (from Compiler Theory):
    δ*(q₀, w) = δ(δ*(...δ(q₀, t₁), t₂), ...), tₙ)
    where w = t₁t₂...tₙ (string decomposition into tokens)

Key Insight:
    Multi-token input can be decomposed into sequential single-token processing.
    This moves complexity from FSM logic (detector code) to data pipeline (token iteration).

Usage:
    # Created fresh each iteration (stateless)
    normalizer = TokenNormalizer(tokenizer)
    for delta_text in normalizer.normalize_tokens(prev_tokens, new_tokens):
        result = detector.parse_streaming_increment(delta_text, tools)
        # process result...
"""

import logging
from typing import Generator, List

logger = logging.getLogger(__name__)

# Max number of tokens to include in the sliding window when resolving incomplete
# UTF-8 sequences.  UTF-8 uses at most 4 bytes per code-point; with byte-level BPE
# each byte can be its own token, so a single character may span up to 4 tokens.
# Qwen-style tokenizers sometimes merge a preceding space into the same token as the
# first byte (e.g. " 薯" → 3 tokens), giving 5 tokens for space + 4-byte char.
# 6 provides a small safety margin.
_MAX_UTF8_WINDOW = 6


class TokenNormalizer:
    """
    Normalizes multi-token input into single-token text deltas for detectors.

    This is a stateless normalizer - it does not maintain state across calls.
    The caller is responsible for tracking context via prev_token_ids.

    Handles edge cases like incomplete UTF-8 sequences (\\uFFFD) by using
    a sliding window to combine tokens when necessary.
    """

    def __init__(self, tokenizer):
        """
        Initialize the token normalizer.

        Args:
            tokenizer: Tokenizer instance with decode() method
        """
        self.tokenizer = tokenizer

    def normalize_tokens(
        self, prev_token_ids: List[int], new_token_ids: List[int]
    ) -> Generator[str, None, None]:
        """
        Decompose multi-token input into single-token text deltas.

        Uses an adaptive sliding window to handle multi-byte UTF-8 characters split
        across tokens (common in ChatGLM and Qwen tokenizers).

        Args:
            prev_token_ids: Previously decoded token IDs (for context)
            new_token_ids: New token IDs from current iteration (may be 1+ tokens with MTP)

        Yields:
            str: Text delta for each token (decoded incrementally)
        """
        if not new_token_ids:
            return

        yielded_length = self._calculate_yielded_length(prev_token_ids)

        for i in range(len(new_token_ids)):
            cumulative_tokens = prev_token_ids + new_token_ids[: i + 1]
            decoded_cumulative = self.tokenizer.decode(cumulative_tokens)
            delta_text = decoded_cumulative[yielded_length:]

            if delta_text and "\uFFFD" in delta_text:
                # Try to resolve with a sliding window that includes future tokens
                valid_delta = self._try_resolve_with_future_tokens(
                    prev_token_ids, new_token_ids, i, yielded_length
                )
                if valid_delta and valid_delta != "\uFFFD":
                    yield valid_delta
                    yielded_length += len(valid_delta)
                # If can't resolve, skip this token - caller's prev_token_ids will
                # include it next iteration, allowing resolution then
                continue

            if delta_text:
                yield delta_text
                yielded_length += len(delta_text)

    def _calculate_yielded_length(self, prev_token_ids: List[int]) -> int:
        """
        Calculate the length of text that has already been yielded.

        Key insight: If prev_decoded ends with \uFFFD, it means the last token(s)
        produced incomplete output that was NOT yielded. In that case, we need to
        find where the incomplete portion starts and only count the complete part.

        We do this by iteratively decoding shorter prefixes of prev_token_ids
        until we find one that produces complete UTF-8 (no replacement characters).

        IMPORTANT: We must check token boundaries, not just strip \uFFFD from the
        end, because the character(s) before \uFFFD may also be part of the
        incomplete sequence (e.g., a space that's part of a multi-token encoding).

        Args:
            prev_token_ids: Previously decoded token IDs

        Returns:
            int: Length of the already-yielded portion
        """
        if not prev_token_ids:
            return 0

        prev_decoded = self.tokenizer.decode(prev_token_ids)

        # If no replacement character, everything was yielded
        if "\uFFFD" not in prev_decoded:
            return len(prev_decoded)

        # There's an incomplete sequence. We need to find the last COMPLETE token.
        # Iterate backwards through the tokens to find the longest prefix
        # that decodes without any replacement characters.
        for split_point in range(len(prev_token_ids) - 1, -1, -1):
            prefix_tokens = prev_token_ids[:split_point]
            if not prefix_tokens:
                return 0
            prefix_decoded = self.tokenizer.decode(prefix_tokens)
            if "\uFFFD" not in prefix_decoded:
                return len(prefix_decoded)

        # All tokens produce incomplete output
        return 0

    def _try_resolve_with_future_tokens(
        self,
        prev_token_ids: List[int],
        new_token_ids: List[int],
        current_index: int,
        yielded_length: int,
    ) -> str:
        """
        Try to resolve incomplete UTF-8 by looking ahead in the token sequence.

        This method looks at multiple tokens together to find valid UTF-8.
        It's designed for MTP (Multi-Token-Per-step) where multiple tokens arrive together.

        Args:
            prev_token_ids: Previously decoded token IDs (for context)
            new_token_ids: New token IDs including the current and future tokens
            current_index: Index of the current token within new_token_ids
            yielded_length: Length of already-yielded text

        Returns:
            str: Valid UTF-8 text if resolved, empty string otherwise
        """
        max_window_size = _MAX_UTF8_WINDOW

        # Combine prev and new tokens for sliding window context
        combined_tokens = prev_token_ids + new_token_ids
        current_absolute_index = len(prev_token_ids) + current_index

        # How many future tokens we can see (for MTP)
        max_lookahead = len(new_token_ids) - current_index - 1

        # First, check if new tokens alone decode cleanly (no prev context needed)
        # This handles cases where prev has incomplete UTF-8 but new tokens are complete
        # Example: prev=[107] -> '\uFFFD', new=[34718] -> '片' (complete alone)
        for lookahead in range(max_lookahead + 1):
            new_only_tokens = new_token_ids[
                current_index : current_index + 1 + lookahead
            ]
            if new_only_tokens:
                new_only_decoded = self.tokenizer.decode(new_only_tokens)
                if "\uFFFD" not in new_only_decoded:
                    # New tokens alone are valid! Return them directly
                    return new_only_decoded

        # Try different combinations of window size and lookahead with prev context
        for lookahead in range(max_lookahead + 1):
            window_end = current_absolute_index + 1 + lookahead

            for window_size in range(2, max_window_size + 1):
                window_start = max(0, window_end - window_size)

                if window_end > len(combined_tokens):
                    continue

                window_tokens = combined_tokens[window_start:window_end]
                window_decoded = self.tokenizer.decode(window_tokens)

                if "\uFFFD" not in window_decoded:
                    # Found valid UTF-8! Calculate what to emit
                    # Get the part that's from the current position onwards
                    full_decoded = self.tokenizer.decode(combined_tokens[:window_end])
                    new_text = full_decoded[yielded_length:]

                    if new_text and "\uFFFD" not in new_text:
                        return new_text

        return ""
