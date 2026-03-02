"""
Token-level input normalization for MTP (Multi-Token-Per-step) streaming.

This module implements token-level decomposition to make detectors MTP-safe by ensuring
they receive text decoded from individual tokens rather than multi-token chunks.

Theoretical Foundation:
    Extended Transition Function Decomposition (from Compiler Theory):
    δ*(q₀, w) = δ(δ*(...δ(q₀, t₁), t₂), ...), tₙ)
    where w = t₁t₂...tₙ (string decomposition into tokens)

Key Insight:
    Multi-token input can be decomposed into sequential single-token processing.
    This moves complexity from FSM logic (detector code) to data pipeline (token iteration).

Usage:
    normalizer = TokenNormalizer(tokenizer)
    for delta_text in normalizer.normalize_tokens(prev_tokens, new_tokens):
        result = detector.parse_streaming_increment(delta_text, tools)
        # process result...
"""

import logging
from typing import Generator, List

logger = logging.getLogger(__name__)


class TokenNormalizer:
    """
    Normalizes multi-token input into single-token text deltas for detectors.

    Handles edge cases like incomplete UTF-8 sequences (\\uFFFD) by maintaining
    minimal context from previous tokens.
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
        across tokens (common in ChatGLM tokenizers).

        Args:
            prev_token_ids: Previously decoded token IDs (for context)
            new_token_ids: New token IDs from current iteration (may be 1+ tokens with MTP)

        Yields:
            str: Text delta for each token (decoded incrementally)
        """
        if not new_token_ids:
            return

        # NOTE: Some tokenizers (e.g. ChatGLM style) may decode trailing incomplete UTF-8
        # sequences as the replacement character (\uFFFD). We intentionally avoid emitting
        # \uFFFD and instead wait for enough context to decode a valid character. Because
        # of this, we must not treat trailing \uFFFD from prev_token_ids as "already
        # emitted" when slicing deltas for the current step.
        prev_decoded = self.tokenizer.decode(prev_token_ids)
        yielded_length = len(prev_decoded.rstrip("\uFFFD"))

        for i in range(len(new_token_ids)):
            cumulative_tokens = prev_token_ids + new_token_ids[: i + 1]
            decoded_cumulative = self.tokenizer.decode(cumulative_tokens)
            delta_text = decoded_cumulative[yielded_length:]

            if delta_text and "\uFFFD" in delta_text:
                valid_delta = self._resolve_incomplete_utf8(new_token_ids, i)
                if valid_delta and valid_delta != "\uFFFD":
                    yield valid_delta
                    yielded_length += len(valid_delta)
                continue

            if delta_text:
                yield delta_text
                yielded_length += len(delta_text)

    def _resolve_incomplete_utf8(
        self, new_token_ids: List[int], current_index: int
    ) -> str:
        """
        Resolve incomplete UTF-8 using adaptive sliding window.

        Tries increasing context windows (2, 3, 4 tokens) until finding valid UTF-8.
        """
        # Cap window size to avoid O(n^2) behavior for long MTP chunks.
        max_window_size = 4
        upper_bound = min(current_index + 1, max_window_size)
        for window_size in range(2, upper_bound + 1):
            window_start = max(0, current_index + 1 - window_size)
            window_tokens = new_token_ids[window_start : current_index + 1]
            window_decoded = self.tokenizer.decode(window_tokens)

            if "\uFFFD" not in window_decoded:
                if window_size > 1:
                    prev_window_decoded = self.tokenizer.decode(window_tokens[:-1])
                    return window_decoded[len(prev_window_decoded) :]
                return window_decoded

        return ""


def normalize_and_process(
    tokenizer,
    detector,
    prev_token_ids: List[int],
    new_token_ids: List[int],
    tools,
    is_streaming: bool,
):
    """
    Convenience function to normalize tokens and feed them to a detector.

    This function wraps the normalization logic and detector invocation,
    accumulating results across all token deltas.

    Args:
        tokenizer: Tokenizer instance
        detector: Format detector instance
        prev_token_ids: Previously decoded token IDs
        new_token_ids: New token IDs from current iteration
        tools: Tool definitions for detector
        is_streaming: Whether in streaming mode

    Returns:
        List of StreamingParseResult objects produced by feeding each normalized
        delta_text into the detector.

    Example:
        results, remaining = normalize_and_process(
            tokenizer, detector, [1,2,3], [4,5,6], tools, True
        )
        # Process results...
    """
    normalizer = TokenNormalizer(tokenizer)
    results = []

    for delta_text in normalizer.normalize_tokens(prev_token_ids, new_token_ids):
        result = detector.parse_streaming_increment(delta_text, tools)
        results.append(result)

    return results
