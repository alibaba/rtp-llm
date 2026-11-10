"""Capacity regression test for the '/' pipeline prompt-prefix cache sizing.

The frontend worker constructs the Pipeline's ``TokenizerPrefixCache``; if it is
left at the class defaults, a served workload of many concurrent append-only
sessions (each re-sending its whole history every step) exceeds the default
``max_cached_ids`` and the LRU evicts entries right before their reuse, so most
steps fall back to a full O(history) re-encode on the request path.  These tests
pin the workload-sized configuration (32 entries / 8M ids) that the frontend
worker must use, and document the default-config thrash the sizing removes.
"""

import random
import unittest

from rtp_llm.frontend.tokenizer_prefix_cache import TokenizerPrefixCache

# Mirrors the frontend worker's construction (frontend_worker.py): the sizing
# this test pins.  Keep in sync with the production constants.
FRONTEND_CACHE_MAX_ENTRIES = 32
FRONTEND_CACHE_MAX_CACHED_IDS = 8_000_000

# A served-worst-case append-only workload: many concurrent sessions whose
# latest prompts together exceed the class defaults.
SESSIONS = 20
PROMPT_CHARS = 200_000  # ~200K one-char tokens per session: 20 x 200K = 4M ids
ROUNDS = 3


class CharTokenizer:
    """One token per character: every cut is aligned, appends never merge."""

    def __init__(self):
        self.calls = 0
        self.encoded_chars = 0

    def encode(self, text: str):
        self.calls += 1
        self.encoded_chars += len(text)
        return [ord(c) % 100003 for c in text]


def build_rounds(rng):
    """ROUNDS snapshots of SESSIONS append-only prompts (the served shape)."""
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 .,\n"
    sessions = ["".join(rng.choice(alphabet) for _ in range(PROMPT_CHARS))
                for _ in range(SESSIONS)]
    rounds = [sessions[:]]  # round 0: the initial prompts
    for _ in range(ROUNDS - 1):
        for i in range(SESSIONS):
            sessions[i] += "".join(rng.choice(alphabet) for _ in range(4000))
        rounds.append(sessions[:])
    return rounds


def session_rounds(rng):
    """ROUNDS snapshots of SESSIONS append-only prompts (the served shape)."""
    return build_rounds(rng)


def run_workload(cache, rounds, tokenizer):
    for round_idx in range(ROUNDS):
        for session_idx in range(SESSIONS):
            prompt = rounds[round_idx][session_idx]
            cached = cache.encode(prompt)
            expected = [ord(c) % 100003 for c in prompt]
            assert cached == expected
    return dict(cache.stats)


class FrontendPrefixCacheCapacityTest(unittest.TestCase):

    def test_frontend_sized_cache_serves_interleaved_sessions(self):
        """32 entries / 8M ids: every steady append splices, no fallback."""
        tokenizer = CharTokenizer()
        rounds = session_rounds(random.Random(1234))
        cache = TokenizerPrefixCache(
            tokenizer,
            max_entries=FRONTEND_CACHE_MAX_ENTRIES,
            max_cached_ids=FRONTEND_CACHE_MAX_CACHED_IDS,
        )
        stats = run_workload(cache, rounds, tokenizer)
        # Round 1 populates (one no_entry per session); rounds 2-3 must splice.
        self.assertEqual(stats["hit_splice"], SESSIONS * (ROUNDS - 1))
        self.assertEqual(stats["no_entry"], SESSIONS)
        self.assertEqual(stats["miss_fallback"], 0)
        # The tokenizer still ran only for misses/suffixes: far less than the
        # full 3x re-encode of every prompt.
        full_chars = 3 * SESSIONS * PROMPT_CHARS
        self.assertLess(tokenizer.encoded_chars, full_chars // 2)

    def test_default_sized_cache_thrashes_on_the_same_workload(self):
        """Class defaults (24/4M) evict before reuse: documents the removed bug.

        This asserts the DEFAULT configuration falls back on the served
        workload shape, i.e. that the frontend worker's explicit sizing is
        load-bearing and must not silently revert to defaults.
        """
        tokenizer = CharTokenizer()
        rounds = session_rounds(random.Random(1234))
        cache = TokenizerPrefixCache(tokenizer)  # defaults: 24 entries / 4M ids
        stats = run_workload(cache, rounds, tokenizer)
        # The working set (20 x ~200K + growth) exceeds 4M ids: some steady
        # steps must have fallen back to a full encode.
        self.assertGreater(stats["no_entry"] + stats["miss_fallback"], SESSIONS)


if __name__ == "__main__":
    unittest.main()
