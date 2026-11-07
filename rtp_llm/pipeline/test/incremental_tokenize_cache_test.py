"""Hermetic unit tests for IncrementalTokenizeCache.

The fake tokenizer is chunk-local by construction (ids(text) == concatenation
of per-chunk ids over the pre-tokenizer chunks), which is exactly the property
the production splice relies on and which was verified 100/100 on the real
served tokenizer at chunk-boundary cuts. These tests pin the cache machinery:
the accumulated production recurrence, fallbacks, eviction, shadow failsafe
and the env/compatibility gates.
"""

import os
import re
import threading
import unittest

from rtp_llm.pipeline.incremental_tokenize_cache import IncrementalTokenizeCache

_CHUNK_RE = re.compile(r"\S+\s*")


class FakePreTokenizer:
    def pre_tokenize_str(self, text):
        return [(m.group(0), (m.start(0), m.end(0))) for m in _CHUNK_RE.finditer(text)]


class FakeRaw:
    def __init__(self):
        self.pre_tokenizer = FakePreTokenizer()


def _chunk_ids(chunk):
    return [1000000 + len(chunk)] + [ord(c) for c in chunk]


class FakeFast:
    def __init__(self, adds_special=False):
        self._adds_special = adds_special

    def encode(self, text, add_special_tokens=True):
        ids = []
        if add_special_tokens and self._adds_special:
            ids.append(1)
        for m in _CHUNK_RE.finditer(text):
            ids.extend(_chunk_ids(m.group(0)))
        return ids


class FakeBaseTokenizer:
    """Mirrors BaseTokenizer: .encode passthrough + .tokenizer wrapper."""

    def __init__(self, adds_special=False):
        self.tokenizer = FakeFast(adds_special=adds_special)
        self.tokenizer.backend_tokenizer = FakeRaw()

    def encode(self, prompt, **kwargs):
        return self.tokenizer.encode(prompt, **kwargs)


WORDS = [
    "alpha",
    "beta42",
    "1234567890",
    "tool_call",
    "hello world",
    "缓存",
    "トークン",
    '{"role": "assistant"}',
    "line\nbreak",
    "x",
]


def _append(rng, text):
    piece = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 6)))
    sep = rng.choice([" ", "\n", "  ", "\t"])
    return text + sep + piece


class _EnvGuard:
    def __init__(self, **values):
        self._values = values
        self._saved = {}

    def __enter__(self):
        for k, v in self._values.items():
            self._saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class IncrementalTokenizeCacheTest(unittest.TestCase):
    def setUp(self):
        import random

        self.rng = random.Random(7)
        self.tok = FakeBaseTokenizer()

    def _cache(self, **kw):
        return IncrementalTokenizeCache(self.tok, **kw)

    def test_from_env_disabled_by_default(self):
        with _EnvGuard(RTP_TOKENIZE_CACHE_ENABLE=None):
            self.assertIsNone(IncrementalTokenizeCache.from_env(self.tok))

    def test_from_env_enabled_truthy_values(self):
        for value in ("1", "true", "TRUE", "yes", "on"):
            with _EnvGuard(
                RTP_TOKENIZE_CACHE_ENABLE=value,
                RTP_TOKENIZE_CACHE_MAX_BYTES="1048576",
                RTP_TOKENIZE_CACHE_SHADOW_RATE="0.5",
            ):
                cache = IncrementalTokenizeCache.from_env(self.tok)
                self.assertIsNotNone(cache, f"enable={value} should build a cache")

    def test_from_env_bad_numeric_disables(self):
        with _EnvGuard(
            RTP_TOKENIZE_CACHE_ENABLE="1", RTP_TOKENIZE_CACHE_MAX_BYTES="abc"
        ):
            self.assertIsNone(IncrementalTokenizeCache.from_env(self.tok))

    def test_special_tokens_tokenizer_disabled(self):
        special_tok = FakeBaseTokenizer(adds_special=True)
        with _EnvGuard(RTP_TOKENIZE_CACHE_ENABLE="1"):
            self.assertIsNone(IncrementalTokenizeCache.from_env(special_tok))

    def test_missing_raw_tokenizer_disabled(self):
        class Opaque:
            def encode(self, prompt, **kwargs):
                return [ord(c) for c in prompt]

        with _EnvGuard(RTP_TOKENIZE_CACHE_ENABLE="1"):
            self.assertIsNone(IncrementalTokenizeCache.from_env(Opaque()))

    def test_append_sequence_exactness(self):
        cache = self._cache()
        text = "Session run/0/0 header seed words"
        for step in range(40):
            text = _append(self.rng, text)
            self.assertEqual(
                cache.get_token_ids(text),
                list(self.tok.encode(text)),
                f"step {step}: spliced ids must equal the full encode",
            )

    def test_multiple_sessions_interleaved(self):
        cache = self._cache()
        texts = [f"Session run/{i}/0 seed text number {i}" for i in range(3)]
        for step in range(25):
            for i in range(len(texts)):
                texts[i] = _append(self.rng, texts[i])
                self.assertEqual(
                    cache.get_token_ids(texts[i]), list(self.tok.encode(texts[i]))
                )

    def test_repeated_identical_text(self):
        cache = self._cache()
        text = "some session text with several words inside " + " ".join(WORDS) * 20
        first = cache.get_token_ids(text)
        for _ in range(3):
            self.assertEqual(cache.get_token_ids(text), first)
        self.assertEqual(first, list(self.tok.encode(text)))

    def test_cold_reset_changes_key_and_stays_exact(self):
        cache = self._cache()
        text = "Session run/0/0 " + " ".join(WORDS) * 5
        cache.get_token_ids(text)
        reset = "Session run/0/1 " + "completely different cold start body words"
        self.assertEqual(cache.get_token_ids(reset), list(self.tok.encode(reset)))

    def test_prefix_hash_mismatch_falls_back(self):
        cache = self._cache()
        base = "shared prefix header " + " ".join(WORDS) * 8
        cache.get_token_ids(base)
        # Same first 256-char key bucket is impossible to forge cheaply, so
        # corrupt the stored entry hash to simulate any verification failure.
        key = next(iter(cache._entries))
        cache._entries[key].prefix_sha = "0" * 64
        grown = base + " appended words after corruption"
        self.assertEqual(cache.get_token_ids(grown), list(self.tok.encode(grown)))

    def test_shorter_text_than_cut_falls_back(self):
        cache = self._cache()
        text = "long enough session text " + " ".join(WORDS) * 10
        cache.get_token_ids(text)
        short = text[:30]
        self.assertEqual(cache.get_token_ids(short), list(self.tok.encode(short)))

    def test_lru_eviction_respects_cap_and_exactness(self):
        cache = self._cache(max_bytes=4096)
        for i in range(40):
            text = f"session {i} header " + " ".join(WORDS) * 6 + f" tail {i}"
            grown = _append(self.rng, text)
            self.assertEqual(cache.get_token_ids(text), list(self.tok.encode(text)))
            self.assertEqual(cache.get_token_ids(grown), list(self.tok.encode(grown)))
        self.assertLessEqual(cache._total_bytes, 4096)

    def test_shadow_mismatch_fails_safe(self):
        cache = self._cache(shadow_rate=1.0)
        reported = []
        cache._report_acc = lambda name: reported.append(name)
        # base text must exceed the 256-char key window so the bucket key is
        # append-stable (production prompts are >180KB; short texts safely miss)
        text = "session text for shadow verification " + " ".join(WORDS) * 25
        cache.get_token_ids(text)  # populate
        grown = text + " and appended words after the long base"
        original = cache._encode_tail

        def corrupted(tail):
            ids = original(tail)
            return ids + [999999] if tail else ids

        cache._encode_tail = corrupted
        result = cache.get_token_ids(grown)
        cache._encode_tail = original
        self.assertEqual(result, list(self.tok.encode(grown)))
        self.assertIn("TOKENIZE_CACHE_MISMATCH_QPS_METRIC", reported)

    def test_shadow_clean_path_no_mismatch(self):
        cache = self._cache(shadow_rate=1.0)
        reported = []
        cache._report_acc = lambda name: reported.append(name)
        text = "clean shadow session text " + " ".join(WORDS) * 25
        cache.get_token_ids(text)  # first call seeds the entry
        for _ in range(5):
            text = _append(self.rng, text)
            self.assertEqual(cache.get_token_ids(text), list(self.tok.encode(text)))
        self.assertNotIn("TOKENIZE_CACHE_MISMATCH_QPS_METRIC", reported)
        self.assertIn("TOKENIZE_CACHE_HIT_QPS_METRIC", reported)

    def test_edge_inputs(self):
        cache = self._cache()
        for text in ("", "x", "no-trailing-space", "trailing space ", "\n\n\n"):
            self.assertEqual(cache.get_token_ids(text), list(self.tok.encode(text)))

    def test_internal_error_falls_back(self):
        cache = self._cache()
        text = "session text that will hit an internal error path"
        cache.get_token_ids(text)

        def boom(_text):
            raise RuntimeError("simulated internal failure")

        cache._conservative_cut = boom
        grown = text + " appended"
        self.assertEqual(cache.get_token_ids(grown), list(self.tok.encode(grown)))

    def test_concurrent_callers_stay_exact(self):
        cache = self._cache()
        texts = ["thread session one seed words", "thread session two seed words"]
        errors = []

        def worker(idx):
            try:
                text = texts[idx]
                local = __import__("random").Random(100 + idx)
                for _ in range(20):
                    text = _append(local, text)
                    if cache.get_token_ids(text) != list(self.tok.encode(text)):
                        errors.append(idx)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in (0, 1)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
