import os
import random
import threading
import unittest
from pathlib import Path

from rtp_llm.frontend.tokenizer_prefix_cache import PrefixCachingTokenizer


class _FakeHFTokenizer:
    """Word-level fake: keeps a leading space on the following word.

    encode(text) optionally prepends one special head token; the no-special
    path returns the plain word ids. Words never merge, so the fake is
    compositional exactly at word boundaries.
    """

    def __init__(self, head=True):
        self._head = head
        self._vocab = {}

    def words(self, text):
        words = []
        current = ""
        for char in text:
            if char == " ":
                if current:
                    words.append(current)
                current = " "
            else:
                current += char
        if current:
            words.append(current)
        return words

    def _ids(self, text, add_special_tokens=True):
        ids = [1] if (add_special_tokens and self._head) else []
        for word in self.words(text):
            if word not in self._vocab:
                self._vocab[word] = 1000 + len(self._vocab)
            ids.append(self._vocab[word])
        return ids

    def _offsets(self, text):
        offsets = [(0, 0)] if self._head else []
        position = 0
        for word in self.words(text):
            offsets.append((position, position + len(word)))
            position += len(word)
        return offsets

    def encode(self, text, add_special_tokens=True):
        return self._ids(text, add_special_tokens)

    def encode_plus(self, text, return_offsets_mapping=False, add_special_tokens=True):
        assert return_offsets_mapping
        return {
            "input_ids": list(self._ids(text, add_special_tokens)),
            "offset_mapping": self._offsets(text),
        }


class _FakeTokenizer:
    """BaseTokenizer-like wrapper around the fake HF tokenizer."""

    def __init__(self, head=True):
        self.tokenizer = _FakeHFTokenizer(head)

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)


def _fake_text(words, seed):
    rng = random.Random(seed)
    vocabulary = [
        "".join(rng.choices("abcdefgh", k=rng.randint(2, 7))) for _ in range(words)
    ]
    return " ".join(vocabulary)


class PrefixCachingTokenizerFakeTest(unittest.TestCase):
    def _cache(self, tokenizer=None, **kwargs):
        defaults = dict(
            probe_chars=32,
            tail_chars=16,
            window_chars=8,
            min_prefix_chars=64,
            min_prefix_tokens=8,
            stats_interval=10**9,
        )
        defaults.update(kwargs)
        return PrefixCachingTokenizer(tokenizer or _FakeTokenizer(), **defaults)

    def test_structure_probe_passes_with_head_token(self):
        cache = self._cache()
        self.assertTrue(cache._enabled)
        self.assertEqual(cache._head, 1)

    def test_structure_probe_rejects_trailing_special(self):
        class Trailing(_FakeTokenizer):
            def encode(self, text, **kwargs):
                ids = super().encode(text, **kwargs)
                if kwargs.get("add_special_tokens", True):
                    return ids + [2]
                return ids

        tokenizer = Trailing()
        cache = self._cache(tokenizer)
        self.assertFalse(cache._enabled)
        text = _fake_text(50, 1)
        self.assertEqual(cache.encode(text), tokenizer.encode(text))

    def test_structure_probe_rejects_missing_hf_tokenizer(self):
        class Bare:
            def encode(self, text, **kwargs):
                return [1, 2, 3]

        cache = self._cache(Bare())
        self.assertFalse(cache._enabled)
        self.assertEqual(cache.encode("anything"), [1, 2, 3])

    def test_roundtrip_equals_plain_tokenizer(self):
        tokenizer = _FakeTokenizer()
        cache = self._cache(tokenizer)
        base = _fake_text(80, 7)
        extensions = [base + " " + _fake_text(10, seed) for seed in range(8, 14)]
        extensions.append(base)  # exact prefix reuse
        for text in [base] + extensions:
            with self.subTest(text=text[:24]):
                self.assertEqual(cache.encode(text), tokenizer.encode(text))

    def test_hit_composes_only_the_suffix(self):
        tokenizer = _FakeTokenizer()
        cache = self._cache(tokenizer)
        base = _fake_text(80, 7)
        cache.encode(base)
        suffix = " " + _fake_text(10, 99)
        calls = []
        plain = tokenizer.encode

        def counting_encode(text, **kwargs):
            calls.append(len(text))
            return plain(text, **kwargs)

        tokenizer.encode = counting_encode
        composed = cache.encode(base + suffix)
        tokenizer.encode = plain
        self.assertEqual(composed, plain(base + suffix))
        # The long base is never re-encoded on a hit: only the window probes
        # and the suffix pass through the tokenizer.
        self.assertTrue(all(size < len(base) for size in calls))
        self.assertGreater(cache.stats()["hits"], 0)

    def test_divergent_text_falls_back_and_still_matches(self):
        tokenizer = _FakeTokenizer()
        cache = self._cache(tokenizer)
        base = _fake_text(80, 7)
        cache.encode(base)
        divergent = base[:64] + _fake_text(40, 21)
        self.assertEqual(cache.encode(divergent), tokenizer.encode(divergent))

    def test_env_flag_disables_cache(self):
        os.environ["RTP_LLM_TOKENIZER_PREFIX_CACHE"] = "0"
        try:
            cache = self._cache()
            self.assertFalse(cache._enabled)
        finally:
            del os.environ["RTP_LLM_TOKENIZER_PREFIX_CACHE"]

    def test_midword_entry_is_rejected_by_window_test(self):
        tokenizer = _FakeTokenizer()
        cache = self._cache(tokenizer)
        base = _fake_text(80, 7)
        # Manually store a prefix that ends in the middle of a word.
        cut = len(base) - 20
        while base[cut - 1] != " ":
            cut -= 1
        cut += 3  # inside the last word of the kept part
        prefix = base[:cut]
        prefix_ids = tokenizer.encode(prefix)
        cache._insert(prefix, prefix_ids)
        extended = base + " tail words here"
        self.assertEqual(cache.encode(extended), tokenizer.encode(extended))
        self.assertEqual(cache.stats()["fallbacks"], 1)

    def test_lru_eviction_bounds_memory(self):
        tokenizer = _FakeTokenizer()
        cache = self._cache(tokenizer, max_entries=2, max_cached_ids=10**9)
        for seed in range(5):
            cache.encode(_fake_text(80, seed))
        self.assertLessEqual(len(cache._entries), 2)
        self.assertLessEqual(cache.stats()["cached_ids"], 2 * 90)

    def test_total_ids_bound_evicts(self):
        tokenizer = _FakeTokenizer()
        cache = self._cache(tokenizer, max_entries=10, max_cached_ids=150)
        for seed in range(5):
            cache.encode(_fake_text(80, seed))
        self.assertLessEqual(cache.stats()["cached_ids"], 150)
        self.assertTrue(cache._entries)

    def test_short_texts_bypass_cache(self):
        tokenizer = _FakeTokenizer()
        cache = self._cache(tokenizer)
        short = _fake_text(5, 3)
        self.assertEqual(cache.encode(short), tokenizer.encode(short))
        self.assertEqual(cache.stats()["ops"], 0)

    def test_kwargs_bypass_cache(self):
        tokenizer = _FakeTokenizer()
        cache = self._cache(tokenizer)
        text = _fake_text(80, 7)
        self.assertEqual(cache.encode(text), tokenizer.encode(text))
        self.assertEqual(
            cache.encode(text, add_special_tokens=False),
            tokenizer.encode(text, add_special_tokens=False),
        )

    def test_concurrent_hits_are_consistent(self):
        tokenizer = _FakeTokenizer()
        cache = self._cache(tokenizer)
        base = _fake_text(80, 7)
        cache.encode(base)
        texts = [base + " " + _fake_text(10, seed) for seed in range(8, 16)]
        results = [None] * 8
        barrier = threading.Barrier(8)

        def worker(index):
            barrier.wait()
            results[index] = cache.encode(texts[index])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for text, result in zip(texts, results):
            self.assertEqual(result, tokenizer.encode(text))


def _real_tokenizer():
    checkpoint = os.environ.get("DSV41_REFERENCE_CHECKPOINT")
    if not checkpoint or not Path(checkpoint, "tokenizer.json").exists():
        return None
    from transformers import PreTrainedTokenizerFast

    class Tokenizer:
        def __init__(self, fast):
            self.tokenizer = fast

        def encode(self, text, **kwargs):
            return self.tokenizer.encode(text, **kwargs)

    return Tokenizer(
        PreTrainedTokenizerFast(tokenizer_file=str(Path(checkpoint) / "tokenizer.json"))
    )


class PrefixCachingTokenizerRealTest(unittest.TestCase):
    """Differential tests against the real V4.1 tokenizer.

    Skipped unless DSV41_REFERENCE_CHECKPOINT points at the reference
    checkpoint (same convention as deepseekv41_renderer_test.py).
    """

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _real_tokenizer()
        if cls.tokenizer is None:
            raise unittest.SkipTest("DSV41_REFERENCE_CHECKPOINT not set")

    def _cache(self, **kwargs):
        defaults = dict(
            min_prefix_chars=2048, min_prefix_tokens=512, stats_interval=10**9
        )
        defaults.update(kwargs)
        return PrefixCachingTokenizer(self.tokenizer, **defaults)

    def _natural_text(self, words, seed):
        rng = random.Random(seed)
        parts = []
        for _ in range(words):
            length = rng.randint(3, 9)
            parts.append("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=length)))
        text = " ".join(parts)
        return text + "\nOnly reply with PERF_OK."

    def test_structure_probe_accepts_real_tokenizer(self):
        cache = self._cache()
        self.assertTrue(cache._enabled)

    def test_shared_prefix_extensions_match_full_encode(self):
        cache = self._cache()
        rng = random.Random(1234)
        for trial in range(6):
            with self.subTest(trial=trial):
                base = self._natural_text(400, 1000 + trial)
                self.assertEqual(cache.encode(base), self.tokenizer.encode(base))
                for _ in range(4):
                    cut = rng.randint(1, len(base))
                    extended = (
                        base[:cut] + " " + self._natural_text(20, trial * 10 + cut)
                    )
                    self.assertEqual(
                        cache.encode(extended), self.tokenizer.encode(extended)
                    )

    def test_c8_geometry_prime_and_suffixes(self):
        """Prime prompt with template tail, then 8 prompts sharing the prefix."""
        cache = self._cache()
        prefix = self._natural_text(2400, 42)
        header = " headerword"
        template_head = "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>"
        template_tail = "<\uff5cAssistant\uff5c>\n```output\n"
        prime = template_head + prefix + header + template_tail
        self.assertEqual(cache.encode(prime), self.tokenizer.encode(prime))
        suffixes = [" " + self._natural_text(160, 500 + i) for i in range(8)]
        for index, suffix in enumerate(suffixes):
            measured = template_head + prefix + header + suffix + template_tail
            with self.subTest(index=index):
                self.assertEqual(
                    cache.encode(measured), self.tokenizer.encode(measured)
                )
        stats = cache.stats()
        self.assertGreaterEqual(stats["hits"], 8)
        self.assertEqual(stats["fallbacks"], 0)

    def test_prime_replay_uses_exact_match(self):
        cache = self._cache()
        prime = (
            "<\uff5cUser\uff5c>"
            + self._natural_text(2400, 4242)
            + "<\uff5cAssistant\uff5c>\n"
        )
        first = cache.encode(prime)
        self.assertEqual(first, self.tokenizer.encode(prime))
        self.assertEqual(cache.encode(prime), self.tokenizer.encode(prime))
        self.assertGreaterEqual(cache.stats()["hits"], 1)

    def test_boundary_merging_suffix_falls_back(self):
        cache = self._cache()
        base = (
            "<\uff5cUser\uff5c>"
            + self._natural_text(2400, 77)
            + "<\uff5cAssistant\uff5c>\n"
        )
        self.assertEqual(cache.encode(base), self.tokenizer.encode(base))
        entry = next(iter(cache._entries.values()))
        # Extend with a letter directly: if the cached boundary splits a word,
        # the window test must reject and the full encode must be used.
        glued = entry.prefix_text + "gluedextra words follow here"
        self.assertEqual(cache.encode(glued), self.tokenizer.encode(glued))

    def test_concurrent_composition_matches(self):
        cache = self._cache()
        prefix = self._natural_text(2400, 555)
        prime = "<\uff5cUser\uff5c>" + prefix + "<\uff5cAssistant\uff5c>\n"
        self.assertEqual(cache.encode(prime), self.tokenizer.encode(prime))
        texts = [
            "<\uff5cUser\uff5c>"
            + prefix
            + " "
            + self._natural_text(80, 600 + i)
            + "<\uff5cAssistant\uff5c>\n"
            for i in range(8)
        ]
        results = [None] * 8
        barrier = threading.Barrier(8)

        def worker(index):
            barrier.wait()
            results[index] = cache.encode(texts[index])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for text, result in zip(texts, results):
            self.assertEqual(result, self.tokenizer.encode(text))


if __name__ == "__main__":
    unittest.main()
