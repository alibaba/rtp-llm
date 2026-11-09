import unittest

from rtp_llm.utils.prefix_token_cache import PrefixTokenCache


class CharTokenizer:
    """Character-level tokenizer; optionally merges adjacent "ab" into one token
    to simulate BPE merges spanning a splice boundary."""

    BOS = 1

    def __init__(self, merge_ab: bool = False):
        self.merge_ab = merge_ab

    def encode(self, text, add_special_tokens=True):
        ids = [self.BOS] if add_special_tokens else []
        i = 0
        while i < len(text):
            if self.merge_ab and text.startswith("ab", i):
                ids.append(100000)
                i += 2
            else:
                ids.append(ord(text[i]) + 100)
                i += 1
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.BOS:
                continue
            out.append("ab" if i == 100000 else chr(i - 100))
        return "".join(out)


class PrefixTokenCacheTest(unittest.TestCase):
    def make_cache(self, capacity=8, min_tokens=0, window=8):
        return PrefixTokenCache(capacity=capacity, min_tokens=min_tokens, window=window)

    def test_cold_miss_then_exact_hit_chain(self):
        tok = CharTokenizer()
        cache = self.make_cache()
        text = "system: nonce-abc123\n[assistant] step 0\n"
        base = cache.encode(tok, text)
        self.assertEqual(base, tok.encode(text))
        self.assertEqual(cache.misses, 1)
        cur = text
        for i in range(1, 6):
            cur = cur + f"[tool] result {i} " + "x" * (50 * i) + "\n[assistant] next\n"
            got = cache.encode(tok, cur)
            self.assertEqual(got, tok.encode(cur), f"step {i} mismatch")
        self.assertEqual(cache.hits, 5)
        self.assertEqual(cache.fallbacks, 0)

    def test_identical_prompt_is_exact_hit(self):
        tok = CharTokenizer()
        cache = self.make_cache()
        text = "hello world " * 100
        cache.encode(tok, text)
        again = cache.encode(tok, text)
        self.assertEqual(again, tok.encode(text))

    def test_merge_boundary_falls_back_but_stays_exact(self):
        tok = CharTokenizer(merge_ab=True)
        cache = self.make_cache()
        text = "prefix ends with a"
        cache.encode(tok, text)
        extended = text + "b and more content follows here"
        got = cache.encode(tok, extended)
        self.assertEqual(got, tok.encode(extended))
        self.assertGreaterEqual(cache.fallbacks, 1)

    def test_eviction_by_recency(self):
        tok = CharTokenizer()
        cache = self.make_cache(capacity=2)
        texts = ["session a " + "x" * 100, "session b " + "y" * 100, "session c " + "z" * 100]
        for t in texts:
            cache.encode(tok, t)
        self.assertEqual(len(cache._entries), 2)
        # the oldest (session a) was evicted: extending it must miss, not hit
        extended_a = texts[0] + " plus a tail"
        before = cache.hits
        cache.encode(tok, extended_a)
        self.assertEqual(cache.hits, before)

    def test_small_prompt_not_cached(self):
        tok = CharTokenizer()
        cache = self.make_cache(min_tokens=1000)
        text = "tiny"
        cache.encode(tok, text)
        self.assertEqual(len(cache._entries), 0)

    def test_no_false_prefix_across_sessions(self):
        tok = CharTokenizer()
        cache = self.make_cache()
        a = "nonce-AAA " + "shared body " * 200
        b = "nonce-BBB " + "shared" * 200
        cache.encode(tok, a)
        got = cache.encode(tok, b)
        self.assertEqual(got, tok.encode(b))
        # different head: no prefix hit path taken
        self.assertEqual(cache.hits, 0)

    def test_exception_in_tail_falls_back(self):
        class FlakyTail(CharTokenizer):
            def encode(self, text, add_special_tokens=True):
                if not add_special_tokens:
                    raise RuntimeError("boom")
                return super().encode(text, add_special_tokens=add_special_tokens)

        tok = FlakyTail()
        cache = self.make_cache()
        text = "base prompt with enough chars " * 10
        cache.encode(tok, text)
        extended = text + "tail"
        got = cache.encode(tok, extended)
        self.assertEqual(got, CharTokenizer().encode(extended))
        self.assertGreaterEqual(cache.fallbacks, 1)


if __name__ == "__main__":
    unittest.main()
