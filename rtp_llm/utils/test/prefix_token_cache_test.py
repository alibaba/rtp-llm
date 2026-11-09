import unittest
from array import array

from rtp_llm.utils.prefix_token_cache import PrefixTokenCache


class _Enc:
    def __init__(self, ids, offsets):
        self.ids = ids
        self.offsets = offsets


class _Backend:
    """Mirrors the Rust backend: encode() returns an Encoding with .ids/.offsets."""

    def __init__(self, tok):
        self.tok = tok

    def encode(self, text, add_special_tokens=True):
        return self.tok.encode_with_offsets(text, add_special_tokens)


class CharTokenizer:
    """Character-level tokenizer with offset support; optionally merges adjacent
    "ab" into one token to simulate BPE merges spanning a splice boundary."""

    BOS = 1

    def __init__(self, merge_ab: bool = False):
        self.merge_ab = merge_ab
        # mirrors BaseTokenizer: .tokenizer.backend_tokenizer
        self.tokenizer = self
        self.backend_tokenizer = _Backend(self)

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

    def encode_with_offsets(self, text, add_special_tokens=True):
        ids, offsets = ([self.BOS], [(0, 0)]) if add_special_tokens else ([], [])
        i = 0
        while i < len(text):
            if self.merge_ab and text.startswith("ab", i):
                ids.append(100000)
                offsets.append((i, i + 2))
                i += 2
            else:
                ids.append(ord(text[i]) + 100)
                offsets.append((i, i + 1))
                i += 1
        return _Enc(ids, offsets)

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.BOS:
                continue
            out.append("ab" if i == 100000 else chr(i - 100))
        return "".join(out)


class PrefixTokenCacheV2Test(unittest.TestCase):
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

    def test_midtoken_seam_walks_back_and_still_hits(self):
        tok = CharTokenizer()
        cache = self.make_cache()
        # previous text ends mid-token-run ("xxxxx" continues in the tail)
        text = "header nonce-q\n" + "x" * 500
        cache.encode(tok, text)
        extended = text + "x" * 300 + "\n[assistant] reply\n"
        got = cache.encode(tok, extended)
        self.assertEqual(got, tok.encode(extended))
        # v2: no offsets on CharTokenizer.encode -> falls to v1 path; verify must hold
        self.assertGreaterEqual(cache.hits, 1)

    def test_offset_guided_walkback_hits_at_safe_boundary(self):
        tok = CharTokenizer()
        cache = self.make_cache()
        text = "header nonce-q\n" + "x" * 500
        cache.encode(tok, text)
        extended = text + "x" * 300 + "\n[assistant] reply\n"
        got = cache.encode(tok, extended)
        self.assertEqual(got, tok.encode(extended))

    def test_merge_boundary_walkback_avoids_bad_seam(self):
        tok = CharTokenizer(merge_ab=True)
        cache = self.make_cache()
        text = "prefix ends with a"
        cache.encode(tok, text)
        extended = text + "b and more content follows here"
        got = cache.encode(tok, extended)
        # v2 walks back to the space before "a"; the tail "ab and..." merges "ab"
        # exactly as a full encode does - exact, no fallback needed
        self.assertEqual(got, tok.encode(extended))
        self.assertGreaterEqual(cache.hits, 1)

    def test_no_safe_boundary_falls_back_but_stays_exact(self):
        tok = CharTokenizer(merge_ab=True)
        cache = self.make_cache()
        # no whitespace anywhere near the prefix tail: no safe seam exists
        text = "x" * 5000 + "a"
        cache.encode(tok, text)
        extended = text + "b" + "y" * 100
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
