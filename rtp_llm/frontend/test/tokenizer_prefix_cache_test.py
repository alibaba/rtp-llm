import random
import threading
import unittest
from typing import Dict, List

from rtp_llm.frontend.tokenizer_prefix_cache import TokenizerPrefixCache

# Long-enough prompts to pass the module's minimums (no monkeypatching: the
# tests must hold against the production constants).
BASE_WORDS = [
    "alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta",
    "iota", "kappa", "tool_result", "action", "observe", "value", "x",
]


class CharTokenizer:
    """One token per character: every cut is aligned, appends never merge."""

    def __init__(self):
        self.encoded_chars = 0
        self.calls = 0

    def encode(self, text: str) -> List[int]:
        self.calls += 1
        self.encoded_chars += len(text)
        return [ord(c) % 100003 for c in text]


class GreedyVocabTokenizer:
    """Longest-match greedy tokenizer over a fixed vocabulary.

    Behaves like a small BPE: tokens can merge across the append boundary when
    the boundary is mid-word, and the pre-token chunks are word-like.
    """

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = dict(sorted(vocab.items(), key=lambda kv: -len(kv[0])))
        self.encoded_chars = 0
        self.calls = 0

    def encode(self, text: str) -> List[int]:
        self.calls += 1
        self.encoded_chars += len(text)
        ids: List[int] = []
        i = 0
        while i < len(text):
            for token, tid in self.vocab.items():
                if text.startswith(token, i):
                    ids.append(tid)
                    i += len(token)
                    break
            else:
                raise ValueError("unencodable character at %d: %r" % (i, text[i:i + 4]))
        return ids


def build_vocab() -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    nxt = 1
    for w in BASE_WORDS:
        vocab[w] = nxt
        nxt += 1
    vocab[" "] = nxt
    nxt += 1
    vocab["\n"] = nxt
    nxt += 1
    # Merge pairs that can cross an append boundary when text ends mid-word.
    for w in ("gamma_next", "observe_next"):
        vocab[w] = nxt
        nxt += 1
    # Every printable ASCII character is its own token (real BPE families
    # always have full single-character coverage), so mid-word cuts encode.
    import string
    for ch in string.printable:
        if ch not in vocab:
            vocab[ch] = nxt
            nxt += 1
    return vocab


def make_prompt(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(BASE_WORDS) for _ in range(words)) + "\n"


class TokenizerPrefixCacheTest(unittest.TestCase):
    def test_append_only_splice_matches_full_encode(self):
        rng = random.Random(7)
        tok = GreedyVocabTokenizer(build_vocab())
        cache = TokenizerPrefixCache(tok, max_entries=4)
        prompt = make_prompt(rng, 900)
        first = cache.encode(prompt)
        # reference full encodes use a pristine tokenizer
        ref = GreedyVocabTokenizer(build_vocab())
        self.assertEqual(first, ref.encode(prompt))
        for step in range(6):
            prompt = prompt + make_prompt(rng, 40)
            got = cache.encode(prompt)
            self.assertEqual(got, ref.encode(prompt))
        # The splice path must have avoided re-encoding the whole prompt:
        # only overlap (~<=4k chars) + suffix per step after the first store.
        self.assertGreater(cache.stats["hit_splice"], 0)
        self.assertEqual(cache.stats["miss_fallback"], 0)

    def test_char_tokenizer_never_reencodes_prefix(self):
        rng = random.Random(11)
        tok = CharTokenizer()
        cache = TokenizerPrefixCache(tok, max_entries=2)
        prompt = make_prompt(rng, 900)
        cache.encode(prompt)
        before = tok.encoded_chars
        total_appended = 0
        for step in range(5):
            chunk = make_prompt(rng, 30)
            total_appended += len(chunk)
            prompt = prompt + chunk
            got = cache.encode(prompt)
            self.assertEqual(got, [ord(c) % 100003 for c in prompt])
        # encoded characters after the first prompt: at most overlap+suffix per
        # step, never the full prompt again.
        self.assertLess(tok.encoded_chars - before, total_appended + 5 * 8192)

    def test_boundary_merge_falls_back_to_full_encode(self):
        # The cached prompt ends mid-word ("gamma"); the append "amma_next" lets
        # the greedy tokenizer merge across the boundary, so the splice must be
        # rejected and the full encode returned.
        rng = random.Random(3)
        vocab = build_vocab()
        vocab["amma_next"] = 909
        tok = GreedyVocabTokenizer(vocab)
        cache = TokenizerPrefixCache(tok, max_entries=4)
        prompt = make_prompt(rng, 900)
        # force the prompt to end with the partial word "gam" (no trailing
        # newline): strip it and append "ma" so the cached text ends mid-word.
        prompt = prompt.rstrip("\n")
        cached = prompt + "gam"
        cache.encode(cached)
        appended = cached + "ma_next tail " + make_prompt(rng, 20)
        got = cache.encode(appended)
        ref = GreedyVocabTokenizer(vocab)
        self.assertEqual(got, ref.encode(appended))
        # and the following append splices again from the (correct) full result
        nxt = appended + make_prompt(rng, 20)
        self.assertEqual(cache.encode(nxt), ref.encode(nxt))

    def test_interleaved_sessions_each_splice_own_entry(self):
        rng = random.Random(5)
        tok = GreedyVocabTokenizer(build_vocab())
        cache = TokenizerPrefixCache(tok, max_entries=8)
        ref = GreedyVocabTokenizer(build_vocab())
        a = make_prompt(rng, 800)
        b = "nonce-b " + make_prompt(rng, 800)
        cache.encode(a)
        cache.encode(b)
        for step in range(4):
            a = a + make_prompt(rng, 25)
            b = b + make_prompt(rng, 25)
            self.assertEqual(cache.encode(a), ref.encode(a))
            self.assertEqual(cache.encode(b), ref.encode(b))
        self.assertGreaterEqual(cache.stats["hit_splice"], 6)

    def test_eviction_keeps_correctness(self):
        rng = random.Random(13)
        tok = GreedyVocabTokenizer(build_vocab())
        cache = TokenizerPrefixCache(tok, max_entries=2)
        ref = GreedyVocabTokenizer(build_vocab())
        prompts = ["nonce-%d " % i + make_prompt(rng, 800) for i in range(4)]
        for p in prompts:
            cache.encode(p)
        # the first two entries were evicted; re-encoding still matches
        for p in prompts:
            q = p + "tail " + make_prompt(rng, 10)
            self.assertEqual(cache.encode(q), ref.encode(q))
        self.assertLessEqual(len(cache._entries), 2)  # bounded by max_entries

    def test_short_prompts_bypass_cache(self):
        tok = CharTokenizer()
        cache = TokenizerPrefixCache(tok)
        cache.encode("short")
        self.assertEqual(list(cache._entries.items()), [])

    def test_threaded_appends_stay_correct(self):
        rng = random.Random(17)
        tok = GreedyVocabTokenizer(build_vocab())
        cache = TokenizerPrefixCache(tok, max_entries=8)
        ref = GreedyVocabTokenizer(build_vocab())
        base = [make_prompt(rng, 800) + "nonce-%d\n" % i for i in range(4)]
        for b in base:
            cache.encode(b)
        errors = []

        def run(idx: int):
            try:
                prompt = base[idx]
                for _ in range(10):
                    prompt = prompt + make_prompt(rng, 20)
                    got = cache.encode(prompt)
                    want = ref.encode(prompt)
                    if got != want:
                        errors.append((idx, len(got), len(want)))
            except Exception as exc:  # pragma: no cover
                errors.append((idx, repr(exc)))

        threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
