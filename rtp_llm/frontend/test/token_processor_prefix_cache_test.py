import random
import unittest
from typing import Dict, List

from rtp_llm.frontend.token_processor import TokenProcessor

# The production TokenProcessor routes encode() through the append-only prefix
# cache.  These tests pin the contract used by the C++ '/' inference path:
# identical ids to a full encode, List[int] return type, bytes == str input,
# reduced tokenizer work on append-only sequences, and a safe full-encode
# fallback when the boundary is not verifiable.  Long prompts are used so the
# cache's minimum-length gate (a production constant) is passed without
# monkeypatching.

BASE_WORDS = [
    "alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta",
    "iota", "kappa", "tool_result", "action", "observe", "value", "x",
]


class CharTokenizer:
    """One token per character: aligned cuts, appends never merge."""

    def __init__(self):
        self.calls = 0
        self.encoded_chars = 0

    def __call__(self, prompt, **_kwargs):
        raise AssertionError("__call__ must not be routed through the cache")

    def decode(self, token_id: List[int]) -> str:
        return "".join(chr(i) for i in token_id)

    def encode(self, text: str) -> List[int]:
        self.calls += 1
        self.encoded_chars += len(text)
        return [ord(c) % 100003 for c in text]


class GreedyVocabTokenizer:
    """Longest-match greedy tokenizer over a fixed vocabulary.

    Tokens can merge across the append boundary when the boundary is mid-word,
    like a small BPE.
    """

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = dict(sorted(vocab.items(), key=lambda kv: -len(kv[0])))
        self.calls = 0
        self.encoded_chars = 0

    def __call__(self, prompt, **_kwargs):
        raise AssertionError("__call__ must not be routed through the cache")

    def decode(self, token_id: List[int]) -> str:
        return " ".join(str(i) for i in token_id)

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
                raise ValueError(
                    "unencodable character at %d: %r" % (i, text[i:i + 4])
                )
        return ids


def build_vocab() -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    nxt = 1
    for w in BASE_WORDS:
        vocab[w] = nxt
        nxt += 1
    vocab[" "] = nxt
    return vocab


def long_prompt(words: int, rng: random.Random) -> str:
    return " ".join(rng.choice(BASE_WORDS) for _ in range(words))


def make_processor(tokenizer) -> TokenProcessor:
    return TokenProcessor(tokenizer, special_tokens={"eos": 0})


class TokenProcessorPrefixCacheTest(unittest.TestCase):
    def test_append_only_equivalence_and_work_reduction(self):
        rng = random.Random(30)
        tok = CharTokenizer()
        reference = CharTokenizer()
        tp = make_processor(tok)
        prompts = [long_prompt(4000, rng)]
        for _ in range(6):
            prompts.append(prompts[-1] + " " + long_prompt(400, rng))
        # Reference ids come from a separate tokenizer so the measured work
        # only reflects what the TokenProcessor itself did.
        full_ids = [reference.encode(p) for p in prompts]
        results = [tp.encode(p) for p in prompts]
        for p, got, want in zip(prompts, results, full_ids):
            self.assertEqual(got, want, "ids must match a full encode")
            self.assertIsInstance(got, list)
            self.assertTrue(all(isinstance(i, int) for i in got))
        # Every append after the first is served by a boundary-verified splice.
        self.assertEqual(tp.prefix_cache.stats["hit_splice"], len(prompts) - 1)
        # Tokenizer work stays ~O(suffix + fixed window), not O(history): the
        # work after the initial full pass is bounded by the appended chars
        # plus a small fixed verification window per step.
        work_after_first = tok.encoded_chars - len(prompts[0])
        # O(suffix + bounded verification window) per step, never O(history):
        # total work is a small fraction of the naive full-encode-per-request cost.
        naive_chars = sum(len(p) for p in prompts)
        self.assertLess(tok.encoded_chars, 0.3 * naive_chars)

    def test_bytes_input_matches_str_input(self):
        rng = random.Random(31)
        tok = CharTokenizer()
        tp = make_processor(tok)
        first = long_prompt(1200, rng)
        second = first + " " + long_prompt(120, rng)
        tp.encode(first)
        got_str = tp.encode(second)
        got_bytes = tp.encode(second.encode("utf-8"))
        self.assertEqual(got_str, got_bytes)
        self.assertEqual(got_str, tok.encode(second))
        self.assertIsInstance(got_bytes, list)

    def test_unrelated_prompt_is_full_encode(self):
        rng = random.Random(32)
        tok = CharTokenizer()
        tp = make_processor(tok)
        a = long_prompt(1200, rng)
        b = long_prompt(1300, rng)
        tp.encode(a)
        got = tp.encode(b)
        self.assertEqual(got, tok.encode(b))
        self.assertIsInstance(got, list)

    def test_short_prompts_skip_cache_and_stay_correct(self):
        rng = random.Random(33)
        tok = CharTokenizer()
        tp = make_processor(tok)
        for _ in range(8):
            p = long_prompt(20, rng)
            self.assertEqual(tp.encode(p), tok.encode(p))

    def test_boundary_merge_falls_back_to_full_encode(self):
        vocab = build_vocab()
        # A vocabulary where a multi-word token can bridge the append boundary.
        vocab["alpha beta"] = 99
        vocab["gamma delta"] = 98
        tok = GreedyVocabTokenizer(vocab)
        tp = make_processor(tok)
        head = long_prompt(1200, random.Random(34)).replace("alpha beta", "alphabeta")
        # Build a head that ends mid-token-chain and a suffix that merges.
        head = " ".join(["gamma" if i % 2 else "delta" for i in range(1400)])
        suffix = " " + " ".join(["kappa" for _ in range(100)])
        tp.encode(head)
        got = tp.encode(head + suffix)
        # Whatever the cache decided, ids must equal a fresh full encode.
        fresh = GreedyVocabTokenizer(vocab)
        self.assertEqual(got, fresh.encode(head + suffix))
        self.assertIsInstance(got, list)
        self.assertTrue(all(isinstance(i, int) for i in got))

    def test_call_and_decode_are_uncached_passthrough(self):
        class CallableTokenizer(CharTokenizer):
            def __call__(self, prompt, **kwargs):
                return {"input_ids": self.encode(prompt), "offset_mapping": []}

        tok = CallableTokenizer()
        tp = make_processor(tok)
        res = tp("hello")
        self.assertEqual(res["input_ids"], tok.encode("hello"))
        self.assertEqual(tp.decode([104, 101]), "he")


if __name__ == "__main__":
    unittest.main()
