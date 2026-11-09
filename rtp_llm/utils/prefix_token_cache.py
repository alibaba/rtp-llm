import hashlib
import logging
import threading
from array import array
from collections import OrderedDict
from typing import List, Optional, Tuple


class PrefixTokenCache:
    """Exact prefix-incremental tokenization cache.

    Caches full-text encodings per process and, for a new prompt that extends a
    cached one, encodes only the appended tail. The splice is verified by
    re-encoding a token window straddling the boundary; any mismatch (or error)
    falls back to a full encode, so the result is always exactly
    tokenizer.encode(text).
    """

    def __init__(self, capacity: int = 512, min_tokens: int = 10000, window: int = 32):
        self.capacity = capacity
        self.min_tokens = min_tokens
        self.window = window
        self._lock = threading.Lock()
        # full-text sha256 -> (char_len, head sha256 of first 4096 chars, tokens)
        self._entries: "OrderedDict[str, Tuple[int, str, array]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.fallbacks = 0
        self._calls = 0

    @staticmethod
    def _hash(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _store(self, text: str, tokens: List[int]) -> None:
        if len(tokens) < self.min_tokens:
            return
        key = self._hash(text)
        entry = (len(text), self._hash(text[:4096]), array("i", tokens))
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

    def _find_prefix(self, text: str) -> Optional[Tuple[int, array]]:
        head4096 = self._hash(text[:4096])
        with self._lock:
            items = [(k, v) for k, v in self._entries.items() if v[0] < len(text)]
        # longest candidate first; the head hash filters to same-session chains
        items.sort(key=lambda kv: kv[1][0], reverse=True)
        for key, (char_len, head_hash, tokens) in items:
            # entry head = hash of the entry's first min(4096, char_len) chars;
            # probe the same span of the new text
            probe = head4096 if char_len >= 4096 else self._hash(text[:char_len])
            if head_hash != probe:
                continue
            if self._hash(text[:char_len]) == key:
                return char_len, tokens
        return None

    def _verify(self, tokenizer, prefix_tokens: array, tail_tokens: List[int]) -> bool:
        w = self.window
        if len(prefix_tokens) < w or len(tail_tokens) < w:
            return False
        window = list(prefix_tokens[-w:]) + list(tail_tokens[:w])
        try:
            window_text = tokenizer.decode(window)
            reencoded = tokenizer.encode(window_text, add_special_tokens=False)
            return reencoded == window
        except Exception:
            return False

    def encode(self, tokenizer, text: str) -> List[int]:
        self._calls += 1
        if self._calls % 500 == 0:
            logging.info(
                f"PrefixTokenCache stats: calls={self._calls} hits={self.hits} "
                f"misses={self.misses} fallbacks={self.fallbacks} entries={len(self._entries)}"
            )
        try:
            found = self._find_prefix(text)
        except Exception:
            found = None
        if found is None:
            # whole-text hit (identical prompt retried) still saves the encode
            key = self._hash(text)
            with self._lock:
                ent = self._entries.get(key)
                if ent is not None:
                    self._entries.move_to_end(key)
            if ent is not None:
                self.hits += 1
                return list(ent[2])
        if found is not None:
            char_len, prefix_tokens = found
            try:
                tail_tokens = tokenizer.encode(text[char_len:], add_special_tokens=False)
                if self._verify(tokenizer, prefix_tokens, tail_tokens):
                    out = list(prefix_tokens) + tail_tokens
                    self._store(text, out)
                    self.hits += 1
                    return out
                self.fallbacks += 1
                logging.warning("PrefixTokenCache boundary verify failed; full encode")
            except Exception:
                self.fallbacks += 1
        self.misses += 1
        tokens = tokenizer.encode(text)
        try:
            self._store(text, tokens)
        except Exception:
            pass
        return tokens
