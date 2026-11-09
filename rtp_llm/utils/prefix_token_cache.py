import hashlib
import logging
import threading
from array import array
from collections import OrderedDict
from typing import List, Optional, Tuple


class PrefixTokenCache:
    """Exact prefix-incremental tokenization cache.

    Caches full-text encodings per process and, for a new prompt that extends a
    cached one, encodes only the appended tail. The splice point is chosen at the
    last cached-prefix token whose char offset ends at a newline/whitespace (a
    pre-token-safe boundary), found by walking the stored offset-ends; a token
    window straddling the seam is then re-encoded and compared as the final
    guard. Any mismatch (or error) falls back to a full encode, so the result is
    always exactly tokenizer.encode(text).
    """

    SAFE_END_CHARS = ("\n", " ", "\t")

    def __init__(self, capacity: int = 768, min_tokens: int = 10000, window: int = 32):
        self.capacity = capacity
        self.min_tokens = min_tokens
        self.window = window
        self._lock = threading.Lock()
        # full-text sha256 -> (char_len, head sha256 of first 4096 chars, ids, offset ends)
        self._entries: "OrderedDict[str, Tuple[int, str, array, Optional[array]]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.fallbacks = 0
        self.walkbacks = 0
        self._calls = 0

    @staticmethod
    def _hash(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    @staticmethod
    def _offset_ends(tokenizer, text: str) -> Optional[array]:
        try:
            backend = tokenizer.tokenizer.backend_tokenizer
            enc = backend.encode(text, add_special_tokens=True)
            return array("q", (e for _, e in enc.offsets))
        except Exception:
            return None

    def _store(self, text: str, tokens: List[int], ends: Optional[array]) -> None:
        if len(tokens) < self.min_tokens:
            return
        key = self._hash(text)
        entry = (len(text), self._hash(text[:4096]), array("i", tokens), ends)
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

    def _find_prefix(
        self, text: str
    ) -> Optional[Tuple[int, array, Optional[array]]]:
        head4096 = self._hash(text[:4096])
        with self._lock:
            items = [(k, v) for k, v in self._entries.items() if v[0] < len(text)]
        # longest candidate first; the head hash filters to same-session chains
        items.sort(key=lambda kv: kv[1][0], reverse=True)
        for key, (char_len, head_hash, tokens, ends) in items:
            # entry head = hash of the entry's first min(4096, char_len) chars;
            # probe the same span of the new text
            probe = head4096 if char_len >= 4096 else self._hash(text[:char_len])
            if head_hash != probe:
                continue
            if self._hash(text[:char_len]) == key:
                return char_len, tokens, ends
        return None

    def _verify(self, tokenizer, prefix_tokens, tail_tokens: List[int]) -> bool:
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

    def _seam_index(self, text: str, char_len: int, ends: Optional[array]) -> Optional[int]:
        """Last token index whose char end lands on a pre-token-safe boundary."""
        if ends is None:
            return None
        floor = char_len - 4096
        k = len(ends) - 1
        while k >= 0 and ends[k] > floor:
            e = ends[k]
            if e <= char_len and 0 < e <= len(text) and text[e - 1] in self.SAFE_END_CHARS:
                return k
            k -= 1
        return None

    def encode(self, tokenizer, text: str) -> List[int]:
        self._calls += 1
        if self._calls % 500 == 0:
            logging.info(
                f"PrefixTokenCache stats: calls={self._calls} hits={self.hits} "
                f"misses={self.misses} fallbacks={self.fallbacks} "
                f"walkbacks={self.walkbacks} entries={len(self._entries)}"
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
            char_len, prefix_ids, ends = found
            try:
                k = self._seam_index(text, char_len, ends)
                seam = char_len if k is None else ends[k]
                cut = len(prefix_ids) if k is None else k + 1
                if k is not None:
                    self.walkbacks += 1
                tail_ids = tokenizer.encode(text[seam:], add_special_tokens=False)
                if self._verify(tokenizer, prefix_ids[:cut], tail_ids):
                    out = list(prefix_ids[:cut]) + tail_ids
                    tail_ends = self._offset_ends(tokenizer, text[seam:])
                    new_ends = None
                    if ends is not None and tail_ends is not None:
                        new_ends = ends[:cut] + array("q", (e + seam for e in tail_ends))
                    self._store(text, out, new_ends)
                    self.hits += 1
                    return out
                self.fallbacks += 1
                logging.warning("PrefixTokenCache boundary verify failed; full encode")
            except Exception:
                self.fallbacks += 1
        self.misses += 1
        tokens = tokenizer.encode(text)
        try:
            self._store(text, tokens, self._offset_ends(tokenizer, text))
        except Exception:
            pass
        return tokens
