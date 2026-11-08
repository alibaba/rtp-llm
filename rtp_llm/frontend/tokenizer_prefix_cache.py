"""Append-only prompt tokenization cache for the frontend pipeline.

The frontend receives the full prompt text on every request of a session.  For
append-only histories (agentic workloads re-send the whole conversation each
step) the previous prompt is a strict text prefix of the new one, so its
tokenization can be reused instead of re-encoding hundreds of thousands of
tokens per request.

A character overlap ending at the append boundary is re-encoded together with
the appended suffix.  The cached ids are spliced only when the overlap region
encodes identically standalone, at the end of the cached prompt, and at the
start of ``overlap + suffix``; this proves the cut and the append boundary are
hard token breaks in both contexts.  Any mismatch - a merge crossing the
boundary, a prompt that does not extend a cached one, a tokenizer that rewrites
the overlap - falls back to a full encode of the complete prompt, so the cache
is purely an optimization and never changes the returned token ids.

Candidate cut positions are preferably whitespace-aligned (pre-tokenizer chunk
boundaries for BPE families); the fixed-length cut is also tried.  Tokenizers
that append trailing special tokens to every encode defeat the overlap check
and simply always fall back (correct, just not accelerated).
"""

import logging
import threading
from collections import OrderedDict
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Shortest overlap tried (characters): one BPE pre-token "word" with margin.
_MIN_OVERLAP_CHARS = 512
# Longest overlap tried (characters) when scanning back for aligned cuts.
_MAX_OVERLAP_CHARS = 4096
# Whitespace-aligned cut candidates tried before giving up on a splice.
_MAX_CUT_CANDIDATES = 8

# Prompts below this length skip the cache entirely: the machinery is not worth
# it for short prompts and they would evict long histories.
_MIN_PROMPT_CHARS = 4096

_WHITESPACE = " \t\n\r\f\v"


class TokenizerPrefixCache:
    """Thread-safe LRU of (prompt text, token ids) for append-only prompts."""

    def __init__(self, tokenizer, max_entries: int = 24, max_cached_ids: int = 4_000_000):
        self._tokenizer = tokenizer
        self._max_entries = max_entries
        self._max_cached_ids = max_cached_ids
        self._entries: "OrderedDict[str, List[int]]" = OrderedDict()
        self._cached_ids = 0
        self._lock = threading.Lock()
        self.stats = {
            "hit_splice": 0,
            "hit_prefix": 0,
            "no_entry": 0,
            "miss_fallback": 0,
            "cut_candidates": 0,
        }

    # -- public -----------------------------------------------------------

    def encode(self, prompt: str) -> List[int]:
        """Return the tokenizer's ids for ``prompt``, reusing cached prefixes."""
        if len(prompt) >= _MIN_PROMPT_CHARS:
            try:
                spliced = self._try_splice(prompt)
                if spliced is not None:
                    return spliced
            except Exception as exc:  # cache machinery must never fail a request
                logger.warning("tokenizer prefix cache fast path failed: %s", exc)
                self.stats["miss_fallback"] += 1
        ids = self._tokenizer.encode(prompt)
        if len(prompt) >= _MIN_PROMPT_CHARS and len(ids) <= self._max_cached_ids:
            try:
                self._store(prompt, ids)
            except Exception as exc:
                logger.warning("tokenizer prefix cache store failed: %s", exc)
        return ids

    # -- internals ---------------------------------------------------------

    def _full_encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    def _match(self, prompt: str) -> Tuple[Optional[str], Optional[List[int]]]:
        """Longest cached text that is a strict prefix of ``prompt``."""
        with self._lock:
            best_text: Optional[str] = None
            best_ids: Optional[List[int]] = None
            for text, ids in self._entries.items():
                if len(text) < len(prompt) and prompt.startswith(text):
                    if best_text is None or len(text) > len(best_text):
                        best_text, best_ids = text, ids
            if best_text is not None:
                self._entries.move_to_end(best_text)
            return best_text, best_ids

    def _cut_candidates(self, text: str) -> List[int]:
        """Cut positions for the overlap start, whitespace-aligned first.

        The first candidate is the fixed shortest overlap; the remaining ones
        step back to successive whitespace characters (pre-tokenizer chunk
        boundaries for BPE families) within the longest-overlap window.
        """
        end = len(text)
        cuts = [end - _MIN_OVERLAP_CHARS]
        lo = max(0, end - _MAX_OVERLAP_CHARS)
        p = end - _MIN_OVERLAP_CHARS
        while len(cuts) < _MAX_CUT_CANDIDATES and p > lo:
            q = max(
                text.rfind(ch, lo, p) for ch in _WHITESPACE[:3]
            )
            if q < lo:
                break
            cuts.append(q)
            p = q
        return [c for c in cuts if 0 < c < end]

    def _try_splice(self, prompt: str) -> Optional[List[int]]:
        text, ids = self._match(prompt)
        if text is None:
            self.stats["no_entry"] += 1
            return None
        if len(text) < _MIN_OVERLAP_CHARS + 1:
            self.stats["no_entry"] += 1
            return None
        suffix = prompt[len(text):]
        for cut in self._cut_candidates(text):
            overlap = text[cut:]
            overlap_ids = self._full_encode(overlap)
            if not overlap_ids:
                continue
            k = len(overlap_ids)
            self.stats["cut_candidates"] += 1
            # The overlap must be token-aligned at the end of the cached
            # prompt: its standalone encoding equals the cached ids' tail.
            if len(ids) < k or ids[-k:] != overlap_ids:
                continue
            merged = self._full_encode(overlap + suffix)
            # The append must not merge into the overlap region: the merged
            # encoding starts with exactly the overlap's standalone ids.
            if merged[:k] != overlap_ids:
                # A merge crossing the append boundary invalidates every cut;
                # stop scanning and take the full-encode fallback.
                return None
            result = ids[:-k] + merged
            self.stats["hit_splice"] += 1
            self._store(prompt, result)
            return result
        return None

    def _store(self, text: str, ids: List[int]) -> None:
        with self._lock:
            old = self._entries.pop(text, None)
            if old is not None:
                self._cached_ids -= len(old)
            self._entries[text] = ids
            self._cached_ids += len(ids)
            while self._entries and (
                len(self._entries) > self._max_entries
                or self._cached_ids > self._max_cached_ids
            ):
                _, evicted = self._entries.popitem(last=False)
                self._cached_ids -= len(evicted)
