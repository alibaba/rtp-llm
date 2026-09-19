import logging
import os
import threading
from array import array
from bisect import bisect_right
from collections import OrderedDict

logger = logging.getLogger(__name__)

_ENABLE_ENV = "RTP_LLM_TOKENIZER_PREFIX_CACHE"
_DEFAULT_ENABLE = True


def _env_enabled() -> bool:
    raw = os.environ.get(_ENABLE_ENV)
    if raw is None or raw == "":
        return _DEFAULT_ENABLE
    return raw.strip().lower() not in ("0", "false", "off", "no")


class _Entry:
    """Immutable cache entry: the ids of one verified prompt prefix."""

    __slots__ = ("prefix_text", "ids")

    def __init__(self, prefix_text: str, ids: array):
        self.prefix_text = prefix_text
        self.ids = ids


class PrefixCachingTokenizer:
    """Drop-in tokenizer proxy that reuses the ids of recently encoded prefixes.

    Serving prompts that extend a recently encoded text (shared system prompt,
    prefix-cache hits, multi-turn conversation) re-tokenizes the whole shared
    prefix on every request. A 131k-token prompt costs ~0.3 s of CPU in the HF
    fast tokenizer, which dominates cache-hit TTFT and serializes concurrent
    frontend renders. This proxy caches the token ids of a long, verified
    prefix and later encodes only the new suffix.

    Correctness contract: ``encode(text)`` returns exactly what the wrapped
    tokenizer returns. Every cached composition is guarded so a wrong result
    is impossible; on any doubt the proxy falls back to a full encode:

    1. the tokenizer's post-processor structure is probed once at init; only
       the plain "optional leading special token, nothing appended" shape is
       accepted, otherwise the cache disables itself;
    2. an entry is stored only at a token boundary whose split is verified
       against the original continuation with a differential encode;
    3. every cache hit re-verifies the boundary with a window differential
       test built from the actual prefix tail and suffix head.

    The cache is bounded (entries and total ids) and thread-safe; lookups and
    encodes run outside the lock.
    """

    def __init__(
        self,
        tokenizer,
        max_entries: int = 8,
        max_cached_ids: int = 2_000_000,
        probe_chars: int = 2048,
        tail_chars: int = 256,
        window_chars: int = 64,
        min_prefix_chars: int = 4096,
        min_prefix_tokens: int = 1024,
        max_split_tries: int = 12,
        stats_interval: int = 64,
    ):
        self._wrap = tokenizer
        # Underlying HF tokenizer used for offset-mapped encodes; absent on
        # stub tokenizers, which disables the cache transparently.
        self._hf = getattr(tokenizer, "tokenizer", None)
        self._max_entries = max_entries
        self._max_cached_ids = max_cached_ids
        self._probe_chars = probe_chars
        self._tail_chars = tail_chars
        self._window_chars = window_chars
        self._min_prefix_chars = min_prefix_chars
        self._min_prefix_tokens = min_prefix_tokens
        self._max_split_tries = max_split_tries
        self._stats_interval = stats_interval
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._total_ids = 0
        self._head = 0
        self._enabled = _env_enabled() and self._probe_structure()
        self._ops = 0
        self._hits = 0
        self._fallbacks = 0
        if not self._enabled:
            logger.info(
                "tokenizer prefix cache disabled (structure probe failed or %s=0)",
                _ENABLE_ENV,
            )

    # ------------------------------------------------------------------ public

    def encode(self, text, **kwargs):
        """Same result as ``tokenizer.encode(text, **kwargs)``."""
        if (
            kwargs
            or not self._enabled
            or not isinstance(text, str)
            or len(text) < self._min_prefix_chars
        ):
            return self._wrap.encode(text, **kwargs)
        self._ops += 1
        entry = self._lookup(text)
        if entry is not None and text.startswith(entry.prefix_text):
            if len(text) == len(entry.prefix_text):
                # The text is exactly the cached prefix: its ids were verified
                # when the entry was stored (split differential check).
                self._hits += 1
                self._maybe_log_stats()
                return entry.ids.tolist()
            remainder = text[len(entry.prefix_text) :]
            if self._split_ok(entry.prefix_text, remainder):
                suffix_ids = self._wrap.encode(remainder, add_special_tokens=False)
                self._hits += 1
                self._maybe_log_stats()
                return entry.ids.tolist() + suffix_ids
            self._fallbacks += 1
        encoded = self._hf.encode_plus(text, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        self._store(text, ids, offsets)
        self._maybe_log_stats()
        return ids

    def stats(self):
        return {
            "enabled": self._enabled,
            "entries": len(self._entries),
            "cached_ids": self._total_ids,
            "ops": self._ops,
            "hits": self._hits,
            "fallbacks": self._fallbacks,
        }

    # ------------------------------------------------------------- internals

    def _plain_encode(self, text, **kwargs):
        return self._wrap.encode(text, **kwargs)

    def _probe_structure(self) -> bool:
        """Accept only plain tokenizers: optional leading special, no tail.

        ``encode(text)`` must equal ``head + encode(text, add_special_tokens=
        False)`` with the same head for different probes, and the offset-mapped
        encode must agree with the plain one. Anything else (trailing EOS,
        per-call prefix ids, missing offsets) disables the cache.
        """
        if self._hf is None:
            return False
        head = None
        for probe in ("prefix cache structure probe", "second probe 0123456789"):
            try:
                full = self._wrap.encode(probe)
                nos = self._wrap.encode(probe, add_special_tokens=False)
                mapped = self._hf.encode_plus(probe, return_offsets_mapping=True)
                offsets = mapped["offset_mapping"]
            except Exception:
                return False
            if list(mapped["input_ids"]) != list(full) or len(offsets) != len(full):
                return False
            if full == nos:
                probe_head = 0
            elif len(full) == len(nos) + 1 and list(full[1:]) == list(nos):
                probe_head = 1
            else:
                return False
            if head is None:
                head = probe_head
            elif head != probe_head:
                return False
        self._head = head
        return True

    def _lookup(self, text):
        key = text[: self._probe_chars]
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
        return entry

    def _split_ok(self, prefix_text: str, remainder: str) -> bool:
        """Differential window test around the cached split boundary."""
        window = self._window_chars
        head = prefix_text[-window:]
        tail = remainder[:window]
        joined = self._wrap.encode(head + tail, add_special_tokens=False)
        split = self._wrap.encode(head, add_special_tokens=False) + self._wrap.encode(
            tail, add_special_tokens=False
        )
        return list(joined) == list(split)

    def _store(self, text: str, ids, offsets) -> None:
        if len(text) < self._min_prefix_chars or len(ids) < self._min_prefix_tokens:
            return
        limit = len(text) - self._tail_chars
        ends = [int(end) for _, end in offsets]
        count = bisect_right(ends, limit)
        tries = 0
        while (
            count - self._head >= self._min_prefix_tokens
            and tries < self._max_split_tries
        ):
            split = ends[count - 1]
            if (
                0 < split < len(text)
                and not text[split - 1].isspace()
                and text[split].isspace()
            ):
                # Verify the split against the original continuation: the
                # cached ids plus the re-encoded tail must rebuild the full
                # encoding exactly.
                rebuilt = list(ids[:count]) + self._wrap.encode(
                    text[split:], add_special_tokens=False
                )
                if rebuilt == list(ids):
                    self._insert(text[:split], ids[:count])
                    return
            count -= 1
            tries += 1

    def _insert(self, prefix_text: str, ids) -> None:
        key = prefix_text[: self._probe_chars]
        payload = array("i", ids)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._total_ids -= len(previous.ids)
            self._entries[key] = _Entry(prefix_text, payload)
            self._total_ids += len(payload)
            while self._entries and (
                self._total_ids > self._max_cached_ids
                or len(self._entries) > self._max_entries
            ):
                _, evicted = self._entries.popitem(last=False)
                self._total_ids -= len(evicted.ids)

    def _maybe_log_stats(self) -> None:
        if self._ops % self._stats_interval == 0:
            logger.info(
                "tokenizer prefix cache stats: %s",
                self.stats(),
            )
