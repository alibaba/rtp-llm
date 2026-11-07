"""Prefix-cached incremental tokenization for append-only prompt streams.

Agentic workloads resend the full growing session history on every step, so the
frontend re-tokenizes prompts whose 93-95% prefix is byte-identical to the
previous step's prompt. This module caches per-session tokenization state and
splices the cached ids with tail encodes cut at true pre-tokenizer chunk
boundaries, where BPE merges never cross (verified exact on the served
tokenizer family). Every failure path (cold start, session remap, prefix-hash
mismatch, eviction, internal error) falls back to the exact full encode, and an
optional sampled shadow verification reports mismatches and fails safe to the
full-encode result.

Disabled by default; enable with RTP_TOKENIZE_CACHE_ENABLE=1 (plus optional
RTP_TOKENIZE_CACHE_MAX_BYTES and RTP_TOKENIZE_CACHE_SHADOW_RATE).
"""

import array
import hashlib
import logging
import os
import random
import threading
import time
from typing import Dict, List, Optional

try:
    from rtp_llm.metrics import AccMetrics, GaugeMetrics
    from rtp_llm.metrics import kmonitor as _kmonitor
except ImportError:  # keep this module importable in minimal test sandboxes
    AccMetrics = None  # type: ignore[assignment]
    GaugeMetrics = None  # type: ignore[assignment]
    _kmonitor = None

_ENV_ENABLE = "RTP_TOKENIZE_CACHE_ENABLE"
_ENV_MAX_BYTES = "RTP_TOKENIZE_CACHE_MAX_BYTES"
_ENV_SHADOW_RATE = "RTP_TOKENIZE_CACHE_SHADOW_RATE"

_DEFAULT_MAX_BYTES = 128 * 1024 * 1024
# Sessions are bucketed by the hash of the first _KEY_CHARS characters (the
# stable session header). Prompts shorter than the window still work but miss
# on every append (exact full-encode fallback), which is fine: the mechanism
# targets the >100KB agentic prompts it was designed for.
_KEY_CHARS = 256
_COMPAT_PROBE = "rtp-llm tokenize cache compatibility probe 12345"
_TRUTHY = ("1", "true", "yes", "on")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


class _Entry:
    """Cached tokenization state for one session prefix."""

    __slots__ = ("cut", "ids", "prefix_sha", "nbytes", "stamp")

    def __init__(self, cut: int, ids: "array.array[int]", prefix_sha: str, stamp: int):
        self.cut = cut
        self.ids = ids
        self.prefix_sha = prefix_sha
        # Only the id array and hashes are retained (the prefix text itself is
        # never stored), so entries cost ~4 bytes per cached token.
        self.nbytes = len(ids) * 4 + 128
        self.stamp = stamp


class IncrementalTokenizeCache:
    """Splice cached prefix ids with boundary-aligned tail encodes.

    ``tokenizer`` must expose ``encode(text)`` and
    ``encode(text, add_special_tokens=False)`` (the BaseTokenizer contract), and
    its underlying tokenizers object must expose ``pre_tokenizer``. Instances
    are created through :meth:`from_env`, which returns ``None`` whenever the
    cache is disabled or the tokenizer fails the compatibility probes, so the
    caller can keep a plain ``if cache is not None`` gate.
    """

    def __init__(
        self,
        tokenizer,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        shadow_rate: float = 0.0,
    ):
        self._tokenizer = tokenizer
        self._max_bytes = max(1, int(max_bytes))
        self._shadow_rate = min(max(float(shadow_rate), 0.0), 1.0)
        self._entries: Dict[str, _Entry] = {}
        self._total_bytes = 0
        self._stamp = 0
        self._lock = threading.Lock()
        self._rng = random.Random()
        self._raw = self._resolve_raw(tokenizer)

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _resolve_raw(tokenizer):
        base = getattr(tokenizer, "tokenizer", tokenizer)
        for attr in ("backend_tokenizer", "_tokenizer"):
            raw = getattr(base, attr, None)
            if raw is not None and getattr(raw, "pre_tokenizer", None) is not None:
                return raw
        return None

    @classmethod
    def from_env(cls, tokenizer) -> Optional["IncrementalTokenizeCache"]:
        """Build the cache when enabled by env and compatible with tokenizer."""
        if os.environ.get(_ENV_ENABLE, "0").strip().lower() not in _TRUTHY:
            return None
        if tokenizer is None:
            return None
        try:
            max_bytes = int(os.environ.get(_ENV_MAX_BYTES, str(_DEFAULT_MAX_BYTES)))
            shadow_rate = float(os.environ.get(_ENV_SHADOW_RATE, "0"))
        except ValueError:
            logging.warning("tokenize cache: unparseable env value; staying disabled")
            return None
        cache = cls(tokenizer, max_bytes=max_bytes, shadow_rate=shadow_rate)
        if not cache.compatible():
            logging.warning(
                "tokenize cache: tokenizer lacks a usable pre_tokenizer or adds "
                "special tokens on default encode; staying disabled"
            )
            return None
        logging.info(
            "tokenize cache enabled: max_bytes=%d shadow_rate=%.3f",
            cache._max_bytes,
            cache._shadow_rate,
        )
        return cache

    def compatible(self) -> bool:
        """True when splice semantics are safe for this tokenizer.

        Requires a chunk-yielding pre_tokenizer and a default encode that adds
        no special tokens (so cached-prefix ids and tail ids concatenate
        without embedded specials).
        """
        if self._raw is None:
            return False
        pre = getattr(self._raw, "pre_tokenizer", None)
        if pre is None or not callable(getattr(pre, "pre_tokenize_str", None)):
            return False
        try:
            with_specials = list(self._tokenizer.encode(_COMPAT_PROBE))
            without = list(
                self._tokenizer.encode(_COMPAT_PROBE, add_special_tokens=False)
            )
        except TypeError:
            return False
        return with_specials == without

    # ------------------------------------------------------------------- api

    def get_token_ids(self, text: str) -> List[int]:
        """Return the exact ``tokenizer.encode(text)`` ids, incrementally.

        The result is always identical to a full encode: cache state is only
        ever consulted after byte-level prefix verification, and any anomaly
        falls back to (and returns) the full encode.
        """
        begin = time.monotonic()
        try:
            ids, key, shadow = self._lookup(text)
        except Exception as e:
            logging.warning(f"tokenize cache: full-encode fallback after {e}")
            try:
                self._report_acc("TOKENIZE_CACHE_MISS_QPS_METRIC")
            except Exception:
                pass
            ids, key, shadow = self._full_encode(text), None, False
        if shadow and key is not None:
            ids = self._shadow_verify(text, ids, key)
        try:
            self._report_gauge(
                "TOKENIZE_CACHE_RT_METRIC", (time.monotonic() - begin) * 1000.0
            )
        except Exception:
            pass
        return ids

    # --------------------------------------------------------------- internals

    def _full_encode(self, text: str) -> List[int]:
        return list(self._tokenizer.encode(text))

    def _encode_tail(self, text: str) -> List[int]:
        if not text:
            return []
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _conservative_cut(self, text: str) -> int:
        """Largest pre-tokenizer chunk end strictly below len(text), else 0."""
        if not text:
            return 0
        chunks = self._raw.pre_tokenizer.pre_tokenize_str(text)
        n = len(text)
        cut = 0
        for _, offsets in chunks:
            end = offsets[1]
            if end < n:
                cut = end
            else:
                break
        return cut

    def _lookup(self, text: str):
        """Return (ids, key, shadow_sampled) for ``text``."""
        key = _sha256_hex(text[:_KEY_CHARS])
        with self._lock:
            entry = self._entries.get(key)
            if (
                entry is not None
                and entry.cut <= len(text)
                and _sha256_hex(text[: entry.cut]) == entry.prefix_sha
            ):
                ids = self._splice_locked(text, entry)
                self._stamp += 1
                entry.stamp = self._stamp
                shadow = (
                    self._shadow_rate > 0 and self._rng.random() < self._shadow_rate
                )
                self._report_acc("TOKENIZE_CACHE_HIT_QPS_METRIC")
                return ids, key, shadow
        return self._miss(text, key), key, False

    def _splice_locked(self, text: str, entry: _Entry) -> List[int]:
        """Hit path: extend the cached prefix with boundary-aligned tails."""
        tail_start = entry.cut
        rel_cut = self._conservative_cut(text[tail_start:])
        new_cut = tail_start + rel_cut
        mid_ids = self._encode_tail(text[tail_start:new_cut])
        end_ids = self._encode_tail(text[new_cut:])
        ids = list(entry.ids)
        ids.extend(mid_ids)
        ids.extend(end_ids)
        if new_cut > entry.cut:
            entry.ids.extend(array.array("i", mid_ids))
            entry.cut = new_cut
            entry.prefix_sha = _sha256_hex(text[:new_cut])
            entry.nbytes = len(entry.ids) * 4 + 128
        return ids

    def _miss(self, text: str, key: str) -> List[int]:
        """Miss path: exact full encode, then cache a verified prefix state."""
        ids = self._full_encode(text)
        self._report_acc("TOKENIZE_CACHE_MISS_QPS_METRIC")
        cut = self._conservative_cut(text)
        if cut <= 0:
            return ids
        ids_cut = self._encode_tail(text[:cut])
        with self._lock:
            self._stamp += 1
            entry = _Entry(
                cut, array.array("i", ids_cut), _sha256_hex(text[:cut]), self._stamp
            )
            old = self._entries.pop(key, None)
            if old is not None:
                self._total_bytes -= old.nbytes
            self._entries[key] = entry
            self._total_bytes += entry.nbytes
            self._evict_locked()
        return ids

    def _evict_locked(self) -> None:
        while self._total_bytes > self._max_bytes and len(self._entries) > 1:
            key = min(self._entries, key=lambda k: self._entries[k].stamp)
            self._total_bytes -= self._entries.pop(key).nbytes

    def _shadow_verify(self, text: str, ids: List[int], key: str) -> List[int]:
        full = self._full_encode(text)
        if full == ids:
            return ids
        logging.error(
            "tokenize cache: shadow mismatch detected; returning the full-encode "
            "result and dropping the session entry"
        )
        self._report_acc("TOKENIZE_CACHE_MISMATCH_QPS_METRIC")
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._total_bytes -= entry.nbytes
        return full

    # ---------------------------------------------------------------- metrics

    @staticmethod
    def _report_acc(name: str) -> None:
        if _kmonitor is None or AccMetrics is None:
            return
        try:
            _kmonitor.report(getattr(AccMetrics, name), 1)
        except Exception:
            pass

    @staticmethod
    def _report_gauge(name: str, value: float) -> None:
        if _kmonitor is None or GaugeMetrics is None:
            return
        try:
            _kmonitor.report(getattr(GaugeMetrics, name), value)
        except Exception:
            pass
