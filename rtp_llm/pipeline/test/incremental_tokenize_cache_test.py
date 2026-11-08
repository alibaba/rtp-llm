"""Hermetic unit tests for IncrementalTokenizeCache.

The fake tokenizer is chunk-local by construction (ids(text) == concatenation
of per-chunk ids over the pre-tokenizer chunks), which is exactly the property
the production splice relies on and which was verified 100/100 on the real
served tokenizer at chunk-boundary cuts. These tests pin the cache machinery:
the accumulated production recurrence, fallbacks, eviction, shadow failsafe
and the env/compatibility gates.
"""

import array
import os
import re
import shutil
import socket
import struct
import tempfile
import threading
import time
import unittest

from rtp_llm.pipeline.incremental_tokenize_cache import (
    _KEY_CHARS,
    IncrementalTokenizeCache,
    _dir_quit,
    _Entry,
    _sha256_hex,
)

_CHUNK_RE = re.compile(r"\S+\s*")


class FakePreTokenizer:
    def pre_tokenize_str(self, text):
        return [(m.group(0), (m.start(0), m.end(0))) for m in _CHUNK_RE.finditer(text)]


class FakeRaw:
    def __init__(self):
        self.pre_tokenizer = FakePreTokenizer()


def _chunk_ids(chunk):
    return [1000000 + len(chunk)] + [ord(c) for c in chunk]


class FakeFast:
    def __init__(self, adds_special=False):
        self._adds_special = adds_special

    def encode(self, text, add_special_tokens=True):
        ids = []
        if add_special_tokens and self._adds_special:
            ids.append(1)
        for m in _CHUNK_RE.finditer(text):
            ids.extend(_chunk_ids(m.group(0)))
        return ids


class FakeBaseTokenizer:
    """Mirrors BaseTokenizer: .encode passthrough + .tokenizer wrapper."""

    def __init__(self, adds_special=False):
        self.tokenizer = FakeFast(adds_special=adds_special)
        self.tokenizer.backend_tokenizer = FakeRaw()

    def encode(self, prompt, **kwargs):
        return self.tokenizer.encode(prompt, **kwargs)


WORDS = [
    "alpha",
    "beta42",
    "1234567890",
    "tool_call",
    "hello world",
    "缓存",
    "トークン",
    '{"role": "assistant"}',
    "line\nbreak",
    "x",
]


def _append(rng, text):
    piece = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 6)))
    sep = rng.choice([" ", "\n", "  ", "\t"])
    return text + sep + piece


class _EnvGuard:
    def __init__(self, **values):
        self._values = values
        self._saved = {}

    def __enter__(self):
        for k, v in self._values.items():
            self._saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class IncrementalTokenizeCacheTest(unittest.TestCase):
    def setUp(self):
        import random

        self.rng = random.Random(7)
        self.tok = FakeBaseTokenizer()

    def _cache(self, **kw):
        return IncrementalTokenizeCache(self.tok, **kw)

    def test_from_env_disabled_by_default(self):
        with _EnvGuard(RTP_TOKENIZE_CACHE_ENABLE=None):
            self.assertIsNone(IncrementalTokenizeCache.from_env(self.tok))

    def test_from_env_enabled_truthy_values(self):
        # SHARE_DIR points at a scratch dir so from_env's best-effort share
        # socket never lands in the machine-default directory during tests.
        share = tempfile.mkdtemp(prefix="tokcache-fromenv-")
        try:
            for value in ("1", "true", "TRUE", "yes", "on"):
                with _EnvGuard(
                    RTP_TOKENIZE_CACHE_ENABLE=value,
                    RTP_TOKENIZE_CACHE_MAX_BYTES="1048576",
                    RTP_TOKENIZE_CACHE_SHADOW_RATE="0.5",
                    RTP_TOKENIZE_CACHE_SHARE_DIR=share,
                ):
                    cache = IncrementalTokenizeCache.from_env(self.tok)
                    self.assertIsNotNone(cache, f"enable={value} should build a cache")
                    if cache is not None:
                        cache.close_share()
        finally:
            _dir_quit(share)
            shutil.rmtree(share, ignore_errors=True)

    def test_from_env_bad_numeric_disables(self):
        with _EnvGuard(
            RTP_TOKENIZE_CACHE_ENABLE="1", RTP_TOKENIZE_CACHE_MAX_BYTES="abc"
        ):
            self.assertIsNone(IncrementalTokenizeCache.from_env(self.tok))

    def test_from_env_bad_share_timeout_disables(self):
        with _EnvGuard(
            RTP_TOKENIZE_CACHE_ENABLE="1", RTP_TOKENIZE_CACHE_SHARE_TIMEOUT_MS="abc"
        ):
            self.assertIsNone(IncrementalTokenizeCache.from_env(self.tok))

    def test_special_tokens_tokenizer_disabled(self):
        special_tok = FakeBaseTokenizer(adds_special=True)
        with _EnvGuard(RTP_TOKENIZE_CACHE_ENABLE="1"):
            self.assertIsNone(IncrementalTokenizeCache.from_env(special_tok))

    def test_missing_raw_tokenizer_disabled(self):
        class Opaque:
            def encode(self, prompt, **kwargs):
                return [ord(c) for c in prompt]

        with _EnvGuard(RTP_TOKENIZE_CACHE_ENABLE="1"):
            self.assertIsNone(IncrementalTokenizeCache.from_env(Opaque()))

    def test_append_sequence_exactness(self):
        cache = self._cache()
        text = "Session run/0/0 header seed words"
        for step in range(40):
            text = _append(self.rng, text)
            self.assertEqual(
                cache.get_token_ids(text),
                list(self.tok.encode(text)),
                f"step {step}: spliced ids must equal the full encode",
            )

    def test_multiple_sessions_interleaved(self):
        cache = self._cache()
        texts = [f"Session run/{i}/0 seed text number {i}" for i in range(3)]
        for step in range(25):
            for i in range(len(texts)):
                texts[i] = _append(self.rng, texts[i])
                self.assertEqual(
                    cache.get_token_ids(texts[i]), list(self.tok.encode(texts[i]))
                )

    def test_repeated_identical_text(self):
        cache = self._cache()
        text = "some session text with several words inside " + " ".join(WORDS) * 20
        first = cache.get_token_ids(text)
        for _ in range(3):
            self.assertEqual(cache.get_token_ids(text), first)
        self.assertEqual(first, list(self.tok.encode(text)))

    def test_cold_reset_changes_key_and_stays_exact(self):
        cache = self._cache()
        text = "Session run/0/0 " + " ".join(WORDS) * 5
        cache.get_token_ids(text)
        reset = "Session run/0/1 " + "completely different cold start body words"
        self.assertEqual(cache.get_token_ids(reset), list(self.tok.encode(reset)))

    def test_prefix_hash_mismatch_falls_back(self):
        cache = self._cache()
        base = "shared prefix header " + " ".join(WORDS) * 8
        cache.get_token_ids(base)
        # Same first 256-char key bucket is impossible to forge cheaply, so
        # corrupt the stored entry hash to simulate any verification failure.
        key = next(iter(cache._entries))
        cache._entries[key].prefix_sha = "0" * 64
        grown = base + " appended words after corruption"
        self.assertEqual(cache.get_token_ids(grown), list(self.tok.encode(grown)))

    def test_shorter_text_than_cut_falls_back(self):
        cache = self._cache()
        text = "long enough session text " + " ".join(WORDS) * 10
        cache.get_token_ids(text)
        short = text[:30]
        self.assertEqual(cache.get_token_ids(short), list(self.tok.encode(short)))

    def test_lru_eviction_respects_cap_and_exactness(self):
        cache = self._cache(max_bytes=4096)
        for i in range(40):
            text = f"session {i} header " + " ".join(WORDS) * 6 + f" tail {i}"
            grown = _append(self.rng, text)
            self.assertEqual(cache.get_token_ids(text), list(self.tok.encode(text)))
            self.assertEqual(cache.get_token_ids(grown), list(self.tok.encode(grown)))
        self.assertLessEqual(cache._total_bytes, 4096)

    # ---- byte-cap accounting regressions ---------------------------------
    # The cap only works if in-place splice growth is accounted: entries are
    # installed at seed size and then grow ~9x via _splice_locked under the
    # frozen agentic pattern. Unaccounted growth defeated eviction entirely
    # (worker RSS grew without bound); these tests pin the accounting.

    def _seed(self, tag):
        # >256 chars so the bucket key stays append-stable (see _KEY_CHARS),
        # but small enough that cap pressure comes from splice growth, not
        # from the (always correctly accounted) install path.
        return f"session {tag} base seed " + " ".join(WORDS) * 5

    def _grow(self, text, words):
        return text + " " + " ".join(self.rng.choice(WORDS) for _ in range(words))

    @staticmethod
    def _real_bytes(cache):
        return sum(e.nbytes for e in cache._entries.values())

    def test_splice_growth_respects_cap(self):
        # The cap must exceed the install-time footprint (~20 x 7KB here) so
        # that only accounted splice growth can breach it; a smaller cap would
        # degenerate into all-miss install churn, which even unaccounted
        # growth keeps symmetric.
        cap = 300_000
        cache = self._cache(max_bytes=cap)
        texts = [self._seed(i) for i in range(20)]
        for step in range(15):
            for i in range(len(texts)):
                texts[i] = self._grow(texts[i], 120)
                self.assertEqual(
                    cache.get_token_ids(texts[i]),
                    list(self.tok.encode(texts[i])),
                    f"step {step} session {i}: spliced ids must stay exact",
                )
            self.assertEqual(
                cache._total_bytes,
                self._real_bytes(cache),
                f"step {step}: reported total must equal the real entry bytes",
            )
            self.assertLessEqual(self._real_bytes(cache), cap * 1.05)
        self.assertGreaterEqual(cache._total_bytes, 0)

    def test_total_bytes_invariant_mixed_ops(self):
        cache = self._cache(max_bytes=16384, shadow_rate=1.0)

        def invariant(where):
            self.assertEqual(
                cache._total_bytes, self._real_bytes(cache), f"invariant: {where}"
            )
            self.assertGreaterEqual(cache._total_bytes, 0, f"invariant: {where}")

        seed0 = self._seed("mixed0")
        self.assertEqual(cache.get_token_ids(seed0), list(self.tok.encode(seed0)))
        invariant("after install")
        text = seed0
        for g in range(3):
            text = self._grow(text, 40)
            self.assertEqual(cache.get_token_ids(text), list(self.tok.encode(text)))
            invariant(f"after splice growth {g}")
        for i in (1, 2, 3):
            seed = self._seed(f"mixed{i}")
            self.assertEqual(cache.get_token_ids(seed), list(self.tok.encode(seed)))
            invariant(f"after install {i}")
            grown = self._grow(seed, 60)
            self.assertEqual(cache.get_token_ids(grown), list(self.tok.encode(grown)))
            invariant(f"after growth {i}")
        # Peer-style install: exactly what the fetch path hands _install_locked.
        dummy = self._seed("peer") + " peer fetched body"
        dcut = cache._conservative_cut(dummy)
        self.assertGreater(dcut, 0)
        fetched = _Entry(
            dcut,
            array.array("i", cache._encode_tail(dummy[:dcut])),
            _sha256_hex(dummy[:dcut]),
            0,
        )
        with cache._lock:
            cache._install_locked(_sha256_hex(dummy[:_KEY_CHARS]), fetched)
        invariant("after peer-style install")
        # The shadow failsafe pop must keep the accounting exact as well.
        reported = []
        original_report = cache._report_acc
        cache._report_acc = lambda name: reported.append(name)
        probe = self._seed("mismatch")
        cache.get_token_ids(probe)
        grown_probe = self._grow(probe, 40)
        original_encode = cache._encode_tail

        def corrupted(tail):
            ids = original_encode(tail)
            return ids + [999999] if tail else ids

        cache._encode_tail = corrupted
        self.assertEqual(
            cache.get_token_ids(grown_probe), list(self.tok.encode(grown_probe))
        )
        cache._encode_tail = original_encode
        cache._report_acc = original_report
        self.assertIn("TOKENIZE_CACHE_MISMATCH_QPS_METRIC", reported)
        invariant("after shadow-mismatch pop")
        with cache._lock:
            cache._evict_locked()
        invariant("after direct evict")

    def test_eviction_retains_freshest(self):
        # Phase 1 (black box): splice-driven cap pressure must evict, and
        # evicted sessions must still resolve exactly (fallback / re-fetch).
        cap = 60_000
        cache = self._cache(max_bytes=cap)
        texts = [self._seed(i) for i in range(6)]
        for _step in range(5):
            for i in range(len(texts)):
                texts[i] = self._grow(texts[i], 110)
                self.assertEqual(
                    cache.get_token_ids(texts[i]), list(self.tok.encode(texts[i]))
                )
        self.assertLessEqual(self._real_bytes(cache), cap * 1.05)
        self.assertEqual(cache._total_bytes, self._real_bytes(cache))
        self.assertLess(len(cache._entries), 6)
        for i in range(len(texts)):
            self.assertEqual(
                cache.get_token_ids(texts[i]), list(self.tok.encode(texts[i]))
            )
        # Phase 2 (white box): eviction pops strictly oldest-stamp entries
        # first, and accounted growth on the MRU entry never evicts itself.
        cache2 = self._cache(max_bytes=10_000)
        with cache2._lock:
            for i in range(4):
                ids = array.array("i", [i + 1] * 600)  # 600*4+128 = 2528 B
                cache2._install_locked(f"k{i}", _Entry(10, ids, "0" * 64, 0))
            # 4 x 2528 = 10112 > 10000: the oldest (k0) is already out.
            self.assertNotIn("k0", cache2._entries)
            self.assertIn("k1", cache2._entries)
            self.assertIn("k3", cache2._entries)
            self.assertEqual(cache2._total_bytes, self._real_bytes(cache2))
            # Grow the MRU entry the same way _splice_locked now does: the
            # next eviction victim must be k1 (oldest stamp), not k3.
            mru = cache2._entries["k3"]
            old_nbytes = mru.nbytes
            mru.ids.extend(array.array("i", [9] * 700))
            mru.nbytes = len(mru.ids) * 4 + 128
            cache2._total_bytes += mru.nbytes - old_nbytes
            cache2._evict_locked()
            self.assertNotIn("k1", cache2._entries)
            self.assertIn("k2", cache2._entries)
            self.assertIn("k3", cache2._entries)
            self.assertEqual(cache2._total_bytes, self._real_bytes(cache2))

    def test_many_session_growth_bounded(self):
        # Incident scenario in miniature: many long-lived sessions growing
        # step by step under the production default cap (128 MiB).
        cache = self._cache()
        sessions = 300
        steps = 40
        texts = [self._seed(f"soak{i}") for i in range(sessions)]
        for step in range(steps):
            for i in range(sessions):
                texts[i] = self._grow(texts[i], 400)
                cache.get_token_ids(texts[i])
            if step % 10 == 9:
                self.assertEqual(
                    cache._total_bytes,
                    self._real_bytes(cache),
                    f"step {step}: reported total must equal the real bytes",
                )
                self.assertLessEqual(self._real_bytes(cache), cache._max_bytes * 1.05)
        self.assertLessEqual(self._real_bytes(cache), cache._max_bytes)
        self.assertLessEqual(len(cache._entries), sessions)
        self.assertGreaterEqual(cache._total_bytes, 0)
        # Exactness spot-check after the eviction churn.
        for i in (0, sessions // 2, sessions - 1):
            self.assertEqual(
                cache.get_token_ids(texts[i]), list(self.tok.encode(texts[i]))
            )

    def test_shadow_mismatch_fails_safe(self):
        cache = self._cache(shadow_rate=1.0)
        reported = []
        cache._report_acc = lambda name: reported.append(name)
        # base text must exceed the 256-char key window so the bucket key is
        # append-stable (production prompts are >180KB; short texts safely miss)
        text = "session text for shadow verification " + " ".join(WORDS) * 25
        cache.get_token_ids(text)  # populate
        grown = text + " and appended words after the long base"
        original = cache._encode_tail

        def corrupted(tail):
            ids = original(tail)
            return ids + [999999] if tail else ids

        cache._encode_tail = corrupted
        result = cache.get_token_ids(grown)
        cache._encode_tail = original
        self.assertEqual(result, list(self.tok.encode(grown)))
        self.assertIn("TOKENIZE_CACHE_MISMATCH_QPS_METRIC", reported)

    def test_shadow_clean_path_no_mismatch(self):
        cache = self._cache(shadow_rate=1.0)
        reported = []
        cache._report_acc = lambda name: reported.append(name)
        text = "clean shadow session text " + " ".join(WORDS) * 25
        cache.get_token_ids(text)  # first call seeds the entry
        for _ in range(5):
            text = _append(self.rng, text)
            self.assertEqual(cache.get_token_ids(text), list(self.tok.encode(text)))
        self.assertNotIn("TOKENIZE_CACHE_MISMATCH_QPS_METRIC", reported)
        self.assertIn("TOKENIZE_CACHE_HIT_QPS_METRIC", reported)

    def test_edge_inputs(self):
        cache = self._cache()
        for text in ("", "x", "no-trailing-space", "trailing space ", "\n\n\n"):
            self.assertEqual(cache.get_token_ids(text), list(self.tok.encode(text)))

    def test_internal_error_falls_back(self):
        cache = self._cache()
        text = "session text that will hit an internal error path"
        cache.get_token_ids(text)

        def boom(_text):
            raise RuntimeError("simulated internal failure")

        cache._conservative_cut = boom
        grown = text + " appended"
        self.assertEqual(cache.get_token_ids(grown), list(self.tok.encode(grown)))

    def test_concurrent_callers_stay_exact(self):
        cache = self._cache()
        texts = ["thread session one seed words", "thread session two seed words"]
        errors = []

        def worker(idx):
            try:
                text = texts[idx]
                local = __import__("random").Random(100 + idx)
                for _ in range(20):
                    text = _append(local, text)
                    if cache.get_token_ids(text) != list(self.tok.encode(text)):
                        errors.append(idx)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in (0, 1)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])


class ShareLayerTest(unittest.TestCase):
    """Peer cache-entry sharing over real unix sockets (hermetic, tmpdir)."""

    def setUp(self):
        import random

        self.rng = random.Random(11)
        self.tok = FakeBaseTokenizer()
        self.dir = tempfile.mkdtemp(prefix="tokcache-share-test-")
        self.caches = []

    def tearDown(self):
        for cache in self.caches:
            try:
                cache.close_share()
            except Exception:
                pass
        try:
            _dir_quit(self.dir)  # retire this test's directory daemon
        except Exception:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)

    def _wait_dir(self, cache, key, timeout=5.0, retry_call=None):
        """Poll the storage daemon until ``key`` resolves to an entry payload.

        The daemon spawn (double fork + bind) and the first announce can race
        by a few ms; retry_call re-announces, exactly like the next production
        request would.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                payload = cache._dir_get(key, time.monotonic() + 0.5)
            except Exception:
                payload = None
            if payload is not None:
                return payload
            if retry_call is not None:
                retry_call()
            time.sleep(0.05)
        return None

    def _block_daemon(self):
        """Kill the daemon and block respawns so the SWEEP path is exercised."""
        _dir_quit(self.dir)
        for cache in self.caches:
            cache._dir_spawn_attempt = time.monotonic() + 3600.0
            cache._dir_drop()

    def _shared(self, **kw):
        cache = IncrementalTokenizeCache(self.tok, share_dir=self.dir, **kw)
        self.assertTrue(cache.start_share())
        self.caches.append(cache)
        return cache

    def _plain(self, **kw):
        cache = IncrementalTokenizeCache(self.tok, **kw)
        self.caches.append(cache)
        return cache

    def _base(self, seed):
        # must exceed the 256-char key window so bucket keys are append-stable
        return ("session " + seed + " header text beyond the key window ") + " ".join(
            WORDS
        ) * 25

    @staticmethod
    def _spy(cache):
        reported = []
        cache._report_acc = lambda name: reported.append(name)
        return reported

    def test_share_fetch_happy_path(self):
        owner = self._shared()
        fetcher = self._shared()
        reported = self._spy(fetcher)
        base = self._base("1")
        owner.get_token_ids(base)  # seeds the owner's entry
        grown = base + " appended assistant action words"
        result = fetcher.get_token_ids(grown)
        self.assertEqual(result, list(self.tok.encode(grown)))
        key = _sha256_hex(grown[:_KEY_CHARS])
        self.assertIn(key, fetcher._entries)  # fetched entry installed locally
        self.assertIn("TOKENIZE_CACHE_HIT_SHARED_QPS_METRIC", reported)
        self.assertIn("TOKENIZE_CACHE_HIT_QPS_METRIC", reported)
        self.assertNotIn("TOKENIZE_CACHE_MISS_QPS_METRIC", reported)

    def test_share_local_hit_preferred_over_fetch(self):
        owner = self._shared()
        fetcher = self._shared()
        base = self._base("2")
        owner.get_token_ids(base)
        fetcher.get_token_ids(base)  # installs locally (fetch or miss)
        reported = self._spy(fetcher)
        grown = base + " more words appended after the base"
        self.assertEqual(fetcher.get_token_ids(grown), list(self.tok.encode(grown)))
        self.assertIn("TOKENIZE_CACHE_HIT_LOCAL_QPS_METRIC", reported)
        self.assertNotIn("TOKENIZE_CACHE_HIT_SHARED_QPS_METRIC", reported)

    def test_share_stale_local_yields_to_fresher_peer(self):
        # THE E12 core: under random dispatch the local entry lags by the
        # revisit gap; a fresher peer cut must win the splice (not the stale
        # local entry, which would re-create the E10@N=16 pseudo-hit tax).
        owner = self._shared()
        fetcher = self._shared()
        base = self._base("15")
        fetcher.get_token_ids(base)  # fetcher keeps a STALE cut for the key
        text = base
        for i in range(3):
            text = text + " appended chunk %d " % i + " ".join(WORDS) * 8
            owner.get_token_ids(text)  # owner stays current with the session
        reported = self._spy(fetcher)
        grown = text + " final appended words " + " ".join(WORDS) * 4
        key = _sha256_hex(grown[:_KEY_CHARS])
        stale_cut = fetcher._entries[key].cut
        self.assertLess(stale_cut, int(len(grown) * 0.92))  # stale by design
        result = fetcher.get_token_ids(grown)
        self.assertEqual(result, list(self.tok.encode(grown)))
        self.assertIn("TOKENIZE_CACHE_HIT_SHARED_QPS_METRIC", reported)
        self.assertNotIn("TOKENIZE_CACHE_MISS_QPS_METRIC", reported)
        self.assertGreater(fetcher._entries[key].cut, len(text) - 64)

    def test_share_fresh_local_skips_peer_lookup(self):
        owner = self._shared()
        fetcher = self._shared()
        base = self._base("16")
        owner.get_token_ids(base)
        fetcher.get_token_ids(base)  # installed via shared fetch
        lookups = []
        original = fetcher._broadcast_lookups
        fetcher._broadcast_lookups = lambda *a: (lookups.append(a), original(*a))[1]
        grown = base + " tiny"  # local cut stays >= _FRESH_FRAC * len(grown)
        result = fetcher.get_token_ids(grown)
        fetcher._broadcast_lookups = original
        self.assertEqual(result, list(self.tok.encode(grown)))
        self.assertEqual(lookups, [])  # fresh local hit asked no peers

    def test_share_best_cut_wins(self):
        stale_peer = self._shared()
        fresh_peer = self._shared()
        fetcher = self._shared()
        base = self._base("17")
        stale_peer.get_token_ids(base)  # stale holder of the same key
        mid = base + " middle appended words " + " ".join(WORDS) * 10
        fresh_peer.get_token_ids(mid)  # fresher holder
        self._block_daemon()  # exercise sweep best-cut selection, not daemon GET
        grown = mid + " final appended words " + " ".join(WORDS) * 4
        served = []
        reported = self._spy(fetcher)
        original = fetcher._fetch_from_peer
        fetcher._fetch_from_peer = lambda path, k, t, d: (
            served.append(path),
            original(path, k, t, d),
        )[1]
        result = fetcher.get_token_ids(grown)
        fetcher._fetch_from_peer = original
        self.assertEqual(result, list(self.tok.encode(grown)))
        self.assertEqual(served[0], fresh_peer._own_path)  # best cut fetched first
        self.assertIn("TOKENIZE_CACHE_HIT_SHARED_QPS_METRIC", reported)

    def test_share_nack_falls_back_to_full_encode(self):
        owner = self._shared()  # reachable peer that does NOT hold the key
        fetcher = self._shared()
        reported = self._spy(fetcher)
        grown = self._base("3") + " appended words nobody has seen"
        self.assertEqual(fetcher.get_token_ids(grown), list(self.tok.encode(grown)))
        self.assertIsNotNone(owner._own_path)  # the NACK peer was live
        self.assertIn("TOKENIZE_CACHE_MISS_QPS_METRIC", reported)
        self.assertNotIn("TOKENIZE_CACHE_HIT_SHARED_QPS_METRIC", reported)

    def test_share_corrupt_entry_rejected(self):
        owner = self._shared()
        fetcher = self._shared()
        base = self._base("4")
        owner.get_token_ids(base)
        key = _sha256_hex(base[:_KEY_CHARS])
        owner._entries[key].prefix_sha = "0" * 64  # simulate corruption in flight
        self._block_daemon()  # force the sweep to fetch the CORRUPT worker copy
        grown = base + " appended words after corruption"
        self.assertEqual(fetcher.get_token_ids(grown), list(self.tok.encode(grown)))
        # the corrupt entry must NOT have been installed; the local miss path
        # re-seeded a correctly verified entry instead
        self.assertIn(key, fetcher._entries)
        installed = fetcher._entries[key]
        self.assertNotEqual(installed.prefix_sha, "0" * 64)
        self.assertEqual(installed.prefix_sha, _sha256_hex(grown[: installed.cut]))

    def test_share_timeout_falls_back(self):
        dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        dead.bind(os.path.join(self.dir, "worker_999999_deadbeef.sock"))
        dead.listen(1)  # accepts nothing, answers nothing
        try:
            fetcher = self._shared(share_timeout_s=0.05)
            reported = self._spy(fetcher)
            grown = self._base("5") + " appended words"
            begin = time.monotonic()
            self.assertEqual(fetcher.get_token_ids(grown), list(self.tok.encode(grown)))
            self.assertLess(time.monotonic() - begin, 2.0)  # bounded by deadline
            self.assertIn("TOKENIZE_CACHE_MISS_QPS_METRIC", reported)
        finally:
            dead.close()

    def test_share_stale_socket_skipped_untouched(self):
        stale_path = os.path.join(self.dir, "worker_888888_stale0000.sock")
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(stale_path)
        stale.close()  # socket file remains with nobody listening
        fetcher = self._shared()
        grown = self._base("6") + " appended words"
        self.assertEqual(fetcher.get_token_ids(grown), list(self.tok.encode(grown)))
        self.assertTrue(os.path.exists(stale_path))  # peer files never touched

    def test_share_concurrent_fetchers_stay_exact(self):
        owner = self._shared()
        bases = [self._base(str(i)) for i in range(3)]
        for base in bases:
            owner.get_token_ids(base)
        fetcher = self._shared()
        errors = []

        def worker(idx):
            try:
                local = __import__("random").Random(200 + idx)
                text = bases[idx]
                for _ in range(10):
                    text = _append(local, text)
                    if fetcher.get_token_ids(text) != list(self.tok.encode(text)):
                        errors.append(idx)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])

    def test_share_cap_eviction_with_installs(self):
        owner = self._shared()
        fetcher = self._shared(max_bytes=16384)
        for i in range(30):
            base = self._base("cap%d" % i)
            owner.get_token_ids(base)
            grown = base + " appended words for the cap test"
            self.assertEqual(fetcher.get_token_ids(grown), list(self.tok.encode(grown)))
        self.assertLessEqual(fetcher._total_bytes, 16384)

    def test_share_disabled_creates_no_sockets(self):
        cache = self._plain()  # share_dir None
        base = self._base("9")
        grown = base + " appended words"
        self.assertEqual(cache.get_token_ids(grown), list(self.tok.encode(grown)))
        self.assertEqual(os.listdir(self.dir), [])
        self.assertFalse(cache.start_share())  # disabled => no-op, still False
        self.assertEqual(os.listdir(self.dir), [])

    def test_from_env_empty_share_dir_disables_sharing(self):
        with _EnvGuard(RTP_TOKENIZE_CACHE_ENABLE="1", RTP_TOKENIZE_CACHE_SHARE_DIR=""):
            cache = IncrementalTokenizeCache.from_env(self.tok)
        self.assertIsNotNone(cache)
        self.caches.append(cache)
        self.assertIsNone(cache._share)
        base = self._base("10")
        grown = base + " appended words"
        self.assertEqual(cache.get_token_ids(grown), list(self.tok.encode(grown)))

    def test_from_env_starts_and_closes_share_server(self):
        with _EnvGuard(
            RTP_TOKENIZE_CACHE_ENABLE="1", RTP_TOKENIZE_CACHE_SHARE_DIR=self.dir
        ):
            cache = IncrementalTokenizeCache.from_env(self.tok)
        self.assertIsNotNone(cache)
        self.caches.append(cache)
        self.assertIsNotNone(cache._share)
        # only WORKER sockets count here; the directory daemon (if spawned)
        # owns directory.sock and deliberately outlives close_share()
        socks = [
            n
            for n in os.listdir(self.dir)
            if n.startswith("worker_") and n.endswith(".sock")
        ]
        self.assertEqual(len(socks), 1)
        cache.close_share()
        self.assertEqual(
            [
                n
                for n in os.listdir(self.dir)
                if n.startswith("worker_") and n.endswith(".sock")
            ],
            [],
        )

    def test_share_serve_request_roundtrip(self):
        owner = self._shared()
        base = self._base("12")
        owner.get_token_ids(base)
        key = _sha256_hex(base[:_KEY_CHARS])
        entry = owner._entries[key]
        hold = owner._serve_request(b"L", key)
        self.assertEqual(hold, b"H" + struct.pack(">I", entry.cut))
        frame = owner._serve_request(b"F", key)
        self.assertEqual(frame[:1], b"E")
        (cut,) = struct.unpack(">I", frame[1:5])
        sha = frame[5:69].decode("ascii")
        ids = array.array("i")
        ids.frombytes(frame[69:])
        self.assertEqual(cut, entry.cut)
        self.assertEqual(sha, entry.prefix_sha)
        self.assertEqual(list(ids), list(entry.ids))
        self.assertEqual(owner._serve_request(b"L", "f" * 64), b"N")
        self.assertEqual(owner._serve_request(b"F", "f" * 64), b"N")

    def test_share_malformed_request_does_not_kill_server(self):
        owner = self._shared()
        fetcher = self._shared()
        base = self._base("13")
        owner.get_token_ids(base)
        junk = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        junk.settimeout(2.0)
        junk.connect(owner._own_path)
        junk.sendall(struct.pack(">I", 10) + b"X" * 10)  # wrong length => drop
        junk.close()
        grown = base + " appended words after the garbage request"
        self.assertEqual(fetcher.get_token_ids(grown), list(self.tok.encode(grown)))

    def test_share_bidirectional(self):
        a = self._shared()
        b = self._shared()
        base_a = self._base("a")
        base_b = self._base("b")
        a.get_token_ids(base_a)
        b.get_token_ids(base_b)
        grown_a = base_a + " appended to session a"
        grown_b = base_b + " appended to session b"
        self.assertEqual(b.get_token_ids(grown_a), list(self.tok.encode(grown_a)))
        self.assertEqual(a.get_token_ids(grown_b), list(self.tok.encode(grown_b)))

    def test_share_soft_window_bounds_sweep(self):
        # A fast stale peer answers; a dead listener never does. With a
        # best-known hold in hand the soft window must end the sweep instead
        # of burning the full deadline (e40 run-2 tail diagnosis / F5).
        fast_owner = self._shared()
        base = self._base("18")
        fast_owner.get_token_ids(base)
        dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        dead.bind(os.path.join(self.dir, "worker_999998_slow00000.sock"))
        dead.listen(1)
        try:
            fetcher = self._shared(share_timeout_s=2.0, share_soft_window_s=0.02)
            self._block_daemon()  # exercise the sweep path, not the daemon GET
            grown = base + " appended words after the stale seed " + " ".join(WORDS) * 8
            begin = time.monotonic()
            result = fetcher.get_token_ids(grown)
            elapsed = time.monotonic() - begin
            self.assertEqual(result, list(self.tok.encode(grown)))
            self.assertLess(elapsed, 1.0)  # soft window, NOT the 2.0s deadline
        finally:
            dead.close()

    def test_suffix_cut_matches_full_scan(self):
        cache = self._plain()
        long_text = ("session suffix cut header beyond window words ") + " ".join(
            WORDS
        ) * 2000
        self.assertGreater(len(long_text), 16384)
        for size in (
            1,
            100,
            16383,
            16384,
            16385,
            20000,
            40000,
            len(long_text) - 1,
            len(long_text),
        ):
            text = long_text[:size]
            self.assertEqual(
                cache._conservative_cut(text),
                cache._full_scan_cut(text, len(text)),
                f"size {size}: suffix-bounded cut must equal the full scan",
            )

    def test_suffix_cut_inconclusive_falls_back(self):
        cache = self._plain()
        # One huge whitespace-free chunk: the trailing window is untrustworthy
        blob = "x" * 40000
        self.assertIsNone(cache._suffix_cut(blob, len(blob)))
        self.assertEqual(cache._conservative_cut(blob), 0)
        # end-to-end exactness on the pathological text is preserved
        self.assertEqual(cache.get_token_ids(blob), list(self.tok.encode(blob)))

    def test_dir_daemon_spawns_and_get_works(self):
        owner = self._shared()
        dpath = os.path.join(self.dir, "directory.sock")
        deadline = time.monotonic() + 5.0
        while not os.path.exists(dpath) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(os.path.exists(dpath))
        base = self._base("20")
        owner.get_token_ids(base)  # miss install announces the FULL entry
        key = _sha256_hex(base[:_KEY_CHARS])
        payload = self._wait_dir(
            owner, key, retry_call=lambda: owner.get_token_ids(base)
        )
        self.assertIsNotNone(payload)
        entry = IncrementalTokenizeCache._parse_entry(base, payload)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.cut, owner._entries[key].cut)
        self.assertEqual(list(entry.ids), list(owner._entries[key].ids))

    def test_dir_announce_max_cut_wins(self):
        a = self._shared()
        b = self._shared()
        base = self._base("21")
        a.get_token_ids(base)  # a announces cut_a
        key = _sha256_hex(base[:_KEY_CHARS])
        self.assertIsNotNone(
            self._wait_dir(a, key, retry_call=lambda: a.get_token_ids(base))
        )
        grown = base + " appended words for the max cut test " + " ".join(WORDS) * 8
        self.assertEqual(b.get_token_ids(grown), list(self.tok.encode(grown)))
        payload = self._wait_dir(a, key, retry_call=lambda: b.get_token_ids(grown))
        entry = IncrementalTokenizeCache._parse_entry(grown, payload)
        self.assertEqual(entry.cut, b._entries[key].cut)  # b's larger cut won
        b._dir_announce(key, 100, "0" * 64, b"")  # stale PUT never overwrites
        payload2 = self._wait_dir(a, key)
        entry2 = IncrementalTokenizeCache._parse_entry(grown, payload2)
        self.assertEqual(entry2.cut, b._entries[key].cut)

    def test_dir_daemon_death_falls_back_to_sweep(self):
        owner = self._shared()
        fetcher = self._shared()
        base = self._base("22")
        owner.get_token_ids(base)
        key = _sha256_hex(base[:_KEY_CHARS])
        self.assertIsNotNone(
            self._wait_dir(fetcher, key, retry_call=lambda: owner.get_token_ids(base))
        )
        self._block_daemon()  # daemon down + respawn blocked ⇒ sweep carries it
        grown = base + " appended words after the daemon died " + " ".join(WORDS) * 6
        result = fetcher.get_token_ids(grown)
        self.assertEqual(result, list(self.tok.encode(grown)))

    def test_dir_storage_survives_owner_eviction(self):
        # STORAGE daemon: the announced entry outlives the owner's local copy,
        # so an eviction on every worker still serves peers exactly (the
        # upgrade over the pointer-directory variant, whose stale announce
        # degraded into NACK + sweep + full-encode miss).
        owner = self._shared()
        fetcher = self._shared()
        base = self._base("23")
        owner.get_token_ids(base)
        key = _sha256_hex(base[:_KEY_CHARS])
        self.assertIsNotNone(
            self._wait_dir(fetcher, key, retry_call=lambda: owner.get_token_ids(base))
        )
        with owner._lock:  # owner loses the entry; daemon storage keeps it
            owner._entries.clear()
            owner._total_bytes = 0
        grown = base + " appended words after the eviction " + " ".join(WORDS) * 6
        reported = self._spy(fetcher)
        self.assertEqual(fetcher.get_token_ids(grown), list(self.tok.encode(grown)))
        self.assertIn("TOKENIZE_CACHE_HIT_SHARED_QPS_METRIC", reported)
        self.assertNotIn("TOKENIZE_CACHE_MISS_QPS_METRIC", reported)

    def test_dir_three_instances_share_via_daemon(self):
        a = self._shared()
        b = self._shared()
        c = self._shared()
        base = self._base("25")
        a.get_token_ids(base)
        t1 = base + " first append " + " ".join(WORDS) * 6
        self.assertEqual(b.get_token_ids(t1), list(self.tok.encode(t1)))
        t2 = t1 + " second append " + " ".join(WORDS) * 6
        self.assertEqual(c.get_token_ids(t2), list(self.tok.encode(t2)))
        t3 = t2 + " third append " + " ".join(WORDS) * 6
        self.assertEqual(a.get_token_ids(t3), list(self.tok.encode(t3)))
        key = _sha256_hex(base[:_KEY_CHARS])
        payload = self._wait_dir(c, key, retry_call=lambda: a.get_token_ids(t3))
        self.assertIsNotNone(payload)
        entry = IncrementalTokenizeCache._parse_entry(t3, payload)
        self.assertEqual(entry.cut, a._entries[key].cut)  # a's t3 splice won


if __name__ == "__main__":
    unittest.main()
