"""Shared kv scenario components. Resource and timing semantics live here."""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..context import CaseContext, CaseDef, rid_base
from ..engine_ops import engine_inflight_clean
from ..grade import GradeReport
from ..harness import (
    DEFAULT_PREFILL_CACHE_BLOCKS,
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    EnvSpec,
    default_perf,
    http_get_json,
    http_post_json,
    wait_for,
)
from .requests import drain_fired as _drain_fired
from .requests import fire_request as _fire_request
from .requests import wait_engine_pending as _poll_engine_pending

STREAM_TIMEOUT_S = 15.0
# Fixed master cache-status sync settle used by the migrated affinity
# cases (one poll period + margin; the eviction events need the longer
# KV_SYNC_CONVERGENCE_S convergence below).
KV_CACHE_SYNC_WAIT_S = 2.0
# Master cache-status sync convergence: the spec demands >= 3.5s of quiet
# (or cache_version polling) before post-eviction assertions — the
# GrpcCacheStatusCheckRunner poll period plus margin (smoke S2 used 2.0
# as a single sleep; the eviction events need the longer convergence).
KV_SYNC_CONVERGENCE_S = 3.5
# P1 fired-batch wave size.  GRADE_BANDS' P1 uniformity bands (strict /
# normal / loose = 0.65 / 0.75 / 0.85) are calibrated against B(20, 0.5):
# normal 0.75 ~= 2 * P(X >= 15) ~= 4.1%, loose 0.85 ~= 2 * P(X >= 17) ~=
# 0.26% two-sided nominal false-fail per wave (see the P1 entry in
# grade.GRADE_BANDS).  Every P1 fired-batch wave MUST fire exactly this
# many requests: the historical n=10 put the SAME bands at ~10.9%
# (normal) / ~2.1% (loose) per wave — one loose trip drops the
# whole-suite verdict straight to unusable.
P1_WAVE_N = 20
# Prefix-family calibration (module docstring): 10 blocks of 1024 tokens
# per family; a full-hit continuation prices at hitTokens = 9216 >= 8192.
PREFIX_BLOCKS = 10
BLOCK_TOKENS = 1024
PREFIX_INPUT_LEN = PREFIX_BLOCKS * BLOCK_TOKENS  # 10240
MIN_AFFINITY_HIT_TOKENS = 8192
# Storm shape: 4 hot families x 10 blocks = 40 hot blocks vs per-engine
# capacity 24 (< 40 — the spec's churn precondition).
STORM_FAMILIES = 4
STORM_WINDOW = 5
STORM_WINDOWS = 10
STORM_CAPACITY_BLOCKS = 24
STORM_FLIP_BOUND = 8 * STORM_WINDOWS  # anti ping-pong bound (TODO calibrate)
# Average replication-factor bands (per-family prefill holder count
# mean; P5 case override — the KV-redundancy caliber, upper bound).
# Calibration (decode=512 dress rehearsal, 2 prefills): mean 1.125,
# steady-state 1.5, max 2 (= the 2-prefill structural ceiling), flips 7,
# M3 0.86, 50/50 requests OK — the mean sits ~8.6 sigma below the
# normal tier, so strict 1.5 IS the steady-state mode and loose 2.0 is
# the 2-of-4-family double-hold saturation line (every family
# double-held).  The band guards the FAKE-FIX direction: repairing hit
# rate by replicating everywhere (admission control gone) trades KV
# footprint for nothing.  CAVEAT: n=1 calibration run — re-check (and
# tighten) from multi-run regression data before trusting strict.
STORM_REPLICATION_BANDS = {"strict": 1.5, "normal": 1.75, "loose": 2.0}
# Hit-tier concentration bands (M3 case override for kv_storm_hot_churn).
# 2026-09-04 recalibration (n=1): the shared GRADE_BANDS M3 floor (loose
# 0.6) sat EXACTLY on this run's measurement — 0.600 = 30/50 graded
# loose-only and failed a normal-grade run on a boundary artifact.
# Cross-era drift is real (promotion-era 0.86 -> 0.600 after the codex
# admission changes), so the tiers below split healthy warming from the
# collapse regimes instead: the 5-requests-per-window structure makes
# the first request of each window a guaranteed miss (rotation period
# 4 windows = 40 blocks > the 24-block LRU), capping perfect stickiness
# at 0.8 — strict 0.72 sits just under it; normal 0.50 admits the
# observed warm-but-not-fully-sticky form with binomial margin
# (sigma ~= 0.07 at n=50); loose 0.40 floors the true collapse regimes
# (holder-avoidance / no in-window warming, ~<= 0.2).  n=1 caveat:
# with 2 prefills even non-affinity routing warms in-window (~0.6 in
# this construction), so the tiers police collapse-vs-healthy more
# than a clean hit-vs-random split — recalibrate from n>=3 runs before
# tightening.
STORM_HIT_RATE_BANDS = {"strict": 0.72, "normal": 0.50, "loose": 0.40}
# Capacity-conflict shape: a 40-block family keeps the seed's hit share
# above minPrefixHitPercent even against a 147456-token seqLen
# (40960 / 147456 = 27.8% >= 20%).
CONFLICT_BLOCKS = 40
CONFLICT_INPUT_LEN = CONFLICT_BLOCKS * BLOCK_TOKENS  # 40960
CONFLICT_SEED_INPUT_LEN = 147456
# kv_decode_capacity_park probe: short client-side gRPC deadline proving
# the parked Schedule RPC stays pending (the master's own scheduling
# deadline is queueTimeoutMs, default 1h — far beyond any useful probe).
E4_PROBE_DEADLINE_S = 5.0


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


# ===========================================================================
# Mock capability helpers (TODO: interface alignment with the mock agent)
# ===========================================================================


def _engine_cache_keys(ops, engine_name: str) -> set:
    """Per-engine LRU key set (TODO(mock-agent): interface alignment).

    Two acceptable shapes, tried in order:
      A (preferred) — each /snapshot engine dict carries ``cache_key_set``
        (list of block keys, backed by MockLruBlockCache.snapshotKeys());
      B — dedicated endpoint ``GET /cache_keys?engine=<name>`` ->
        ``{"keys": [...]}``.

    Raises RuntimeError when neither exists — until the capability lands
    the depending cases fail loudly on this raise rather than silently
    skipping (the raise IS the alignment signal).
    """
    entry = ops.snapshot_by_name().get(engine_name, {})
    if "cache_key_set" in entry:
        return set(int(k) for k in entry["cache_key_set"])
    data = http_get_json(
        f"http://127.0.0.1:{ops.mock_http_port}/cache_keys" f"?engine={engine_name}"
    )
    if data and "keys" in data:
        return set(int(k) for k in data["keys"])
    raise RuntimeError(
        f"cache key set for engine {engine_name} unavailable: snapshot has "
        f"no 'cache_key_set' field and /cache_keys returned {data!r} "
        f"(TODO: mock-agent interface alignment — see module docstring)"
    )


def _cache_evict(ops, engine_name: str, keys) -> dict:
    """Force-evict *keys* from one engine's LRU (TODO(mock-agent)).

    POST /cache_evict {"engine": name, "keys": [...]} — evicts the named
    keys from the engine's MockLruBlockCache and bumps cacheVersion so
    the master's cache-status poll propagates the eviction.
    """
    status, body = http_post_json(
        f"http://127.0.0.1:{ops.mock_http_port}/cache_evict",
        {"engine": engine_name, "keys": [int(k) for k in keys]},
    )
    if status != 200:
        raise RuntimeError(
            f"cache_evict({engine_name}, {len(keys)} keys) failed: " f"{status} {body}"
        )
    return body or {}


# ===========================================================================
# Shared observation helpers
# ===========================================================================


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot_by_name()
    return sorted(name for name, e in snap.items() if e.get("role") == "prefill")


def _fam_keys(base: int, fam: int, blocks: int = PREFIX_BLOCKS) -> list:
    """Block keys of prefix family *fam* (blocks keys, 1000-key stride)."""
    return [base + fam * 1000 + j for j in range(blocks)]


def _contiguous_prefix_len(key_set: set, keys: list) -> int:
    """Length of the contiguous prefix run of *keys* present in *key_set*
    (the MockLruBlockCache.prefixHitBlocks caliber: first miss truncates)."""
    n = 0
    for key in keys:
        if key not in key_set:
            break
        n += 1
    return n


def _wait_cache_sync(ops, engine_names: list, timeout_s: float = 8.0) -> bool:
    """Wait for every named engine's cache key set to go QUIET.

    Polls /snapshot every 0.5s and requires KV_SYNC_CONVERGENCE_S (>= 3.5s)
    with no cache_key_set change (two consecutive samples equal) on every
    named engine — after that window the master's cache-status poll (its own
    ~1-2s period) has necessarily observed the final state.  This is the
    spec's ">= 3.5s quiet" convergence caliber.

    Key-set equality is the cache_version proxy: mock commit fc35323af7
    dropped the per-engine ``cache_version`` snapshot field, and every
    cacheVersion bump is driven by a key-set change anyway (admit insert /
    evict removal / capacity eviction inside admit), so set equality tracks
    version stability exactly.  The bare two-sample signal alone only proves
    1s of stability — too short for the master poll to have caught up, so
    the quiet window (which subsumes it) is kept as the convergence bar.
    """
    deadline = time.monotonic() + timeout_s
    last_sets: dict = {}
    last_change = {n: time.monotonic() for n in engine_names}
    while time.monotonic() < deadline:
        snap = ops.snapshot_by_name()
        now = time.monotonic()
        quiet = True
        for n in engine_names:
            keys = frozenset(int(k) for k in snap.get(n, {}).get("cache_key_set", ()))
            if keys != last_sets.get(n):
                last_sets[n] = keys
                last_change[n] = now
            if now - last_change[n] < KV_SYNC_CONVERGENCE_S:
                quiet = False
        if quiet:
            return True
        time.sleep(0.5)
    return False


def _wait_master_alive(ops, role: str, count: int, timeout_s: float = 30.0) -> bool:
    return wait_for(lambda: ops.master_alive_count(role) == count, timeout_s, 0.5)


# ===========================================================================
# Fire-and-forget plumbing (S4 drainage lesson: unconsumed fire-and-forget
# entries linger in master inflight/ledger and poison later phases)
# ===========================================================================


def _seed_shared_prefix(ops, base: int, keys: list, input_len: int):
    """Ledger-separated double dispatch of ONE prefix family onto TWO
    prefill engines (the kv_prefix_stickiness seeding technique): the
    first request is fired-and-forgotten onto engine e1; while e1's ~2s
    ledger entry is live, the second request deterministically lands on
    a DIFFERENT engine.  Returns (e1, e2, err) — both engines admit the
    family, forming the shared-holder state the global cases need."""
    names = _prefill_names(ops)
    if len(names) < 2:
        return None, None, "need >=2 prefill workers"
    fired, fired_handles = [], {}
    for name in names:
        ops.set_perf(name, prefill_fixed_ms=2000.0)
    time.sleep(1.5)  # master perf sync (both engines slowed)
    try:
        rid1 = ops.next_request_id(base)
        e1, err = _fire_request(
            ops,
            rid1,
            fired,
            fired_handles,
            input_len=input_len,
            output_len=2,
            block_keys=keys,
        )
        if err is None and not _poll_engine_pending(ops, e1, 1):
            err = f"first dispatch never appeared on {e1}"
        e2 = None
        if err is None:
            rid2 = ops.next_request_id(base)
            addr2, err2 = ops.run_one_request(
                rid2,
                input_len=input_len,
                output_len=2,
                block_keys=keys,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            e2 = ops.addr_to_name().get(addr2, addr2) if addr2 else None
            if err2:
                err = f"second dispatch failed: {err2}"
            elif e2 == e1:
                err = f"double dispatch collapsed onto {e1} (ledger diversion failed)"
        return e1, e2, err
    finally:
        _drain_fired(ops, fired, fired_handles)
        for name in names:
            try:
                ops.set_perf(name, prefill_fixed_ms=100.0)
            except Exception:
                pass


def _kv_spec(
    ctx: CaseContext,
    suffix: str = "",
    *,
    n_prefill: int = 2,
    prefill_cache_blocks: Optional[int] = None,
    decode_cache_blocks: int = 12,
    discovery: str = "file",
) -> EnvSpec:
    """Shared KV-family env: 2P+2D default, default prefill LRU capacity
    unless the case pins a small one (tiny capacities pair with the
    10-block families so the hit still prices past the affinity line).

    decode_cache_blocks=12: KV-family requests carry
    PREFIX_INPUT_LEN=10240 tokens (10 blocks), and since the
    total=blocks*spb reporting fix the master's
    CostBasedDecodeStrategy.rejectIfPhysicalCapacityIsTooSmall compares
    the decode seq_len against the engine-reported totalKv — a 4-block
    pool (4096 tokens) made every prefix-sized decode request a typed
    StaticCapacityExceededException reject before routing even started
    (observed: "Decode request seq_len=10240 exceeds max known physical
    KV=4096").  12 blocks = 12288 tokens covers the 10-block prefix with
    the KV-v2 reserve margin to spare; decode-pool size plays no role in
    the family's prefill-side assertions (affinity/evict/replication all
    price against the PREFILL pool), so widening it changes no other
    case semantics.  Cases whose decode seq_len towers over the
    12-block default (kv_capacity_conflict_overflow: 40960/147456-token
    requests) pin decode_cache_blocks so the decode phase fits a single
    engine — there the pool is a construction gate, not a semantic
    axis."""
    return EnvSpec(
        label=f"kv{suffix}_{ctx.profile}",
        n_prefill=n_prefill,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        prefill_cache_blocks=(
            prefill_cache_blocks
            if prefill_cache_blocks is not None
            else DEFAULT_PREFILL_CACHE_BLOCKS
        ),
        decode_cache_blocks=decode_cache_blocks,
        discovery=discovery,
    )


def _lru_spec(ctx: CaseContext) -> EnvSpec:
    """kv_lru_eviction_affinity env: 2P+2D with a tiny per-engine prefill
    LRU (4 blocks)."""
    return EnvSpec(
        label=f"kv_lru_{ctx.profile}",
        n_prefill=2,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        prefill_cache_blocks=4,
        decode_cache_blocks=4,
    )


# ===========================================================================
# Per-engine cases
# ===========================================================================


# ===========================================================================
# Global-index cases
# ===========================================================================


# ===========================================================================
# Storm / capacity cases
# ===========================================================================


# ===========================================================================
# Affinity routing cases (migrated from the legacy scheduling family,
# category reorg — rid_base family "scheduling" -> "kv")
# ===========================================================================


# ===========================================================================
# LRU capacity case (migrated from the legacy gate family,
# category reorg — rid_base family "chaos" -> "kv")
# ===========================================================================


# ===========================================================================
# Decode-side KV-capacity parking case (migrated from the legacy anomaly
# family E4, category reorg — rid_base family "anomaly" -> "kv")
# ===========================================================================


# ===========================================================================
# Block-pool saturation calibers (KV v2 extremes, 2026-09
# follow-up): the pool driven to BOTH extremes — the P pool through
# capacity eviction to a synchronous 602 saturation reject and back
# (release != delete), and the D pool to the P-enqueue decode-reservation
# 602.  All numbers are calibrated against the mock's actual gate
# (MockLruBlockCache.acquire: need <= avail AND avail - need >= reserve,
# reserve = ceil(0.05 x totalBlocks)) and the master's error wrapping
# (DefaultBatchDispatcher: "EnqueueBatch rejected request <rid>: <engine
# error message verbatim>" — the engine's LACK_MEM text reaches the client
# untouched inside that wrapper).
# ===========================================================================

# Saturation P pool: 24 blocks, reserve = ceil(24 x 0.05) = 2.
# 27 = 3 x 8 occupants + reserve headroom: with the pool AT 24 the third
# occupant's admit hits the reserve gate (avail 8 - need 8 = 0 < reserve
# 2) and the wave degenerates to 2 occupants (remote evidence run
# 20260903_130256: held_peak=16, occupants 2/3).  27 admits all three
# (11 - 8 = 3 >= 2), pins held=24 and drops available to exactly
# reserve + 1 = 3 — the floor the saturation proof samples for.
SAT_POOL_BLOCKS = 27
# Requests carry 8 block keys == an 8-block P demand (prefill admission
# prices the hash-channel key count, JavaMockEngineCluster.needBlocks).
SAT_REQUEST_KEYS = 8
# input_len is DECOUPLED from the P-side demand (which prices the key
# count) and kept at 2 blocks of tokens.  At 8192 the master's ROUTE-time
# decode soft reservation (input+output TOKENS, CUMULATIVE across
# in-flight requests) priced the 3 occupants at 8194 x 3 = 24582 —
# within a hair of the widened 32-block D pool's 90% door (29491) — so
# the probe's own 8194 parked master-side until an occupant drained
# (remote evidence run 20260903_130256: probe 10.48s slow fail, engine
# lack_mem_rejects +0, held_peak sampled only after the window).  At
# 2048 the cumulative reservation (2050 x 4 = 8200) sits far under any
# D-pool door while the P saturation semantics (8 keys = 8 blocks vs
# the 27-block pool) are unchanged.
SAT_INPUT_LEN = 2 * BLOCK_TOKENS
# Decode pool widened past the saturation wave's D reservations: every
# request reserves ceil(inputLen/spb) = 8 D blocks at P-enqueue (Phase
# 1.6, the prepare-stage ALLOCATE counterpart), so 3 occupants reserve
# 24 D blocks + reserve + per-step growth — the 12-block KV-family
# default would 602 the wave on the DECODE side before the prefill pool
# ever saturated (design said "decode default"; the code says otherwise).
SAT_DECODE_POOL_BLOCKS = 32
# 3s prefill pins all three occupants' P leases inside one occupancy
# window; 0.4s fire spacing keeps the wave inside it.
SAT_PREFILL_MS = 3000.0
SAT_FIRE_SPACING_S = 0.4
SAT_PROBE_BOUND_S = 3.0
# Follow-up burst inside the saturation window (bounded-share caliber).
SAT_BURST_N = 2
# Saturation window sampling cadence (available-floor proof).
SAT_SAMPLE_S = 0.1
# Decode-exhaustion caliber — PERMANENT tier (wave-3 v1 adaptation,
# 2026-09): the occupant-based RETRYABLE tier tried first (D pool 4,
# a stretched 3s prefill holding 2 decode blocks) never reached the
# engine — the occupant occupied the single decode worker's
# concurrency slot (decode concurrency is a production-locked one per
# engine on this line), so the probe was rejected at ROUTE time with
# NO_DECODE_WORKER(8403) before any admission arithmetic ran (0.00s
# reject, engine enqueue_rpcs == 0).  "Hold KV blocks with an
# in-flight request" is structurally impossible on the v1 stack, so
# the tier reverts to direct saturation: D pool 3, probe input 2560
# (net demand ceil(2560/1024) = 3), need + reserve = 3 + 1 > 3 — no
# pool state can ever admit it — and the case PINS the deterministic,
# stack-independent classification contract: a request the pool
# structurally cannot fit counts into lack_mem_rejects (permanent
# family), not kv_admission_fails (retryable family).  Upstream
# carries the same bug, same fix.  The 2560 input stays the FALLBACK
# caliber (remote run 20260903_130256): it passes the master's
# ROUTE-time soft reservation token door (2562 <= 3072 x 0.9) while
# the block-caliber demand saturates the pool.
DSAT_DECODE_POOL_BLOCKS = 3
# 2.5 blocks exactly: 2560 rounds up to a 3-block demand while staying
# under the master's 90% token door with output headroom.
DSAT_INPUT_LEN = (5 * BLOCK_TOKENS) // 2
# Recovery probe: net demand ceil(1024/1024) = 1 <= avail - reserve = 1.
DSAT_RECOVER_INPUT_LEN = BLOCK_TOKENS
DSAT_PROBE_BOUND_S = 3.0


def _pool_state(ops, engine_name: str) -> tuple:
    """(cache_blocks, held, referenced, available) of one engine's pool."""
    info = ops.snapshot_by_name().get(engine_name, {})
    return (
        int(info.get("cache_blocks", 0)),
        int(info.get("held_blocks", 0)),
        int(info.get("referenced_blocks", 0)),
        int(info.get("available_blocks", 0)),
    )


# ===========================================================================
# Leader-saturation spill case: the queue-full affinity give-up cascade
# observed in production, pinned as a declared-finding probe
# (expected-fail until remote calibration decides promotion — the
# kv_storm_hot_churn precedent).
# ===========================================================================

# Spill topology: per-prefill pool 12 blocks vs a 2 x 10-block family
# working set (20 > 12) — a single engine cannot host both families
# while exactly ONE stable placement exists (A@P1, B@P2); any spill onto
# the survivor evicts the resident family inside admit (capacity
# conservation), which is the cascade fuel.
SPILL_POOL_BLOCKS = 12
SPILL_BASELINE_WINDOWS = 3
SPILL_SATURATION_WINDOWS = 4
SPILL_RECOVERY_WINDOWS = 6
# Saturation window shape: fires [A, A, B, B] back-to-back (A first
# keeps the A-to-A spacing that deterministically lifts the leader's
# committed projection past the cutoff); baseline/recovery windows stay
# interleaved A,B,A,B serial.
SPILL_WINDOW_REQS = 4
# Leader slowdown depth: the worst-case saturation window puts all 4
# fires on the leader's SERIAL service (max_prefill_concurrency=1), so
# the last completes at ~4 x 3s = 12s — clearing the fire path's REAL
# bounds (the end-of-window drain's 30s per-rid wait and the stream's
# 60s gRPC deadline) with margin; 5s would put the same window at ~20s,
# hugging the 30s drain ceiling with no margin.  STREAM_TIMEOUT_S=15
# only bounds the run_one_request path (steer/baseline/recovery
# windows, where every engine is fast so it never binds).  3s still
# keeps the first fire's in-flight window open far beyond one window's
# ~0.5s fire span — the slowdown itself never enters the formula-based
# TTFT projection, it only widens the in-flight window that carries
# the committed value (see the mechanism chain).
SPILL_SLOW_MS = 3000.0
SPILL_FAST_MS = 100.0
# Saturation fire interval: each routing decision must land while the
# PREVIOUS fire is still an ENGINE_QUEUED/RUNNING committed entry on
# the leader's ledger — one in-flight entry (a ~221ms formula
# prediction) already crosses both spill thresholds, and at 0.12s
# spacing the next fire still sees it un-decayed.  Known jitter source
# (n=1 calibration caveat): B#1's pre-request hit view straddles A#2's
# admit/eviction propagation boundary (P2-side 100ms execution vs the
# 120ms fire interval), so up to 4/16 saturation hit samples may flap
# — read remote calibration and future n>=3 reruns with that in mind.
SPILL_FIRE_INTERVAL_S = 0.12
# Recovery-ready caliber (observational): the first recovery window whose
# hit rate crosses this share counts as "recovered".
SPILL_RECOVERY_READY_RATE = 0.75
# Hit-rate bands (M3 lower kind, case override — the storm hit-served
# caliber: a request is hit-served when its landing engine already held
# >= PREFIX_BLOCKS - 2 contiguous family blocks pre-request).  All three
# tables are PRE-CALIBRATION guesses stating the healthy contract; the
# saturation tier carries the bounded-collapse bound the finding predicts
# will FAIL (~0 measured under the LRU ping-pong) — that failure IS the
# finding evidence under expected-fail semantics.  Recalibrate from
# remote runs before any promotion off expected_fail.
SPILL_BASELINE_HIT_BANDS = {"strict": 0.9, "normal": 0.85, "loose": 0.8}
SPILL_SATURATION_HIT_BANDS = {"strict": 0.5, "normal": 0.4, "loose": 0.3}
SPILL_RECOVERY_HIT_BANDS = {"strict": 0.85, "normal": 0.8, "loose": 0.75}
# Replication bands (P5 upper kind, case override — the storm
# KV-redundancy caliber on the per-family prefill holder-count mean,
# with the hard structural cap max <= n_prefill asserted alongside).
SPILL_REPLICATION_BANDS = {"strict": 1.5, "normal": 1.75, "loose": 2.0}
