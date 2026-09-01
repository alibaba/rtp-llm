"""KV-category cases: the prefix-cache lifecycle contract.

Theme: the KV prefix-cache lifecycle as the master sees it — per-engine
ledger isolation, eviction events, shared (global-index) blocks, sync
convergence, engine-down cleanup, hot-prefix churn storms, the
affinity-vs-capacity conflict, and the decode-side KV-capacity parking
path.  Grouped per the researcher spec (task #84) plus the task #85
category-reorg migrations:

  per-engine (the mock LRU is per engine; the master's cache-status
              index must stay per-engine too)
    kv_pe_admit_isolation        A's admits never widen B's key set
    kv_pe_evict_zero_match       forced evict syncs; no stale stickiness
    kv_pe_prefix_continuity      continuous-prefix matching (gap truncates)

  global (the master-side index maps key -> holder SET)
    kv_g_shared_block_both_match   both holders -> NO_CACHE_LEAD spread
    kv_g_partial_release_redirect  one holder evicts -> redirect to other
    kv_g_full_release_no_ghost     all holders evict -> zero-hit spread
    kv_g_sync_convergence          mixed admit/evict, quiet > 3.5s, routing
                                   matches the engine snapshots
    kv_g_engine_down_cleanup       engine removal keeps shared blocks for
                                   the survivor

  storm / capacity
    kv_storm_hot_churn             rotating hot prefixes vs a small LRU
                                   (FINDING case — see its docstring)
    kv_capacity_conflict_overflow  affinity yields to a full ledger
    kv_decode_capacity_park        every decode KV-exhausted -> the
                                   request parks; Cancel releases it
                                   (migrated from the legacy anomaly
                                   family, E4)

  affinity routing (migrated from the legacy scheduling family)
    kv_prefix_stickiness           prefix-reuse traffic sticks to the
                                   holder (P9 + P2 + P6)
    kv_hot_prefix_tension          70% hot family: stickiness holds AND
                                   holder concentration capped (P9 + M2)
    kv_match_mixed                 full/half-hit concentrate, zero-hit
                                   spreads (M3 + P2)

  LRU capacity (migrated from the legacy gate family)
    kv_lru_eviction_affinity       LRU prefix reuse + capacity eviction +
                                   affinity routing end-to-end

MOCK CAPABILITY NOTE (task #85 reorg wiring): the module IS registered
with the runner, but two mock capabilities the per-engine/global cases
depend on do not exist yet and are being added by a parallel agent (the
TODO markers below are the alignment contract) — until they land, those
cases fail loudly on the capability raise:

  1. per-engine key-set exposure — either ``cache_key_set`` in each
     /snapshot engine dict (backed by MockLruBlockCache.snapshotKeys())
     or a dedicated ``GET /cache_keys?engine=<name>`` endpoint;
  2. ``POST /cache_evict {"engine", "keys"}`` — force-evict the named
     keys from one engine's LRU and bump cacheVersion so the master's
     cache-status poll propagates the eviction.

Assertion policy (core principle): every assertion states the CORRECT
contract, never the current behaviour.  Cases expected to fail (the
storm) carry an explicit FINDING note in their docstring.

Calibration: prefix families are 10 blocks x 1024 tokens; a full-hit
continuation carries hitTokens = 9 * 1024 = 9216 >= 8192 — the
researcher-mandated floor that keeps the estimate discount past the
maxExtraTtftMs=20 affinity gate (small-capacity x short-prefix combos
fall below the line and affinity never fires; when capacity is small,
pair it with a long prefix, never a short one).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..context import CaseContext, CaseDef, rid_base
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

KV_CASES: list[CaseDef] = []

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


def case(name: str, profiles=None, requires=None, source: str = ""):
    def deco(fn):
        KV_CASES.append(
            CaseDef(
                name=name,
                category="kv",
                fn=fn,
                profiles=profiles,
                requires=requires,
                source=source,
            )
        )
        return fn

    return deco


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


def _fire_request(ops, rid: int, fired: list, fired_handles: dict, **kwargs):
    """Schedule without consuming the stream — the ledger entry stays
    live until the case drain.  Returns (engine_name, error).  Under
    NON_BATCH dispatch the engine only sees the request when the CLIENT
    opens the stream, so the direct stream is opened fire-and-forget."""
    try:
        resp = ops.schedule(rid, **kwargs)
    except Exception as exc:
        return None, repr(exc)
    if resp.code != 200 or not resp.success:
        return None, f"schedule failed: {resp.error_message}"
    addr = ops.role_addr(resp, "PREFILL")
    name = ops.addr_to_name().get(addr, addr)
    fired.append((rid, resp))
    if not resp.enqueued_by_master:
        try:
            input_pb = ops.build_generate_input(rid, **kwargs)
            fired_handles[rid] = ops.start_stream(resp, rid, input_pb=input_pb)
        except Exception as exc:
            return name, f"direct stream failed to open: {exc!r}"
    return name, None


def _poll_engine_pending(
    ops, engine_name: str, min_pending: int, timeout_s: float = 6.0
) -> bool:
    """Engine-side proof that a fired request was really dispatched."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = ops.snapshot_by_name().get(engine_name, {})
        if info.get("waiting", 0) + info.get("running", 0) >= min_pending:
            return True
        time.sleep(0.1)
    return False


def _drain_fired(ops, fired: list, fired_handles: dict, wait_s: float = 30.0) -> list:
    """Consume every fired request to terminal state (cancel fallback).
    Returns [(rid, engine_name, completed, err)]."""
    outcomes = []
    for rid, resp in fired:
        name = ops.addr_to_name().get(ops.role_addr(resp, "PREFILL"), "")
        completed = False
        err = None
        try:
            handle = (
                fired_handles[rid]
                if rid in fired_handles
                else ops.start_stream(resp, rid)
            )
            ended = handle.wait_end(wait_s)
            completed = ended and handle.snap.completed and not handle.snap.error
            if not completed:
                err = handle.snap.error or "stream did not complete"
        except Exception as exc:
            err = repr(exc)
        if not completed:
            try:
                ops.cancel(rid, resp)
            except Exception:
                pass
        outcomes.append((rid, name, completed, err))
    return outcomes


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
    discovery: str = "file",
) -> EnvSpec:
    """Shared KV-family env: 2P+2D default, default prefill LRU capacity
    unless the case pins a small one (tiny capacities pair with the
    10-block families so the hit still prices past the affinity line)."""
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
        decode_cache_blocks=4,
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


@case(
    "kv_pe_admit_isolation",
    source="kv family: per-engine cache ledger isolation (task #84)",
)
def kv_pe_admit_isolation(ctx: CaseContext):
    """[per-engine] A's admissions never widen B's key set.

    Scenario: ledger-separated seeding (the kv_prefix_stickiness
    technique) pins family-0 on engine A and family-5 on engine B; with
    B slowed to 5s, families 1..4 are zero-hit and their ledger pricing
    admits them on A only.  Behaviour: per-engine admit accounting in
    the master's cache-status index.  Expected (contract): only A's
    cache_key_set grows with the four families — B's stays exactly its
    own seed family — and subsequent same-prefix continuations stick to
    A (P9); a global broadcast of A's admits would equalize the hit and
    dissolve the affinity into spread.  Prediction: passes.
    """
    env = ctx.env_manager.ensure(_kv_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    fired, fired_handles = [], {}
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"

        # -- ledger-separated seeding: fam0 -> A, fam5 -> the other engine.
        for name in names:
            ops.set_perf(name, prefill_fixed_ms=2000.0)
        time.sleep(1.5)  # master perf sync
        fam0 = _fam_keys(base, 0)
        rid_a = ops.next_request_id(base)
        a_name, err = _fire_request(
            ops,
            rid_a,
            fired,
            fired_handles,
            input_len=PREFIX_INPUT_LEN,
            output_len=2,
            block_keys=fam0,
        )
        if err is None and not _poll_engine_pending(ops, a_name, 1):
            err = f"seed A never appeared on {a_name}"
        _drain_fired(ops, fired, fired_handles)
        fired, fired_handles = [], {}
        b_name, err_b = None, "seed A failed"
        if err is None:
            fam5 = _fam_keys(base, 5)
            rid_b = ops.next_request_id(base)
            addr_b, err_b = ops.run_one_request(
                rid_b,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=fam5,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            b_name = ops.addr_to_name().get(addr_b, addr_b)
        for name in names:
            ops.set_perf(name, prefill_fixed_ms=100.0)
        if err or err_b:
            return False, f"seeding failed: {err or err_b}"
        if a_name == b_name:
            return False, f"family separation failed: both seeds on {a_name}"

        # -- B stays heavy: zero-hit families 1..4 divert onto A only.
        ops.set_perf(b_name, prefill_fixed_ms=5000.0)
        time.sleep(1.5)
        fams = [_fam_keys(base, i) for i in range(1, 5)]
        for keys in fams:
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=keys,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                return False, f"family admit request failed: {err}"
            if ops.addr_to_name().get(addr, addr) != a_name:
                return False, (
                    f"zero-hit family landed on {addr} instead of {a_name} "
                    f"(B-slow diversion failed)"
                )
        ops.set_perf(b_name, prefill_fixed_ms=100.0)
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after family admits"

        # -- contract: A carries the four families, B does not.
        expected = {k for keys in fams for k in keys}
        a_keys = _engine_cache_keys(ops, a_name)
        b_keys = _engine_cache_keys(ops, b_name)
        missing_on_a = sorted(expected - a_keys)[:4]
        leaked_to_b = sorted(expected & b_keys)[:4]
        isolation_ok = not missing_on_a and not leaked_to_b

        # -- P9: same-prefix continuations stick to the sole holder A.
        addrs = []
        for i in range(10):
            keys = fams[i % 4]
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=keys,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                report.invariant("P6", False, detail=f"continuation failed: {err}")
                break
            addrs.append(ops.addr_to_name().get(addr, addr))
        if addrs:
            hits = sum(1 for n in addrs if n == a_name)
            report.check(
                "P9",
                hits / len(addrs),
                context="fam1_4",
                detail=f"holder={a_name}, hits={hits}/{len(addrs)}, " f"other={b_name}",
            )

        # -- final mock view: B never picked up A's families (a stray
        #    continuation onto B would admit there — caught here too).
        b_keys_after = _engine_cache_keys(ops, b_name)
        leaked = sorted((expected | set(fam0)) & b_keys_after)[:4]
        mock_ok = not leaked
        passed, detail, rep = report.finish(
            f"holder={a_name}, other={b_name}, " f"grades: {report.summary()}"
        )
        return (
            passed and isolation_ok and mock_ok,
            f"isolation_ok={isolation_ok} "
            f"(A missing={len(expected - a_keys)}, B leaked="
            f"{len(expected & b_keys)}), final_B_leak={len(leaked)}"
            f"{' e.g. ' + str(leaked) if leaked else ''}, {detail}",
            rep,
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _drain_fired(ops, fired, fired_handles)
        for name in _prefill_names(ops):
            try:
                ops.set_perf(name, prefill_fixed_ms=100.0)
            except Exception:
                pass


@case(
    "kv_pe_evict_zero_match",
    source="kv family: eviction event syncs through the master index (task #84)",
)
def kv_pe_evict_zero_match(ctx: CaseContext):
    """[per-engine] A forced evict syncs through: no stale stickiness.

    Scenario: family-0 is primed on its landing engine X (capacity 16 —
    the spec's tiny-capacity spirit, sized up from 4 so the 10-block
    prefix survives the LRU and prices past the affinity line); the
    sync converges; a positive-control request sticks to X; then
    /cache_evict removes the whole family from X and the sync converges
    again (>= 3.5s of cache_version quiet).  Behaviour: master-side
    propagation of the eviction event.  Expected (contract): the
    post-evict batch carries the same prefix but must NOT stick to X —
    zero-hit tie-window spread over the fired batch (P1); a stale
    master index would keep routing the family onto X (max-share 1.0).
    Prediction: passes.
    """
    env = ctx.env_manager.ensure(_kv_spec(ctx, "_evict", prefill_cache_blocks=16))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    fired, fired_handles = [], {}
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        fam0 = _fam_keys(base, 0)

        # -- prime: wherever family-0 first lands is the holder X.
        rid_prime = ops.next_request_id(base)
        addr_x, err = ops.run_one_request(
            rid_prime,
            input_len=PREFIX_INPUT_LEN,
            output_len=2,
            block_keys=fam0,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err:
            return False, f"prime failed: {err}"
        x_name = ops.addr_to_name().get(addr_x, addr_x)
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after prime"

        # -- positive control: affinity is live — the replay sticks to X.
        rid_ctl = ops.next_request_id(base)
        addr_ctl, err_ctl = ops.run_one_request(
            rid_ctl,
            input_len=PREFIX_INPUT_LEN,
            output_len=2,
            block_keys=fam0,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        ctl_name = ops.addr_to_name().get(addr_ctl, addr_ctl)
        if err_ctl:
            return False, f"positive control failed: {err_ctl}"
        if ctl_name != x_name:
            return False, (
                f"positive control landed on {ctl_name} instead of holder "
                f"{x_name} — affinity was not live before the evict"
            )

        # -- evict the whole family from X, wait for the sync to settle.
        _cache_evict(ops, x_name, fam0)
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after evict"
        x_keys = _engine_cache_keys(ops, x_name)
        evicted_ok = not (set(fam0) & x_keys)

        # -- negative control: a fired batch (two-phase, decisions inside
        #    one sync window) must spread — no stale stickiness to X.
        wave_names = []
        for _ in range(10):
            rid = ops.next_request_id(base)
            name, err = _fire_request(
                ops,
                rid,
                fired,
                fired_handles,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=fam0,
            )
            if err:
                report.invariant("P6", False, detail=f"wave fire failed: {err}")
                break
            wave_names.append(name)
            time.sleep(0.12)
        outcomes = _drain_fired(ops, fired, fired_handles)
        fired, fired_handles = [], {}
        if wave_names:
            dist = {}
            for n in wave_names:
                dist[n] = dist.get(n, 0) + 1
            max_share = max(dist.values()) / len(wave_names)
            share_x = dist.get(x_name, 0) / len(wave_names)
            report.check(
                "P1",
                max_share,
                context="post_evict",
                detail=(
                    f"holder_evicted={x_name}, share_x={share_x:.2f}, "
                    f"dist={json.dumps(dist, sort_keys=True)}"
                ),
            )
            passed, detail, rep = report.finish(
                f"evicted_holder={x_name}, grades: {report.summary()}"
            )
            return (
                passed and evicted_ok,
                f"evicted_from_holder={evicted_ok}, {detail}",
                rep,
            )
        passed, detail, rep = report.finish("wave never fired")
        return False, detail, rep
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _drain_fired(ops, fired, fired_handles)


@case(
    "kv_pe_prefix_continuity",
    source="kv family: continuous-prefix matching, gap truncates (task #84)",
)
def kv_pe_prefix_continuity(ctx: CaseContext):
    """[per-engine] Continuous-prefix matching: a gap truncates the hit.

    Scenario: the ledger-separated double dispatch shares family-0
    between e1 and e2; /cache_evict then removes k2 from e1 (leaving 9
    blocks with a GAP) and k9,k10 from e2 (leaving the first 8 blocks
    CONTIGUOUS).  Behaviour: prefix-hit accounting.  Expected
    (contract): a request for [k1..k10] matches only e2's contiguous
    run — 8 blocks, hitTokens 8192, the M3 half-hit frame (a partial
    hit still concentrates on its holder); a block-COUNT caliber
    instead would rank e1 (9 blocks) over e2 (8) and route there, and a
    suffix-match caliber would equalize the two.  The 5-request batch
    must therefore land on e2 (M3).  Prediction: passes.
    """
    env = ctx.env_manager.ensure(_kv_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        fam0 = _fam_keys(base, 0)

        e1, e2, err = _seed_shared_prefix(ops, base, fam0, PREFIX_INPUT_LEN)
        if err:
            return False, f"shared seeding failed: {err}"
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after seeding"

        # -- carve the two views: e1 loses k2 (gap), e2 loses k9,k10
        #    (trailing blocks — its k1..k8 run stays contiguous).
        _cache_evict(ops, e1, [fam0[1]])
        _cache_evict(ops, e2, [fam0[8], fam0[9]])
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after carving"
        e1_run = _contiguous_prefix_len(_engine_cache_keys(ops, e1), fam0)
        e2_run = _contiguous_prefix_len(_engine_cache_keys(ops, e2), fam0)
        carved_ok = e1_run == 1 and e2_run == 8

        # -- M3: the 5-request batch concentrates on the LONGEST
        #    CONTIGUOUS run holder (e2).  A count caliber would send it
        #    to e1 (9 blocks); a suffix caliber would spread it.
        addrs = []
        for _ in range(5):
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=fam0,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                report.invariant("P6", False, detail=f"request failed: {err}")
                break
            addrs.append(ops.addr_to_name().get(addr, addr))
        if addrs:
            hits = sum(1 for n in addrs if n == e2)
            report.check(
                "M3",
                hits / len(addrs),
                context="continuity",
                detail=(
                    f"contiguous_holder={e2}(run={e2_run}), "
                    f"gapped={e1}(run={e1_run}), hits={hits}/{len(addrs)}"
                ),
            )
        passed, detail, rep = report.finish(
            f"runs: {e1}={e1_run}, {e2}={e2_run}, grades: {report.summary()}"
        )
        return passed and carved_ok, f"carved_ok={carved_ok}, {detail}", rep
    except Exception as exc:
        return False, f"exception: {exc!r}"


# ===========================================================================
# Global-index cases
# ===========================================================================


@case(
    "kv_g_shared_block_both_match",
    source="kv family: shared holder set -> equal-hit tie (task #84)",
)
def kv_g_shared_block_both_match(ctx: CaseContext):
    """[global] Both holders of a shared block match: tie, not fight.

    Scenario: the double dispatch shares family-0 between e1 and e2
    (the master's global index maps the blocks to a holder SET).
    Behaviour: affinity with two equal max-hit candidates.  Expected
    (contract): maxHit == minHit -> NO_CACHE_LEAD — subsequent
    same-prefix requests spread across the holders (P1 max-share over
    20 serial requests, P2 both engines used) and the holder-union
    share stays 100%; a one-holder-only index would instead pin every
    request onto a single engine (max-share ~1.0).  Prediction: passes.
    """
    env = ctx.env_manager.ensure(_kv_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        fam0 = _fam_keys(base, 0)

        e1, e2, err = _seed_shared_prefix(ops, base, fam0, PREFIX_INPUT_LEN)
        if err:
            return False, f"shared seeding failed: {err}"
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after seeding"

        shared_ok = set(fam0) <= _engine_cache_keys(ops, e1) and set(
            fam0
        ) <= _engine_cache_keys(ops, e2)

        # -- 20 serial same-prefix requests: equal-hit tie spreads.
        addrs = []
        for _ in range(20):
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=fam0,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                report.invariant("P6", False, detail=f"request failed: {err}")
                break
            addrs.append(ops.addr_to_name().get(addr, addr))
        if addrs:
            dist = {}
            for n in addrs:
                dist[n] = dist.get(n, 0) + 1
            max_share = max(dist.values()) / len(addrs)
            used = len(dist)
            union_share = (dist.get(e1, 0) + dist.get(e2, 0)) / len(addrs)
            report.check(
                "P1",
                max_share,
                context="shared_both_match",
                detail=f"dist={json.dumps(dist, sort_keys=True)}",
            )
            report.invariant(
                "P2",
                used >= 2,
                context="shared_both_match",
                detail=f"workers={used}",
            )
            report.invariant(
                "P6",
                union_share == 1.0,
                context="holder_union",
                detail=f"holder-union share={union_share:.2f} (e1+e2)",
            )
        passed, detail, rep = report.finish(
            f"holders={{{e1}, {e2}}}, grades: {report.summary()}"
        )
        return passed and shared_ok, f"shared_ok={shared_ok}, {detail}", rep
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "kv_g_partial_release_redirect",
    source="kv family: partial release of a shared block (task #84)",
)
def kv_g_partial_release_redirect(ctx: CaseContext):
    """[global] Partial release: one holder evicts -> redirect to the other.

    Scenario: family-0 shared between e1 and e2; /cache_evict releases
    it from e1 only (e1's own churn — the spec's small-A/big-A surface,
    expressed through the eviction endpoint because EnvSpec capacities
    are uniform across engines).  Behaviour: partial release of a
    shared block.  Expected (contract): after convergence e2 is the
    SOLE holder and the sole max-hit candidate — every subsequent
    same-prefix request redirects to e2 (P9 over 10 serial requests);
    a stale e1 entry would either pin traffic on e1 or equalize the hit
    into a spread.  Prediction: passes (blank-slate lock contract).
    """
    env = ctx.env_manager.ensure(_kv_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        fam0 = _fam_keys(base, 0)

        e1, e2, err = _seed_shared_prefix(ops, base, fam0, PREFIX_INPUT_LEN)
        if err:
            return False, f"shared seeding failed: {err}"
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after seeding"

        # -- e1 releases its copy; e2 keeps the shared family.
        _cache_evict(ops, e1, fam0)
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after partial release"
        released_ok = not (set(fam0) & _engine_cache_keys(ops, e1)) and set(
            fam0
        ) <= _engine_cache_keys(ops, e2)

        # -- P9: every continuation redirects onto the surviving holder.
        addrs = []
        for _ in range(10):
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=fam0,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                report.invariant("P6", False, detail=f"request failed: {err}")
                break
            addrs.append(ops.addr_to_name().get(addr, addr))
        if addrs:
            hits = sum(1 for n in addrs if n == e2)
            report.check(
                "P9",
                hits / len(addrs),
                context="partial_release",
                detail=f"survivor={e2}, hits={hits}/{len(addrs)}, " f"released={e1}",
            )
        passed, detail, rep = report.finish(
            f"survivor={e2}, released={e1}, grades: {report.summary()}"
        )
        return passed and released_ok, f"released_ok={released_ok}, {detail}", rep
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "kv_g_full_release_no_ghost",
    source="kv family: full release leaves no ghost entries (task #84)",
)
def kv_g_full_release_no_ghost(ctx: CaseContext):
    """[global] Full release leaves no ghost entries.

    Scenario: family-0 shared between e1 and e2; BOTH holders evict it;
    the sync converges (>= 3.5s of cache_version quiet).  Behaviour:
    full release of a shared block.  Expected (contract): the master's
    index carries no residue — same-prefix requests behave zero-hit
    (P1 spread over the fired batch, no engine pinned); a ghost entry
    would keep the family stuck on one engine (max-share ~1.0).
    Prediction: passes.
    """
    env = ctx.env_manager.ensure(_kv_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    fired, fired_handles = [], {}
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        fam0 = _fam_keys(base, 0)

        e1, e2, err = _seed_shared_prefix(ops, base, fam0, PREFIX_INPUT_LEN)
        if err:
            return False, f"shared seeding failed: {err}"
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after seeding"

        _cache_evict(ops, e1, fam0)
        _cache_evict(ops, e2, fam0)
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after full release"
        released_ok = all(
            not (set(fam0) & _engine_cache_keys(ops, n)) for n in (e1, e2)
        )

        # -- fired batch (two-phase, decisions inside one sync window):
        #    zero-hit tie-window spread — no ghost stickiness.
        wave_names = []
        for _ in range(10):
            rid = ops.next_request_id(base)
            name, err = _fire_request(
                ops,
                rid,
                fired,
                fired_handles,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=fam0,
            )
            if err:
                report.invariant("P6", False, detail=f"wave fire failed: {err}")
                break
            wave_names.append(name)
            time.sleep(0.12)
        _drain_fired(ops, fired, fired_handles)
        fired, fired_handles = [], {}
        if wave_names:
            dist = {}
            for n in wave_names:
                dist[n] = dist.get(n, 0) + 1
            max_share = max(dist.values()) / len(wave_names)
            report.check(
                "P1",
                max_share,
                context="full_release",
                detail=(
                    f"both_evicted={{{e1}, {e2}}}, "
                    f"dist={json.dumps(dist, sort_keys=True)}"
                ),
            )
            passed, detail, rep = report.finish(f"grades: {report.summary()}")
            return (
                passed and released_ok,
                f"released_ok={released_ok}, {detail}",
                rep,
            )
        passed, detail, rep = report.finish("wave never fired")
        return False, detail, rep
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _drain_fired(ops, fired, fired_handles)


@case(
    "kv_g_sync_convergence",
    source="kv family: mixed admit/evict stream converges to snapshot truth (task #84)",
)
def kv_g_sync_convergence(ctx: CaseContext):
    """[global] Mixed admit/evict sequences converge to snapshot truth.

    Scenario: an INTERLEAVED event stream — family-0 double-dispatched
    (2 admits) then evicted from e1; family-1 admitted once (landing
    spot s1 recorded); family-0 evicted from e2 too; family-2
    double-dispatched then evicted from one side — then >= 3.5s of
    silence.  Behaviour: incremental sync under a mixed event stream
    (the out-of-order / dropped-update paths have never been
    exercised).  Expected (contract): post-silence routing matches the
    ENGINE snapshots — family-0 (no holder) spreads (P1 over the fired
    batch), family-1 (sole holder s1) sticks (P9), family-2 (sole
    holder d2) sticks (P9); a ghost or a lost update lands in the wrong
    bucket.  Prediction: UNCERTAIN — the reorder/drop path is
    untested; a failure here is a finding, not a flake to retry away.
    """
    env = ctx.env_manager.ensure(_kv_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    fired, fired_handles = [], {}
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        fam0 = _fam_keys(base, 0)
        fam1 = _fam_keys(base, 1)
        fam2 = _fam_keys(base, 2)

        # -- op 1-2: admit fam0 on both engines (double dispatch).
        e1, e2, err = _seed_shared_prefix(ops, base, fam0, PREFIX_INPUT_LEN)
        if err:
            return False, f"fam0 double dispatch failed: {err}"

        # -- op 3: partial evict of fam0 from e1.
        _cache_evict(ops, e1, fam0)

        # -- op 4: admit fam1 on its natural landing spot.
        rid_f1 = ops.next_request_id(base)
        addr_s1, err = ops.run_one_request(
            rid_f1,
            input_len=PREFIX_INPUT_LEN,
            output_len=2,
            block_keys=fam1,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err:
            return False, f"fam1 admit failed: {err}"
        s1 = ops.addr_to_name().get(addr_s1, addr_s1)

        # -- op 5: full evict of fam0 (the e2 side).
        _cache_evict(ops, e2, fam0)

        # -- op 6-7: admit fam2 on both, then evict one side.
        d1, d2, err = _seed_shared_prefix(ops, base, fam2, PREFIX_INPUT_LEN)
        if err:
            return False, f"fam2 double dispatch failed: {err}"
        _cache_evict(ops, d1, fam2)

        # -- silence: >= 3.5s of cache_version quiet, then the state is
        #    what it is — routing must agree with the engine snapshots.
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after the mixed stream"
        holder_keys = {n: _engine_cache_keys(ops, n) for n in names}
        fam0_ghost = any(set(fam0) & ks for ks in holder_keys.values())
        fam1_sole = sorted(n for n, ks in holder_keys.items() if set(fam1) <= ks)
        fam2_sole = sorted(n for n, ks in holder_keys.items() if set(fam2) <= ks)
        snapshot_ok = not fam0_ghost and fam1_sole == [s1] and fam2_sole == [d2]

        # -- fam0 (no holder): fired batch spreads (no ghost stickiness).
        wave_names = []
        for _ in range(10):
            rid = ops.next_request_id(base)
            name, err = _fire_request(
                ops,
                rid,
                fired,
                fired_handles,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=fam0,
            )
            if err:
                report.invariant("P6", False, detail=f"wave fire failed: {err}")
                break
            wave_names.append(name)
            time.sleep(0.12)
        _drain_fired(ops, fired, fired_handles)
        fired, fired_handles = [], {}
        if wave_names:
            dist = {}
            for n in wave_names:
                dist[n] = dist.get(n, 0) + 1
            report.check(
                "P1",
                max(dist.values()) / len(wave_names),
                context="fam0_no_holder",
                detail=f"dist={json.dumps(dist, sort_keys=True)}",
            )

        # -- fam1/fam2 (sole holders): continuations stick (P9 x2).
        for label, keys, holder in (
            ("fam1_sole", fam1, s1),
            ("fam2_sole", fam2, d2),
        ):
            addrs = []
            for _ in range(5):
                rid = ops.next_request_id(base)
                addr, err = ops.run_one_request(
                    rid,
                    input_len=PREFIX_INPUT_LEN,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    report.invariant(
                        "P6", False, detail=f"{label} request failed: {err}"
                    )
                    break
                addrs.append(ops.addr_to_name().get(addr, addr))
            if addrs:
                hits = sum(1 for n in addrs if n == holder)
                report.check(
                    "P9",
                    hits / len(addrs),
                    context=label,
                    detail=f"holder={holder}, hits={hits}/{len(addrs)}",
                )
        passed, detail, rep = report.finish(
            f"final_holders: fam1={s1}, fam2={d2}, grades: {report.summary()}"
        )
        return (
            passed and snapshot_ok,
            f"snapshot_ok={snapshot_ok} (fam0_ghost={fam0_ghost}, "
            f"fam1_holders={fam1_sole}, fam2_holders={fam2_sole}), {detail}",
            rep,
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _drain_fired(ops, fired, fired_handles)


@case(
    "kv_g_engine_down_cleanup",
    source="kv family: engine removal cleans only its own entries (task #84)",
)
def kv_g_engine_down_cleanup(ctx: CaseContext):
    """[global] Engine-down cleanup keeps shared blocks for the survivor.

    Scenario: 3-prefill env with dynamic file discovery; family-0 is
    double-dispatched onto h1 and h2 (the third engine holds nothing);
    remove_engine takes h1 down permanently.  Behaviour: index cleanup
    on engine removal.  Expected (contract): ONLY h1's holder entries
    are dropped — the shared family stays attributed to h2, so
    post-removal same-prefix requests keep landing on h2 (P9 over 5
    requests) and the survivor's key set is untouched; a cleanup that
    drops the whole key entry would orphan h2's cache and scatter the
    family uniformly across the remaining engines.  Prediction:
    UNCERTAIN — the removal -> index-cleanup wiring has never been
    verified; a failure is a finding (over-cleanup or leak).
    """
    env = ctx.env_manager.ensure(
        _kv_spec(ctx, "_down", n_prefill=3, discovery="discovery_file")
    )
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    removed = False
    try:
        names = _prefill_names(ops)
        if len(names) < 3:
            return False, "need >=3 prefill workers"
        fam0 = _fam_keys(base, 0)

        # -- share fam0 across two of the three engines.
        h1, h2, err = _seed_shared_prefix(ops, base, fam0, PREFIX_INPUT_LEN)
        if err:
            return False, f"shared seeding failed: {err}"
        other = [n for n in names if n not in (h1, h2)]
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after seeding"
        shared_ok = (
            set(fam0) <= _engine_cache_keys(ops, h1)
            and set(fam0) <= _engine_cache_keys(ops, h2)
            and all(not (set(fam0) & _engine_cache_keys(ops, n)) for n in other)
        )

        # -- h1 goes down; the master's alive count must follow.
        status, body = ops.remove_engine(engine_name=h1)
        removed = status == 200
        if not removed:
            return False, f"remove_engine({h1}) failed: {status} {body}"
        if not _wait_master_alive(ops, "PREFILL", len(names) - 1):
            return False, (
                f"master did not converge to {len(names) - 1} alive prefill "
                f"engines after removal (alive="
                f"{ops.master_alive_count('PREFILL')})"
            )

        # -- P9: the family keeps routing to the surviving holder h2.
        addrs = []
        for _ in range(5):
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=PREFIX_INPUT_LEN,
                output_len=2,
                block_keys=fam0,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                report.invariant("P6", False, detail=f"request failed: {err}")
                break
            addrs.append(ops.addr_to_name().get(addr, addr))
        survivor_kept = set(fam0) <= _engine_cache_keys(ops, h2)
        if addrs:
            hits = sum(1 for n in addrs if n == h2)
            report.check(
                "P9",
                hits / len(addrs),
                context="engine_down",
                detail=(
                    f"survivor={h2}, removed={h1}, hits={hits}/{len(addrs)}, "
                    f"other_alive={other}"
                ),
            )
        passed, detail, rep = report.finish(
            f"survivor={h2}, removed={h1}, grades: {report.summary()}"
        )
        return (
            passed and shared_ok and survivor_kept,
            f"shared_ok={shared_ok}, survivor_kept={survivor_kept}, {detail}",
            rep,
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if removed:
            # env hygiene: restore the prefill count for later reuse
            try:
                ops.add_engine("prefill")
            except Exception:
                pass


# ===========================================================================
# Storm / capacity cases
# ===========================================================================


@case(
    "kv_storm_hot_churn",
    source="kv family: hot-prefix churn storm vs small LRU (task #84)",
)
def kv_storm_hot_churn(ctx: CaseContext):
    """[storm] Rotating hot prefixes vs a small LRU — FINDING case.

    Scenario: 4 hot families (10 blocks each = 40 hot blocks) rotate
    one per window (5 requests/window, 10 windows = 50 requests) while
    each engine's LRU holds only 24 blocks — every rotation must evict
    yesterday's hot family somewhere.  Behaviour: churn-driven
    replication and the master index's stability under it.  Expected
    (contract): (a) the REPLICATION FACTOR (holders per family via
    cache_key_set) stays bounded — first run is OBSERVATION-MODE,
    recorded not asserted, TODO calibrate a band (<= 2 to start);
    (b) holder FLIPS stay within the traffic-driven bound (an
    unbounded ping-pong means sync thrash); (c) the hit-tier
    concentration M3 keeps >= loose (a request counts as hit-served
    when its landing engine already held the family's contiguous
    prefix, >= 8 blocks).  Prediction: EXPECTED TO FAIL — nothing
    suppresses replication, so every miss re-admits the family onto a
    second engine, the doubled footprint accelerates self-eviction
    and the hit rate collapses; THAT COLLAPSE IS THE FINDING (no
    replication suppression / admission control in the KV sync
    layer), not a flake to retry away.
    """
    env = ctx.env_manager.ensure(
        _kv_spec(ctx, "_storm", prefill_cache_blocks=STORM_CAPACITY_BLOCKS)
    )
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        fams = [_fam_keys(base, f) for f in range(STORM_FAMILIES)]

        def holder_mask(keys: list) -> dict:
            return {n: bool(set(keys) & _engine_cache_keys(ops, n)) for n in names}

        prev_hold = None
        flips = 0
        m3_hits = 0
        m3_total = 0
        failures = []
        for w in range(STORM_WINDOWS):
            hot = w % STORM_FAMILIES
            keys = fams[hot]
            for _ in range(STORM_WINDOW):
                # pre-request view: does the landing engine hold a
                # contiguous run long enough to price past the line?
                runs = {
                    n: _contiguous_prefix_len(_engine_cache_keys(ops, n), keys)
                    for n in names
                }
                rid = ops.next_request_id(base)
                addr, err = ops.run_one_request(
                    rid,
                    input_len=PREFIX_INPUT_LEN,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                m3_total += 1
                if err:
                    failures.append(f"w{w} rid={rid}: {err}")
                    continue
                landed = ops.addr_to_name().get(addr, addr)
                if runs.get(landed, 0) >= PREFIX_BLOCKS - 2:
                    m3_hits += 1
            # end-of-window flip accounting (any-block holder mask)
            cur = {}
            for f in range(STORM_FAMILIES):
                mask = holder_mask(fams[f])
                for n in names:
                    cur[(n, f)] = mask[n]
            if prev_hold is not None:
                flips += sum(1 for k in cur if cur[k] != prev_hold.get(k))
            prev_hold = cur

        # -- replication factor: OBSERVATION-MODE this round (spec):
        #    record the per-family holder-count distribution, assert
        #    nothing yet.  TODO(calibrate): band <= 2 to start once the
        #    distribution is known (meaningful at 3+ engines; with 2 it
        #    saturates trivially).
        replication = {
            f: sum(1 for n in names if prev_hold.get((n, f)))
            for f in range(STORM_FAMILIES)
        }
        max_replication = max(replication.values()) if replication else 0

        hit_rate = m3_hits / m3_total if m3_total else 0.0
        flips_ok = flips <= STORM_FLIP_BOUND
        report.invariant("P6", not failures, detail=f"failures={failures[:2]}")
        report.check(
            "M3",
            hit_rate,
            context="storm_hit_rate",
            detail=(
                f"hits={m3_hits}/{m3_total}, flips={flips}"
                f"(<= {STORM_FLIP_BOUND}), replication={replication}"
            ),
        )
        passed, detail, rep = report.finish(
            f"windows={STORM_WINDOWS}, grades: {report.summary()}"
        )
        return (
            passed and flips_ok,
            f"flips={flips} (bound {STORM_FLIP_BOUND}, ok={flips_ok}), "
            f"max_replication={max_replication} (observation-mode, "
            f"TODO band), {detail}",
            rep,
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "kv_capacity_conflict_overflow",
    source="kv family: affinity yields to a full holder ledger (task #84)",
)
def kv_capacity_conflict_overflow(ctx: CaseContext):
    """[capacity] Affinity yields when the holder's ledger is full.

    Scenario: a 40-block family-0 is primed on engine e1 (40960-token
    seqLen keeps the seed's hit share at 27.8% >= the 20% gate even
    against a 147456-token seed); both prefills are slowed to 5s; the
    seed CARRIES the family-0 prefix so affinity pins it on e1 — the
    holder IS the hot engine (the balance_overload_avoid_prefill seed
    technique); the cool engine e2 is then restored while the seed's
    ~2.06s predicted ledger keeps e1 hot, and a 5-request same-prefix
    wave fires through the live ledger (two-phase, 0.12s spacing).
    Behaviour: the affinity-vs-capacity conflict — every wave request
    is full-hit on e1, but e1's projected TTFT sits ~2s above e2
    (>> maxExtraTtftMs).  Expected (contract): OVER_CAP overflow — the
    affinity gate steps aside and the wave spills onto the
    non-matching engine (hot_share < 1, P5); short requests stay
    protected (P7 vs the unloaded baseline) and NOTHING parks on a
    queue timeout (P6 — the routing decision returns immediately).
    Prediction: UNCERTAIN — both sides of the switch point are
    verified in isolation (balance_overload_avoid_prefill, the kv
    affinity cases) but were never exercised in the same frame; a
    failure is a finding.
    """
    env = ctx.env_manager.ensure(_kv_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    is_batch = ctx.batch_dispatch()
    caliber = "completion_duration" if is_batch else "client_ttft"
    fired, fired_handles = [], {}
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        fam0 = _fam_keys(base, 0, CONFLICT_BLOCKS)

        # -- prime the long family on its natural landing engine.
        rid_prime = ops.next_request_id(base)
        addr_e1, err = ops.run_one_request(
            rid_prime,
            input_len=CONFLICT_INPUT_LEN,
            output_len=2,
            block_keys=fam0,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err:
            return False, f"prime failed: {err}"
        e1 = ops.addr_to_name().get(addr_e1, addr_e1)
        e2 = next(n for n in names if n != e1)
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after prime"

        # -- the seed carries the family prefix: affinity pins it on e1.
        for name in names:
            ops.set_perf(name, prefill_fixed_ms=5000.0)
        time.sleep(1.5)  # master perf sync
        seed_keys = fam0 + [base + 9000 + j for j in range(104)]
        rid_seed = ops.next_request_id(base)
        seed_name, err = _fire_request(
            ops,
            rid_seed,
            fired,
            fired_handles,
            input_len=CONFLICT_SEED_INPUT_LEN,
            output_len=2,
            block_keys=seed_keys,
        )
        if err:
            return False, f"seed failed: {err}"
        if seed_name != e1:
            return False, (
                f"seed carried the holder prefix but landed on {seed_name} "
                f"instead of holder {e1} (affinity pin failed)"
            )
        if not _poll_engine_pending(ops, e1, 1):
            return False, f"seed never appeared on {e1}"

        # -- cool engine fast again; baseline anchors the P7 denominator.
        ops.set_perf(e2, prefill_fixed_ms=100.0)
        time.sleep(0.3)

        def timed_request(rid: int, **kwargs):
            t_send = time.monotonic()
            try:
                resp = ops.schedule(rid, **kwargs)
            except Exception as exc:
                return None, None, None, repr(exc)
            if resp.code != 200 or not resp.success:
                return None, None, None, f"schedule failed: {resp.error_message}"
            name = ops.addr_to_name().get(ops.role_addr(resp, "PREFILL"), "")
            input_pb = (
                None
                if resp.enqueued_by_master
                else ops.build_generate_input(rid, **kwargs)
            )
            try:
                handle = ops.start_stream(resp, rid, input_pb=input_pb)
            except Exception as exc:
                return name, None, None, f"stream failed to open: {exc!r}"
            ended = handle.wait_end(STREAM_TIMEOUT_S)
            snap = handle.snap
            ttft = snap.first_received_s - t_send if snap.first_received_s else None
            dur = snap.terminated_s - t_send if snap.terminated_s else None
            if not ended or snap.error or not snap.completed:
                return name, ttft, dur, (snap.error or "stream did not complete")
            return name, ttft, dur, None

        rid_base_line = ops.next_request_id(base)
        base_name, base_ttft, base_dur, base_err = timed_request(
            rid_base_line, output_len=2
        )
        if base_err:
            return False, f"baseline failed: {base_err}"

        # -- wave: 5 same-prefix requests fired back-to-back through the
        #    live seed ledger (two-phase — decisions first, then collect).
        wave = []
        for i in range(5):
            rid = ops.next_request_id(base)
            name, err = _fire_request(
                ops,
                rid,
                fired,
                fired_handles,
                input_len=CONFLICT_INPUT_LEN,
                output_len=2,
                block_keys=fam0,
            )
            wave.append((rid, name, err))
            if err:
                report.invariant("P6", False, detail=f"wave fire failed: {err}")
            if i < 4:
                time.sleep(0.12)
        outcomes = {rid: (name, err) for rid, name, err in wave}
        _drain_fired(ops, fired, fired_handles)
        fired, fired_handles = [], {}

        wave_names = [name for _, name, err in wave if err is None]
        failures = [f"rid={rid}: {err}" for rid, _, err in wave if err]
        hot_count = sum(1 for n in wave_names if n == e1)
        hot_share = hot_count / len(wave) if wave else 1.0
        overflow_ok = hot_share < 1.0

        report.invariant("P6", not failures, detail=f"failures={failures[:2]}")
        report.check(
            "P5",
            hot_share,
            context="capacity_conflict",
            detail=(
                f"hot=holder={e1}({hot_count}/" f"{max(len(wave_names), 1)}), cool={e2}"
            ),
        )
        # P7 dual caliber (profile-dependent measurement, one band
        # table).  The wave requests were drained without per-request
        # timing capture, so the protection caliber is measured by ONE
        # timed same-prefix probe through the SAME live seed ledger:
        # the wave's routing outcome already proved the overflow, the
        # probe proves the short-request protection (a parked request
        # would blow past every tier).
        metric_base = (base_dur if is_batch else base_ttft) or 0.0
        rid_probe = ops.next_request_id(base)
        probe_name, probe_ttft, probe_dur, probe_err = timed_request(
            rid_probe,
            input_len=CONFLICT_INPUT_LEN,
            output_len=2,
            block_keys=fam0,
        )
        if probe_err:
            report.invariant(
                "P6", False, detail=f"protection probe failed: {probe_err}"
            )
            p7_value = float("inf")
            p7_detail = f"caliber={caliber}, probe failed: {probe_err}"
        else:
            probe_metric = (probe_dur if is_batch else probe_ttft) or 0.0
            if metric_base > 0 and probe_metric > 0:
                p7_value = probe_metric / metric_base
                p7_detail = (
                    f"caliber={caliber}, base={metric_base:.3f}s, "
                    f"probe={probe_metric:.3f}s, probe_landed={probe_name}"
                )
            else:
                p7_value = float("inf")
                p7_detail = f"caliber={caliber}, missing timing"
        report.check("P7", p7_value, context=caliber, detail=p7_detail)

        passed, detail, rep = report.finish(
            f"hot=holder={e1}, cool={e2}, grades: {report.summary()}"
        )
        return (
            passed and overflow_ok,
            f"overflow_ok={overflow_ok} (hot_share={hot_share:.2f} < 1), " f"{detail}",
            rep,
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _drain_fired(ops, fired, fired_handles)
        for name in _prefill_names(ops):
            try:
                ops.set_perf(name, prefill_fixed_ms=100.0)
            except Exception:
                pass


# ===========================================================================
# Affinity routing cases (migrated from the legacy scheduling family,
# task #85 category reorg — rid_base family "scheduling" -> "kv")
# ===========================================================================


@case(
    "kv_prefix_stickiness",
    source="scheduling_smoke.py S2+S5 (merged, task #61; M1 generalized, task #62)",
)
def kv_prefix_stickiness(ctx: CaseContext):
    """Prefix-reuse traffic sticks to the engine that holds the prefix cache —
    multi-family + free-mixing generalization (M1).

    Result properties (graded): P9 affinity fidelity (family-A followers
    landing on the family-A seed engine), P2 free-flow multi-engine spread,
    P6 completeness.

    Construction:
      1. seed A (keys 1001-1008, input_len=8192) fired while both prefills
         are slowed to 2s — its ~231ms production-fit estimate keeps the
         landing engine's ledger entry live (tie-window override is
         impossible: the doubled ledger ~463ms vs ~231ms dwarfs the
         ~23ms tie window);
      2. seed B (keys 2001-2008, same shape) scheduled while A is still
         in flight -> deterministically lands on the OTHER engine — the
         family separation the design calls for (a plain serial seeding
         would put both families on the same engine half the time);
      3. after both seeds complete and the master cache syncs
         (KV_CACHE_SYNC_WAIT_S), the main phase runs ~30 serial requests:
         60% family-A continuations (same keys, deterministic stickiness —
         the production-fit estimate prices the hit engine only ~6ms above
         the all-miss engine, but the bounded cache-affinity gate
         (maxExtraTtftMs=20) keeps the cache leader preferred; the legacy
         1ms/token default instead relied on its 0.7*hitTokens discount
         pushing the hit engine ~5s BELOW the tie window) interleaved
         with 40% unique-key free requests (no cache lead on either
         engine -> uniform tie-window spread).

    The legacy S5 cache_keys>0 assertion stays demoted to an observational
    log: mock-internal cache accounting is the mock's own unit-tested
    behaviour, not an LB contract.  Hit-latency benefits are NOT asserted
    (mock execution time is length/cache-blind — framework fact).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "kv")
    family_a_keys = list(range(1001, 1009))
    family_b_keys = list(range(2001, 2009))
    prefill_names: list[str] = []
    fired: list[tuple[int, object]] = []
    fired_handles: dict[int, object] = {}
    try:
        prefill_names = _prefill_names(ops)
        if len(prefill_names) < 2:
            return False, "need >=2 prefill workers"
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=2000.0)
        time.sleep(1.5)  # master perf sync

        # -- seed A: fire-and-forget, engine-side proof of its ledger entry.
        rid_a = ops.next_request_id(base)
        seed_a_name, err = _fire_request(
            ops,
            rid_a,
            fired,
            fired_handles,
            input_len=8192,
            output_len=2,
            block_keys=family_a_keys,
        )
        if err:
            report.invariant("P6", False, detail=f"seed A failed: {err}")
            return report.finish(f"seed A failed: {err}")
        if not _poll_engine_pending(ops, seed_a_name, 1):
            report.invariant(
                "P6", False, detail=f"seed A never appeared on {seed_a_name}"
            )
            return report.finish(f"seed A never appeared on {seed_a_name}")

        # -- seed B: deterministic away from seed A's live ledger.
        rid_b = ops.next_request_id(base)
        seed_b_addr, seed_b_err = ops.run_one_request(
            rid_b,
            input_len=8192,
            output_len=2,
            block_keys=family_b_keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        seed_b_name = ops.addr_to_name().get(seed_b_addr, seed_b_addr)
        if seed_b_err:
            report.invariant("P6", False, detail=f"seed B failed: {seed_b_err}")
            return report.finish(f"seed B failed: {seed_b_err}")

        # -- drain seed A, restore fast perf, let the master sync both caches.
        outcomes = _drain_fired(ops, fired, fired_handles)
        fired.clear()
        fired_handles.clear()
        seed_a_ok = outcomes and outcomes[0][2]
        if not seed_a_ok:
            report.invariant(
                "P6", False, detail=f"seed A did not complete: {outcomes[0][3]}"
            )
            return report.finish(f"seed A did not complete: {outcomes[0][3]}")
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=100.0)
        time.sleep(KV_CACHE_SYNC_WAIT_S)  # master cache sync

        if seed_a_name == seed_b_name:
            # The ledger technique makes this practically impossible (the
            # ~8s ledger gap dwarfs the tie window); keep the design's
            # "report it" clause as a loud observation.
            report.invariant(
                "P6",
                False,
                detail=(
                    f"family separation failed: both seeds landed on "
                    f"{seed_a_name} (ledger diversion did not fire)"
                ),
            )
            return report.finish(
                f"family separation failed: both seeds on {seed_a_name}"
            )

        # -- main phase: 60% family-A continuations + 40% unique-key free.
        cont_n, free_n = 18, 12
        addrs_a, addrs_free, failures = [], [], []
        for i in range(cont_n + free_n):
            rid = ops.next_request_id(base)
            if i % 5 < 3:  # 3:2 interleave -> 18 continuations / 12 free
                addr, err = ops.run_one_request(
                    rid,
                    input_len=8192,
                    output_len=2,
                    block_keys=family_a_keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"cont rid={rid}: {err}")
                else:
                    addrs_a.append(addr)
            else:
                keys = [rid * 100 + j for j in range(8)]
                addr, err = ops.run_one_request(
                    rid,
                    input_len=8192,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"free rid={rid}: {err}")
                else:
                    addrs_free.append(addr)

        addr_map = ops.addr_to_name()
        hits = sum(1 for a in addrs_a if addr_map.get(a, a) == seed_a_name)
        stick_share = hits / len(addrs_a) if addrs_a else 0.0
        free_engines = len({addr_map.get(a, a) for a in addrs_free})

        # Observational only (legacy S5 demoted to log).
        cache_keys_a = ops.snapshot_by_name().get(seed_a_name, {}).get("cache_keys", -1)

        report.invariant(
            "P6",
            not failures and len(addrs_a) == cont_n and len(addrs_free) == free_n,
            detail=f"failures={failures[:2]}",
        )
        report.check(
            "P9",
            stick_share,
            context="family_a",
            detail=(
                f"seed_a={seed_a_name}, seed_b={seed_b_name} (ledger-forced "
                f"apart), hits={hits}/{len(addrs_a)}, "
                f"cache_keys={cache_keys_a} (observational)"
            ),
        )
        report.invariant(
            "P2",
            free_engines >= 2,
            context="free_flow",
            detail=f"engines={free_engines}, free_n={len(addrs_free)}",
        )
        return report.finish(
            f"seed_a={seed_a_name}, seed_b={seed_b_name}, "
            f"stick={hits}/{len(addrs_a)}, free_engines={free_engines}, "
            f"cache_keys={cache_keys_a}(log), grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for name in prefill_names:
                ops.set_perf(name, prefill_fixed_ms=100.0)
        except Exception:
            pass
        if fired or fired_handles:
            _drain_fired(ops, fired, fired_handles)
        try:
            # Best-effort residue drain (task #87): a drain-fallback cancel
            # that fails leaves slots settling on the stale-TTL +
            # ExpirationTimer path (worst ~90s) — the legacy 30s window
            # stopped short of it and the residue poisoned later cases on
            # this shared env.  Still not asserted (this finally is
            # hygiene, the case's own contract lives in its verdict).
            AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)
        except Exception:
            pass


@case(
    "kv_hot_prefix_tension",
    source="hot-prefix tension M2 (task #62)",
)
def kv_hot_prefix_tension(ctx: CaseContext):
    """A 70%-traffic hot prefix family: stickiness holds AND the holder's
    concentration stays capped.

    Result properties (graded combination): P9 family stickiness (graded —
    the design's tension axis 1), M2 holder total-share cap (graded upper
    bound — tension axis 2, first calibrated measurement of the M2 band),
    P2 free-flow no-starvation (the other engine still takes free traffic),
    P6 completeness.

    Construction: family F shares a 16-block long prefix (keys 3001-3016,
    input_len=16384).  One seed request lands on engine X (uniform initial
    pick); after the master cache sync the main phase runs 40 serial
    requests in a fixed 7:3 interleave — 28 family continuations (every one
    carries the ~10.7s estimate discount on X: est = 16384 - 0.7*15360 =
    5619 vs 16384 elsewhere, a gap ~20x the tie window -> deterministic
    stickiness) and 12 unique-key free requests (no affinity on either
    engine -> uniform tie-window spread).

    On X's accumulating state: each completed family request re-admits the
    same 16 blocks (LRU-refreshed, idempotent) and adds its inputLen to the
    mock's KV accounting — the holder's cache/KV footprint keeps growing
    across the phase (observational), while the routing ledger itself
    resets between serial requests (each completes before the next fires),
    which is what keeps P9 deterministic and pins the M2 model to the free
    flow's binomial spread.

    M2 caliber: family and free requests share input_len=16384, so token
    share and request share coincide; the holder's TOTAL share counts seed,
    family continuations AND the free requests that tie-window scatter onto
    it: (29 + k)/41 with k ~ B(12, 0.5) over the free flow (29 = seed + 28
    continuations deterministically on X when stickiness is perfect), i.e.
    ~0.854 ± 0.042 (1σ) — see the M2 calibration note in grade.GRADE_BANDS
    for the false-fail derivation of the band values.

    Free-flow starvation (P2): if all 12 free requests were swallowed by
    X the other engine would idle — that is the starvation this property
    forbids (probability 0.5**12 under correct uniform spread).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "kv")
    family_keys = list(range(3001, 3017))
    input_len = 16384
    try:
        # -- seed: family F prefix lands on X (uniform initial pick).
        rid_seed = ops.next_request_id(base)
        seed_addr, seed_err = ops.run_one_request(
            rid_seed,
            input_len=input_len,
            output_len=2,
            block_keys=family_keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if seed_err:
            report.invariant("P6", False, detail=f"seed failed: {seed_err}")
            return report.finish(f"seed request failed: {seed_err}")
        addr_map = ops.addr_to_name()
        holder = addr_map.get(seed_addr, seed_addr)
        other_names = [n for n in _prefill_names(ops) if n != holder]
        time.sleep(KV_CACHE_SYNC_WAIT_S)  # master cache sync

        # -- main phase: 40 serial, fixed 7:3 interleave (28 family + 12 free).
        cont_n, free_n = 28, 12
        cont_addrs, free_addrs, failures = [], [], []
        for i in range(cont_n + free_n):
            rid = ops.next_request_id(base)
            if i % 10 < 7:  # 7 family : 3 free per decade
                addr, err = ops.run_one_request(
                    rid,
                    input_len=input_len,
                    output_len=2,
                    block_keys=family_keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"cont rid={rid}: {err}")
                else:
                    cont_addrs.append(addr)
            else:
                keys = [rid * 100 + j for j in range(16)]
                addr, err = ops.run_one_request(
                    rid,
                    input_len=input_len,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"free rid={rid}: {err}")
                else:
                    free_addrs.append(addr)

        holder_hits = sum(1 for a in cont_addrs if addr_map.get(a, a) == holder)
        stick_share = holder_hits / len(cont_addrs) if cont_addrs else 0.0
        free_on_other = sum(1 for a in free_addrs if addr_map.get(a, a) != holder)
        # M2 caliber: the holder's TOTAL share — seed + family continuations
        # + free requests scattered onto it by the tie window (token share ==
        # request share by uniform input_len).
        free_on_holder = len(free_addrs) - free_on_other
        holder_total = holder_hits + 1 + free_on_holder  # + seed
        total = 1 + len(cont_addrs) + len(free_addrs)
        holder_share = holder_total / total if total else 1.0
        holder_token_share = (
            holder_total * input_len / (total * input_len) if total else 1.0
        )

        report.invariant(
            "P6",
            not failures and len(cont_addrs) == cont_n and len(free_addrs) == free_n,
            detail=f"failures={failures[:2]}",
        )
        report.check(
            "P9",
            stick_share,
            context="hot_family",
            detail=(
                f"holder={holder}, hits={holder_hits}/{len(cont_addrs)}, "
                f"other={other_names}"
            ),
        )
        report.check(
            "M2",
            holder_share,
            context="holder_total_share",
            detail=(
                f"holder={holder}: {holder_total}/{total} requests "
                f"(token share {holder_token_share:.3f} — equal by uniform "
                f"input_len), free_on_other={free_on_other}/{len(free_addrs)}"
            ),
        )
        report.invariant(
            "P2",
            free_on_other >= 1,
            context="free_flow",
            detail=(
                f"free requests landing off-holder={free_on_other}/"
                f"{len(free_addrs)} (other engine must not be starved)"
            ),
        )
        return report.finish(
            f"holder={holder}, stick={holder_hits}/{len(cont_addrs)}, "
            f"holder_share={holder_share:.3f}, "
            f"free_off_holder={free_on_other}/{len(free_addrs)}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "kv_match_mixed",
    source="hit-rate tier contrast M3 (task #62)",
)
def kv_match_mixed(ctx: CaseContext):
    """Prefix hit-rate tiers: full-hit and half-hit traffic concentrate on
    the holder while zero-hit traffic spreads — a graded contrast.

    Result properties: M3 soft contrast bound (graded lower band on the
    same-engine concentration of the full-hit and half-hit tiers), P2
    zero-hit multi-engine spread, P6 completeness.  Hit-latency benefits
    are NOT asserted (mock execution time is length/cache-blind).

    Construction (fixed input_len=8192, three tiers, all serial):
      * full-hit tier — seed family keys 4001-4008 on engine X1, then 10
        continuations reusing the SAME 8 blocks: hitTokens = 7168 (the last
        partial block is excluded: rawHit >= seqLen -> seqLen - blockSize),
        estimate discount ~5.0s vs tie window ~0.3s -> deterministic
        concentration on X1;
      * half-hit tier — seed keys 5001-5004 (input_len=4096, 4 blocks) on
        X2, then 10 requests carrying [5001-5004 + 4 fresh keys]: the
        continuous prefix match stops at 4 blocks -> hitTokens = 4096,
        discount ~2.9s vs tie window ~0.5s -> deterministic concentration
        on X2 (a 50% hit rate still clears the affinity threshold — the
        contrast with the zero-hit tier is the point, not a partial
        stickiness);
      * zero-hit tier — 10 requests with fresh unique keys on both
        engines: no discount anywhere -> uniform tie-window spread.

    Why P2 covers only the zero-hit tier: P2 forbids starving an engine
    with INDISTINGUISHABLE traffic; full/half-hit requests landing on
    their holder is correct affinity routing, not starvation.  The
    zero-hit tier is exactly the indistinguishable population, so its
    spread carries the P2 contract (probability of a single-engine
    collapse under correct spread: 2 * 0.5**10 ~= 0.2%).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "kv")
    full_keys = list(range(4001, 4009))
    half_shared_keys = list(range(5001, 5005))
    try:

        def run_tier_cont(n: int, keys_fn, label: str):
            """Serial run of *n* requests, each keys from keys_fn(rid, i)."""
            addrs, failures = [], []
            for i in range(n):
                rid = ops.next_request_id(base)
                addr, err = ops.run_one_request(
                    rid,
                    input_len=8192,
                    output_len=2,
                    block_keys=keys_fn(rid, i),
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"{label} rid={rid}: {err}")
                else:
                    addrs.append(addr)
            return addrs, failures

        # -- tier 1: full-hit (8-block family).
        rid_seed1 = ops.next_request_id(base)
        seed1_addr, seed1_err = ops.run_one_request(
            rid_seed1,
            input_len=8192,
            output_len=2,
            block_keys=full_keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if seed1_err:
            report.invariant("P6", False, detail=f"seed1 failed: {seed1_err}")
            return report.finish(f"full-hit seed failed: {seed1_err}")
        time.sleep(KV_CACHE_SYNC_WAIT_S)
        full_addrs, full_fail = run_tier_cont(10, lambda rid, i: full_keys, "full")

        # -- tier 2: half-hit (4 shared + 4 fresh per request).
        rid_seed2 = ops.next_request_id(base)
        seed2_addr, seed2_err = ops.run_one_request(
            rid_seed2,
            input_len=4096,
            output_len=2,
            block_keys=half_shared_keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if seed2_err:
            report.invariant("P6", False, detail=f"seed2 failed: {seed2_err}")
            return report.finish(f"half-hit seed failed: {seed2_err}")
        time.sleep(KV_CACHE_SYNC_WAIT_S)
        half_addrs, half_fail = run_tier_cont(
            10,
            lambda rid, i: half_shared_keys + [rid * 100 + 40 + j for j in range(4)],
            "half",
        )

        # -- tier 3: zero-hit (fresh unique keys everywhere).
        zero_addrs, zero_fail = run_tier_cont(
            10, lambda rid, i: [rid * 100 + j for j in range(8)], "zero"
        )

        addr_map = ops.addr_to_name()
        failures = full_fail + half_fail + zero_fail

        def concentration(addrs, anchor_addr) -> float:
            if not addrs:
                return 0.0
            anchor = addr_map.get(anchor_addr, anchor_addr)
            return sum(1 for a in addrs if addr_map.get(a, a) == anchor) / len(addrs)

        full_conc = concentration(full_addrs, seed1_addr)
        half_conc = concentration(half_addrs, seed2_addr)
        zero_dist = Counter(addr_map.get(a, a) for a in zero_addrs)
        zero_engines = len(zero_dist)
        zero_max = max(zero_dist.values()) / len(zero_addrs) if zero_addrs else 1.0

        report.invariant(
            "P6",
            not failures
            and len(full_addrs) == 10
            and len(half_addrs) == 10
            and len(zero_addrs) == 10,
            detail=f"failures={failures[:2]}",
        )
        report.check(
            "M3",
            full_conc,
            context="full_hit",
            detail=(
                f"concentration on full-hit seed engine={full_conc:.2f} "
                f"(vs zero-hit baseline ~0.5)"
            ),
        )
        report.check(
            "M3",
            half_conc,
            context="half_hit",
            detail=(
                f"concentration on half-hit seed engine={half_conc:.2f} "
                f"(50% hit still clears the affinity threshold)"
            ),
        )
        report.invariant(
            "P2",
            zero_engines >= 2,
            context="zero_hit",
            detail=(
                f"zero-hit spread: engines={zero_engines}, "
                f"max_share={zero_max:.2f} (observational, expected ~0.5-0.7)"
            ),
        )
        return report.finish(
            f"full_conc={full_conc:.2f}, half_conc={half_conc:.2f}, "
            f"zero_dist={json.dumps(dict(zero_dist), sort_keys=True)}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


# ===========================================================================
# LRU capacity case (migrated from the legacy gate family, task #85
# category reorg — rid_base family "chaos" -> "kv")
# ===========================================================================


@case(
    "kv_lru_eviction_affinity",
    source="gap G10: LRU prefix reuse + capacity eviction + affinity routing end-to-end",
)
def kv_lru(ctx: CaseContext):
    """Drive the mock's per-engine MockLruBlockCache end to end:

    1. R1 primes [k1,k2] on its landing engine X — snapshot proves
       cache_keys >= 2 with zero evictions.
    2. R2 replays the SAME keys: master-side cache-status sync must route
       it back to X (S2-style affinity, one retry for sync lag).
    3. R3 replays the prefix [k1,k2] plus three fresh keys: five keys
       admitted into the capacity-4 LRU evict exactly the eldest block —
       snapshot proves evictions >= 1 and cache_keys capped at 4, and the
       prefix hit keeps R3 on X as well.
    """
    env = ctx.env_manager.ensure(_lru_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")

    def run(rid, keys, input_len):
        return ops.run_one_request(
            rid,
            input_len=input_len,
            output_len=2,
            block_keys=keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )

    try:
        keys_a = [base + 1, base + 2]
        rid1 = ops.next_request_id(base)
        addr1, err1 = run(rid1, keys_a, 2048)
        if err1:
            return False, f"R1 (prime) failed: {err1}"
        time.sleep(KV_CACHE_SYNC_WAIT_S)

        # Affinity: same keys must return to the priming engine.
        rid2 = ops.next_request_id(base)
        addr2, err2 = run(rid2, keys_a, 2048)
        if err2:
            return False, f"R2 (replay) failed: {err2}"
        affinity = addr1 == addr2
        if not affinity:
            # S2-style retry: the cache-status sync may lag one poll.
            time.sleep(KV_CACHE_SYNC_WAIT_S)
            rid2b = ops.next_request_id(base)
            addr2b, err2b = run(rid2b, keys_a, 2048)
            if err2b:
                return False, f"R2 retry failed: {err2b}"
            affinity = addr1 == addr2b

        addr_map = ops.addr_to_name()
        engine_x = addr_map.get(addr1, "?")
        snap = ops.snapshot_by_name()
        keys_after_prime = snap.get(engine_x, {}).get("cache_keys", 0)
        evictions_after_prime = snap.get(engine_x, {}).get("cache_evictions", 0)

        # Capacity pressure: prefix [k1,k2] + 3 fresh keys -> 5 admits into
        # a capacity-4 LRU -> exactly the eldest block evicted.
        keys_ext = keys_a + [base + 3, base + 4, base + 5]
        rid3 = ops.next_request_id(base)
        addr3, err3 = run(rid3, keys_ext, 4096)
        if err3:
            return False, f"R3 (pressure) failed: {err3}"
        time.sleep(0.5)  # admit lands at prefill completion
        snap = ops.snapshot_by_name()
        engine_z = addr_map.get(addr3, "?")
        keys_after_pressure = snap.get(engine_z, {}).get("cache_keys", 0)
        evictions_after_pressure = snap.get(engine_z, {}).get("cache_evictions", 0)
        prefix_affinity = addr3 == addr1

        prime_ok = keys_after_prime >= 2 and evictions_after_prime == 0
        eviction_ok = evictions_after_pressure >= 1 and keys_after_pressure <= 4
        passed = affinity and prime_ok and eviction_ok and prefix_affinity
        return passed, (
            f"engine_x={engine_x}, affinity_r2={affinity}, "
            f"after_prime: keys={keys_after_prime}, evictions={evictions_after_prime}, "
            f"pressure_landed_on={engine_z}, prefix_affinity_r3={prefix_affinity}, "
            f"after_pressure: keys={keys_after_pressure} (<=4), "
            f"evictions={evictions_after_pressure} (>=1)"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


# ===========================================================================
# Decode-side KV-capacity parking case (migrated from the legacy anomaly
# family E4, task #85 category reorg — rid_base family "anomaly" -> "kv")
# ===========================================================================


@case(
    "kv_decode_capacity_park",
    source="decode-side anomaly gap (G1) — new (anomaly E4 until task #85)",
)
def kv_decode_capacity_park(ctx: CaseContext):
    """Decode-side anomaly: every decode engine KV-exhausted -> the request
    is parked undelivered, a master Cancel releases it without residue, and
    clearing the pressure recovers routing.

    Why KV pressure and not the E-series ops.inject faults: a decode engine
    in the Java mock never receives traffic through any gRPC entry point.
    After prefill completes, the request is handed off IN-PROCESS
    (JavaMockEngineCluster.FastRpcService.startDecode ->
    scheduleDecodeCompletion), so enqueue_error / generate_error /
    fetch_error — all checked at the enqueueBatch / generateStreamCall /
    fetchResponse RPC entries — never fire for a decode engine, and
    no_respond on decode only suppresses the "intermediate first-step
    output" which the mock never produces (each request yields exactly one
    finished message).  The one decode-side anomaly observable end-to-end
    is KV capacity: the delivery-capacity admission hard-filters every
    decode endpoint whose available_kv_tokens < seq_len, so exhausting
    every decode engine's KV must block delivery.

    v2 contract (task #55, source-verified — supersedes the v1 fail-fast
    assertion): the QUEUE scheduler treats decode KV exhaustion as a WAIT
    condition, not a fail-fast rejection:
      * FixedWindowBatcherAlgorithm parks the head when delivery capacity
        cannot be reserved ("Dynamic KV pressure is a wait condition, not a
        rejection"; BatcherContext.admitAndDeliverCapacityFeasiblePrefix
        returns CapacityBlocked and the worker loop waits for the exact
        resource-change event).
      * The scheduling deadline is owned by the queue config
        (QueueSchedulerConfig.queueTimeoutMs, default 1h), not the caller,
        so the Schedule RPC stays pending while parked — the client
        observes its own gRPC DEADLINE_EXCEEDED instead of a rejection
        response.  The pre-v2 fail-fast NO_AVAILABLE_WORKER contract
        belonged to the v1 non-QUEUE flow and does not exist in v2.
      * A client-side RPC deadline/cancellation does NOT release the
        parked entry (it lingered until the stale-inflight TTL eviction in
        the repro); an explicit master Cancel does
        (PriorityScheduler.cancelRequest -> isLocallyReversible -> local
        cleanup).

    Scenario:
      1. set active_kv_tokens = total on every decode engine
      2. probe Schedule with a short client-side deadline: it must stay
         pending (client DEADLINE_EXCEEDED, no rejection response) and the
         parked rid must NOT be delivered to any engine
      3. master Cancel must release the parked request and leave no
         inflight residue
      4. clear the pressure -> a fresh request must complete again

    Profile semantics (v2): the decision and dispatcher axes are invisible
    to the decode-side delivery capacity gate — both delivery modes share
    the per-worker batcher and the same capacity admission — so the case
    runs under all profiles.  The no-residue assertion is a pre-probe
    WATERMARK comparison rather than a global zero check: under NON_BATCH
    dispatch a client-side Cancel cannot safely release a delivered
    request's master ledger entry (the fence probe's NOT_FOUND ack is not
    a safe-release fact — the client connects to the engine
    asynchronously after RouteDecision), so earlier requests on the
    shared env may leave contract-parked entries; this case only owns
    the residue of ITS OWN parked probe.
    """
    ops = ctx.ops()
    base = rid_base(ctx, "kv")
    injected: list[str] = []
    try:
        snap = ops.snapshot_by_name()
        decode_names = sorted(
            name for name, e in snap.items() if e.get("role") == "decode"
        )
        if not decode_names:
            return False, "no decode workers found"

        # Exhaust every decode engine: active = total -> available = 0.
        for name in decode_names:
            info = snap[name]
            total_kv = int(info.get("available_kv_tokens", 0)) + int(
                info.get("active_kv_tokens", 0)
            )
            ops.set_kv_pressure(name, total_kv)
            injected.append(name)
        time.sleep(1.5)  # master worker-status sync

        # 1. The probe stays pending: a short client-side deadline fires
        #    instead of the master returning a rejection.
        base_view = ops.master_inflight() or {}

        def _inflight_totals(view: dict) -> tuple[int, int, int]:
            return (
                int(view.get("scheduler_inflight", 0) or 0),
                sum(
                    int(ep.get("inflight_batches", 0) or 0)
                    for ep in view.get("prefill_endpoints", []) or []
                ),
                sum(
                    int(ep.get("inflight_requests", 0) or 0)
                    for ep in view.get("decode_endpoints", []) or []
                ),
            )

        base_sched, base_prefill, base_decode = _inflight_totals(base_view)
        rid = ops.next_request_id(base)
        probe: dict = {}

        def _probe() -> None:
            try:
                resp = ops.schedule(rid, timeout_s=E4_PROBE_DEADLINE_S)
                probe["returned"] = (
                    f"code={resp.code}, success={resp.success}, "
                    f"error={resp.error_message!r}"
                )
            except Exception as exc:  # client deadline while parked
                code_fn = getattr(exc, "code", None)
                probe["grpc_code"] = str(code_fn()) if callable(code_fn) else ""
                probe["exc"] = repr(exc)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_probe).result(timeout=E4_PROBE_DEADLINE_S + 10.0)
        parked_ok = probe.get("grpc_code") == "StatusCode.DEADLINE_EXCEEDED"
        parked_detail = probe.get("returned") or probe.get(
            "grpc_code", probe.get("exc", "no outcome")
        )

        # 2. The parked request must not have been delivered to any engine.
        time.sleep(0.5)
        snap2 = ops.snapshot()
        delivered = [
            engine["name"]
            for engine in snap2.get("engines", [])
            if str(rid) in engine.get("request_lifecycle", {})
        ]
        not_delivered_ok = not delivered

        # 3. An explicit master Cancel releases the parked request: master
        #    inflight must return to the pre-probe watermark (scheduler
        #    entry + decode shadow reservation both released).
        cancel_err = None
        try:
            ops.cancel(rid, None)
        except Exception as exc:
            cancel_err = repr(exc)
        time.sleep(0.5)
        inflight_ok, inflight_detail = False, "no inflight view"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            view = ops.master_inflight()
            if view is not None:
                sched, pre, dec = _inflight_totals(view)
                if sched <= base_sched and pre <= base_prefill and dec <= base_decode:
                    inflight_ok = True
                    inflight_detail = (
                        f"back to pre-probe watermark "
                        f"(scheduler={sched}/{base_sched}, "
                        f"prefill_batches={pre}/{base_prefill}, "
                        f"decode_reservations={dec}/{base_decode})"
                    )
                    break
                inflight_detail = (
                    f"scheduler={sched} (base {base_sched}), "
                    f"prefill_batches={pre} (base {base_prefill}), "
                    f"decode_reservations={dec} (base {base_decode})"
                )
            time.sleep(0.5)

        # 4. Clear the pressure on every decode engine; recovery must be
        #    functional, not just cosmetic: a fresh request must schedule
        #    and complete again.
        for name in injected:
            try:
                ops.set_kv_pressure(name, 0)
            except Exception:
                pass
        time.sleep(2.0)  # master worker-status sync (recovery view)

        rid_rec = ops.next_request_id(base)
        rec_addr, rec_err = ops.run_one_request(
            rid_rec, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            parked_ok
            and not_delivered_ok
            and cancel_err is None
            and inflight_ok
            and rec_err is None
            and recovery_ok
        )
        return passed, (
            f"parked_pending={parked_ok} ({parked_detail}), "
            f"delivered_while_parked={delivered or 'none'}, "
            f"cancel_err={cancel_err}, "
            f"recovered_request_ok={rec_err is None}"
            f"(prefill={rec_addr}, err={rec_err}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for name in injected:
            try:
                ops.set_kv_pressure(name, 0)
            except Exception:
                pass
