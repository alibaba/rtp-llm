"""KV-cache event / storm / capacity-conflict cases (task #84 family).

Ten cases around the KV prefix-cache lifecycle as the master sees it:
per-engine ledger isolation, eviction events, shared (global-index)
blocks, sync convergence, engine-down cleanup, hot-prefix churn storms
and the affinity-vs-capacity conflict.  Grouped per the researcher spec:

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

DORMANT BY DESIGN: this module is NOT imported by flexlb_functional_tests
yet — the runner wiring lands with the suite reorg (task #85).  Two mock
capabilities it depends on do not exist yet and are being added by a
parallel agent (the TODO markers below are the alignment contract):

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
from typing import Optional

from .context import CaseContext, CaseDef, rid_base
from .grade import GradeReport
from .harness import (
    DEFAULT_PREFILL_CACHE_BLOCKS,
    EnvSpec,
    default_perf,
    http_get_json,
    http_post_json,
    wait_for,
)

KV_CACHE_CASES: list[CaseDef] = []

STREAM_TIMEOUT_S = 15.0
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


def case(
    name: str, profiles=None, requires=None, source: str = "", suite: str = "smoke"
):
    """Register into KV_CACHE_CASES; *suite* drives the runner grouping
    (routing-semantics cases -> smoke, the churn storm -> chaos),
    following the injection_gate_cases precedent."""

    def deco(fn):
        KV_CACHE_CASES.append(
            CaseDef(
                name=name,
                suite=suite,
                fn=fn,
                profiles=profiles,
                requires=requires,
                source=source,
            )
        )
        return fn

    return deco


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
    this module stays dormant (not imported by the runner), so the raise
    is a loud alignment failure rather than a broken suite.
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


def _wait_cache_sync(ops, engine_names: list, timeout_s: float = 15.0) -> bool:
    """Wait for the mock-side cache_version to go QUIET.

    Polls every 0.5s and requires KV_SYNC_CONVERGENCE_S (>= 3.5s) with
    no cache_version change on every named engine — after that window
    the master's cache-status poll (its own ~1-2s period) has necessarily
    observed the final state.  This is the spec's ">= 3.5s quiet or
    cache_version polling" convergence caliber.
    """
    deadline = time.monotonic() + timeout_s
    last_versions = {n: None for n in engine_names}
    last_change = {n: time.monotonic() for n in engine_names}
    while time.monotonic() < deadline:
        snap = ops.snapshot_by_name()
        now = time.monotonic()
        quiet = True
        for n in engine_names:
            version = snap.get(n, {}).get("cache_version")
            if version != last_versions[n]:
                last_versions[n] = version
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
    prefill engines (the aff_prefix_stickiness seeding technique): the
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


# ===========================================================================
# Per-engine cases
# ===========================================================================


@case(
    "kv_pe_admit_isolation",
    source="kv family: per-engine cache ledger isolation (task #84)",
)
def kv_pe_admit_isolation(ctx: CaseContext):
    """[per-engine] A's admissions never widen B's key set.

    Scenario: ledger-separated seeding (the aff_prefix_stickiness
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
        fired, fired_handles = [], []
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
        fired, fired_handles = [], []
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


@case(
    "kv_storm_hot_churn",
    suite="chaos",
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
    holder IS the hot engine (the bal_overload_avoid_prefill seed
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
    verified in isolation (bal_overload_avoid_prefill, the aff_*
    cases) but were never exercised in the same frame; a failure is a
    finding.
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
        fired, fired_handles = [], []

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
