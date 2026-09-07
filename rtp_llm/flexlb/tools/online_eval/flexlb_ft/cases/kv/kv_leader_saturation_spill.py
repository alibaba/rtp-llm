from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.kv import (
    KV_CACHE_SYNC_WAIT_S,
    PREFIX_BLOCKS,
    PREFIX_INPUT_LEN,
    SPILL_BASELINE_HIT_BANDS,
    SPILL_BASELINE_WINDOWS,
    SPILL_FAST_MS,
    SPILL_FIRE_INTERVAL_S,
    SPILL_POOL_BLOCKS,
    SPILL_RECOVERY_HIT_BANDS,
    SPILL_RECOVERY_READY_RATE,
    SPILL_RECOVERY_WINDOWS,
    SPILL_REPLICATION_BANDS,
    SPILL_SATURATION_HIT_BANDS,
    SPILL_SATURATION_WINDOWS,
    SPILL_SLOW_MS,
    SPILL_WINDOW_REQS,
    STREAM_TIMEOUT_S,
    _contiguous_prefix_len,
    _drain_fired,
    _engine_cache_keys,
    _fam_keys,
    _fire_request,
    _kv_spec,
    _prefill_names,
    _wait_cache_sync,
)


@case(
    "kv_leader_saturation_spill",
    category="kv",
    source=(
        "kv family: leader saturation spill — queue-full affinity give-up "
        "cascades eviction onto the surviving engine (production observation)"
    ),
    expected_fail=True,
)
def kv_leader_saturation_spill(ctx: CaseContext):
    """[spill] A saturated cache leader spills its whole family onto the
    survivor — expected-fail probe of a production finding.

    FINDING (declared): sustained saturation of a cache-leader prefill
    engine spills the ENTIRE family traffic to the surviving engine;
    the survivor's LRU ping-pongs between the competing families,
    collapsing the global prefix hit rate for the whole saturation
    duration — no self-stabilization while the leader stays saturated.
    The saturated leader's own copy does NOT survive (remote data,
    FINDING-CONFIRMED): family A ends evicted from the whole fleet —
    both engines hold only family B (spill_share=0.50) — and the
    [A, A, B, B] symmetry cuts both ways: B#2 also OVER_CAP-spills
    onto the leader (B#1's ledger entry un-decayed), whose admit then
    capacity-evicts A's copy there, so A's copies are destroyed in
    BOTH directions during saturation.  Recovery is still fast in this
    minimal topology (observed recovery_window=1): one re-admit plus
    affinity re-stick rebuilds the placement once perf is restored.
    CAVEAT: under production multi-engine fan-out every receiver
    destroys copies the same way, so recovery is expected to be
    significantly slower there — the minimal topology only bounds the
    re-stick mechanics, not the fan-out recovery cost.

    Verified mechanism chain (design note, code-level; corrected after
    the first remote run): the master's TTFT projection is the DSV4
    token-formula prediction PLUS the in-flight committed predictions
    carried in the master's own work ledger — it is blind to the
    engine's /set_perf slowdown (WorkSnapshot sums COMMITTED /
    ENGINE_QUEUED predictions verbatim and decays ENGINE_RUNNING by
    wall clock from the formula value, which is decoupled from the
    real service time).  Saturation must therefore be created on the
    LEDGER, not the clock: the saturation phase fires back-to-back
    fire-and-forget (0.12s spacing) so every routing decision lands
    while the previous fire is still an ENGINE_QUEUED/RUNNING entry —
    the leader's committed sum lifts its projected TTFT past the
    affinity cutoff (best-candidate TTFT + maxExtraTtftMs = 20 on this
    line) -> the preferred set empties -> OVER_CAP -> the baseline tie
    window (max(0.1 * minTtft, 20ms)) excludes the leader ENTIRELY ->
    the whole family deterministically spills onto the surviving
    engine -> the survivor's admit evicts the resident family by
    capacity conservation -> the resident family's holders empty ->
    NO_CACHE_LEAD random -> still lands on the same survivor ->
    ping-pong.  First-run lesson (the serial await-each-request form):
    a clean ledger at every decision lets the cutoff track minTtft
    upward — affinity is then mathematically unbreakable, and the
    measured spill_share was 0.00 / hit rate 1.0.

    Construction: 2P+2D, each prefill pool 12 blocks vs two 10-block
    families (20 > 12 — a single engine cannot host both families while
    exactly one stable placement exists).  Four phases:
      1. steer — deterministic placement via per-engine /set_perf (slow
         P2 -> A lands P1 and admits; restore P2, slow P1 -> B lands P2
         and admits; restore P1), each step settled by the
         KV_SYNC_CONVERGENCE_S quiet window (the
         kv_capacity_conflict_overflow steer technique);
      2. baseline — both engines fast, interleaved A/B windows: near-full
         hit rate (each landing engine already holds >= 8 contiguous
         blocks pre-request — the storm hit-served caliber);
      3. saturation — P1 (A's leader) slowed to 3s (worst-case window
         depth: 4 serial services ~= 12s, inside the 30s per-rid drain
         wait with margin), P2 fast: each window fires [A, A, B, B]
         back-to-back fire-and-forget (0.12s spacing, drained per
         window); the chain above collapses the global hit rate for the
         whole phase.  The saturation band states the HEALTHY contract
         (bounded collapse) — the finding predicts it FAILS;
      4. recovery — P1 restored: one re-admit rebuilds A's placement
         (its copies were evicted in BOTH directions during saturation,
         so this phase measures the re-convergence window count, not a
         sticky re-attach), B re-sticks to P2; the steady-state (last
         1/3 of the recovery windows) hit rate returns to baseline; the
         recovery window count is observational.

    Assertions: P6 zero failures (hard); M3 hit-rate bands x3 (baseline /
    saturation / recovery steady-state — case overrides); P5 replication
    mean (case override) with the hard structural cap max <= n_prefill.
    Observations recorded in detail: saturation-phase spill share
    (A-family requests landing off their leader), holder flip count,
    recovery window count, per-phase per-engine cache-key digests.
    """
    env = ctx.env_manager.ensure(
        _kv_spec(ctx, "_spill", prefill_cache_blocks=SPILL_POOL_BLOCKS)
    )
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "kv")
    report = GradeReport(run_grade=ctx.grade)
    sat_fired, sat_handles = [], {}
    try:
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill workers"
        p1, p2 = names[0], names[1]
        fams = [_fam_keys(base, f) for f in range(2)]  # A = fam0, B = fam1

        def holder_mask(keys: list) -> dict:
            return {n: bool(set(keys) & _engine_cache_keys(ops, n)) for n in names}

        def snapshot_digest() -> dict:
            """Per-engine contiguous-run lengths of both families."""
            key_sets = {n: _engine_cache_keys(ops, n) for n in names}
            return {
                n: {f: _contiguous_prefix_len(key_sets[n], fams[f]) for f in range(2)}
                for n in names
            }

        failures = []
        digests = {}
        phase_hits = {"baseline": 0, "saturation": 0, "recovery": 0}
        phase_total = {"baseline": 0, "saturation": 0, "recovery": 0}
        recovery_windows = []  # per-window (hits, total)
        sat_a_total = 0
        sat_a_spilled = 0
        prev_hold = None
        flips = 0

        def run_window(phase: str) -> tuple:
            """One interleaved A,B,A,B serial window (baseline/recovery —
            storm calibers: pre-request contiguous-run hit view +
            end-of-window any-block flip mask)."""
            nonlocal prev_hold, flips
            hits = 0
            total = 0
            for i in range(SPILL_WINDOW_REQS):
                fam = i % 2
                keys = fams[fam]
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
                if err:
                    failures.append(f"{phase} rid={rid}: {err}")
                    continue
                total += 1
                landed = ops.addr_to_name().get(addr, addr)
                if runs.get(landed, 0) >= PREFIX_BLOCKS - 2:
                    hits += 1
            cur = {}
            for f in range(2):
                mask = holder_mask(fams[f])
                for n in names:
                    cur[(n, f)] = mask[n]
            if prev_hold is not None:
                flips += sum(1 for k in cur if cur[k] != prev_hold.get(k))
            prev_hold = cur
            return hits, total

        # -- phase 1: steer — deterministic placement (A@P1, B@P2).
        ops.set_perf(p2, prefill_fixed_ms=SPILL_SLOW_MS)
        time.sleep(1.5)  # master perf sync
        rid = ops.next_request_id(base)
        addr_a, err = ops.run_one_request(
            rid,
            input_len=PREFIX_INPUT_LEN,
            output_len=2,
            block_keys=fams[0],
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err:
            return False, f"steer A failed: {err}"
        if ops.addr_to_name().get(addr_a, addr_a) != p1:
            return False, f"steer A landed on {addr_a}, expected {p1}"
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after steer A"

        ops.set_perf(p2, prefill_fixed_ms=SPILL_FAST_MS)
        ops.set_perf(p1, prefill_fixed_ms=SPILL_SLOW_MS)
        time.sleep(1.5)  # master perf sync
        rid = ops.next_request_id(base)
        addr_b, err = ops.run_one_request(
            rid,
            input_len=PREFIX_INPUT_LEN,
            output_len=2,
            block_keys=fams[1],
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err:
            return False, f"steer B failed: {err}"
        if ops.addr_to_name().get(addr_b, addr_b) != p2:
            return False, f"steer B landed on {addr_b}, expected {p2}"
        ops.set_perf(p1, prefill_fixed_ms=SPILL_FAST_MS)
        time.sleep(1.5)  # perf restore sync
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after steer B"
        digests["steer"] = snapshot_digest()
        if not (
            digests["steer"][p1][0] >= PREFIX_BLOCKS
            and digests["steer"][p2][1] >= PREFIX_BLOCKS
        ):
            return False, f"steer placement incomplete: {digests['steer']}"

        # -- phase 2: baseline — both engines fast, interleaved windows.
        for _ in range(SPILL_BASELINE_WINDOWS):
            hits, total = run_window("baseline")
            phase_hits["baseline"] += hits
            phase_total["baseline"] += total
            time.sleep(KV_CACHE_SYNC_WAIT_S)
        digests["baseline"] = snapshot_digest()

        # -- phase 3: saturation — P1 (A's leader) slow, P2 fast; fires
        #    back-to-back so every routing decision lands while the
        #    previous fire is still a live in-flight ledger entry (the
        #    spill trigger — see the mechanism chain).
        ops.set_perf(p1, prefill_fixed_ms=SPILL_SLOW_MS)
        time.sleep(1.5)  # master perf sync
        for _ in range(SPILL_SATURATION_WINDOWS):
            # [A, A, B, B]: A first keeps the A-to-A spacing that
            # deterministically lifts the leader's committed projection
            # past the cutoff while the first A is still in flight.
            for i in range(SPILL_WINDOW_REQS):
                fam = 0 if i < 2 else 1
                keys = fams[fam]
                runs = {
                    n: _contiguous_prefix_len(_engine_cache_keys(ops, n), keys)
                    for n in names
                }
                rid = ops.next_request_id(base)
                landed, err = _fire_request(
                    ops,
                    rid,
                    sat_fired,
                    sat_handles,
                    input_len=PREFIX_INPUT_LEN,
                    output_len=2,
                    block_keys=keys,
                )
                if err:
                    failures.append(f"saturation rid={rid}: {err}")
                else:
                    phase_total["saturation"] += 1
                    if runs.get(landed or "", 0) >= PREFIX_BLOCKS - 2:
                        phase_hits["saturation"] += 1
                    if fam == 0:
                        sat_a_total += 1
                        if landed != p1:
                            sat_a_spilled += 1
                if i < SPILL_WINDOW_REQS - 1:
                    time.sleep(SPILL_FIRE_INTERVAL_S)
            for rid_done, _name, completed, err in _drain_fired(
                ops, sat_fired, sat_handles
            ):
                if not completed:
                    failures.append(
                        f"saturation rid={rid_done}: {err or 'no completion'}"
                    )
            sat_fired, sat_handles = [], {}
            cur = {}
            for f in range(2):
                mask = holder_mask(fams[f])
                for n in names:
                    cur[(n, f)] = mask[n]
            if prev_hold is not None:
                flips += sum(1 for k in cur if cur[k] != prev_hold.get(k))
            prev_hold = cur
        ops.set_perf(p1, prefill_fixed_ms=SPILL_FAST_MS)
        time.sleep(1.5)  # perf restore sync
        if not _wait_cache_sync(ops, names):
            return False, "cache sync never converged after saturation"
        digests["saturation"] = snapshot_digest()

        # -- phase 4: recovery — P1 restored, interleaved windows.
        for _ in range(SPILL_RECOVERY_WINDOWS):
            hits, total = run_window("recovery")
            phase_hits["recovery"] += hits
            phase_total["recovery"] += total
            recovery_windows.append((hits, total))
            time.sleep(KV_CACHE_SYNC_WAIT_S)
        digests["recovery"] = snapshot_digest()

        # -- metrics roll-up.
        base_rate = (
            phase_hits["baseline"] / phase_total["baseline"]
            if phase_total["baseline"]
            else 0.0
        )
        sat_rate = (
            phase_hits["saturation"] / phase_total["saturation"]
            if phase_total["saturation"]
            else 0.0
        )
        steady_n = max(1, SPILL_RECOVERY_WINDOWS // 3)  # last 1/3 windows
        steady_hits = sum(h for h, _ in recovery_windows[-steady_n:])
        steady_total = sum(t for _, t in recovery_windows[-steady_n:])
        steady_rate = steady_hits / steady_total if steady_total else 0.0
        recovery_at = next(
            (
                w + 1
                for w, (h, t) in enumerate(recovery_windows)
                if t and h / t >= SPILL_RECOVERY_READY_RATE
            ),
            None,
        )
        recovery_str = str(recovery_at) if recovery_at else f">{SPILL_RECOVERY_WINDOWS}"
        # Replication: end-state per-family holder count (the storm
        # any-block mask caliber) — mean band + hard structural cap.
        replication = {
            f: sum(1 for n in names if prev_hold.get((n, f))) for f in range(2)
        }
        max_replication = max(replication.values()) if replication else 0
        mean_replication = (
            sum(replication.values()) / len(replication) if replication else 0.0
        )
        spill_share = sat_a_spilled / sat_a_total if sat_a_total else 0.0
        max_repl_ok = max_replication <= len(names)

        report.invariant("P6", not failures, detail=f"failures={failures[:2]}")
        report.check(
            "M3",
            base_rate,
            context="spill_baseline_hit_rate",
            bands=SPILL_BASELINE_HIT_BANDS,
            detail=f"hits={phase_hits['baseline']}/{phase_total['baseline']}",
        )
        report.check(
            "M3",
            sat_rate,
            context="spill_saturation_hit_rate",
            bands=SPILL_SATURATION_HIT_BANDS,
            detail=(
                f"hits={phase_hits['saturation']}/"
                f"{phase_total['saturation']}, spill_share="
                f"{spill_share:.2f} ({sat_a_spilled}/{sat_a_total} "
                f"A-family requests off leader {p1})"
            ),
        )
        report.check(
            "M3",
            steady_rate,
            context="spill_recovery_steady_hit_rate",
            bands=SPILL_RECOVERY_HIT_BANDS,
            detail=(
                f"steady(last {steady_n} windows)={steady_hits}/"
                f"{steady_total}, recovery_window={recovery_str}"
            ),
        )
        report.check(
            "P5",
            mean_replication,
            context="spill_replication",
            bands=SPILL_REPLICATION_BANDS,
            detail=(
                f"replication={replication}, max={max_replication}/"
                f"{len(names)} (hard cap)"
            ),
        )
        passed, detail, rep = report.finish(f"grades: {report.summary()}")
        return (
            passed and max_repl_ok,
            f"baseline={base_rate:.2f}, saturation={sat_rate:.2f} "
            f"(spill_share={spill_share:.2f}), recovery_steady="
            f"{steady_rate:.2f} (recovery_window={recovery_str}), "
            f"flips={flips} (observational), "
            f"mean_replication={mean_replication:.2f} "
            f"(bands {SPILL_REPLICATION_BANDS}), max_replication="
            f"{max_replication}/{len(names)} (hard cap, ok={max_repl_ok}), "
            f"digests={digests}, {detail}",
            rep,
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if sat_fired:
            try:
                _drain_fired(ops, sat_fired, sat_handles)
            except Exception:
                pass
        for name in _prefill_names(ops):
            try:
                ops.set_perf(name, prefill_fixed_ms=SPILL_FAST_MS)
            except Exception:
                pass
