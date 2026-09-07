from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import TTL_DRAIN_TIMEOUT_S, AssertUtils
from ...registry import case
from ...support.kv import (
    KV_CACHE_SYNC_WAIT_S,
    STREAM_TIMEOUT_S,
    _drain_fired,
    _fire_request,
    _master_http,
    _poll_engine_pending,
    _prefill_names,
)


@case(
    "kv_prefix_stickiness",
    category="kv",
    source="scheduling_smoke.py S2+S5 (merged; M1 generalized)",
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
            # Best-effort residue drain (integration-round cascade hygiene): a drain-fallback cancel
            # that fails leaves slots settling on the stale-TTL +
            # ExpirationTimer path (worst ~90s) — the legacy 30s window
            # stopped short of it and the residue poisoned later cases on
            # this shared env.  Still not asserted (this finally is
            # hygiene, the case's own contract lives in its verdict).
            AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)
        except Exception:
            pass
