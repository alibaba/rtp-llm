from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import (
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    _cleanup_dynamic,
    _elastic_env,
    _run_batch,
    _ttft_p50,
    http_get_status,
    wait_for,
)
from ...registry import case
from ...support.engine_fault import MASTER_EVICT_S, _master_http


@case(
    "engine_fault_down_phases",
    category="engine_fault",
    profiles=["batch-window"],  # _elastic_env pins the legacy fault axes
    source="flexlb_behavior_test.sh S2/S4 merged — five-phase engine-down assertion set",
)
def engine_down_http_stop_prefill(ctx: CaseContext):
    """Five phases with the uniform engine-down assertion set (core 3 of 7):
    master stays up + surviving engines take over + recovery rate.
    """
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "engine_fault")
    try:
        _cleanup_dynamic(ops, env)

        # Integration-round cascade hygiene: residues from the
        # preceding elastic cases on this shared env settle via the
        # stale-TTL + ExpirationTimer path (worst ~90s).  Drain them
        # BEFORE the baseline batch so an earlier case's TTL settle cannot
        # fail this case's Phase-1 gate (best-effort — a true leak is
        # caught by the case's own end-of-run drain assertion below).
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)

        def master_up() -> bool:
            return (
                http_get_status(
                    f"{_master_http(ops)}/rtp_llm/inflight_status", timeout=5
                )
                == 200
            )

        # Phase 1 — baseline: 20 requests, all succeed.  TTFT p50 is
        # recorded for the post-recovery regression gate (Phase 5).
        ok1, err1, _ = _run_batch(ops, base, 20, collect_ttft=True)
        base_ttft_p50 = _ttft_p50(getattr(_run_batch, "last_ttfts", []))
        master_ok1 = master_up()
        if err1:
            return False, f"baseline had {err1} errors (master_up={master_ok1})"

        # Phase 2 — http-stop prefill-0.
        ops.stop_engine("prefill-0")
        # Phase 3 — downtime: wait for the 3-consecutive-failure eviction,
        # then 20 more requests must still succeed (2P redundancy).
        evicted = wait_for(
            lambda: ops.master_alive_count("PREFILL") <= 1,
            MASTER_EVICT_S,
            0.5,
        )
        ok2, err2, _ = _run_batch(ops, base, 20)
        master_ok2 = master_up()
        rate2 = ok2 / 20 if err2 == 0 else ok2 / 20
        takeover_ok = err2 <= 2  # ≥90% success while one prefill is down
        downtime_err_types = list(getattr(_run_batch, "last_error_types", []))[:3]

        # Phase 4 — restart the engine and wait for re-discovery.
        ops.start_engine("prefill-0")
        alive_back = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= 2,
            MASTER_EVICT_S,
            0.5,
        )
        # Channel recovery settle (S2's reconnect window).
        time.sleep(3.0)

        # Phase 5 — recovery: 20 requests ≥95%, and TTFT must fall back to
        # within 1.5x of the baseline p50 (master_recovery_ttft_test.sh
        # semantics: once the fault heals, TTFT returns to baseline — the
        # legacy analyzer tolerates an early 1.5x spike and only degrades
        # the verdict when the *stable* window stays above 1.2x; this batch
        # sits past the 3s channel-settle window, so the 1.5x gate is the
        # conservative bound on steady-state recovery).
        ok5, err5, _ = _run_batch(ops, base, 20, collect_ttft=True)
        recovery_ttft_p50 = _ttft_p50(getattr(_run_batch, "last_ttfts", []))
        ttft_ok, ttft_detail = AssertUtils.ttft_degradation(
            base_ttft_p50, recovery_ttft_p50, threshold_pct=50.0
        )
        master_ok5 = master_up()
        recovery_ok = ok5 >= 19  # ≥95%

        # Drain guard: the tolerated Phase-3 failures (err2<=2,
        # requests routed onto the stopped engine) settle through the
        # TTL path — wait for the worst-case window so this case does not
        # leak its residue into engine_fault_flap/master_kill on the same
        # env; a slot that never settles still FAILs this case here.
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )

        passed = (
            master_ok1
            and master_ok2
            and master_ok5
            and evicted
            and takeover_ok
            and alive_back
            and recovery_ok
            and ttft_ok
            and inflight_ok
        )
        return passed, (
            f"baseline=20/20, evicted_after_stop={evicted}"
            f"(alive={ops.master_alive_count('PREFILL')}), "
            f"downtime={ok2}/20({rate2:.0%}, err_types={downtime_err_types}), "
            f"alive_restored={alive_back}, "
            f"recovery={ok5}/20({ok5 / 20:.0%}), "
            f"ttft_recovery=[{ttft_detail}], "
            f"master_up=(p1:{master_ok1}, p3:{master_ok2}, p5:{master_ok5}), "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            snap = ops.snapshot_by_name()
            if snap.get("prefill-0", {}).get("stopped"):
                ops.start_engine("prefill-0")
        except Exception:
            pass
