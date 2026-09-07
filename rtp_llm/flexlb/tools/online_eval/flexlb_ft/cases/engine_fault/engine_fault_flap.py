from __future__ import annotations

import time
from typing import Optional

from ...context import CaseContext, rid_base
from ...harness import (
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    _BackgroundFlow,
    _cleanup_dynamic,
    _elastic_env,
    _run_batch,
    _wait_master_topology,
    http_get_status,
)
from ...registry import case
from ...support.engine_fault import MASTER_EVICT_S, _master_http


@case(
    "engine_fault_flap",
    category="engine_fault",
    profiles=["batch-window"],  # _elastic_env pins the legacy fault axes
    source="gap G2: rapid /stop_engine+/start_engine oscillation, 3-strike eviction vs re-discovery race",
)
def engine_flap(ctx: CaseContext):
    """Connection flapping: >=5 rapid stop/start cycles on one prefill.

    Exercises the race window between the master's 3-strike health eviction
    and the engine's re-discovery: each cycle stops prefill-0, holds it down
    long enough for the health poller (20ms interval) to accumulate strikes,
    then brings it back WITHOUT waiting for convergence (the flap).  A
    background flow keeps traffic live throughout.

    Assertions (user-mandated):
      * master stays healthy the whole time — HTTP 200 probe every cycle,
        no hang;
      * after the flapping stops: the engine is re-discovered
        (discovered == alive == initial topology), routing and requests
        recover (>=95% batch), and no inflight leaks (global drain to zero
        within the TTL_DRAIN_TIMEOUT_S cap that covers the 30s
        stale-inflight TTL plus the 60s ExpirationTimer sweep).

    The per-cycle alive count is observational evidence of the eviction vs
    re-discovery race: dipping below 2 means the 3-strike demotion landed,
    staying at 2 means recovery won the race — both are correct; the
    contract is only about final convergence and no leak.
    """
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "engine_fault")
    flow: Optional[_BackgroundFlow] = None
    try:
        _cleanup_dynamic(ops, env)

        flow = _BackgroundFlow(ops, base, interval_s=0.2)
        flow.start()
        time.sleep(1.0)  # let the flow ramp up before the first stop

        cycles = 6
        cycle_log: list[str] = []
        master_200_all = True
        evict_landings = 0
        for i in range(1, cycles + 1):
            ops.stop_engine("prefill-0")
            # Hold down ~0.8s: the health poller runs every 20ms, so the
            # 3-strike counter fires well inside this window, while the
            # EngineSyncRunner endpoint-eviction threshold (max(3*20ms, 1s)
            # from the last successful status) lands at the tail of the
            # window or right after the restart — exactly the race under
            # test.
            time.sleep(0.8)
            alive_mid = ops.master_alive_count("PREFILL")
            if alive_mid < 2:
                evict_landings += 1
            probe = http_get_status(
                f"{_master_http(ops)}/rtp_llm/inflight_status", timeout=5
            )
            if probe != 200:
                master_200_all = False
            ops.start_engine("prefill-0")
            time.sleep(0.4)  # short gap — flap, no convergence wait
            cycle_log.append(f"c{i}[alive={alive_mid}, master={probe}]")

        total, ok = flow.stop()
        rate = ok / total if total else 0.0

        # Post-flap convergence: full re-discovery of the flapped engine
        # (discovered count covers the eviction side of the race — see
        # elastic_rebalance for why alive alone is not a safe signal).
        topology_ok = _wait_master_topology(
            ops, "PREFILL", env.spec.n_prefill, MASTER_EVICT_S
        )
        # Routing/request recovery: 20 requests, >=95%.
        ok_batch, _, _ = _run_batch(ops, base, 20)
        recovery_ok = ok_batch >= 19
        # No inflight leak: global drain to zero (covers the 30s TTL plus
        # the 60s ExpirationTimer sweep — the legacy 90s cap sat
        # below the worst-phase settle and let residue poison the next
        # case on this shared env).
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )

        passed = (
            master_200_all
            and topology_ok
            and recovery_ok
            and inflight_ok
            and total > 0
            and rate >= 0.5  # availability floor — a total blackout must fail
        )
        return passed, (
            f"cycles={cycles}, evictions_landed={evict_landings}/{cycles}, "
            f"flap=[{'; '.join(cycle_log)}], "
            f"flow_success={ok}/{total}({rate:.0%}), "
            f"topology_converged={topology_ok}, "
            f"post_flap_batch={ok_batch}/20, "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if flow is not None:
            flow.stop()
        try:
            snap = ops.snapshot_by_name()
            if snap.get("prefill-0", {}).get("stopped"):
                ops.start_engine("prefill-0")
        except Exception:
            pass
