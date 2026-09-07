from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import inject_type
from ...harness import (
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    _cleanup_dynamic,
    _wait_master_alive,
    wait_for,
)
from ...registry import case
from ...support.engine_fault import (
    E5_GAP_S,
    RECOVERY_EVICT_S,
    RECOVERY_SETTLE_S,
    _consume_fired,
    _created_generation_count,
    _engine_ip_port,
    _ensure_started,
    _fire_inflight,
    _master_http,
    _master_log_offset,
    _recovery_env,
    _retire_count,
)


@case(
    "engine_fault_status_gap_long_retire",
    category="engine_fault",
    profiles=["batch-window"],  # _recovery_spec pins the fault axes
    source="E5: a long status gap must retire the generation and fence its ledger",
)
def status_gap_long_retire(ctx: CaseContext):
    """E5 — expected behaviour: a status gap LONGER than the retire
    threshold (5s of no_respond ≈ 4+ timed-out polls > 3 consecutive
    failures) is a crash — the master must actively retire the endpoint's
    generation and FENCE its queue/ledger/inflight:

      * the retire landed ("marked dead after 3 consecutive gRPC
        failures") and a fresh generation is created once reporting
        resumes;
      * the fenced engine's master ledger/inflight settles to zero within
        the TTL cap (fence or stale-TTL — an entry that never settles is
        the F7 pending-drain gap);
      * once reporting resumes, the new generation serves fresh traffic.

    In-flight requests fired BEFORE the gap are the fence payload: their
    engine-side execution completes but the master's status channel is
    dead, so their ledger release must come from the retire fence (or the
    stale-TTL), never from a stale post-recovery resurrection.

    FINDING if it fails: no retire on a long gap, or an unfenced ledger
    that neither the retire nor the TTL ever clears.
    """
    env, ops = _recovery_env(ctx)
    base = rid_base(ctx, "engine_fault")
    names = ["prefill-0", "prefill-1"]
    try:
        _cleanup_dynamic(ops, env)
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)

        ip = _engine_ip_port(ops, "prefill-0")
        log_offset = _master_log_offset(env)
        created_before = _created_generation_count(env, ip, log_offset)

        # Fence payload: slow requests in flight on the engines when the
        # status channel dies.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=2000.0)
        time.sleep(1.5)
        fired = _fire_inflight(ops, base, 8, input_len=512, output_len=2)
        time.sleep(0.5)

        # Long gap: 5s of no_respond ≈ 4+ consecutive timed-out status
        # polls — past the 3-failure retire threshold.
        inject_type(ops, "prefill-0", "status_no_respond", enabled=True)
        retired = wait_for(
            lambda: _retire_count(env, ip, log_offset) > 0, E5_GAP_S + 10.0, 0.2
        )
        # Hold the gap a little past the retire so the retire/re-create
        # cycle is observable, then resume reporting.
        time.sleep(1.0)
        inject_type(ops, "prefill-0", "status_no_respond", enabled=False)

        alive_back = _wait_master_alive(
            ops, "PREFILL", env.spec.n_prefill, RECOVERY_EVICT_S
        )
        time.sleep(RECOVERY_SETTLE_S)
        created_after = _created_generation_count(env, ip, log_offset)
        generation_bumped = created_after > created_before

        # Consume the fence payload to terminal states.
        outcomes = _consume_fired(ops, fired, wait_s=5.0)

        # Fenced ledger/inflight must settle within the TTL cap.
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )

        # New generation serves fresh traffic.
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            retired and alive_back and generation_bumped and inflight_ok and recovery_ok
        )
        return passed, (
            f"ip={ip}, created_generations={created_before}->{created_after}, "
            f"retired={retired}, alive_restored={alive_back}, "
            f"fired={len(fired)}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            inject_type(ops, "prefill-0", "status_no_respond", enabled=False)
        except Exception:
            pass
        _ensure_started(ops, names)
        for n in names:
            try:
                ops.set_perf(n, prefill_fixed_ms=100.0)
            except Exception:
                pass
