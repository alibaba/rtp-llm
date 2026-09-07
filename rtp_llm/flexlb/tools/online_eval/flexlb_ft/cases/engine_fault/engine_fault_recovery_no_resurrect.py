from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean, inject_type
from ...harness import (
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    _cleanup_dynamic,
    _wait_master_alive,
    wait_for,
)
from ...registry import case
from ...support.engine_fault import (
    CRASH_TRIGGER_WINDOW_S,
    RECOVERY_EVICT_S,
    RECOVERY_SETTLE_S,
    _consume_fired,
    _engine_ip_port,
    _ensure_started,
    _fire_inflight,
    _master_http,
    _master_log_offset,
    _prefill_endpoint_ledger,
    _recovery_env,
    _retire_count,
)


@case(
    "engine_fault_recovery_no_resurrect",
    category="engine_fault",
    profiles=["batch-window"],  # _recovery_spec pins the fault axes
    source="E3: pre-outage inflight requests must not resurrect after recovery",
)
def recovery_no_resurrect(ctx: CaseContext):
    """E3 — expected behaviour: requests that were in flight on an engine
    when it CRASHED must not leak into the recovered generation's
    bookkeeping:

      * the master's per-endpoint ledger for the recovered engines must
        read zero inflight (old entries fenced or TTL-settled), and the
        global inflight must drain to zero within the TTL cap;
      * the engine side must come back EMPTY — a true crash wipes the
        process memory, so after /start_engine the engine has no running
        tasks, no held blocks and an empty KV cache (recovery == a reboot
        from zero, not an in-place resume);
      * the pre-outage rids must never complete — their engine-side state
        is gone (no resurrection); they settle through the master's
        fence/TTL paths, and fresh traffic must schedule normally on the
        recovered engines.

    Mechanism: crash_after with TRUE-CRASH semantics — each target engine
    is armed to die on its NEXT EnqueueBatch (every queue, running task,
    KV lease and LRU entry is wiped, the gRPC port is killed), then
    trigger requests are fired until every target reports stopped.  The
    master observes the dead port (3 consecutive gRPC failures) and
    retires the endpoint; /start_engine rebuilds the gRPC server on clean
    state (the mock control-plane HTTP server survives the crash, which
    is what makes the restart path reachable).

    FINDING if the asserted master-side bars fail: F1 permanent-ledger-leak
    elastic variant — old generation requests surviving into the new one's
    ledger.
    """
    env, ops = _recovery_env(ctx)
    base = rid_base(ctx, "engine_fault")
    names = ["prefill-0", "prefill-1"]
    try:
        _cleanup_dynamic(ops, env)
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)

        # Widen the in-flight window so the outage lands mid-execution:
        # slow prefills keep requests waiting/running on the engines while
        # we take them down.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=2000.0)
        time.sleep(1.5)  # master perf sync

        fired = _fire_inflight(ops, base, 8, input_len=512, output_len=2)
        if len(fired) < 4:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
            return False, f"only {len(fired)}/8 requests fired successfully"
        time.sleep(0.5)  # let the batcher dispatch them onto the engines

        targets = sorted({name for _rid, _resp, name in fired})
        target_ips = {n: _engine_ip_port(ops, n) for n in targets}
        log_offset = _master_log_offset(env)

        # Crash: arm crash_after at each target's NEXT EnqueueBatch (the
        # counter already includes the fired requests' batches), then fire
        # trigger requests until every target reports stopped — the crash
        # only fires when a fresh EnqueueBatch lands on the armed engine.
        # The trigger requests are small and unconsumed; ones that land on
        # an armed target become the crash-triggering empty ack (uncertain
        # fence, TTL-settled by the inflight bar below).
        for n in targets:
            snap = ops.snapshot_by_name().get(n, {})
            n_batches = int(snap.get("rpc_counts", {}).get("enqueue_batch", 0))
            inject_type(ops, n, "crash_after", n=n_batches + 1)
        crashed_all = False
        deadline = time.monotonic() + CRASH_TRIGGER_WINDOW_S
        while time.monotonic() < deadline:
            snaps = ops.snapshot_by_name()
            if all(snaps.get(n, {}).get("stopped") for n in targets):
                crashed_all = True
                break
            try:
                ops.schedule(ops.next_request_id(base), input_len=64, output_len=2)
            except Exception:
                pass  # a trigger may hit an engine mid-crash; keep firing
            time.sleep(0.2)

        # The master observes the dead ports and retires the endpoints
        # (same 3-strike transport path as stop_engine).
        retired_all = True
        for n, ip in target_ips.items():
            if not wait_for(
                lambda ip=ip: _retire_count(env, ip, log_offset) > 0,
                RECOVERY_EVICT_S,
                0.2,
            ):
                retired_all = False

        # Recovery: /start_engine rebuilds the gRPC server on CLEAN state
        # (it also disarms the fault config and resets the enqueue count).
        for n in targets:
            ops.start_engine(n)
        alive_back = _wait_master_alive(
            ops, "PREFILL", env.spec.n_prefill, RECOVERY_EVICT_S
        )
        time.sleep(RECOVERY_SETTLE_S)

        # The recovered generation's ledger must start from zero.
        ledger_clean = True
        ledger_detail = {}
        for n, ip in target_ips.items():
            ledger = _prefill_endpoint_ledger(ops, ip)
            zero = bool(
                ledger
                and int(ledger.get("inflight_requests", -1)) == 0
                and int(ledger.get("inflight_batches", -1)) == 0
            )
            ledger_clean = ledger_clean and zero
            ledger_detail[n] = ledger

        # True-crash wipe: the recovered engine must have NO memory of the
        # pre-crash world — no running tasks, no held blocks, an empty LRU,
        # zero inflight and a zeroed accept counter (a fresh process).
        wipe_ok = True
        wipe_detail = {}
        for n in targets:
            snap = ops.snapshot_by_name().get(n, {})
            clean = (
                int(snap.get("running", -1)) == 0
                and int(snap.get("inflight", -1)) == 0
                and list(snap.get("cache_key_set") or []) == []
                and int(snap.get("held_blocks", -1)) == 0
                and int(snap.get("accepted", -1)) == 0
            )
            wipe_ok = wipe_ok and clean
            wipe_detail[n] = {
                "running": snap.get("running"),
                "inflight": snap.get("inflight"),
                "cache_keys": len(snap.get("cache_key_set") or []),
                "held_blocks": snap.get("held_blocks"),
                "accepted": snap.get("accepted"),
            }

        # Consume the fired requests to terminal states: with the true
        # crash their engine-side state is GONE, so NONE may complete — a
        # completion would be a resurrection (asserted, no longer just an
        # observation).  Their client streams fail against the wiped
        # engine and settle through the master's fence/TTL.
        outcomes = _consume_fired(ops, fired, wait_s=2.0)
        resurrected = [
            (rid, name)
            for rid, name, completed in outcomes
            if completed and name in targets
        ]

        # Engine side must not keep the old rids registered.
        engine_clean, engine_detail = engine_inflight_clean(ops, targets)

        # Master global ledger drains within the TTL cap (fence or TTL).
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )

        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            crashed_all
            and retired_all
            and alive_back
            and ledger_clean
            and wipe_ok
            and not resurrected
            and engine_clean
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={len(fired)} onto {targets}, "
            f"crashed_all={crashed_all}, retired_all={retired_all}, "
            f"alive_restored={alive_back}, "
            f"recovered_ledger_zero={ledger_clean}({ledger_detail}), "
            f"engine_wipe_clean={wipe_ok}({wipe_detail}), "
            f"resurrected={len(resurrected)}/{len(fired)} (bar: 0), "
            f"engine_inflight_clean={engine_clean}({engine_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        # Disarm any undelivered crash_after before the hygiene restart
        # (start_engine would clear it too, but the arm must not survive
        # an early-exit path that skips the restart).
        for n in names:
            try:
                inject_type(ops, n, "crash_after", enabled=False)
            except Exception:
                pass
        _ensure_started(ops, names)
        for n in names:
            try:
                ops.set_perf(n, prefill_fixed_ms=100.0)
            except Exception:
                pass
