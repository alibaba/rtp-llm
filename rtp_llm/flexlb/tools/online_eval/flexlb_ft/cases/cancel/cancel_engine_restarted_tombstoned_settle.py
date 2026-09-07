from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import _fence_residue_stable, engine_inflight_clean
from ...harness import wait_for
from ...registry import case
from ...support.cancel import (
    CANCEL_SETTLE_BOUND_S,
    _all_engine_names,
    _cancel_rpc_total,
    _crash_and_restart,
    _direct_enqueue,
    _fence_rejected_8429,
    _ha_env,
    _restore_engines,
)


@case(
    "cancel_engine_restarted_tombstoned_settle",
    category="cancel",
    profiles=["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"],
)
def cancel_engine_restarted_tombstoned_settle(ctx: CaseContext):
    """Engine restart + pre-restart cancel: TOMBSTONED settles immediately.

    Scenario (BATCH): R1 is handed to decode (first output received, so
    the slot lives on the decode side and survives the prefill generation
    retire — prefill retire never closes decode-owned slots) when its
    original prefill TRUE-CRASHES (crash_after: memory wipe + port kill)
    and is restarted.  The master's cancel for R1 then reaches the FRESH
    instance, which has never seen the rid: the three-branch contract
    answers TOMBSTONED and installs the ABSENT_FENCE tombstone.

    Expected (contract, EngineFenceCoordinator ground truth):
      * resumeTombstoned settles the slot IMMEDIATELY — the client stream
        closes as a typed cancelled well inside 5s, never via the 95s TTL
        drain net (settle-latency bound asserted);
      * the master really sent the cancel: the engine's Cancel RPC counter
        increases by >= 1;
      * the installed fence rejects a DIRECT late Enqueue of the same rid
        with the typed 8429, pre-admission (no success ack, no engine
        state, no inflight residue).  A master-routed re-schedule of the
        settled rid cannot be probed here — the master answers from its
        already-settled ledger (the documented no-resurrection semantics
        pinned by cancel_fencing_lost_on_engine_restart instead);
      * the decode leg finishes its bounded orphan computation, the
        engines report inflight 0 with no leak, the master ledger keeps
        only the sacrificial crash trigger's bounded uncertain residue
        (non-growing), and a follow-up request completes normally.
    """
    ops, _ = _ha_env(ctx, "restart")
    base = rid_base(ctx, "cancel")
    handle = None
    try:
        rid = ops.next_request_id(base)
        # output_len=5000 (~38s of decode at the production-fit step
        # pricing) outlives the whole crash/restart cycle, so the slot is
        # still inflight (decode-owned) when the cancel fires.
        response = ops.schedule(rid, output_len=5000)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None
            if response.enqueued_by_master
            else ops.build_generate_input(rid, output_len=5000)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            return False, "no output before the crash window"

        dropped, restored = _crash_and_restart(ops, "prefill-0")
        if not (dropped and restored):
            return False, (
                f"crash/restart failed: dropped={dropped}, " f"restored={restored}"
            )

        baseline_cancel = _cancel_rpc_total(ops)
        settle_t0 = time.monotonic()
        # Response only under BATCH (its batch_id rides the master Cancel);
        # a NON_BATCH response would add the worker_cancel direct connect
        # — a second engine-side Cancel and a dead-port RpcError risk.
        ops.cancel(rid, response if response.enqueued_by_master else None)
        settled_fast = handle.wait_end(CANCEL_SETTLE_BOUND_S)
        settle_latency = time.monotonic() - settle_t0
        # The master settles the slot locally within milliseconds, but the
        # engine-side Cancel forward registers on the census asynchronously
        # (observed ~8-10s under load) — poll the snapshot until the Cancel
        # RPC count grows instead of sampling a stale value (the coordinator
        # is one-shot, so the counter moves exactly once, late).  The poll
        # must also land BEFORE the fence probe below: the tombstone is
        # installed engine-side only when the Cancel RPC is processed.
        cancel_reached = wait_for(
            lambda: _cancel_rpc_total(ops) > baseline_cancel, 15.0, 0.5
        )
        cancel_delta = _cancel_rpc_total(ops) - baseline_cancel

        # The ABSENT_FENCE tombstone from the TOMBSTONED cancel rejects a
        # direct late Enqueue of the same rid with the typed 8429.
        fence_ok, fence_detail = False, "no probe"
        try:
            probe = ops.build_generate_input(rid, output_len=2)
            ops._copy_role_addrs(probe, response)
            ack = _direct_enqueue(ops, ops.prefill_addr(response), probe, rid * 10 + 1)
            fence_ok, fence_detail = _fence_rejected_8429(ack, rid)
        except Exception as exc:
            fence_detail = repr(exc)

        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 45.0
        )
        residue_ok, residue_detail = _fence_residue_stable(ops, 1)
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = (
            settled_fast
            and not handle.snap.completed
            and cancel_reached
            and cancel_delta >= 1
            and fence_ok
            and engine_clean
            and residue_ok
            and recovery_ok
        )
        return passed, (
            f"tombstoned_settle: settled_fast={settled_fast}"
            f"({settle_latency:.3f}s <= {CANCEL_SETTLE_BOUND_S:.0f}s, "
            f"completed={handle.snap.completed}), "
            f"cancel_rpc_delta={cancel_delta} (>=1), "
            f"fence_8429={fence_ok}({fence_detail}), "
            f"engine_clean={engine_clean}({engine_detail}), "
            f"master_residue={residue_ok}({residue_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if handle is not None:
            handle.cancel()
        _restore_engines(ops)
