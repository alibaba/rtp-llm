from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import _fence_residue_stable, engine_inflight_clean
from ...harness import wait_for, wait_for_port
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
    "cancel_fencing_lost_on_engine_restart",
    category="cancel",
    requires=["enqueue_batch"],  # mock crash_after is an EnqueueBatch hook
)
def cancel_fencing_lost_on_engine_restart(ctx: CaseContext):
    """Design boundary: fencing is engine memory — a second crash drops it.

    Scenario (BATCH): stage 1 replays the tombstoned-settle contract — R1
    (decode-owned, decode still running) survives its prefill's TRUE
    crash + restart; the master cancel answers TOMBSTONED on the fresh
    instance (never-seen rid), installs the ABSENT_FENCE tombstone (the
    direct late-Enqueue probe is 8429-rejected — the fence WORKS at this
    point) and settles the slot.  Stage 2 crashes the prefill AGAIN: the
    tombstone lived only in engine memory, so the second restart comes
    up fence-less and the SAME late Enqueue is now ACCEPTED by the fresh
    instance.

    Expected (contract — a DOCUMENTED DESIGN TRADE-OFF, not a bug-fix
    expectation): engine-side fencing is memory-only with no persistence.
    The master ledger has already settled the rid, so nothing resurrects
    master-side, and the orphan computation the fresh instance now runs
    is bounded by its own execution.  Assertions:
      (a) stage-1 fence rejects the probe with the typed 8429 (control);
      (b) after the second crash+restart the same probe is ADMITTED
          (>= 1 success, no 8429) — the trade-off made executable;
      (c) the master ledger does NOT resurrect the settled rid: the
          residue stays bounded at the two sacrificial crash triggers'
          uncertain entries and never grows while the orphan completes
          and its terminal is (correctly) reconciled as a no-op;
      (d) the orphan computation is bounded — every engine reports
          inflight 0 with no leak; a follow-up request completes.
    """
    ops, _ = _ha_env(ctx, "fence_lost")
    base = rid_base(ctx, "cancel")
    handle = None
    try:
        rid = ops.next_request_id(base)
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
            return False, "no output before the first crash window"

        # Stage 1: crash + restart; the cancel lands TOMBSTONED on the
        # fresh instance and settles the slot (typed cancelled, fast).
        dropped1, restored1 = _crash_and_restart(ops, "prefill-0")
        if not (dropped1 and restored1):
            return False, (f"first crash/restart failed: {dropped1}/{restored1}")
        baseline_cancel = _cancel_rpc_total(ops)
        settle_t0 = time.monotonic()
        # Response only under BATCH (its batch_id rides the master Cancel);
        # a NON_BATCH response would add the worker_cancel direct connect
        # — a second engine-side Cancel and a dead-port RpcError risk.
        ops.cancel(rid, response if response.enqueued_by_master else None)
        settled_fast = handle.wait_end(CANCEL_SETTLE_BOUND_S)
        settle_latency = time.monotonic() - settle_t0
        # The ABSENT_FENCE tombstone is installed engine-side only when the
        # Cancel RPC is PROCESSED — which the census records asynchronously
        # (observed ~8-10s under load) — so wait for the RPC to register
        # BEFORE probing the fence: a probe racing ahead of the tombstone
        # would be admitted and masquerade as a lost fence.
        cancel_reached = wait_for(
            lambda: _cancel_rpc_total(ops) > baseline_cancel, 15.0, 0.5
        )

        # Control: the ABSENT_FENCE tombstone IS armed — a direct probe of
        # the settled rid is 8429-rejected.
        fence_armed = False
        fence_detail = "no probe"
        try:
            probe = ops.build_generate_input(rid, output_len=100)
            ops._copy_role_addrs(probe, response)
            ack = _direct_enqueue(ops, ops.prefill_addr(response), probe, rid * 10 + 1)
            fence_armed, fence_detail = _fence_rejected_8429(ack, rid)
        except Exception as exc:
            fence_detail = repr(exc)

        # Stage 2: crash AGAIN — the tombstone dies with the memory.
        dropped2, restored2 = _crash_and_restart(ops, "prefill-0")
        if not (dropped2 and restored2):
            return False, (f"second crash/restart failed: {dropped2}/{restored2}")

        # The trade-off, executable: the same probe is now ADMITTED.
        orphan_accepted = False
        orphan_detail = "no probe"
        try:
            probe2 = ops.build_generate_input(rid, output_len=100)
            ops._copy_role_addrs(probe2, response)
            # Transport readiness: the SECOND crash killed the engine's
            # gRPC listener and the fresh bind lags the master's alive view
            # by a few seconds (observed _InactiveRpcError/UNAVAILABLE
            # "Socket closed" on an immediate probe).  Stage 1 needs no
            # such gate — its census poll already proves the engine's gRPC
            # server is processing RPCs.  Wait for the port here; single
            # probe, no retry (a re-sent rid could be admitted twice).
            probe2_addr = ops.prefill_addr(response)
            probe2_host, _, probe2_port = probe2_addr.rpartition(":")
            wait_for_port(probe2_host, int(probe2_port), 10.0)
            ack2 = _direct_enqueue(ops, probe2_addr, probe2, rid * 10 + 2)
            orphan_accepted = len(ack2.successes) >= 1
            orphan_detail = (
                f"successes={len(ack2.successes)}, errors="
                f"{[(e.request_id, e.error_info.error_code) for e in ack2.errors]}"
            )
        except Exception as exc:
            orphan_detail = repr(exc)

        # The orphan (output_len=100) and the surviving decode leg are
        # both bounded; 60s covers the worst case with margin.
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 60.0
        )
        # Master ledger stays settled: bounded at the two sacrificial
        # uncertain entries, never growing (the orphan's terminal is
        # reconciled as a no-op — no resurrection).
        residue_ok, residue_detail = _fence_residue_stable(ops, 2)
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = (
            settled_fast
            and not handle.snap.completed
            and cancel_reached
            and fence_armed
            and orphan_accepted
            and engine_clean
            and residue_ok
            and recovery_ok
        )
        return passed, (
            f"fencing_lost: settled_fast={settled_fast}"
            f"({settle_latency:.3f}s), "
            f"stage1_fence_8429={fence_armed}({fence_detail}), "
            f"stage2_orphan_accepted={orphan_accepted}({orphan_detail}) "
            "[documented design trade-off: engine-memory-only fencing], "
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
