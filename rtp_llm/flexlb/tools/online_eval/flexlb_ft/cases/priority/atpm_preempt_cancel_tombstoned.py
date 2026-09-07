from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import _fence_residue_stable, engine_inflight_clean
from ...grade import GradeReport
from ...harness import wait_for
from ...registry import case
from ...support.priority import (
    CODE_OK,
    _all_engine_names,
    _cancel_rpc_total,
    _crash_and_restart,
    _direct_enqueue,
    _fence_rejected_8429,
    _restore_engines,
    _ts_spec,
)


@case(
    "atpm_preempt_cancel_tombstoned",
    category="priority",
    profiles=["single-batch"],
    requires=["enqueue_batch"],
    source="preemption-stages audit (2026-09) — Cancel TOMBSTONED branch",
)
def atpm_preempt_cancel_tombstoned(ctx: CaseContext):
    """Preemption-chain Cancel TOMBSTONED branch: the victim's original
    prefill TRUE-CRASHED and restarted (memory wipe) before the
    preemption Cancel fires, so the FRESH instance has never seen the
    rid — the engine answers TOMBSTONED, installs the ABSENT_FENCE
    tombstone, and the master's coordinator settles the PREEMPTION
    through it (resumeTombstoned → committed: the incoming wins the
    freed slot and completes).

    Consumer-side difference vs the client-chain twin
    (cancel.py cancel_engine_restarted_tombstoned_settle): there the
    trigger is a CLIENT cancel settling one stream; here the trigger is
    the ADMISSION preemption and what must close is the preemption
    state machine — the eviction is COMMITTED by the tombstone, the
    victim's slot is freed for the incoming, and the victim itself
    settles at the master as a priority terminal.

    Choreography (BATCH — crash_after only arms on the EnqueueBatch
    handler): victim P30 (input=512, output=5000 ≈ 38s of decode,
    outliving the whole crash/restart cycle) schedules, dispatches and
    hands off to decode (first output received — the slot lives
    decode-side and survives the prefill generation retire); prefill-0
    then true-crashes (crash_after n=1 via a sacrificial request — the
    victim's FetchResponse stream, whose pump thread lives in the
    engine JVM, is cut by the crash) and restarts as a FRESH instance
    (same address, empty memory).  The P70 incoming then fires: the
    master view still has the victim RUNNING on decode (decode-0 is
    healthy and reporting) → decode placement BLOCKED
    (maxEngineRequests=1) → DECODE_ENGINE_OWNED eviction → the
    tokenized Cancel reaches the FRESH prefill → never-seen branch →
    TOMBSTONED + ABSENT_FENCE.

    Contract:
      * the incoming completes 200 with real FetchResponse output
        (tombstone-settled preemption);
      * the victim's stream is TERMINATED not completed (cut by the
        crash) and the victim's engine-side decode leg finishes its
        bounded orphan computation (engine inflight drains);
      * the Cancel really reached the fresh instance (post-restart
        Cancel RPC counter delta >= 1);
      * the installed ABSENT_FENCE rejects a DIRECT late Enqueue of
        the victim rid with exactly the typed 8429 (pre-admission);
      * the sacrificial crash request leaves only a bounded,
        non-growing master residue; recovery works.
    """
    env = ctx.env_manager.ensure(_ts_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    victim_handle = None
    inc_handle = None
    try:
        victim = ops.next_request_id(base)
        victim_resp = ops.schedule(
            victim, priority=30, input_len=512, output_len=5000, timeout_s=90.0
        )
        if victim_resp.code != CODE_OK or not victim_resp.success:
            return False, f"victim schedule failed: {victim_resp.error_message}"
        victim_handle = ops.start_stream(victim_resp, victim)
        if not victim_handle.wait_first_output():
            return False, "no output before the crash window"

        dropped, restored = _crash_and_restart(ops, "prefill-0")
        if not (dropped and restored):
            return False, (
                f"crash/restart failed: dropped={dropped}, " f"restored={restored}"
            )
        # The crash cut the victim's FetchResponse (pump thread lived in
        # the engine JVM): terminated, not completed.
        victim_cut = bool(
            victim_handle.wait_end(10.0) and not victim_handle.snap.completed
        )

        baseline_cancel = _cancel_rpc_total(ops)
        inc = ops.next_request_id(base)
        inc_resp = ops.schedule(
            inc, priority=70, input_len=512, output_len=2, timeout_s=90.0
        )
        inc_ok = inc_resp.code == CODE_OK and inc_resp.success
        inc_completed = False
        if inc_ok:
            inc_handle = ops.start_stream(inc_resp, inc)
            inc_ended = inc_handle.wait_end(45.0)
            inc_completed = inc_ended and inc_handle.snap.completed
        # The engine-side Cancel forward registers on the census
        # asynchronously — poll the counter instead of sampling a
        # stale value (cancel.py precedent).
        cancel_reached = wait_for(
            lambda: _cancel_rpc_total(ops) > baseline_cancel, 15.0, 0.5
        )
        cancel_delta = _cancel_rpc_total(ops) - baseline_cancel

        # The ABSENT_FENCE tombstone installed by the TOMBSTONED answer
        # rejects a direct late Enqueue of the victim rid with 8429.
        fence_ok, fence_detail = False, "no probe"
        try:
            probe = ops.build_generate_input(victim, output_len=2)
            ops._copy_role_addrs(probe, victim_resp)
            ack = _direct_enqueue(
                ops, ops.prefill_addr(victim_resp), probe, victim * 10 + 1
            )
            fence_ok, fence_detail = _fence_rejected_8429(ack, victim)
        except Exception as exc:
            fence_detail = repr(exc)

        # Victim decode (output=5000 ≈ 38s) runs its bounded orphan
        # computation — the 60s window covers it.
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 60.0
        )
        residue_ok, residue_detail = _fence_residue_stable(ops, 1)
        recovery_ok, recovery_msg = ops.verify_recovery()

        report.invariant(
            "PR10",
            victim_cut and inc_ok and inc_completed and cancel_delta >= 1,
            context="tombstoned_preemption_settlement",
            detail=(
                f"victim stream cut by crash={victim_cut} "
                f"(terminated, completed={victim_handle.snap.completed}), "
                f"incoming schedule={inc_resp.code} FetchResponse "
                f"completed={inc_completed} (tombstone-settled preemption "
                f"— the freed slot went to the incoming), "
                f"cancel_rpc_delta={cancel_delta} (>=1 on the fresh "
                f"instance)"
            ),
        )
        report.invariant(
            "PR6",
            fence_ok and cancel_reached,
            context="tombstoned_absent_fence_installed",
            detail=(
                f"ABSENT_FENCE direct-enqueue rejection 8429={fence_ok} "
                f"({fence_detail}), cancel reached engine={cancel_reached}"
            ),
        )
        report.invariant(
            "P6",
            engine_clean and residue_ok and recovery_ok,
            detail=(
                f"engine={'ok' if engine_clean else engine_detail}, "
                f"master residue stable={residue_ok} "
                f"({residue_detail}), recovery={recovery_msg}"
            ),
        )
        return report.finish(
            f"tombstoned preemption: victim cut+8429-fenced, incoming "
            f"completed={inc_completed}, cancel_delta={cancel_delta}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if victim_handle is not None:
            victim_handle.cancel()
        if inc_handle is not None:
            inc_handle.cancel()
        _restore_engines(ops)
