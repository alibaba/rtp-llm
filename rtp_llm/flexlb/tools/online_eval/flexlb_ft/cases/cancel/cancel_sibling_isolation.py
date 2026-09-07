from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ...context import CaseContext, rid_base
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.cancel import _master_http


@case("cancel_sibling_isolation", category="cancel", source="cancel_smoke.py T3")
def cancel_sibling_isolation(ctx: CaseContext):
    ops = ctx.ops()
    base = rid_base(ctx, "cancel")
    rids = [ops.next_request_id(base) for _ in range(3)]
    cancel_rid = rids[1]  # B
    # B runs a long decode so the cancel lands while it is still decoding.
    # The default output_len=10 finishes decode in ~80ms and races the
    # cancel: the engine finishes first, the master then (correctly) returns
    # the terminal state idempotently without forwarding an engine cancel,
    # and verify_engine_cancelled flaps.  500 tokens ≈ 3.8s of decode at the
    # default production-fit step pricing (ceil(500/2.6)=193 steps ×
    # (19.5+0.175×running) ms), comfortably spanning the cancel window.
    long_output_len = 500
    try:

        def _schedule(rid: int):
            if rid == cancel_rid:
                return ops.schedule(rid, output_len=long_output_len)
            return ops.schedule(rid)

        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(_schedule, rids))
        for i, resp in enumerate(responses):
            if resp.code != 200 or not resp.success:
                # Drainage discipline (S4 lesson, 2026-08-27
                # post-mortem): a sibling that was already scheduled must not
                # be left behind as an unconsumed entry — under BATCH dispatch
                # the leaked EnqueueBatch result sits in the engine's fetch
                # queue and the master's inflight/ledger far past the 30s
                # stale TTL (fence-quarantine family), poisoning later cases
                # on the shared env (observed cascade: this case's leak
                # -> kv_prefix_stickiness / balance_len_mixed /
                # admission_gate_no_starvation failures in the batch-window
                # full run, all solo-PASS). Cancel every scheduled sibling
                # before failing the case; the streams were never opened, so
                # the master-side cancel is a clean local release on both
                # dispatch modes.
                for j, sibling in enumerate(responses):
                    if j != i and sibling.code == 200 and sibling.success:
                        try:
                            ops.cancel(rids[j], sibling)
                        except Exception:
                            pass
                return False, f"schedule failed for rid={rids[i]}: {resp.error_message}"

        handles = []
        for rid, resp in zip(rids, responses):
            if resp.enqueued_by_master:
                input_pb = None
            else:
                # Shape fidelity (finding-⑥ family, 2026-08-28
                # post-mortem): under NON_BATCH dispatch the direct stream's
                # GenerateInputPB must carry the SAME output_len the
                # ScheduleRequest carried.  A default-shape rebuild
                # (output_len=10) finishes B's decode in ~80ms — inside the
                # cancel-path latency (~150-250ms) — turning the docstring's
                # "comfortably spanning" cancel window into a coin flip on
                # every NON_BATCH run (observed: wn full-run + wn solo FAIL
                # with B completed before the engine-side cancel landed,
                # while sn flapped run-dependent).  With output_len=500 the
                # ~3.8s decode dwarfs the cancel path on both dispatch modes.
                kwargs = {"output_len": long_output_len} if rid == cancel_rid else {}
                input_pb = ops.build_generate_input(rid, **kwargs)
            handles.append(ops.start_stream(resp, rid, input_pb=input_pb))

        # Wait for the SHORT requests' (A, C) first output only.  In batch
        # mode the mock engine's FetchResponse surfaces the first message
        # only after decode completes, so waiting for B (output_len=500)
        # would mean B is already terminal when the cancel fires — the
        # master then (correctly) answers REQUEST_STATE_COMPLETED
        # idempotently and never forwards an engine cancel.  A/C finish in
        # ~1s while B still has ~3.3s of decode left, so cancelling right
        # after A/C's first output lands the cancel mid-decode.
        if not all(handles[i].wait_first_output(15.0) for i in (0, 2)):
            for h in handles:
                h.cancel()
            return False, "short requests (A, C) did not receive first output"

        ops.cancel(cancel_rid, responses[1])
        b_ended = handles[1].wait_end(5.0)

        # Master mid-flight leg (P1): once B's client stream terminated,
        # its scheduler ledger entry must release — poll for
        # scheduler_inflight <= 2 (only the surviving siblings A/C may
        # remain; a stale 3 pins B's cancellation never settling the
        # master ledger).  A/C completing meanwhile only LOWERS the
        # count, so the assertion direction is monotone-safe on both
        # dispatch timings; the 2s window tolerates the 20ms event-driven
        # reconcile chain.
        sched_isolated = True
        sched_final = -1
        if responses[1].enqueued_by_master:
            sched_isolated = wait_for(
                lambda: ops.master_scheduler_inflight() <= 2, 2.0, 0.1
            )
            sched_final = ops.master_scheduler_inflight()

        a_complete = handles[0].wait_end(30.0)
        c_complete = handles[2].wait_end(30.0)

        recovery_ok, recovery_msg = ops.verify_recovery()
        method = (
            "enqueue_batch" if responses[1].enqueued_by_master else "generate_stream"
        )
        engine_recv, recv_detail = ops.verify_engine_received(cancel_rid, method)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(cancel_rid)
        if responses[1].enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"

        a_snap, b_snap, c_snap = handles[0].snap, handles[1].snap, handles[2].snap
        passed = (
            b_ended
            and a_complete
            and a_snap.completed
            and c_complete
            and c_snap.completed
            and not b_snap.completed
            and engine_cancelled
            and recovery_ok
        )
        if responses[1].enqueued_by_master:
            # P1 master legs: mid-flight isolation (B's ledger entry
            # released without waiting for the siblings) + the closing
            # drain of the whole inflight ledger.
            passed = passed and sched_isolated and inflight_ok
        return passed, (
            f"A_completed={a_snap.completed}(outputs={len(a_snap.outputs)}), "
            f"B_cancelled={b_ended}(completed={b_snap.completed}), "
            f"C_completed={c_snap.completed}(outputs={len(c_snap.outputs)}), "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"master_isolation(sched<=2)={sched_isolated}(final={sched_final}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
