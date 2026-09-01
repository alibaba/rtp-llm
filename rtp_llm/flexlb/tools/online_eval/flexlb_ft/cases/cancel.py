"""Cancel-category cases: the request-cancellation contract.

Theme: a client Cancel must terminate the stream, free the engine-side
request state and (under master-enqueued delivery) drain the master
inflight ledger — idempotently, in isolation from sibling requests, at
every lifecycle stage, and even for requests the master has never seen.
The legacy cancel_smoke.py T1-T6 scripts port 1:1; the anomaly E1
cancel-path case joins this family because it is the same contract seen
from the client side of a failed request (cancel_anomaly_path).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ..context import CaseContext, CaseDef, rid_base
from ..harness import AssertUtils

CANCEL_CASES: list[CaseDef] = []


def case(name: str, profiles=None, requires=None, source: str = ""):
    def deco(fn):
        CANCEL_CASES.append(
            CaseDef(
                name=name,
                category="cancel",
                fn=fn,
                profiles=profiles,
                requires=requires,
                source=source,
            )
        )
        return fn

    return deco


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


# ===========================================================================
# Cancel cases (cancel_smoke.py T1-T6, ported 1:1)
# ===========================================================================


@case("cancel_t1", source="cancel_smoke.py T1")
def t1_basic_cancel(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            handle.cancel()
            return False, "no output received before cancel window"
        cancel_at = time.monotonic()
        ops.cancel(rid, response)
        ended = handle.wait_end(5.0)
        cancel_latency = time.monotonic() - cancel_at
        recovery_ok, recovery_msg = ops.verify_recovery()
        method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
        engine_recv, recv_detail = ops.verify_engine_received(rid, method)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        passed = ended and recovery_ok
        return passed, (
            f"cancel_latency={cancel_latency:.3f}s, stream_terminated={ended}, "
            f"outputs={len(handle.snap.outputs)}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("cancel_t2", source="cancel_smoke.py T2")
def t2_cancel_idempotency(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            handle.cancel()
            return False, "no output received before cancel window"
        ops.cancel(rid, response)
        second_cancel_ok, second_cancel_err = True, ""
        try:
            ops.cancel(rid, response)
        except Exception as exc:
            second_cancel_ok, second_cancel_err = False, repr(exc)
        ended = handle.wait_end(5.0)
        recovery_ok, recovery_msg = ops.verify_recovery()
        method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
        engine_recv, recv_detail = ops.verify_engine_received(rid, method)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        passed = second_cancel_ok and ended and recovery_ok
        return passed, (
            f"second_cancel_ok={second_cancel_ok} {second_cancel_err}, "
            f"stream_terminated={ended}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("cancel_t3", source="cancel_smoke.py T3")
def t3_multi_request_isolation(ctx: CaseContext):
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
                # Drainage discipline (S4 lesson, 2026-08-27 task #63
                # post-mortem): a sibling that was already scheduled must not
                # be left behind as an unconsumed entry — under BATCH dispatch
                # the leaked EnqueueBatch result sits in the engine's fetch
                # queue and the master's inflight/ledger far past the 30s
                # stale TTL (fence-quarantine family), poisoning later cases
                # on the shared env (observed cascade: T3 leak ->
                # kv_prefix_stickiness / balance_len_mixed /
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
                # Shape fidelity (finding-⑥ family, 2026-08-28 task #63
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
        return passed, (
            f"A_completed={a_snap.completed}(outputs={len(a_snap.outputs)}), "
            f"B_cancelled={b_ended}(completed={b_snap.completed}), "
            f"C_completed={c_snap.completed}(outputs={len(c_snap.outputs)}), "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("cancel_t4", source="cancel_smoke.py T4")
def t4_cancel_after_completion(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid, output_len=1)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        deadline = time.monotonic() + 30.0
        while not handle.snap.completed and time.monotonic() < deadline:
            time.sleep(0.05)
        if not handle.snap.completed:
            handle.cancel()
            return False, "request did not complete before timeout"
        cancel_ok, cancel_err = True, ""
        try:
            ops.cancel(rid, response)
        except Exception as exc:
            cancel_ok, cancel_err = False, repr(exc)
        handle.wait_end(2.0)
        recovery_ok, recovery_msg = ops.verify_recovery()
        method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
        engine_recv, recv_detail = ops.verify_engine_received(rid, method)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        passed = cancel_ok and recovery_ok
        return passed, (
            f"cancel_ok={cancel_ok} {cancel_err}, completed={handle.snap.completed}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("cancel_t5", source="cancel_smoke.py T5")
def t5_cancel_nonexistent(ctx: CaseContext):
    ops = ctx.ops()
    try:
        fake_rid = 99999
        cancel_ok, cancel_err = True, ""
        try:
            ops.cancel(fake_rid)
        except Exception as exc:
            cancel_ok, cancel_err = False, repr(exc)
        return cancel_ok, (
            f"cancel(rid={fake_rid}) ok={cancel_ok} {cancel_err}, "
            f"engine_verify=N/A (nonexistent request)"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("cancel_t6", source="cancel_smoke.py T6")
def t6_cancel_at_prefill_vs_decode(ctx: CaseContext):
    ops = ctx.ops()
    base = rid_base(ctx, "cancel")
    try:
        # A: cancel in prefill phase (before any output)
        rid_a = ops.next_request_id(base)
        resp_a = ops.schedule(rid_a)
        if resp_a.code != 200 or not resp_a.success:
            return False, f"schedule A failed: {resp_a.error_message}"
        input_pb_a = (
            None if resp_a.enqueued_by_master else ops.build_generate_input(rid_a)
        )
        handle_a = ops.start_stream(resp_a, rid_a, input_pb=input_pb_a)
        time.sleep(0.1)
        a_in_prefill = not handle_a.snap.first_received
        ops.cancel(rid_a, resp_a)
        a_ended = handle_a.wait_end(5.0)

        # B: cancel in decode phase (after first output)
        rid_b = ops.next_request_id(base)
        resp_b = ops.schedule(rid_b)
        if resp_b.code != 200 or not resp_b.success:
            return False, f"schedule B failed: {resp_b.error_message}"
        input_pb_b = (
            None if resp_b.enqueued_by_master else ops.build_generate_input(rid_b)
        )
        handle_b = ops.start_stream(resp_b, rid_b, input_pb=input_pb_b)
        b_got_first = handle_b.wait_first_output()
        if not b_got_first:
            handle_b.cancel()
            return False, "B never received first output (decode phase)"
        ops.cancel(rid_b, resp_b)
        b_ended = handle_b.wait_end(5.0)

        recovery_ok, recovery_msg = ops.verify_recovery()
        method_a = "enqueue_batch" if resp_a.enqueued_by_master else "generate_stream"
        method_b = "enqueue_batch" if resp_b.enqueued_by_master else "generate_stream"
        engine_recv_a, _ = ops.verify_engine_received(rid_a, method_a)
        engine_cancelled_a, _ = ops.verify_engine_cancelled(rid_a)
        engine_recv_b, _ = ops.verify_engine_received(rid_b, method_b)
        engine_cancelled_b, _ = ops.verify_engine_cancelled(rid_b)
        if resp_a.enqueued_by_master or resp_b.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"

        passed = a_ended and b_ended and a_in_prefill and recovery_ok
        return passed, (
            f"A_prefill_phase={a_in_prefill}, A_terminated={a_ended}, "
            f"A_outputs={len(handle_a.snap.outputs)}, "
            f"B_decode_phase={b_got_first}, B_terminated={b_ended}, "
            f"B_outputs={len(handle_b.snap.outputs)}, "
            f"engine_recv_A={engine_recv_a}, engine_cancel_A={engine_cancelled_a}, "
            f"engine_recv_B={engine_recv_b}, engine_cancel_B={engine_cancelled_b}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


# ===========================================================================
# Cancel-path anomaly case (anomaly_smoke.py E1 — the same contract seen
# from the client side of a failed request; rid_base family "anomaly" ->
# "cancel" in the task #85 category reorg)
# ===========================================================================


@case(
    "cancel_anomaly_path",
    source="anomaly_smoke.py E1",
)
def e1_cancel_path(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            handle.cancel()
            return False, "no output received before cancel window"
        cancel_at = time.monotonic()
        ops.cancel(rid, response)
        ended = handle.wait_end(5.0)
        cancel_latency = time.monotonic() - cancel_at
        recovery_ok, recovery_msg = ops.verify_recovery()
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            # NON_BATCH: a client Cancel on a delivered request cannot
            # safely release the master ledger entry (fence probe NOT_FOUND
            # is not a safe-release fact), so immediate-zero is not a
            # contract here — see kv_decode_capacity_park's watermark
            # rationale.
            inflight_ok, inflight_detail = True, "N/A (NON_BATCH residue contract)"
        passed = ended and recovery_ok
        return passed, (
            f"cancel_latency={cancel_latency:.3f}s, stream_terminated={ended}, "
            f"outputs={len(handle.snap.outputs)}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
