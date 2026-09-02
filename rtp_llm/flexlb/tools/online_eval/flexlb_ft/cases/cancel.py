"""Cancel-category cases: the request-cancellation contract.

Theme: a client Cancel must terminate the stream, free the engine-side
request state and (under master-enqueued delivery) drain the master
inflight ledger — idempotently, in isolation from sibling requests, at
every lifecycle stage, and even for requests the master has never seen.
The legacy cancel_smoke.py T1-T6 scripts port 1:1; the anomaly E1
cancel-path case joins this family because it is the same contract seen
from the client side of a failed request (cancel_anomaly_path).

Git-session gap analysis additions (2026-09, task #87 cancel-family
completion; assertions pin the CONTRACT, not the current behaviour —
cases predicted to fail carry a finding note in their docstring):

    cancel_deadline_exempt_inflight     M2: deadline fires after claim → exempt
    cancel_schedule_drop_delivered       Schedule-stream drop → real engine Cancel
    cancel_engine_notfound_settle        late Cancel vs finished request (idempotent)
    cancel_preemption_victim             M3: P70 evicts RUNNING P30 victim (8429)
    cancel_stream_break_prefill_autonomous  C1: engine-side stream-break cleanup
    cancel_stream_break_decode_autonomous   C2: decode autonomous terminal on break

The claim boundary (``deliveryClaimKind``) is the single point of no
return: NONE = still owned by the master, BATCH_ENQUEUE / ROUTE_DECISION
= already delivered to an engine.  Every case above probes what a cancel
means on each side of that boundary.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

from ..context import CaseContext, CaseDef, rid_base
from ..engine_ops import clear_type_all, engine_inflight_clean, inject_type_all
from ..harness import (
    AssertUtils,
    EnvSpec,
    default_perf,
    flexlb_config_for_profile,
    wait_for,
)

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


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"]


def _all_engine_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", [])]


def _schedule_with_priority(ops, request_id: int, priority: int, **kwargs):
    """Schedule RPC carrying an explicit priority (proto field 14).

    EngineOps.build_schedule_request does not expose the priority kwarg
    yet; the legacy flexlb_smoke_base._build_schedule_request proved the
    proto carries it ("Priority must be carried by the schedule protocol;
    embedding it only in unique_key metadata does not reach Auto-TPM
    admission").  Rather than widening engine_ops.py from the cancel
    category (other agents own the neighbouring modules), set the field
    on the built message here — protobuf messages are mutable.
    """
    req = ops.build_schedule_request(request_id, **kwargs)
    req.priority = priority
    stub = ops.schedule_pb2_grpc.FlexlbServiceStub(ops._channel(ops.master_target()))
    return stub.Schedule(req, timeout=30.0)


def _schedule_future(ops, request_id: int, **kwargs):
    """Fire-and-forget Schedule: a grpc Future whose cancel() aborts the
    in-flight Schedule RPC itself.

    Under BATCH dispatch the master completes the Schedule response only
    after the EnqueueBatch ACK (RequestRegistry.deliveryPublication runs
    on the ACK path), so an injected prefill ``enqueue_delay`` keeps the
    Schedule RPC in flight while the batch has already been claimed —
    exactly the window in which cancelling the client-side RPC triggers
    the master's inbound-context CancellationListener
    (FlexlbServiceImpl: Context.addListener → cancelUndeliveredRoute).
    """
    req = ops.build_schedule_request(request_id, **kwargs)
    stub = ops.schedule_pb2_grpc.FlexlbServiceStub(ops._channel(ops.master_target()))
    return stub.Schedule.future(req, timeout=30.0)


def _cancel_rpc_total(ops) -> int:
    """Sum of per-engine Cancel RPC counters from /snapshot."""
    snap = ops.snapshot()
    return sum(
        int(e.get("rpc_counts", {}).get("cancel", 0)) for e in snap.get("engines", [])
    )


# ===========================================================================
# Cancel cases (cancel_smoke.py T1-T6, ported 1:1)
# ===========================================================================


@case("cancel_t1", source="cancel_smoke.py T1")
def t1_basic_cancel(ctx: CaseContext):
    """T1: mid-flight client Cancel terminates stream + engine state.

    Scenario: one request is streaming its first outputs; the client
    issues the explicit Cancel RPC while the request is still running.

    Behaviour: master Cancel (typed CLIENT_CANCELLED) → under BATCH
    dispatch the master walks the real GrpcEngineCancelChannel and the
    engine records the cancellation (cancelled_rids / lifecycle).

    Expected (contract): stream terminates, engine-side cancel is
    OBSERVED for NON_BATCH but CONTRACT-GUARANTEED for BATCH (the
    production cancel channel is a real gRPC wiring, so the engine
    seeing the cancel is not an implementation accident), master
    inflight ledger drains, a follow-up request completes normally.

    Prediction: passes (t1-t6 kept engine verification observational
    while the cancel channel wiring was under construction; the BATCH
    hard assertion is the 2026-09 upgrade — see the family docstring).
    """
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
        # BATCH: engine cancellation is contract-guaranteed (real cancel
        # channel wiring) — hard assertion.  NON_BATCH keeps the legacy
        # client-driven worker-cancel path observational for compatibility.
        if response.enqueued_by_master:
            passed = ended and recovery_ok and engine_cancelled
        else:
            passed = ended and recovery_ok
        return passed, (
            f"cancel_latency={cancel_latency:.3f}s, stream_terminated={ended}, "
            f"outputs={len(handle.snap.outputs)}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail})"
            f"[{'hard' if response.enqueued_by_master else 'observational'}], "
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


# ===========================================================================
# Git-session gap-analysis cases (task #87): the cancel contract around the
# deliveryClaimKind boundary.  Assertions pin the CONTRACT behaviour; cases
# predicted to fail before a parallel mock-engine capability lands carry an
# explicit finding note (docstring Prediction).
# ===========================================================================


@case("cancel_deadline_exempt_inflight")
def cancel_deadline_exempt_inflight(ctx: CaseContext):
    """M2 exemption: a queue deadline that fires AFTER the claim must not
    cancel the request.

    Scenario: queueTimeoutMs=2000; every prefill engine is injected with
    enqueue_delay=3000, so under BATCH dispatch the EnqueueBatch ACK (and
    with it the Schedule response) is in flight when the deadline expires.
    One request is sent synchronously (output_len=500 keeps decode alive
    across the deadline so the NON_BATCH variant also exercises the
    post-claim expiry path rather than racing request completion).

    Behaviour: ExpirationTimer fires cancelForDeadline(DEADLINE_EXCEEDED)
    while the request holds deliveryClaimKind != NONE.  Under BATCH the
    cancel is deferred by the open admission mutation and promoted after
    the ACK; under NON_BATCH it fires directly on the running request.
    Either way RequestRegistry.cancelRequest hits the exemption
    (RequestRegistry.java: "DEADLINE_EXCEEDED && claim != NONE → return
    current") — the deadline is NOT a cancel reason past the boundary.

    Expected (contract): the request is NOT cancelled — it completes
    normally with its full output; no engine-side cancel record exists
    (cancelled_rids / lifecycle end_state); the master inflight ledger
    settles through the ordinary completion path; a follow-up request
    completes normally.

    Prediction: passes (the Java side carries the same contract in its
    unit tests; the deadline path only ever CANCELS pre-claim — M1).
    """
    spec = EnvSpec(
        label=f"cancel_exempt_{ctx.profile}",
        n_prefill=2,
        n_decode=4,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={
            "FLEXLB_CONFIG": flexlb_config_for_profile(
                ctx.profile, queue_timeout_ms=2_000
            )
        },
    )
    env = ctx.env_manager.ensure(spec)
    ops = ctx.engine_ops(env)
    prefill_names = _prefill_names(ops)
    # enqueue_delay(3000) > queueTimeout(2000): the deadline expires while
    # the EnqueueBatch ACK is still in flight (BATCH) — the canonical M2
    # window.  Harmless no-op under NON_BATCH (no EnqueueBatch path).
    inject_type_all(ops, prefill_names, "enqueue_delay", delay_ms=3_000)
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    handle = None
    try:
        # Long client deadline: the Schedule call itself blocks on the
        # delayed ACK (~3s) and must still return success (exemption keeps
        # the request alive; enqueue_delay 3000 < enqueueRpcTimeout 5000).
        response = ops.schedule(rid, output_len=500)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None
            if response.enqueued_by_master
            else ops.build_generate_input(rid, output_len=500)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        handle.wait_end(45.0)
        snap = handle.snap
        completed = snap.completed and not snap.error
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = completed and not engine_cancelled and inflight_ok and recovery_ok
        return passed, (
            f"deadline_exempt: completed={snap.completed}, "
            f"outputs={len(snap.outputs)}, error={snap.error}, "
            f"engine_cancelled={engine_cancelled}({cancel_detail})"
            "[expect False — exemption], "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if handle is not None:
            handle.cancel()
        clear_type_all(ops, prefill_names, "enqueue_delay")


@case("cancel_schedule_drop_delivered", requires=["enqueue_batch"])
def cancel_schedule_drop_delivered(ctx: CaseContext):
    """Schedule-stream drop on a DELIVERED request → master still sends the
    real engine Cancel.

    Scenario (BATCH only): every prefill is injected with enqueue_delay,
    which holds the Schedule RPC in flight (the master completes the
    Schedule response only after the EnqueueBatch ACK).  The request is
    sent fire-and-forget; after the batch has been claimed (claim =
    BATCH_ENQUEUE, set at tryClaimForDelivery before the RPC even
    leaves), the client CANCELS THE SCHEDULE RPC ITSELF
    (stub.Schedule.future(...).cancel()) — the gRPC CANCEL propagates to
    the master's inbound context and arms the CancellationListener that
    FlexlbServiceImpl attaches per Schedule call.

    Behaviour: cancelUndeliveredRoute → cancelRequest(rid, 0,
    CLIENT_CANCELLED).  While the admission mutation is still open the
    cancel is deferred and resumes after the ACK
    (resumeCancellationAfterAdmission); CLIENT_CANCELLED gets NO M2-style
    exemption (that courtesy is DEADLINE_EXCEEDED-only), so with claim !=
    NONE the master MUST send the real Cancel RPC to the original
    prefill — the delivered-request twin of the undelivered local
    rollback.

    Expected (contract): (a) the original prefill's Cancel RPC counter
    increases — the master really sent an engine cancel; (b) the engine
    records the cancellation (cancelled_rids / lifecycle end_state), the
    cancellation landing on the tracked request after the delayed
    enqueue processed; (c) the master inflight ledger settles through
    the typed CANCELLED reconcile; (d) a follow-up request completes.

    Trigger note: this is the Schedule-RPC drop (the only stream the
    master's CancellationListener observes); dropping the FetchResponse
    stream instead never reaches the master — that variant is the C1/C2
    autonomous-cleanup cases below.

    Prediction: expected to pass — the chain (listener → defer →
    resume-after-ACK → fence cancel channel → engine tracked-cancel →
    typed CANCELLED reconcile) is all production wiring; if it fails,
    the finding is in the master's resume/fence path, not the test.
    """
    ops = ctx.ops()
    prefill_names = _prefill_names(ops)
    inject_type_all(ops, prefill_names, "enqueue_delay", delay_ms=2_000)
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    future = None
    try:
        baseline_cancel = _cancel_rpc_total(ops)
        future = _schedule_future(ops, rid, output_len=500)
        # Batch collection (~10ms) + EnqueueBatch dispatch: by 0.5s the
        # batch is claimed; the delayed ACK keeps Schedule in flight.
        time.sleep(0.5)
        future.cancel()
        try:
            future.result(timeout=5.0)
        except Exception:
            pass  # cancelled future — the expected outcome

        def cancel_reached() -> bool:
            return _cancel_rpc_total(ops) - baseline_cancel >= 1

        engine_cancel_rpc = wait_for(cancel_reached, 15.0, 0.2)
        # The promoted cancel lands after the delayed enqueue processed
        # (tracked), so the engine must record it.
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 15.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = engine_cancel_rpc and engine_cancelled and inflight_ok
        passed = passed and recovery_ok
        return passed, (
            f"schedule_drop_delivered: engine_cancel_rpc_delta="
            f"{_cancel_rpc_total(ops) - baseline_cancel}, "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if future is not None:
            future.cancel()
        clear_type_all(ops, prefill_names, "enqueue_delay")


@case("cancel_engine_notfound_settle")
def cancel_engine_notfound_settle(ctx: CaseContext):
    """Late Cancel vs an already-finished request: idempotent everywhere,
    no double settlement.

    Scenario: a minimal request (output_len=1) runs to completion; the
    Cancel then arrives LATE (the fence/probe raced the terminal) — once
    through the master Cancel RPC, once directly against the original
    prefill engine.

    Behaviour: master-side RequestRegistry.cancelRequest returns the
    terminal snapshot untouched (state is terminal → no second
    settlement, no engine cancel is forwarded); engine-side
    JavaMockEngineCluster.cancelRequest classifies the request as
    alreadyFinished and answers CANCEL_STATUS_NOT_FOUND without
    republishing any terminal.

    Expected (contract): the master Cancel RPC succeeds (idempotent);
    the direct engine Cancel answers NOT_FOUND; the engine's recorded
    terminal stays a completion (no cancelled_rids entry / lifecycle
    rewrite); the master inflight ledger stays clean (nothing re-opened);
    a follow-up request completes normally.

    Prediction: passes (t4 already covers the master-idempotent half;
    the engine NOT_FOUND branch is the mock's documented three-branch
    cancel semantics).
    """
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid, output_len=1)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None
            if response.enqueued_by_master
            else ops.build_generate_input(rid, output_len=1)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        deadline = time.monotonic() + 30.0
        while not handle.snap.completed and time.monotonic() < deadline:
            time.sleep(0.05)
        if not handle.snap.completed:
            handle.cancel()
            return False, "request did not complete before late-cancel window"

        master_cancel_ok, master_cancel_err = True, ""
        try:
            ops.cancel(rid, response)
        except Exception as exc:
            master_cancel_ok, master_cancel_err = False, repr(exc)

        # Direct engine probe (bypass the master): the fence arriving at
        # the engine AFTER the terminal must read NOT_FOUND.
        engine_status_ok, engine_status_detail = False, "no probe"
        try:
            stub = ops.pb2_grpc.RpcServiceStub(ops._channel(ops.prefill_addr(response)))
            ack = stub.Cancel(ops.pb2.CancelRequestPB(request_id=rid), timeout=10.0)
            engine_status_ok = ack.status == ops.pb2.CANCEL_STATUS_NOT_FOUND
            engine_status_detail = f"status={ack.status}"
        except Exception as exc:
            engine_status_detail = repr(exc)

        handle.wait_end(2.0)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = (
            master_cancel_ok
            and engine_status_ok
            and not engine_cancelled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"late_cancel_settle: master_cancel_ok={master_cancel_ok} "
            f"{master_cancel_err}, engine_probe={engine_status_ok}"
            f"({engine_status_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail})"
            "[expect False — terminal preserved], "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("cancel_preemption_victim")
def cancel_preemption_victim(ctx: CaseContext):
    """M3 preemption: a P70 arrival evicts a RUNNING P30 victim through the
    master → original-Prefill weak-Cancel protocol.

    Scenario: dedicated 1P+1D environment, PRIORITY ordering with the
    production preemption block (allowedVictimStages PREFILL_QUEUED +
    DECODE_RESERVED — master_fixed_window.json values) and
    decode maxEngineRequests=1 so the single decode slot makes the
    capacity contest deterministic.  The victim (priority 30,
    input_len=512 / output_len=200 — long decode) is scheduled first and
    waits RUNNING on the decode engine; the preemptor (priority 70,
    output_len=2) then arrives.

    Behaviour: the preemptor's ordinary placement is BLOCKED (decode
    capacity exhausted), so RequestScheduler.attemptPlacement escalates
    to EvictionManager.tryAdmit; the victim is selected and the master
    sends the real (weak) Cancel to the ORIGINAL prefill, which
    propagates to decode (P→D stream-cancel conduction).

    Expected (contract): the victim's stream terminates in a
    non-completion terminal carrying the typed 8429
    (PRIORITY_PREEMPTED / CANCELLED) error; the engine records the
    victim as cancelled (cancelled_rids / lifecycle); the original
    prefill's Cancel RPC counter increased (the weak cancel really
    went out); the P70 request completes normally once the slot frees;
    the master inflight ledger drains with no leak; recovery works.

    Prediction: expected to pass — this is the priority_preemption_smoke
    scenario (RUNNING decode victim, batch default) ported onto the
    flexlb_ft framework; capacity here comes from maxEngineRequests=1
    instead of the smoke line's KV pressure so the eviction trigger is
    deterministic.  Priority rides the Schedule proto's priority field
    (see _schedule_with_priority).
    """
    config = json.loads(flexlb_config_for_profile(ctx.profile, ordering="priority"))
    ordering = config["scheduler"]["ordering"]
    ordering["defaultPriority"] = 50
    ordering["preemption"] = {
        "allowedVictimStages": ["PREFILL_QUEUED", "DECODE_RESERVED"],
    }
    config["router"]["roles"]["decode"]["availability"]["maxEngineRequests"] = 1
    spec = EnvSpec(
        label=f"cancel_preempt_{ctx.profile}",
        n_prefill=1,
        n_decode=1,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={"FLEXLB_CONFIG": json.dumps(config, separators=(",", ":"))},
    )
    env = ctx.env_manager.ensure(spec)
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "cancel")
    victim_handle = None
    high_handle = None
    try:
        victim_rid = ops.next_request_id(base)
        victim_keys = [victim_rid * 100 + 1]
        victim_resp = _schedule_with_priority(
            ops,
            victim_rid,
            30,
            input_len=512,
            output_len=200,
            block_keys=victim_keys,
        )
        if victim_resp.code != 200 or not victim_resp.success:
            return False, f"victim schedule failed: {victim_resp.error_message}"
        victim_input = (
            None
            if victim_resp.enqueued_by_master
            else ops.build_generate_input(
                victim_rid,
                input_len=512,
                output_len=200,
                block_keys=victim_keys,
            )
        )
        victim_handle = ops.start_stream(victim_resp, victim_rid, input_pb=victim_input)

        # Victim must be RUNNING on the decode engine before the preemptor
        # arrives — otherwise the preemptor would simply take the free slot.
        def victim_running() -> bool:
            snap = ops.snapshot_by_name()
            return any(
                e.get("role") == "decode"
                and e.get("request_lifecycle", {})
                .get(str(victim_rid), {})
                .get("end_state")
                == "running"
                for e in snap.values()
            )

        running = wait_for(victim_running, 10.0, 0.1)
        if not running:
            victim_handle.cancel()
            return False, "victim never reached RUNNING on decode"

        baseline_cancel = _cancel_rpc_total(ops)
        high_rid = ops.next_request_id(base)
        high_keys = [high_rid * 100 + 1]
        with ThreadPoolExecutor(max_workers=1) as pool:
            high_future = pool.submit(
                _schedule_with_priority,
                ops,
                high_rid,
                70,
                input_len=512,
                output_len=2,
                block_keys=high_keys,
            )
            # The victim's terminal: preemption ends its stream in a
            # non-completion state (typed 8429 surfaces as the stream
            # error under both dispatch modes).
            victim_ended = victim_handle.wait_end(20.0)
            try:
                high_resp = high_future.result(timeout=40.0)
            except Exception as exc:
                return False, f"high-priority schedule failed: {exc!r}"
        if high_resp.code != 200 or not high_resp.success:
            return False, f"high schedule failed: {high_resp.error_message}"
        high_input = (
            None
            if high_resp.enqueued_by_master
            else ops.build_generate_input(
                high_rid,
                input_len=512,
                output_len=2,
                block_keys=high_keys,
            )
        )
        high_handle = ops.start_stream(high_resp, high_rid, input_pb=high_input)
        high_handle.wait_end(30.0)

        victim_cancelled, victim_cancel_detail = ops.verify_engine_cancelled(victim_rid)
        weak_cancel_delta = _cancel_rpc_total(ops) - baseline_cancel
        if victim_resp.enqueued_by_master or high_resp.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 15.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        engine_clean, engine_clean_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 15.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = (
            victim_ended
            and not victim_handle.snap.completed
            and victim_cancelled
            and weak_cancel_delta >= 1
            and high_handle.snap.completed
            and not high_handle.snap.error
            and inflight_ok
            and engine_clean
            and recovery_ok
        )
        return passed, (
            f"preemption_victim: victim_terminated={victim_ended}"
            f"(completed={victim_handle.snap.completed}, "
            f"error={victim_handle.snap.error}), "
            f"victim_engine_cancelled={victim_cancelled}"
            f"({victim_cancel_detail}), weak_cancel_delta={weak_cancel_delta}, "
            f"high_completed={high_handle.snap.completed}"
            f"(outputs={len(high_handle.snap.outputs)}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_clean_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if victim_handle is not None:
            victim_handle.cancel()
        if high_handle is not None:
            high_handle.cancel()


@case("cancel_stream_break_prefill_autonomous", requires=["enqueue_batch"])
def cancel_stream_break_prefill_autonomous(ctx: CaseContext):
    """C1: the client drops the FetchResponse stream mid-request; the
    ENGINE must sense the break and clean the request up on its own.

    Scenario (BATCH only): a long request (output_len=500) is dispatched
    and its first output has been received (the FetchResponse stream is
    established, decode is running); the client then cancels the stream
    itself (StreamHandle.call.cancel()) WITHOUT the explicit Cancel RPC.

    Behaviour (production C++ semantics, being ported to the mock in
    parallel): the engine's output loop observes the dead consumer
    context, tears the request down, and reports the typed CANCELLED
    terminal through WorkerStatus so the master can reconcile; any
    decode downstream is cancelled by the P→D stream-cancel conduction.

    Expected (contract): the engine records the cancellation
    (cancelled_rids / lifecycle end_state = cancelled) and the rid
    leaves the running set; every engine reports inflight 0 with no
    leak; the master ledger settles through the CANCELLED reconcile; a
    follow-up request completes normally.

    Prediction: FINDING — depends on the mock engine's stream-break
    sensing (output loop checking the consumer context's isCancelled),
    which is being implemented in parallel.  Until that lands, the
    engine keeps executing the request to completion and the
    cancelled-record assertion fails by design; rerun this case in the
    follow-up integration round once the C1 capability merges.
    """
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid, output_len=500)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        handle = ops.start_stream(response, rid)  # BATCH: FetchResponse
        if not handle.wait_first_output():
            handle.cancel()
            return False, "no output received before stream-break window"

        # Drop the consumer stream itself — NOT ops.cancel.
        handle.cancel()

        def engine_sensed_break() -> bool:
            ok, _ = ops.verify_engine_cancelled(rid)
            return ok

        engine_cancelled = wait_for(engine_sensed_break, 10.0, 0.2)
        _, cancel_detail = ops.verify_engine_cancelled(rid)
        engine_clean, engine_clean_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 15.0
        )
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 15.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = engine_cancelled and engine_clean and inflight_ok
        passed = passed and recovery_ok
        return passed, (
            f"stream_break_prefill: engine_sensed={engine_cancelled}"
            f"({cancel_detail}), engine_clean={engine_clean}"
            f"({engine_clean_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg} "
            "[expected FINDING until mock C1 stream-break sensing lands]"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("cancel_stream_break_decode_autonomous", requires=["generate_stream"])
def cancel_stream_break_decode_autonomous(ctx: CaseContext):
    """C2: mid-decode stream drop on the frontend-sent stream — decode
    cleans itself up and reports the terminal early instead of waiting
    for the stale-inflight TTL.

    Scenario (NON_BATCH only): the request is delivered via
    GenerateStreamCall (frontend → engine direct); the first output has
    been received, so the request is decoding; the client cancels the
    stream itself (no explicit Cancel RPC).

    Behaviour (production C++ semantics, being ported to the mock in
    parallel): the engine senses the broken consumer context, the
    prefill leg cleans up and cancels downstream; decode stops early,
    frees its state and reports the terminal through WorkerStatus —
    the master reconciles without waiting for the stale-inflight TTL
    (production 5min; the framework config keeps 30s).

    Expected (contract): the engine records the cancellation
    (cancelled_rids / lifecycle end_state = cancelled); no engine-side
    residue (inflight 0 everywhere, no leak); the master ledger settles;
    a follow-up request completes normally.

    Prediction: FINDING — depends on the mock engine's stream-break
    sensing for the frontend-sent stream (C2 capability, implemented in
    parallel).  Until it lands the request simply runs to completion and
    the cancelled-record assertion fails by design; rerun in the
    follow-up integration round.
    """
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid, output_len=500)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = ops.build_generate_input(rid, output_len=500)
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            handle.cancel()
            return False, "no output received before stream-break window"

        # Drop the consumer stream itself — NOT ops.cancel.
        handle.cancel()

        def engine_sensed_break() -> bool:
            ok, _ = ops.verify_engine_cancelled(rid)
            return ok

        engine_cancelled = wait_for(engine_sensed_break, 10.0, 0.2)
        _, cancel_detail = ops.verify_engine_cancelled(rid)
        engine_clean, engine_clean_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 15.0
        )
        # NON_BATCH ledger residue contract: the stale-TTL is the safety
        # net, but the C2 terminal should settle well inside it.
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 15.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = engine_cancelled and engine_clean and inflight_ok
        passed = passed and recovery_ok
        return passed, (
            f"stream_break_decode: engine_sensed={engine_cancelled}"
            f"({cancel_detail}), engine_clean={engine_clean}"
            f"({engine_clean_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg} "
            "[expected FINDING until mock C2 stream-break sensing lands]"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
