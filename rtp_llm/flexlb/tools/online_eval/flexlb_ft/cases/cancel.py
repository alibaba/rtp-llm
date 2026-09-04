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

HA family (2026-09, task #26 — assertions pin the audited production
ground truth: EngineFenceCoordinator is one-shot and owns no timer; a
TOMBSTONED ack settles immediately via resumeTombstoned while FAILED /
exception acks park in awaitAuthoritativeTerminal):

    cancel_engine_restarted_tombstoned_settle  true-crash restart → fresh
                                   instance never saw the rid → TOMBSTONED
                                   + ABSENT_FENCE + immediate typed settle
    cancel_prefill_dead_await_terminal   cancel vs dead prefill: decode
                                   WorkerStatus terminal is the authority
    cancel_decode_retire_closes_fence    decode generation retire closes an
                                   AWAIT_TERMINAL cancel fence
    cancel_fencing_lost_on_engine_restart  design boundary: fencing is
                                   engine memory — a second crash drops it
                                   and admits the late Enqueue (documented
                                   trade-off, master ledger stays settled)
    cancel_transport_failure_one_shot    cancel_no_respond: exactly one
                                   cancel RPC, decode settles the request
    cancel_unexpected_status_await_terminal  cancel_unexpected_status: no
                                   false success, no false terminal

The claim boundary (``deliveryClaimKind``) is the single point of no
return: NONE = still owned by the master, BATCH_ENQUEUE / ROUTE_DECISION
= already delivered to an engine.  Every case above probes what a cancel
means on each side of that boundary.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import grpc

from ..context import CaseContext, CaseDef, rid_base
from ..engine_ops import (
    _fence_residue_stable,
    clear_type_all,
    engine_inflight_clean,
    inject_type,
    inject_type_all,
)
from ..harness import (
    AssertUtils,
    EnvSpec,
    default_perf,
    flexlb_config_for_profile,
    wait_for,
    wait_for_port,
)

CANCEL_CASES: list[CaseDef] = []


def case(
    name: str,
    profiles=None,
    requires=None,
    source: str = "",
    skip_reason=None,
    expected_fail: bool = False,
):
    """Register into CANCEL_CASES (category is always "cancel").

    ``skip_reason`` (dsv4 stack gate): non-empty -> SKIP without executing
    (a contract the dsv4 v1 stack does not implement) — the kv.py
    precedent.
    ``expected_fail=True`` declares a declared-finding probe (task #101):
    failing confirms the finding, passing resolves it — neither counts
    toward failed_count / the suite verdict / the exit code."""

    def deco(fn):
        CANCEL_CASES.append(
            CaseDef(
                name=name,
                category="cancel",
                fn=fn,
                profiles=profiles,
                requires=requires,
                source=source,
                skip_reason=skip_reason,
                expected_fail=expected_fail,
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
    yet; the legacy smoke client's schedule builder (since removed with the
    rest of the smoke family) proved the proto carries it ("Priority must be
    carried by the schedule protocol;
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


def _engine_cancel_receipt_within(
    ops, rid: int, timeout_s: float = 5.0, since: Optional[float] = None
) -> tuple[bool, str]:
    """Engine-side cancel receipt within a tight propagation bound.

    The master→engine cancel channel is a real gRPC wiring
    (GrpcEngineCancelChannel), so the engine recording the rid in
    cancelled_rids is a second-scale expectation.  The 95s TTL drain
    window is a leak safety net, NOT an acceptable propagation path:
    waiting "eventually" conflates a correct cancel with TTL-swept
    cleanup — exactly where F1/F8-class findings hide (2026-09 eval
    batch A timing-contract upgrade).  *since* anchors the clock at the
    ops.cancel() issuance instant for callers whose poll starts after
    intermediate waits; the measured receipt latency lands in the
    detail.
    """
    t0 = time.monotonic() if since is None else since
    detail = "no poll yet"
    while True:
        ok, detail = ops.verify_engine_cancelled(rid)
        elapsed = time.monotonic() - t0
        if ok:
            if elapsed <= timeout_s:
                return True, (
                    f"{detail}, receipt={elapsed:.3f}s "
                    f"(within {timeout_s:.0f}s propagation contract)"
                )
            # Receipt exists but landed outside the window — the
            # TTL/fence sweep, not a timely cancel.
            return False, (
                f"{detail} but receipt={elapsed:.3f}s exceeds the "
                f"{timeout_s:.0f}s propagation contract (TTL sweep is NOT "
                "an acceptable cancel path)"
            )
        if elapsed >= timeout_s:
            break
        time.sleep(0.05)
    return False, (
        f"{detail}; no engine receipt within {timeout_s:.0f}s of cancel "
        f"issuance ({time.monotonic() - t0:.3f}s elapsed) — TTL sweep is "
        "NOT an acceptable cancel path"
    )


def _inflight_fingerprint(ops):
    """Master inflight fingerprint: scheduler count + per-endpoint
    (ip_port, inflight_batches, inflight_requests) rows.

    Same construction as the status family's homonym (equal
    fingerprints mean "no ledger mutation"); copied locally to keep the
    cancel category decoupled from status.py (parallel edits, 2026-09
    eval batch A).
    """
    data = ops.master_inflight()
    if data is None:
        return None

    def ep_rows(eps) -> tuple:
        rows = []
        for ep in eps or []:
            batches = ep.get("inflight_batches", 0)
            counted = len(batches) if isinstance(batches, list) else int(batches)
            rows.append(
                (
                    ep.get("ip_port", "?"),
                    counted,
                    int(ep.get("inflight_requests", 0) or 0),
                )
            )
        return tuple(rows)

    return (
        int(data.get("scheduler_inflight", 0)),
        ep_rows(data.get("prefill_endpoints")),
        ep_rows(data.get("decode_endpoints")),
    )


# ===========================================================================
# Cancel cases (cancel_smoke.py T1-T6, ported 1:1)
# ===========================================================================


@case("cancel_basic", source="cancel_smoke.py T1")
def cancel_basic(ctx: CaseContext):
    """Mid-flight client Cancel terminates stream + engine state.

    Scenario: one request is streaming its first outputs; the client
    issues the explicit Cancel RPC while the request is still running.

    Behaviour: master Cancel (typed CLIENT_CANCELLED) → under BATCH
    dispatch the master walks the real GrpcEngineCancelChannel and the
    engine records the cancellation (cancelled_rids / lifecycle).

    Expected (contract): stream terminates; engine-side cancel receipt
    is OBSERVED for NON_BATCH but CONTRACT-GUARANTEED for BATCH with a
    5s propagation bound (the cancel channel is a real gRPC wiring —
    engine receipt within 5s of cancel issuance; the TTL sweep is NOT
    an acceptable path); the master inflight ledger drains (asserted
    for BATCH — fixed in the 2026-09 eval batch A: the docstring
    previously promised more than the verdict checked); a follow-up
    request completes normally.

    Prediction: passes (the six legacy-ported cases kept engine
    verification observational while the cancel channel wiring was under
    construction; the BATCH
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
        # 修复（eval batch A + 时效契约）：BATCH 分支 engine 收证从"最终
        # 出现"收紧为 cancel 发出后 5s 内出现——cancel 走真实 gRPC 通道
        # 秒级应然；95s TTL 兜底把"正确取消"与"TTL 清理"混成同一通过态。
        if response.enqueued_by_master:
            engine_cancelled, cancel_detail = _engine_cancel_receipt_within(
                ops, rid, timeout_s=5.0, since=cancel_at
            )
        else:
            engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        recovery_ok, recovery_msg = ops.verify_recovery()
        method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
        engine_recv, recv_detail = ops.verify_engine_received(rid, method)
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        # BATCH: engine cancellation is contract-guaranteed (real cancel
        # channel wiring) with the 5s propagation bound — hard assertion.
        # 修复（eval batch A）：docstring 承诺的 master inflight ledger
        # drains 进 passed（BATCH）；NON_BATCH 保持 observational（口径
        # 限制见 family docstring）。
        if response.enqueued_by_master:
            passed = ended and recovery_ok and engine_cancelled and inflight_ok
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


@case("cancel_idempotent", source="cancel_smoke.py T2")
def cancel_idempotent(ctx: CaseContext):
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
        # 修复（eval batch A）：以 engine 侧 Cancel RPC 计数
        # （_cancel_rpc_total / snapshot rpc_counts.cancel）为收证口径——
        # 二次 cancel 不重复触发 engine 取消正是本用例语义核心。
        baseline_cancel = _cancel_rpc_total(ops)
        first_cancel_at = time.monotonic()
        ops.cancel(rid, response)
        # 时效契约：第一次 cancel 的 engine 收证须在 5s 内（BATCH）。
        if response.enqueued_by_master:
            engine_cancelled, cancel_detail = _engine_cancel_receipt_within(
                ops, rid, timeout_s=5.0, since=first_cancel_at
            )
        else:
            engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        first_cancel_delta = _cancel_rpc_total(ops) - baseline_cancel
        second_cancel_ok, second_cancel_err = True, ""
        try:
            ops.cancel(rid, response)
        except Exception as exc:
            second_cancel_ok, second_cancel_err = False, repr(exc)
        ended = handle.wait_end(5.0)
        recovery_ok, recovery_msg = ops.verify_recovery()
        method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
        engine_recv, recv_detail = ops.verify_engine_received(rid, method)
        # recovery 之后读取：二次 cancel 若错误地又触发 engine 取消，异步
        # 转发已在 recovery 窗口内落地，此处计数已覆盖。
        second_cancel_delta = (
            _cancel_rpc_total(ops) - baseline_cancel - first_cancel_delta
        )
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        # 修复（eval batch A）：BATCH 分支升格 engine 收证（5s 时效）+
        # 收证计数恰为 1（第二次 cancel 命中终态快照，不得再触发 engine
        # 取消）；NON_BATCH 二次 cancel 直发 worker（幂等 NOT_FOUND），
        # 计数断言不适用，保持现状。
        if response.enqueued_by_master:
            passed = (
                second_cancel_ok
                and ended
                and recovery_ok
                and engine_cancelled
                and first_cancel_delta >= 1
                and second_cancel_delta == 0
            )
        else:
            passed = second_cancel_ok and ended and recovery_ok
        return passed, (
            f"second_cancel_ok={second_cancel_ok} {second_cancel_err}, "
            f"stream_terminated={ended}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"cancel_rpc_delta=first:{first_cancel_delta}/"
            f"second:{second_cancel_delta}"
            f"[{'hard' if response.enqueued_by_master else 'observational'}], "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("cancel_sibling_isolation", source="cancel_smoke.py T3")
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
                # Drainage discipline (S4 lesson, 2026-08-27 task #63
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


@case("cancel_after_terminal", source="cancel_smoke.py T4")
def cancel_after_terminal(ctx: CaseContext):
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
        # 修复（eval batch A）：终态请求的 cancel 不应产生 engine 取消
        # 动作（cancel_engine_notfound_settle 同契约已证可断）+ master
        # 账本不被重开（inflight_ok；NON_BATCH 下恒 N/A 不放大失败面）。
        passed = cancel_ok and recovery_ok and not engine_cancelled and inflight_ok
        return passed, (
            f"cancel_ok={cancel_ok} {cancel_err}, completed={handle.snap.completed}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail})"
            "[expect False — terminal preserved], "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("cancel_unknown_rid", source="cancel_smoke.py T5")
def cancel_unknown_rid(ctx: CaseContext):
    """Cancel for a rid the master has never seen: typed NOT_FOUND, zero
    ledger mutation.

    Rewritten in the 2026-09 eval batch A: the legacy port only asserted
    "the Cancel RPC did not raise" — nearly vacuous (any well-formed
    gRPC error also satisfies it).  The real contract has two layers:

      * response semantics: the master answers the unknown-rid Cancel
        with a typed NOT_FOUND — either an OK response carrying
        found=false (RequestRegistry has no slot for the rid) or a gRPC
        NOT_FOUND status; any other outcome (a found=true hallucination,
        INTERNAL/UNAVAILABLE/...) fails;
      * ledger invariant: the master inflight fingerprint is
        bit-identical before vs after the cancel (scheduler count +
        every endpoint row — same construction as the status family's
        _inflight_fingerprint, copied locally to avoid cross-category
        import churn).
    """
    ops = ctx.ops()
    try:
        fake_rid = 99999
        # 清零基线（status_unknown_rid_finished 先例）：共享 env 的前序
        # 残渣排空后，"逐位不变"才可观察。
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        before = _inflight_fingerprint(ops)
        semantics_ok, semantics_detail = False, "no attempt"
        try:
            stub = ops.schedule_pb2_grpc.FlexlbServiceStub(
                ops._channel(ops.master_target())
            )
            ack = stub.Cancel(
                ops.schedule_pb2.FlexlbCancelRequestPB(
                    request_id=fake_rid,
                    reason=ops.schedule_pb2.CANCEL_REASON_CLIENT_CANCELLED,
                ),
                timeout=10.0,
            )
            if not ack.found:
                semantics_ok = True
                semantics_detail = "typed NOT_FOUND (found=false)"
            else:
                semantics_detail = (
                    f"found=true for unknown rid={fake_rid} "
                    f"(lifecycle={ack.lifecycle})"
                )
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.NOT_FOUND:
                semantics_ok = True
                semantics_detail = "gRPC NOT_FOUND status"
            else:
                semantics_detail = f"gRPC {exc.code()}: {exc.details()}"
        except Exception as exc:
            semantics_detail = repr(exc)

        after = _inflight_fingerprint(ops)
        # 修复（eval batch A）：两层真断言——响应语义（NOT_FOUND 或幂等
        # found=false，其它 gRPC 状态皆 fail）+ 账本指纹逐位不变。
        ledger_unchanged = before is not None and after is not None and before == after
        passed = clean0 and semantics_ok and ledger_unchanged
        return passed, (
            f"cancel(rid={fake_rid}): semantics={semantics_ok}"
            f"({semantics_detail}), "
            f"baseline_clean={clean0}({clean0_detail}), "
            f"ledger_unchanged={ledger_unchanged} "
            f"(before={before}, after={after})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("cancel_phase_timing", source="cancel_smoke.py T6")
def cancel_phase_timing(ctx: CaseContext):
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
        a_cancel_at = time.monotonic()
        ops.cancel(rid_a, resp_a)
        a_ended = handle_a.wait_end(5.0)
        # 时效契约：A 的 engine 收证须在 A cancel 发出后 5s 内（BATCH）。
        if resp_a.enqueued_by_master:
            engine_cancelled_a, cancel_detail_a = _engine_cancel_receipt_within(
                ops, rid_a, timeout_s=5.0, since=a_cancel_at
            )
        else:
            engine_cancelled_a, cancel_detail_a = ops.verify_engine_cancelled(rid_a)

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
        b_cancel_at = time.monotonic()
        ops.cancel(rid_b, resp_b)
        b_ended = handle_b.wait_end(5.0)
        # 时效契约：B 的 engine 收证须在 B cancel 发出后 5s 内（BATCH）。
        if resp_b.enqueued_by_master:
            engine_cancelled_b, cancel_detail_b = _engine_cancel_receipt_within(
                ops, rid_b, timeout_s=5.0, since=b_cancel_at
            )
        else:
            engine_cancelled_b, cancel_detail_b = ops.verify_engine_cancelled(rid_b)

        recovery_ok, recovery_msg = ops.verify_recovery()
        method_a = "enqueue_batch" if resp_a.enqueued_by_master else "generate_stream"
        method_b = "enqueue_batch" if resp_b.enqueued_by_master else "generate_stream"
        engine_recv_a, _ = ops.verify_engine_received(rid_a, method_a)
        engine_recv_b, _ = ops.verify_engine_received(rid_b, method_b)
        if resp_a.enqueued_by_master or resp_b.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"

        # 修复（eval batch A）：engine 收证（5s 时效契约）与 master 账本
        # 排空升格进 passed——BATCH 分支按各自交付模式断言。
        passed = a_ended and b_ended and a_in_prefill and recovery_ok
        if resp_a.enqueued_by_master:
            passed = passed and engine_cancelled_a
        if resp_b.enqueued_by_master:
            passed = passed and engine_cancelled_b
        if resp_a.enqueued_by_master or resp_b.enqueued_by_master:
            passed = passed and inflight_ok
        return passed, (
            f"A_prefill_phase={a_in_prefill}, A_terminated={a_ended}, "
            f"A_outputs={len(handle_a.snap.outputs)}, "
            f"B_decode_phase={b_got_first}, B_terminated={b_ended}, "
            f"B_outputs={len(handle_b.snap.outputs)}, "
            f"engine_recv_A={engine_recv_a}, "
            f"engine_cancel_A={engine_cancelled_a}({cancel_detail_a}), "
            f"engine_recv_B={engine_recv_b}, "
            f"engine_cancel_B={engine_cancelled_b}({cancel_detail_b}), "
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
def cancel_anomaly_path(ctx: CaseContext):
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
    republishing any terminal — production ground truth (C++ Cancel
    handler): NOT_FOUND means "seen but already terminal" (the
    completion record stays deliverable from the retain-window
    backlog).

    Expected (contract): the master Cancel RPC succeeds (idempotent);
    the direct engine Cancel answers NOT_FOUND (seen-and-terminal; the
    retain window keeps the completion deliverable); the engine's
    recorded terminal stays a completion (no cancelled_rids entry /
    lifecycle rewrite); the master inflight ledger stays clean (nothing
    re-opened); a follow-up request completes normally.

    Prediction: passes (cancel_after_terminal already covers the
    master-idempotent half; the engine branch is the mock's
    production-faithful three-branch cancel semantics: ACCEPTED (live
    or active-cancel tombstone) / NOT_FOUND (seen but already terminal
    — this case) / TOMBSTONED (never-seen rid, absent fence installed).
    The production 10-minute recently-seen TTL is simplified away in
    the mock: every cancel in these cases is a sub-second race, far
    inside that window).
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
        # the engine AFTER the terminal must read NOT_FOUND (the
        # seen-and-terminal branch of the production three-branch cancel
        # map; TOMBSTONED is reserved for never-seen rids whose absent
        # fence blocks later Enqueues).
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

    Prediction: expected to pass — this is the legacy priority-preemption
    smoke scenario (RUNNING decode victim, batch default) ported onto the
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


# ===========================================================================
# HA cancel family (task #26, 2026-09): the cancel contract across engine
# restarts, dead-prefill windows, decode retirement and transport-layer
# faults.  Production ground truth (code-audited):
#   * the master cancel is ONE-SHOT — EngineFenceCoordinator "never
#     retries and never owns a timer"; a TOMBSTONED ack settles the slot
#     immediately (resumeTombstoned) while ACCEPTED / NOT_FOUND / FAILED /
#     exceptions park in awaitAuthoritativeTerminal;
#   * a TRUE engine crash (crash_after) wipes all per-engine memory — a
#     restarted instance has NEVER SEEN pre-restart rids, so the master's
#     cancel answers TOMBSTONED and installs the ABSENT_FENCE tombstone
#     (later same-rid enqueues are 8429-rejected pre-admission);
#   * decode WorkerStatus terminals and decode generation retire close
#     open cancellation fences; prefill retire never closes fenced slots;
#     there is no fallback sweeper for cancellation first-cause slots (a
#     known production gap — the windows below are sized to the REAL
#     settle paths instead of relying on a nonexistent safety net);
#   * fencing is in-engine memory only: a second crash drops the tombstone
#     and a late Enqueue of the settled rid is ACCEPTED by the fresh
#     instance (documented design trade-off: the master ledger stays
#     settled — no resurrection — and the engine-side orphan computation
#     is bounded).
# ===========================================================================

# 3-strike health demotion + eviction window (engine_fault precedent).
MASTER_EVICT_S = 30.0
# Engine restart channel-reconnect settle window (engine_fault precedent).
ENGINE_RECOVERY_WAIT_S = 3.0
# A TOMBSTONED cancel settles the slot immediately — the client stream must
# close well inside this bound, far away from the 95s TTL drain net.
CANCEL_SETTLE_BOUND_S = 5.0


def _direct_enqueue(ops, addr: str, input_pb, batch_id: int):
    """Client-side EnqueueBatch probe straight at one engine's gRPC port.

    Bypasses the master entirely — the late-Enqueue / fence probes below
    must observe the ENGINE's admission decision, not the master's
    already-settled ledger view.
    """
    stub = ops.pb2_grpc.RpcServiceStub(ops._channel(addr))
    request = ops.pb2.EnqueueBatchRequestPB(
        batch_id=batch_id,
        dp_slots=[
            ops.pb2.EnqueueBatchDpSlotPB(
                dp_rank=0,
                requests=[ops.pb2.EnqueueBatchExternalInputPB(input=input_pb)],
            )
        ],
        fetch_attach_timeout_ms=30_000,
    )
    return stub.EnqueueBatch(request, timeout=10.0)


def _fence_rejected_8429(ack, rid: int) -> tuple:
    """True when the direct-enqueue ack carries exactly the typed 8429
    absent-fence rejection for rid (no successes, no admission)."""
    errors = list(ack.errors)
    rejected = (
        not ack.successes
        and len(errors) == 1
        and errors[0].request_id == rid
        and errors[0].error_info.error_code == 8429
    )
    detail = (
        f"successes={len(ack.successes)}, errors="
        f"{[(e.request_id, e.error_info.error_code) for e in errors]}"
    )
    return rejected, detail


def _crash_and_restart(ops, engine_name: str) -> tuple:
    """True-crash + restart cycle on one engine (crash_after n=1).

    The sacrificial request's own fate is the empty-ack uncertain path and
    is deliberately not asserted (engine_fault_crash_after precedent);
    what matters here is that ALL per-engine memory — running tasks,
    cancel tombstones, absent-fence records, RPC counters — is wiped, so
    the restarted instance has never seen any pre-restart rid.  Returns
    (alive_dropped, alive_restored).
    """
    inject_type(ops, engine_name, "crash_after", n=1)
    try:
        sacrificial = ops.next_request_id()
        ops.schedule(sacrificial, timeout_s=8.0)
    except Exception:
        # The crash may cut the RPC mid-flight — either way the port dies.
        pass
    dropped = wait_for(
        lambda: ops.master_alive_count("PREFILL") <= 0, MASTER_EVICT_S, 0.5
    )
    ops.start_engine(engine_name)  # clears fault config + enqueue counter
    restored = wait_for(
        lambda: ops.master_alive_count("PREFILL") >= 1, MASTER_EVICT_S, 0.5
    )
    time.sleep(ENGINE_RECOVERY_WAIT_S)
    return dropped, restored


def _engine_grpc_ready(ops, addr: str) -> bool:
    """gRPC-level engine readiness: the idempotent CheckHealth RPC answers
    only when the engine's gRPC server is actually serving requests — a
    bound TCP socket (wait_for_port) can still reject RPCs for a beat
    after a restart bind (_InactiveRpcError / UNAVAILABLE "Socket
    closed")."""
    try:
        stub = ops.pb2_grpc.RpcServiceStub(ops._channel(addr))
        stub.CheckHealth(ops.pb2.EmptyPB(), timeout=2.0)
        return True
    except grpc.RpcError:
        return False


def _restore_engines(ops) -> None:
    """Best-effort topology + fault restore for finally blocks."""
    try:
        for name, engine in ops.snapshot_by_name().items():
            if engine.get("stopped"):
                try:
                    ops.start_engine(name)
                except Exception:
                    pass
            try:
                inject_type(ops, name, "crash_after", enabled=False)
            except Exception:
                pass
    except Exception:
        pass


def _ha_env(ctx: CaseContext, label_suffix: str) -> tuple:
    """1P/1D dedicated env for the HA cancel family.

    One prefill keeps the crash trigger deterministic (every enqueue lands
    on prefill-0); one decode keeps the handoff target unambiguous.  The
    label embeds the profile so each profile gets its own env instance.
    """
    spec = EnvSpec(
        label=f"cancel_{label_suffix}_{ctx.profile}",
        n_prefill=1,
        n_decode=1,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={"FLEXLB_CONFIG": flexlb_config_for_profile(ctx.profile)},
    )
    env = ctx.env_manager.ensure(spec)
    return ctx.engine_ops(env), env


@case("cancel_engine_restarted_tombstoned_settle", requires=["enqueue_batch"])
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
        ops.cancel(rid, response)
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


@case("cancel_prefill_dead_await_terminal", requires=["enqueue_batch"])
def cancel_prefill_dead_await_terminal(ctx: CaseContext):
    """Dead prefill mid-cancel-window: the decode leg is the authority.

    Scenario (BATCH, stable ordering — stop FIRST, then cancel): R1 is
    handed to decode (first output received) when its original prefill is
    stopped (gRPC port closed; per-engine memory retained — only the
    restart cases need the wipe).  The client cancel is then issued: the
    master's one-shot cancel RPC hits the dead port, fails at the
    transport layer and parks the fence in awaitAuthoritativeTerminal —
    no retry, no timer.

    Expected (contract): the client outcome is BOUNDED — the stream ends
    inside the decode terminal-delivery horizon (typed cancelled OR
    completed are both correct: whether the cancel could ever reach the
    engine is exactly the race this case keeps open on purpose); the slot
    settles through the decode WorkerStatus terminal; the master ledger
    drains; the engines report inflight 0 with no leak; the restored
    prefill serves a follow-up request normally.

    Assertion-window rationale: the settle path is R1's remaining decode
    (~4s at output_len=500) plus the WorkerStatus delivery period —
    bounded well inside the 30s windows below; the 95s TTL drain is the
    safety net, not an acceptable path.
    """
    ops, _ = _ha_env(ctx, "prefill_dead")
    base = rid_base(ctx, "cancel")
    handle = None
    try:
        rid = ops.next_request_id(base)
        response = ops.schedule(rid, output_len=500)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None
            if response.enqueued_by_master
            else ops.build_generate_input(rid, output_len=500)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            return False, "no output before the dead-prefill window"

        # Stable ordering: stop first, then cancel — the cancel RPC fails
        # at the transport layer for sure (port closed), exercising the
        # awaitAuthoritativeTerminal path deterministically.
        ops.stop_engine("prefill-0")
        ops.cancel(rid, response)

        ended = handle.wait_end(30.0)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 20.0
        )

        # Restore the topology before the recovery probe.
        _restore_engines(ops)
        # Readiness gate (_crash_and_restart precedent): verify_recovery
        # is a single-shot probe, and the stop->start endpoint replacement
        # races the fresh engine's startup — an early probe can hit the
        # old, already-shutdown batcher's stopped queue.  Wait for the
        # master to see the restored prefill alive, then hand the engine
        # its recovery settle window before probing.
        wait_for(lambda: ops.master_alive_count("PREFILL") >= 1, MASTER_EVICT_S, 0.5)
        time.sleep(ENGINE_RECOVERY_WAIT_S)
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = ended and inflight_ok and engine_clean and recovery_ok
        return passed, (
            f"prefill_dead_await_terminal: stream_ended={ended}"
            f"(completed={handle.snap.completed}, "
            f"error={str(handle.snap.error)[:60]}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if handle is not None:
            handle.cancel()
        _restore_engines(ops)


@case("cancel_decode_retire_closes_fence", requires=["enqueue_batch"])
def cancel_decode_retire_closes_fence(ctx: CaseContext):
    """Decode generation retire closes an AWAIT_TERMINAL cancel fence.

    Scenario (BATCH, stable ordering): R1 is handed to decode (first
    output received); its prefill is stopped first so the client cancel
    fails at the transport layer and the fence parks in
    awaitAuthoritativeTerminal (one-shot, no retry, no timer).  The
    decode engine is then stopped too — BEFORE R1's decode completes — so
    the master's health poller accumulates the 3-strike failures and
    retires the decode generation, and reduceDecodeGenerationRetired is
    the production close path for the open cancellation fence.

    Expected (contract): the decode retire closes R1's fence — the slot
    settles and the client stream ends as a typed cancelled inside the
    retire horizon (3-strike eviction <= 30s + retire processing); the
    master ledger drains; the engines (restarted, memory retained by the
    mock stop) report inflight 0 with no leak once the orphan decode
    finishes; the restored topology serves a follow-up request normally.

    Prediction: passes — decode retire closing fenced slots is explicit
    production wiring.  The window where BOTH engines sit between stop
    and retire has no fallback sweeper for cancellation first-cause slots
    (a known production gap this case deliberately does NOT probe — the
    retire path itself is the contract under test).
    """
    ops, _ = _ha_env(ctx, "decode_retire")
    base = rid_base(ctx, "cancel")
    handle = None
    try:
        rid = ops.next_request_id(base)
        # output_len=1000 (~7.7s of decode) leaves room for the
        # stop-prefill → cancel → stop-decode sequence (~1s) to land
        # while decode still runs.
        response = ops.schedule(rid, output_len=1000)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None
            if response.enqueued_by_master
            else ops.build_generate_input(rid, output_len=1000)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            return False, "no output before the retire window"

        # Fence parks in AWAIT_TERMINAL (cancel to the dead prefill port
        # fails at the transport layer), then the decode dies too.
        ops.stop_engine("prefill-0")
        ops.cancel(rid, response)
        ops.stop_engine("decode-0")

        ended = handle.wait_end(45.0)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 45.0
        )
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 30.0
        )
        _restore_engines(ops)
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = (
            ended
            and not handle.snap.completed
            and inflight_ok
            and engine_clean
            and recovery_ok
        )
        return passed, (
            f"decode_retire_closes_fence: stream_ended={ended}"
            f"(completed={handle.snap.completed}, "
            f"error={str(handle.snap.error)[:60]}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if handle is not None:
            handle.cancel()
        _restore_engines(ops)


@case("cancel_fencing_lost_on_engine_restart", requires=["enqueue_batch"])
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
        ops.cancel(rid, response)
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
            # server is processing RPCs.  Two-level gate here: the TCP port
            # probe, then the idempotent CheckHealth RPC — a bound socket
            # can still reject RPCs for a beat, and the probe itself stays
            # single-shot, no retry (a re-sent rid could be admitted twice).
            probe2_addr = ops.prefill_addr(response)
            probe2_host, _, probe2_port = probe2_addr.rpartition(":")
            wait_for_port(probe2_host, int(probe2_port), 10.0)
            wait_for(lambda: _engine_grpc_ready(ops, probe2_addr), 10.0, 0.2)
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


@case(
    "cancel_transport_failure_one_shot",
    requires=["enqueue_batch"],
    skip_reason=(
        "v2-only: the strict one-shot cancel contract (EngineFenceCoordinator "
        "'never retries and never owns a timer') is not present in the dsv4 "
        "v1 stack, whose EngineFencePolicy runs a bounded fast-probe ladder "
        "(8 probes, 100ms->5s exponential backoff) plus a 60s quarantine "
        "sweep, so the cancel RPC delta is ladder-sized, not 1"
    ),
)
def cancel_transport_failure_one_shot(ctx: CaseContext):
    """One-shot cancel under transport failure: no retry, decode settles.

    Scenario (BATCH): R1 is handed to decode (first output received) when
    its prefill is armed with cancel_no_respond — the engine's Cancel RPC
    handler counts the arrival and HANGS (an RPC-layer fault injected
    BEFORE the engine cancel state machine: no fence, no tombstone, the
    request keeps running untouched).  The client cancel is issued; the
    master's short cancel-ack timeout (50ms) fails the future and the
    fence parks in awaitAuthoritativeTerminal.

    Expected (contract — EngineFenceCoordinator "never retries and never
    owns a timer"; the no-retry design is explicit: a retry would flip an
    already-ACCEPTED cancel into a NOT_FOUND false negative):
      * the engine records EXACTLY ONE cancel RPC arrival — hard one-shot
        assertion on the engine-side counter, re-sampled after the settle
        window so a hidden retry would surface;
      * the request settles through the decode leg's authoritative
        terminal (client outcome bounded — typed cancelled or completed
        are both correct: the cancel never reached the engine);
      * the master ledger drains through that terminal; no engine-side
        leak; after the injection is cleared a follow-up request
        completes normally.
    """
    ops, _ = _ha_env(ctx, "oneshot")
    base = rid_base(ctx, "cancel")
    handle = None
    prefill_names = None
    try:
        rid = ops.next_request_id(base)
        response = ops.schedule(rid, output_len=500)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None
            if response.enqueued_by_master
            else ops.build_generate_input(rid, output_len=500)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            return False, "no output before the injection window"

        prefill_names = _prefill_names(ops)
        inject_type_all(ops, prefill_names, "cancel_no_respond")
        baseline_cancel = _cancel_rpc_total(ops)
        ops.cancel(rid, response)
        # Settle window: the decode leg finishes R1 (~4s) and its
        # WorkerStatus terminal settles the slot; a master retry would
        # move the engine counter past 1 inside this window — the
        # post-settle re-sample below is the one-shot proof.
        ended = handle.wait_end(30.0)
        time.sleep(2.0)
        cancel_delta = _cancel_rpc_total(ops) - baseline_cancel
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 20.0
        )
        clear_type_all(ops, prefill_names, "cancel_no_respond")
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = (
            ended and cancel_delta == 1 and inflight_ok and engine_clean and recovery_ok
        )
        return passed, (
            f"transport_failure_one_shot: stream_ended={ended}"
            f"(completed={handle.snap.completed}, "
            f"error={str(handle.snap.error)[:60]}), "
            f"cancel_rpc_delta={cancel_delta} (== 1, one-shot contract), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if handle is not None:
            handle.cancel()
        if prefill_names:
            clear_type_all(ops, prefill_names, "cancel_no_respond")


@case(
    "cancel_unexpected_status_await_terminal",
    requires=["enqueue_batch"],
    skip_reason=(
        "v2-only: the strict one-shot cancel contract (EngineFenceCoordinator "
        "'never retries and never owns a timer') is not present in the dsv4 "
        "v1 stack, whose EngineFencePolicy runs a bounded fast-probe ladder "
        "(8 probes, 100ms->5s exponential backoff) plus a 60s quarantine "
        "sweep, so the cancel RPC delta is ladder-sized, not 1"
    ),
)
def cancel_unexpected_status_await_terminal(ctx: CaseContext):
    """Out-of-contract cancel ack: no false success, no false terminal.

    Scenario (BATCH): R1 is handed to decode (first output received) when
    its prefill is armed with cancel_unexpected_status — the Cancel RPC
    "succeeds" but answers a status outside the cancel contract
    (CANCEL_STATUS_UNSPECIFIED).  The fault is injected before the engine
    cancel state machine, so no fence and no tombstone are installed; the
    master's response mapping must FAIL this ack (never accept it as
    success) and the fence parks in awaitAuthoritativeTerminal — the
    same one-shot, no-retry, no-timer contract as a transport failure.

    Expected (contract): the master neither misreads the ack as success
    (which would settle the slot on a cancel the engine never applied)
    nor fails the request outright on the cancel alone — the request
    settles through the decode leg's authoritative terminal (client
    outcome bounded, typed cancelled or completed both correct); the
    engine records exactly one cancel arrival; the master ledger drains;
    no engine-side leak; no exception escapes the master (the follow-up
    probe completing normally is the liveness proof); after the injection
    is cleared everything recovers.
    """
    ops, _ = _ha_env(ctx, "unexpected")
    base = rid_base(ctx, "cancel")
    handle = None
    prefill_names = None
    try:
        rid = ops.next_request_id(base)
        response = ops.schedule(rid, output_len=500)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None
            if response.enqueued_by_master
            else ops.build_generate_input(rid, output_len=500)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            return False, "no output before the injection window"

        prefill_names = _prefill_names(ops)
        inject_type_all(ops, prefill_names, "cancel_unexpected_status")
        baseline_cancel = _cancel_rpc_total(ops)
        ops.cancel(rid, response)
        # Settle window: the UNSPECIFIED ack fails the master's mapping,
        # the fence parks in awaitAuthoritativeTerminal and the decode
        # terminal settles the slot; re-sample the counter afterwards so
        # a hidden retry would surface.
        ended = handle.wait_end(30.0)
        time.sleep(2.0)
        cancel_delta = _cancel_rpc_total(ops) - baseline_cancel
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 20.0
        )
        clear_type_all(ops, prefill_names, "cancel_unexpected_status")
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = (
            ended and cancel_delta == 1 and inflight_ok and engine_clean and recovery_ok
        )
        return passed, (
            f"unexpected_status_await_terminal: stream_ended={ended}"
            f"(completed={handle.snap.completed}, "
            f"error={str(handle.snap.error)[:60]}), "
            f"cancel_rpc_delta={cancel_delta} (== 1, one-shot contract), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_detail}), "
            f"recovery={recovery_msg} (master liveness: no exception leak)"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if handle is not None:
            handle.cancel()
        if prefill_names:
            clear_type_all(ops, prefill_names, "cancel_unexpected_status")
