from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import AssertUtils
from ...registry import case
from ...support.cancel import _engine_cancel_receipt_within, _master_http


@case("cancel_basic", category="cancel", source="cancel_smoke.py T1")
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
