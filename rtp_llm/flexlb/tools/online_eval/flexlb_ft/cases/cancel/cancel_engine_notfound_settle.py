from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import AssertUtils
from ...registry import case
from ...support.cancel import _master_http


@case("cancel_engine_notfound_settle", category="cancel")
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
