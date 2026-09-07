from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import AssertUtils, ConfigOverride, EnvSpec, default_perf
from ...registry import case
from ...support.cancel import _master_http, _prefill_names


@case("cancel_deadline_exempt_inflight", category="cancel")
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
        config_overrides=ConfigOverride(queue_timeout_ms=2_000),
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
