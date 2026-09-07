from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.cancel import _all_engine_names, _master_http


@case(
    "cancel_stream_break_prefill_autonomous",
    category="cancel",
    requires=["enqueue_batch"],
)
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
