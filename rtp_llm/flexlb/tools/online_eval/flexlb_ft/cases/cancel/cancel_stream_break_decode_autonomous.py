from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.cancel import _all_engine_names, _master_http


@case(
    "cancel_stream_break_decode_autonomous",
    category="cancel",
    requires=["generate_stream"],
)
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
