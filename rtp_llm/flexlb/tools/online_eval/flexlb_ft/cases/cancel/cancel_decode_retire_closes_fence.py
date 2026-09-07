from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean
from ...harness import AssertUtils
from ...registry import case
from ...support.cancel import _all_engine_names, _ha_env, _master_http, _restore_engines


@case(
    "cancel_decode_retire_closes_fence",
    category="cancel",
    profiles=["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"],
)
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
        # Response only under BATCH — a NON_BATCH response would add the
        # worker_cancel direct connect at the now-dead prefill port.
        ops.cancel(rid, response if response.enqueued_by_master else None)
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
