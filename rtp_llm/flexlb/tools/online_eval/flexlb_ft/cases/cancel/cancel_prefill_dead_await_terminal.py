from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean
from ...harness import AssertUtils
from ...registry import case
from ...support.cancel import _all_engine_names, _ha_env, _master_http, _restore_engines


@case(
    "cancel_prefill_dead_await_terminal", category="cancel", requires=["enqueue_batch"]
)
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
