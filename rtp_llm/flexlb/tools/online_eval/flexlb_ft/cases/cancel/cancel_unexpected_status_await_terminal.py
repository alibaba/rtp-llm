from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, engine_inflight_clean, inject_type_all
from ...harness import AssertUtils
from ...registry import case
from ...support.cancel import (
    _all_engine_names,
    _cancel_rpc_total,
    _ha_env,
    _master_http,
    _prefill_names,
)


@case(
    "cancel_unexpected_status_await_terminal",
    category="cancel",
    profiles=["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"],
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
        # Response only under BATCH — a NON_BATCH response would add the
        # worker_cancel direct connect: a SECOND engine-side Cancel that
        # breaks the hard cancel_delta == 1 one-shot assertion below.
        ops.cancel(rid, response if response.enqueued_by_master else None)
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
