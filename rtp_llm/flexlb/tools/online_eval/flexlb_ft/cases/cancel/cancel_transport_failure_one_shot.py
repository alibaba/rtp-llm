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
    "cancel_transport_failure_one_shot",
    category="cancel",
    profiles=["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"],
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
        # Response only under BATCH — a NON_BATCH response would add the
        # worker_cancel direct connect: a SECOND engine-side Cancel that
        # breaks the hard cancel_delta == 1 one-shot assertion below.
        ops.cancel(rid, response if response.enqueued_by_master else None)
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
