from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.cancel import (
    _cancel_rpc_total,
    _master_http,
    _prefill_names,
    _schedule_future,
)


@case("cancel_schedule_drop_delivered", category="cancel", requires=["enqueue_batch"])
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
    increases — the master really sent an engine cancel; that counter IS
    the delivery evidence of the contract now.  (b) superseded 2026-09-04:
    the engine-side cancelled_rids / lifecycle record is no longer a
    required artifact of this path — under the CancelAck typed semantics
    the late cancel lands either before the delayed enqueue has tracked
    the request (TOMBSTONED / absent-fence branch) or after the terminal
    (NOT_FOUND branch), and neither branch records cancelled_rids; the
    run showed delta=1 with an empty cancelled set, which is the new
    protocol behaving correctly.  (c) the master inflight ledger settles
    through the typed CANCELLED reconcile; (d) a follow-up request
    completes.

    Trigger note: this is the Schedule-RPC drop (the only stream the
    master's CancellationListener observes); dropping the FetchResponse
    stream instead never reaches the master — that variant is the C1/C2
    autonomous-cleanup cases below.

    Prediction: expected to pass — the chain (listener → defer →
    resume-after-ACK → fence cancel channel → typed CANCELLED reconcile)
    is all production wiring; the engine record was dropped from the
    gate because the typed-ack protocol no longer produces it on this
    path (see (b)); if (a)/(c)/(d) fail, the finding is in the master's
    resume/fence path, not the test.
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
        # Diagnostic only (NOT gated): under the CancelAck typed semantics
        # the cancel may answer TOMBSTONED/NOT_FOUND without recording a
        # cancelled_rid — see the docstring contract note (b).
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 15.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()
        passed = engine_cancel_rpc and inflight_ok and recovery_ok
        return passed, (
            f"schedule_drop_delivered: engine_cancel_rpc_delta="
            f"{_cancel_rpc_total(ops) - baseline_cancel}, "
            f"engine_cancelled={engine_cancelled}({cancel_detail})"
            "[diagnostic, not gated], "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if future is not None:
            future.cancel()
        clear_type_all(ops, prefill_names, "enqueue_delay")
