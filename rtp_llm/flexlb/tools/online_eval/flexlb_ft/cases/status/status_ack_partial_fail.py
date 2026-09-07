from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import AssertUtils
from ...registry import case
from ...support.status import (
    _enqueue_rpc_count,
    _fire_and_forget,
    _master_http,
    _master_ok,
    _prefill_names,
    _prefill_requests_sum,
    _run_requests,
    _status_spec,
    _timeout_typed,
)


@case(
    "status_ack_partial_fail",
    category="status",
    profiles=["batch-window"],  # _status_spec pins the legacy fault axes
    source="P0 status fault family: enqueue_ack_partial_fail(k=1) on a 4-request batch",
    expected_fail=True,  # MIXED form (see docstring) — whole-case probe
)
def status_ack_partial_fail(ctx: CaseContext):
    """Scenario: a 4-request enqueue batch lands on prefills whose ack marks
    k=1 members failed — first with the default transient-class code 13,
    then with a permanent-class code 8431 (enqueue_ack_partial_fail +
    enqueue_ack_error_code co-injected; the mock executes every member
    either way, only the ACK lies).

    Behaviour: the mock answers EnqueueBatch with a partial failure — the
    k members carry a terminal error, the rest are acknowledged.

    Expectation (contract), three layers:
    1. Isolation + ledger release: the k members receive a TERMINAL error
       while the remaining members STILL SUCCEED; the failed members leave
       the master's prefill member ledger PROMPTLY (the ledger peak across
       the execution window never exceeds the surviving member count — 3
       for a single 4-member batch) and the ledger drains (inflight_clean).
    2. Retry dispatch shape (observation): IF a failed member is retried,
       the retry must surface as NEW EnqueueBatch RPCs — a fresh dispatch
       entry, never silently folded back into the original batch.  The
       engine snapshot exposes no batch-composition field, so the
       new-dispatch dimension is observed via the EnqueueBatch RPC-count
       delta and reported in the detail line.
    3. Retry policy matrix: a TRANSIENT code (13) with SLO budget left
       (scheduler.queueTimeoutMs=10s) MUST be retried to success or to an
       SLO-shaped terminal (deadline/timeout class — never the raw
       injected error surfaced straight through); a PERMANENT code (8431)
       must terminate FAST (1-2 failed members carrying 8431, >= 2
       survivors) with no retry RPCs beyond the original batch dispatch.

    PREDICTED FINDING (retry policy missing): the current master turns
    EngineRejectedException into an immediate terminal for EVERY error
    code, so the transient arm's failed members surface the raw injected
    error — the transient_ok assertion is EXPECTED TO FAIL and that
    failure is the finding.  The permanent arm, the isolation and the
    ledger layers pass against the current implementation.

    Expected-fail marking (MIXED form): the case mixes
    should-pass layers (Layer 1 isolation/ledger, Layer 3b permanent)
    with the predicted-fail retry dimension (Layer 3a transient), and the
    expected_fail granularity is whole-case — so the whole case is
    marked expected_fail: its expected failure classifies as
    finding-confirmed (the retry-policy finding stands), its unexpected
    pass as finding-resolved (the retry policy landed).  CAVEAT: a
    Layer-1/3b regression ALSO shows up as finding-confirmed — read the
    detail flags (hang_free / drained_a / permanent_fast) to tell a
    regression apart from the declared finding; the predicted-fail arm's
    own verdict stays visible as transient_ok=<bool> in the detail.

    Grade: P0 (isolation/ledger) + P2 retry-policy probe."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"

    def _rpc_now() -> int:
        return _enqueue_rpc_count(ops, names)

    try:
        # ── Layer 1: isolation + prompt ledger release (default code 13).
        # Fire-and-forget so the ledger is sampled WHILE the surviving
        # members still execute: the failed members get their terminal
        # from the ack itself and must leave the prefill member ledger
        # immediately — a hung member pushes the peak to the full batch
        # size (4) while the surviving shape is 3 (fewer when the 4
        # requests split into 2 batches, one failed member each).
        rpc_a0 = _rpc_now()
        inject_type_all(ops, names, "enqueue_ack_partial_fail", k=1)
        try:
            rids, sched_err = _fire_and_forget(ops, base, 4)
            if sched_err:
                return False, f"layer1 schedule failed: {sched_err}"
            samples: list[int] = []
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                samples.append(_prefill_requests_sum(ops))
                if len(samples) >= 5 and all(v == 0 for v in samples[-5:]):
                    break
                time.sleep(0.2)
            ledger_peak = max(samples) if samples else -1
            hang_free = ledger_peak <= 3
            # Soft observation (detail only): the surviving-member shape
            # (a 3-member batch ledger) was actually caught on screen.
            member_shape_seen = 3 in samples
        finally:
            clear_type_all(ops, names, "enqueue_ack_partial_fail")
        rpc_a1 = _rpc_now()
        # Layer-2 observation: with no master-side retry the dispatch count
        # stays at the original batch count (1-2 RPCs for 4 requests); a
        # retry re-dispatches and shows up as NEW EnqueueBatch RPCs.
        retry_rpc_layer1 = rpc_a1 - rpc_a0
        drained_a, drained_a_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )

        # ── Layer 3a: transient arm (code 13) — retry-policy contract.
        # enqueue_ack_error_code only takes effect when co-injected with
        # enqueue_ack_partial_fail k>0 (mock applyEnqueueAckFaults gate).
        rpc_b0 = _rpc_now()
        inject_type_all(ops, names, "enqueue_ack_partial_fail", k=1)
        inject_type_all(ops, names, "enqueue_ack_error_code", code=13)
        try:
            errs_t = _run_requests(ops, base, 4, concurrency=4)
        finally:
            clear_type_all(ops, names, "enqueue_ack_partial_fail")
            clear_type_all(ops, names, "enqueue_ack_error_code")
        time.sleep(3.0)  # late-retry observation window
        rpc_b1 = _rpc_now()
        failed_t = [e for e in errs_t if e is not None]
        # Contract: with SLO budget left a transient code must end in
        # success or an SLO-shaped terminal — the legal_terminal caliber
        # used by the suppress family.  A raw injected-error passthrough
        # is the missing-retry finding.
        transient_ok = all(e is None or _timeout_typed(e) for e in errs_t)
        transient_raw_leak = sorted(
            {str(e)[:70] for e in failed_t if not _timeout_typed(e)}
        )[:3]
        transient_rpc = rpc_b1 - rpc_b0

        # ── Layer 3b: permanent arm (code 8431) — fast terminal, no retry.
        rpc_c0 = _rpc_now()
        inject_type_all(ops, names, "enqueue_ack_partial_fail", k=1)
        inject_type_all(ops, names, "enqueue_ack_error_code", code=8431)
        try:
            errs_p = _run_requests(ops, base, 4, concurrency=4)
        finally:
            clear_type_all(ops, names, "enqueue_ack_partial_fail")
            clear_type_all(ops, names, "enqueue_ack_error_code")
        rpc_c1 = _rpc_now()
        failed_p = [e for e in errs_p if e is not None]
        ok_p = len(errs_p) - len(failed_p)
        permanent_fast = (
            1 <= len(failed_p) <= 2
            and ok_p >= 2
            and all("8431" in str(e) for e in failed_p)
        )
        # No retry RPCs beyond the original batch dispatch (4 requests
        # form 1-2 batches → 1-2 EnqueueBatch RPCs; a retry adds more).
        permanent_no_retry = 1 <= (rpc_c1 - rpc_c0) <= 2
        permanent_rpc = rpc_c1 - rpc_c0

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        master_ok = _master_ok(ops)

        passed = (
            hang_free
            and drained_a
            and transient_ok  # PREDICTED FINDING arm (missing retry)
            and permanent_fast
            and permanent_no_retry
            and inflight_ok
            and master_ok
        )
        return passed, (
            f"ledger_hang_free={hang_free} (peak={ledger_peak}, "
            f"member_shape_3_seen={member_shape_seen}), "
            f"layer1_drained={drained_a}({drained_a_detail}), "
            f"retry_rpc(layer1_obs)={retry_rpc_layer1}, "
            f"transient_ok={transient_ok} (failed={len(failed_t)}, "
            f"raw_leak={transient_raw_leak}, rpc_delta={transient_rpc}), "
            f"permanent_fast={permanent_fast} "
            f"(failed={len(failed_p)}, ok={ok_p}), "
            f"permanent_no_retry={permanent_no_retry} "
            f"(rpc_delta={permanent_rpc}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "enqueue_ack_partial_fail")
        clear_type_all(ops, names, "enqueue_ack_error_code")
