from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import AssertUtils
from ...registry import case
from ...support.status import (
    STALE_INFLIGHT_TTL_S,
    TTL_MARGIN_S,
    _inflight_fingerprint,
    _master_http,
    _master_ok,
    _prefill_names,
    _recovery_rate,
    _run_requests,
    _status_spec,
)


@case(
    "status_batch_async_partial_fail",
    category="status",
    profiles=["batch-window"],  # _status_spec pins the legacy fault axes
    source="P0 status fault family: prefill_async_partial_fail(k=1, code=8500) "
    "on 4 concurrent requests (execution-phase in-batch failure)",
)
def status_batch_async_partial_fail(ctx: CaseContext):
    """Scenario: 4 concurrent requests (output_len=2, concurrency=4) land on
    prefills injected with prefill_async_partial_fail k=1 code=8500: the
    FIRST non-cancelled member of every executing prefill batch fails AT
    the batch completion callback (execution phase), while the EnqueueBatch
    ack stays fully successful for all members — the execution-phase
    counterpart of status_ack_partial_fail.

    Behaviour (mock, production-aligned): the batch executes normally, then
    the injected member surfaces a TYPED terminal
    (task_info.error_code=8500, "injected prefill_async_partial_fail") on
    the same finished_task_list channel the survivors' terminals ride; the
    failed member starts no decode, leaves no output frame and no KV lease;
    the survivors hand off to decode and finish normally.

    Expectation (contract), five layers:
    1. Batch isolation: the survivors of each affected batch complete
       NORMALLY (>= 2 of 4 requests succeed).
    2. Typed failure terminal: every failed request's client-visible error
       carries the engine error code 8500 (master path:
       WorkerTerminalObservation -> "worker error code 8500" ->
       WORKER_EXECUTION_FAILED); 1 <= failed <= 2 — 4 requests form 1-2
       batches, each losing exactly k=1 member (the ack_partial_fail
       interval calibre).
    3. Ledger integrity: the master inflight ledger drains (inflight_clean)
       — a failure terminal missing or mismatching the dispatched batchId
       falls into the reconcile mismatch branch and leaks the slot, so this
       layer also covers batchId correctness. The failed member's slot
       settles immediately through the typed FAILED terminal (scheduler +
       prefill ledger drain fast), while its decode-side reservation is a
       soft hold reclaimed by the master's stale-inflight eviction
       (stale_inflight_ms TTL plus the eviction sweep's fixed-rate
       interval), so the drain window uses the relaxed TTL calibre
       (STALE_INFLIGHT_TTL_S + TTL_MARGIN_S + 60.0), not a fast path.
    4. No resurrection / duplicate settle: the master stays HTTP-200 and
       the drained ledger fingerprint stays stable across a late-terminal
       observation window.
    5. Batch lease closure: mixed-terminal batches release their dispatch
       capacity (BatchCompletionProjection counts members outcome-blind,
       the LAST member closes the lease) — a fresh healthy round of 4
       requests recovers >= 95%.

    Optional second form (prefill_fixed_ms=3000): stretching prefill
    execution past the batching window forces queued-serial batches — the
    same typed-terminal interval must hold when batch boundaries are
    unambiguous.

    Production alignment notes (ground truth survey):
    * Splitting is master-transparent: production EnqueueBatch has no
      split (each request enters FIFOScheduler as its own stream; partial
      admit / residual groups only reorganize execution) and the status
      report is a per-request terminal stream — batch_id travels unchanged
      from the stream group_id into TaskInfo. The mock models no split:
      whether the 4 requests form 1 or 2 batches, the shape above is the
      legal outcome.
    * In-batch async failure semantics: after admission an execution-phase
      failure goes stream->reportError -> the dequeue fills
      task_info.error_code / error_message -> finished_task_list,
      independent of the same batch's surviving members (no batch-level
      failure propagation); each member settles on its own.
      stream->reportError is a TWO-channel terminal — the client-visible
      half terminates the request's own output stream with an error frame
      (the mock engine delivers RpcErrorPB carrying the injected code),
      the master half rides finished_task_list into the reconcile; layer 2
      asserts the former (client error string carries 8500), layer 3 the
      latter (ledger drains through the typed FAILED terminal).
    * Master settlement: a P-side terminal fact with errorCode != 0 maps
      to Kind.FAILED -> WorkerStatusFact.terminal -> the slot's failure
      terminal (errorCode == 0 non-cancel prefill returns null — the
      success terminal is left for decode, which is exactly what the
      survivors exercise in parallel); no re-dispatch, no whole-batch
      abandon.

    Grade: P0."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"

    try:
        # ── Arm 1 (default perf): the batching window forms 1-2 batches.
        # typed_stream_error: the injected member's output stream terminates
        # with an in-band error frame (production stream->reportError's
        # client-visible half) — the failed request's error carries the
        # engine code 8500 instead of the generic stream-timeout form.
        inject_type_all(ops, names, "prefill_async_partial_fail", k=1, code=8500)
        try:
            errs = _run_requests(
                ops,
                base,
                4,
                output_len=2,
                concurrency=4,
                typed_stream_error=True,
            )
        finally:
            clear_type_all(ops, names, "prefill_async_partial_fail")
        failed = [e for e in errs if e is not None]
        ok = len(errs) - len(failed)
        typed_terminal = (
            1 <= len(failed) <= 2 and ok >= 2 and all("8500" in str(e) for e in failed)
        )
        error_shapes = sorted({str(e)[:80] for e in failed})[:3]

        # ── Layers 3+4: ledger integrity + no resurrection.
        # Relaxed window: the failed member's decode-side reservation is a
        # soft hold reclaimed by stale-inflight eviction (TTL + the sweep's
        # fixed-rate interval), so use the relaxed TTL calibre like the
        # other relaxed precedents in this file.
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), STALE_INFLIGHT_TTL_S + TTL_MARGIN_S + 60.0
        )
        fingerprint = _inflight_fingerprint(ops)
        time.sleep(3.0)  # late-terminal / duplicate-report window
        stable = _inflight_fingerprint(ops) == fingerprint
        master_ok = _master_ok(ops)

        # ── Layer 5: batch lease closure via a healthy round.
        recovery_ok, recovery_msg = _recovery_rate(ops, base, 4)

        # ── Arm 2 (prefill_fixed_ms=3000): queued-serial batches.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)
        inject_type_all(ops, names, "prefill_async_partial_fail", k=1, code=8500)
        try:
            errs2 = _run_requests(
                ops,
                base,
                4,
                output_len=2,
                concurrency=4,
                typed_stream_error=True,
            )
        finally:
            clear_type_all(ops, names, "prefill_async_partial_fail")
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        failed2 = [e for e in errs2 if e is not None]
        ok2 = len(errs2) - len(failed2)
        typed_terminal_2 = (
            1 <= len(failed2) <= 2
            and ok2 >= 2
            and all("8500" in str(e) for e in failed2)
        )
        inflight_ok2, inflight_detail2 = AssertUtils.inflight_clean(
            _master_http(ops), STALE_INFLIGHT_TTL_S + TTL_MARGIN_S + 60.0
        )

        passed = (
            typed_terminal
            and inflight_ok
            and stable
            and master_ok
            and recovery_ok
            and typed_terminal_2
            and inflight_ok2
        )
        return passed, (
            f"arm1_typed_terminal={typed_terminal} "
            f"(failed={len(failed)}, ok={ok}, shapes={error_shapes}), "
            f"arm1_inflight_clean={inflight_ok}({inflight_detail}), "
            f"no_resurrection={stable}, master_200={master_ok}, "
            f"recovery={recovery_msg}, "
            f"arm2_serial_typed_terminal={typed_terminal_2} "
            f"(failed={len(failed2)}, ok={ok2}), "
            f"arm2_inflight_clean={inflight_ok2}({inflight_detail2})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "prefill_async_partial_fail")
        for n in names:
            try:
                ops.set_perf(n, prefill_fixed_ms=100.0)
            except Exception:
                pass
