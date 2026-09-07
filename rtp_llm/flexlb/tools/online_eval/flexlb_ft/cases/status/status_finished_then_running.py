from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type
from ...harness import AssertUtils
from ...registry import case
from ...support.status import (
    STREAM_TIMEOUT_S,
    _inflight_fingerprint,
    _master_http,
    _master_ok,
    _prefill_names,
    _status_spec,
)


@case(
    "status_finished_then_running",
    category="status",
    profiles=["batch-window"],
    source="P1 status fault family: fake_task sequence — finished replay then persistent RUNNING for a settled rid",
)
def status_finished_then_running(ctx: CaseContext):
    """Scenario: after a request completes, the engine FIRST replays its
    finished fact (fake_task phase=finished), THEN keeps reporting it
    RUNNING (fake_task phase=RUNNING, persistent).

    Behaviour: a terminal replay followed by an out-of-order ACTIVE for an
    already-settled request.

    Expectation (contract): no resurrection, no rollback — the terminal is
    final: the inflight ledger stays clean across both injections (the
    settled slot must not re-enter inflight) and the master stays HTTP 200.

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        rid = ops.next_request_id(base)
        _, err0 = ops.run_one_request(
            rid, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
        )
        if err0:
            return False, f"baseline request failed: {err0}"
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        fp_baseline = _inflight_fingerprint(ops)

        # Phase 1 — replay the terminal (idempotency).
        inject_type(ops, names[0], "status_fake_task", rid=rid, phase="finished")
        try:
            time.sleep(2.0)
        finally:
            clear_type_all(ops, names, "status_fake_task")

        # Phase 2 — persistent RUNNING for the settled rid (resurrection
        # attempt).  Hold it open across several poll rounds.  The window
        # fingerprint is a hard assertion, not an observation: a transient
        # resurrection that only disappears after the injection is cleared
        # would escape a post-clear-only check.
        inject_type(ops, names[0], "status_fake_task", rid=rid, phase="RUNNING")
        try:
            time.sleep(5.0)
            fp_during = _inflight_fingerprint(ops)
        finally:
            clear_type_all(ops, names, "status_fake_task")
        no_resurrect_during = fp_during is not None and fp_during == fp_baseline

        clean_final, clean_final_detail = AssertUtils.inflight_clean(
            _master_http(ops), 20.0
        )
        master_ok = _master_ok(ops)

        passed = clean_ok and no_resurrect_during and clean_final and master_ok
        return passed, (
            f"terminal_final={clean_final} "
            f"(baseline_clean={clean_ok}, no_resurrect_during_window="
            f"{no_resurrect_during}, after_resurrection_attempt="
            f"{clean_final_detail}, fingerprint_during_running={fp_during}, "
            f"fingerprint_baseline={fp_baseline}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")
