from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...registry import case
from ...support.status import (
    _master_ok,
    _prefill_names,
    _run_requests,
    _status_spec,
    _wait_scheduler_zero,
)


@case(
    "status_ack_multi_error",
    category="status",
    profiles=["batch-window"],
    source="P0 status fault family: enqueue_ack_error_code with two distinct codes",
)
def status_ack_multi_error(ctx: CaseContext):
    """Scenario: two enqueue batches fail with DIFFERENT injected error
    codes (enqueue_ack_error_code 8431 then 8510).

    Behaviour: each batch's ack carries its own injected code for every
    member of that batch.

    Expectation (contract): the error code is passed through PER REQUEST —
    every failed request of batch A surfaces 8431 and every one of batch B
    surfaces 8510 in its client-visible error; the master never crashes
    (HTTP 200 throughout); the failed batches do not resurrect (no retry:
    scheduler inflight settles to zero and stays there).

    Grade: P0."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        results = []
        for code in (8431, 8510):
            inject_type_all(ops, names, "enqueue_ack_error_code", code=code)
            try:
                errs = _run_requests(ops, base, 2, concurrency=2)
            finally:
                clear_type_all(ops, names, "enqueue_ack_error_code")
            results.append((code, errs))

        # Per-request passthrough: every request fails and carries ITS
        # batch's injected code in the client-visible error text.
        passthrough = all(
            errs and all(str(code) in str(e) for e in errs) for code, errs in results
        )
        err_samples = {
            code: [str(e)[:70] for e in errs if e][:2] for code, errs in results
        }

        # No resurrection: the failed batches must not re-enter the ledger.
        settle_ok = _wait_scheduler_zero(ops, 15.0)
        time.sleep(3.0)
        stable = ops.master_scheduler_inflight() == 0
        no_resurrect = settle_ok and stable

        master_ok = _master_ok(ops)
        passed = passthrough and no_resurrect and master_ok
        return passed, (
            f"code_passthrough={passthrough} (samples={err_samples}), "
            f"no_resurrect={no_resurrect} "
            f"(settle={settle_ok}, stable_after_3s={stable}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "enqueue_ack_error_code")
