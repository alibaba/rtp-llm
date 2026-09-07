from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type
from ...harness import AssertUtils
from ...registry import case
from ...support.status import (
    GHOST_RID_OFFSET,
    _inflight_fingerprint,
    _master_http,
    _master_ok,
    _prefill_names,
    _status_spec,
    _wait_scheduler_zero,
)


@case(
    "status_unknown_rid_running",
    category="status",
    profiles=["batch-window"],
    source="P1 status fault family: status_fake_task(running, unknown rid), one-shot",
)
def status_unknown_rid_running(ctx: CaseContext):
    """Scenario: an engine reports a RUNNING fact for a request id the
    master has never seen (status_fake_task, one-shot on the first
    prefill).

    Behaviour: correct masters ignore the ghost ACTIVE (no slot exists);
    an implementation that registers it creates a resident ghost entry —
    which the stale TTL must still reclaim once the one-shot report stops.

    Expectation (contract): the master stays HTTP 200 and the scheduler
    inflight is ZERO within TTL+margin after the one-shot injection (both
    implementations converge to zero — the contract forbids a permanent
    ghost resident either way).

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    ghost_rid = base + GHOST_RID_OFFSET + 1
    try:
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        before = _inflight_fingerprint(ops)

        inject_type(ops, names[0], "status_fake_task", rid=ghost_rid, phase="RUNNING")
        try:
            time.sleep(3.0)  # the one-shot report has landed
        finally:
            clear_type_all(ops, names, "status_fake_task")

        # Did the ghost register? (observational — either answer must
        # still converge to zero below).
        peak = _inflight_fingerprint(ops)
        registered = peak is not None and peak != before
        drained = _wait_scheduler_zero(ops)
        final = ops.master_scheduler_inflight()
        master_ok = _master_ok(ops)

        passed = clean0 and drained and final == 0 and master_ok
        return passed, (
            f"baseline_clean={clean0}({clean0_detail}), "
            f"ghost_registered={registered} "
            f"(before={before}, after_injection={peak}), "
            f"scheduler_zero_within_ttl={drained} (final={final}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")
