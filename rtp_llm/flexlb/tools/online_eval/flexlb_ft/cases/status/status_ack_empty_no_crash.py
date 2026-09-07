from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import _fence_residue_stable, clear_type_all, inject_type_all
from ...registry import case
from ...support.status import (
    _master_ok,
    _prefill_names,
    _run_requests,
    _status_spec,
    _wait_scheduler_zero,
)


@case(
    "status_ack_empty_no_crash",
    category="status",
    profiles=["batch-window"],
    source="P0 status fault family: enqueue_ack_drop — empty ack (dispatch-uncertain)",
    expected_fail=True,
)
def status_ack_empty_no_crash(ctx: CaseContext):
    """Scenario: the prefill drops the whole enqueue ack
    (enqueue_ack_drop) — the master sees an EMPTY ack for a batch it
    dispatched.

    Behaviour: the master classifies the batch dispatch-uncertain and
    installs a BATCH_ACK_UNCERTAIN engine fence.

    Expectation (contract): the fence residue stays BOUNDED and
    non-growing (reuse _fence_residue_stable), AND the quarantined entries
    are ultimately clearable — the scheduler inflight must drain to zero
    within TTL+margin.  NOTE: the verified current behaviour parks
    uncertain-fence entries in quarantine forever (cleanupInflight skips
    engineFence entries from the stale TTL), so the drain assertion is a
    declared contract-level candidate to FAIL — that failure is the
    finding.  The master itself must stay up (HTTP 200) regardless.

    Expected-fail marking: the quarantine-forever behaviour
    is the DECLARED finding, so the case is marked expected_fail — a
    failure classifies as finding-confirmed (the finding stands, exit
    0), an unexpected pass as finding-resolved (the fence-TTL drain
    landed; review the mark).

    Grade: P0."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        inject_type_all(ops, names, "enqueue_ack_drop")
        try:
            errs = _run_requests(ops, base, 4, concurrency=4)
        finally:
            clear_type_all(ops, names, "enqueue_ack_drop")

        # Dispatch-uncertain: the requests themselves end (their fate is
        # reported, not asserted); the ledger contract is below.
        failed = sum(1 for e in errs if e is not None)
        # 4 requests -> at most 4 fence entries (one slot per request).
        residue_ok, residue_detail = _fence_residue_stable(ops, 4)
        # Contract: no permanently-resident entries — TTL+margin drain.
        drained = _wait_scheduler_zero(ops)
        final = ops.master_scheduler_inflight()
        master_ok = _master_ok(ops)

        passed = residue_ok and drained and final == 0 and master_ok
        return passed, (
            f"request_fate: {4 - failed}/4 ok, "
            f"err_types={getattr(_run_requests, 'last_error_types', [])[:2]}, "
            f"fence_residue={residue_ok}({residue_detail}), "
            f"ttl_drained={drained} (final={final}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "enqueue_ack_drop")
