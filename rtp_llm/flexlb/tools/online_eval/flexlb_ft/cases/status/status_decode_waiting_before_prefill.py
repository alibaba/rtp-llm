from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type, inject_type_all
from ...harness import TTL_DRAIN_TIMEOUT_S, AssertUtils, wait_for
from ...registry import case
from ...support.status import (
    _decode_names,
    _master_http,
    _master_ok,
    _prefill_batches_sum,
    _prefill_names,
    _status_spec,
)


@case(
    "status_decode_waiting_before_prefill",
    category="status",
    profiles=["batch-window"],
    source=(
        "decode-before-prefill matrix, waiting arm: decode reports an "
        "early-phase RECEIVED fact only (status_suppress_rids + "
        "status_fake_task(RECEIVED) on decodes) while prefills suppress "
        "the whole batch (status_suppress_rids)"
    ),
)
def status_decode_waiting_before_prefill(ctx: CaseContext):
    """Scenario (waiting arm of the decode-before-prefill matrix): the
    prefills suppress ALL facts for the batch's rids AND every decode
    engine suppresses the real facts for those rids
    (status_suppress_rids on both roles), while a synthetic EARLY-PHASE
    fact per rid (status_fake_task, phase=RECEIVED) is appended on every
    decode — the decode side reports the requests as
    received-but-never-executing, forever, with no terminal.

    Behaviour: the only D-side fact the master ever sees for these rids
    is a RECEIVED intermediate (the queued/waiting shape); the prefill
    ledger holds the frozen batch entries.

    Expectation (contract — "D intermediate states drive NOTHING"): the
    RECEIVED fact must neither settle the requests (no early client
    terminal) nor release the P entries early (the prefill ledger stays
    populated while the synthetic fact streams — the finished arm pins
    the terminal-driven direction, this arm pins the
    intermediate-driven prohibition); after the injections are cleared
    the synthetic fact stops and the frozen slots converge to zero
    within the TTL-aware window; the master stays HTTP 200.

    Grade: P1."""
    env = ctx.env_manager.ensure(_status_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    pnames = _prefill_names(ops)
    dnames = _decode_names(ops)
    if not pnames or not dnames:
        return False, (f"engines missing (prefill={len(pnames)}, decode={len(dnames)})")
    rids = [ops.next_request_id(base) for _ in range(4)]
    try:
        inject_type_all(ops, pnames, "status_suppress_rids", rids=rids)
        inject_type_all(ops, dnames, "status_suppress_rids", rids=rids)
        # Synthetic RECEIVED per rid on every decode (fake facts are
        # appended AFTER the rid-suppression filter — mock
        # applyStatusReportFaults ordering — so the real facts are dropped
        # while the synthetic early-phase fact streams on every poll).
        for dname in dnames:
            for rid in rids:
                inject_type(ops, dname, "status_fake_task", rid=rid, phase="RECEIVED")
        try:
            for rid in rids:
                resp = ops.schedule(rid, output_len=2)
                if resp.code != 200 or not resp.success:
                    return False, (
                        f"schedule failed for rid={rid}: {resp.error_message}"
                    )
            dispatched = wait_for(lambda: _prefill_batches_sum(ops) > 0, 15.0, 0.5)
            # Observation window: the RECEIVED facts stream on every poll —
            # the P entries must survive them (no intermediate cleanup) and
            # the requests must stay unsettled (no intermediate settle).
            time.sleep(10.0)
            p_batches_held = _prefill_batches_sum(ops)
            p_held = p_batches_held > 0
            sched_held = ops.master_scheduler_inflight() > 0
        finally:
            clear_type_all(ops, pnames, "status_suppress_rids")
            clear_type_all(ops, dnames, "status_suppress_rids")
            clear_type_all(ops, dnames, "status_fake_task")

        # After release: the real decode terminal is permanently lost (the
        # completion-cursor head-trim ran under the suppression), so the
        # frozen slots must expire via the stale TTL within the TTL-aware
        # window — convergence, not a hang.
        drained, drained_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )
        master_ok = _master_ok(ops)

        passed = dispatched and p_held and sched_held and drained and master_ok
        return passed, (
            f"dispatched={dispatched}, "
            f"p_entries_held_under_received={p_held} "
            f"(batches={p_batches_held}), "
            f"requests_unsettled={sched_held}, "
            f"post_release_drained={drained}({drained_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, pnames, "status_suppress_rids")
        clear_type_all(ops, dnames, "status_suppress_rids")
        clear_type_all(ops, dnames, "status_fake_task")
