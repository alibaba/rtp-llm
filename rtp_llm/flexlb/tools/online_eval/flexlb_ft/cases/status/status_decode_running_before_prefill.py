from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
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
    "status_decode_running_before_prefill",
    category="status",
    profiles=["batch-window"],
    source=(
        "decode-before-prefill matrix, RUNNING arm: decode ACTIVE-only "
        "(status_suppress_finished on decodes) + full prefill suppression "
        "(status_suppress_rids)"
    ),
)
def status_decode_running_before_prefill(ctx: CaseContext):
    """Scenario (RUNNING arm of the decode-before-prefill matrix): the
    prefills suppress ALL facts for the batch's rids (status_suppress_rids)
    AND every decode engine suppresses its finished facts
    (status_suppress_finished) — the decode side reports the requests as
    RUNNING (real ACTIVE facts while executing, then silence once the
    mock finishes) but never a terminal.

    Behaviour: with the D-side terminal absent, the master's live facts
    for these rids are the short-lived decode ACTIVE reports and then
    nothing; the prefill ledger holds the frozen batch entries.

    Expectation (contract — "D intermediate states drive NOTHING"): while
    no D terminal exists the P entries must NOT be cleaned early — an
    ACTIVE-fact-driven or silence-driven early cleanup would corrupt the
    bookkeeping (cleanup is terminal-driven only; the finished arm pins
    the positive direction, this arm pins the negative) — and the
    requests must stay unsettled (scheduler inflight keeps them); after
    the injections are cleared the frozen slots converge to zero within
    the TTL-aware window (no permanent hang); the master stays HTTP 200.

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
        inject_type_all(ops, dnames, "status_suppress_finished")
        try:
            # Schedule without consuming streams: the observation target is
            # the LEDGER, not the client outcome (the terminal is absent by
            # construction, so the streams would only hang).
            for rid in rids:
                resp = ops.schedule(rid, output_len=2)
                if resp.code != 200 or not resp.success:
                    return False, (
                        f"schedule failed for rid={rid}: {resp.error_message}"
                    )
            dispatched = wait_for(lambda: _prefill_batches_sum(ops) > 0, 15.0, 0.5)
            # Observation window: the decode execution finishes
            # (sub-second) and its ACTIVE facts disappear — the P entries
            # must survive BOTH the ACTIVE period and the silence after.
            time.sleep(10.0)
            p_batches_held = _prefill_batches_sum(ops)
            p_held = p_batches_held > 0
            sched_held = ops.master_scheduler_inflight() > 0
        finally:
            clear_type_all(ops, pnames, "status_suppress_rids")
            clear_type_all(ops, dnames, "status_suppress_finished")

        # After release: the suppressed decode terminal is permanently lost
        # (the completion-cursor head-trim ran under the injection), so the
        # frozen slots must expire via the stale TTL within the TTL-aware
        # window — convergence, not a hang.
        drained, drained_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )
        master_ok = _master_ok(ops)

        passed = dispatched and p_held and sched_held and drained and master_ok
        return passed, (
            f"dispatched={dispatched}, "
            f"p_entries_held_without_d_terminal={p_held} "
            f"(batches={p_batches_held}), "
            f"requests_unsettled={sched_held}, "
            f"post_release_drained={drained}({drained_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, pnames, "status_suppress_rids")
        clear_type_all(ops, dnames, "status_suppress_finished")
