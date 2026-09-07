from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...registry import case
from ...support.status import (
    _decode_names,
    _decode_requests_sum,
    _master_ok,
    _run_requests,
    _stale_inflight_clean,
    _status_spec,
)


@case(
    "status_zombie_completed_running",
    category="status",
    source="P1 status fault family: status_zombie_running — completed tasks re-reported RUNNING",
)
def status_zombie_completed_running(ctx: CaseContext):
    """Scenario: the DECODE engines keep re-reporting already-completed
    tasks as RUNNING (status_zombie_running) while a batch of requests
    completes.

    Behaviour: every terminal the master settles is followed by zombie
    ACTIVE facts for the same reservations — the tombstone path must absorb
    them without re-confirming.

    Expectation (contract): master stays HTTP 200; NO new confirmed entries
    leak on the decode side (decode inflight_requests stays drained); the
    whole ledger stays clean (inflight_clean).  The unknown/zombie counter
    behaviour is RECORDED as an observational item (defensive field reads),
    not asserted.

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    dnames = _decode_names(ops)
    if not dnames:
        return False, "no decode engines found"
    try:
        inject_type_all(ops, dnames, "status_zombie_running")
        try:
            errs = _run_requests(ops, base, 4, concurrency=4)
            # Hold the zombie window open across several poll rounds.
            time.sleep(10.0)
        finally:
            clear_type_all(ops, dnames, "status_zombie_running")

        ok = sum(1 for e in errs if e is None)
        clean_ok, clean_detail = _stale_inflight_clean(ops)
        d_requests = _decode_requests_sum(ops)
        master_ok = _master_ok(ops)

        # Observational: unknown/zombie counter fields (defensive reads).
        snap = ops.snapshot_by_name()
        zombie_obs = {
            n: {
                f: snap.get(n, {}).get(f)
                for f in (
                    "unknown_tasks",
                    "unknown_count",
                    "zombie_reports",
                    "tombstone_hits",
                    "confirmed",
                )
                if f in snap.get(n, {})
            }
            for n in dnames
        }

        passed = ok == 4 and clean_ok and d_requests == 0 and master_ok
        return passed, (
            f"requests_ok={ok}/4, "
            f"inflight_clean={clean_ok}({clean_detail}), "
            f"decode_confirmed_leak={d_requests} (need 0), "
            f"zombie_counters_observed={zombie_obs}, "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, dnames, "status_zombie_running")
