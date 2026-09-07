from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type
from ...harness import AssertUtils
from ...registry import case
from ...support.status import (
    GHOST_RID_OFFSET,
    STALE_INFLIGHT_TTL_S,
    _master_http,
    _master_ok,
    _prefill_names,
    _status_spec,
    _wait_scheduler_zero,
)


@case(
    "status_zombie_fake_running",
    category="status",
    source="P2 status fault family (DECLARED FINDING PROBE): persistent fake RUNNING for N ghost rids, >= 2x TTL",
    expected_fail=True,
)
def status_zombie_fake_running(ctx: CaseContext):
    """Scenario: the engine PERSISTENTLY reports RUNNING facts for several
    request ids the master has never seen (status_fake_task, ghost rids,
    held for >= 2x the stale TTL).

    Behaviour: every status poll re-delivers the ghost ACTIVE facts.  On
    the current implementation each report refreshes the entry's activity
    clock (lastWorkerStatusAtMs), so the stale TTL can NEVER fire — the
    expected failure mode is permanently-resident inflight entries
    (ConfirmedTask-style) that survive the whole observation window.

    Expectation (contract — this is the probe): the master must NOT retain
    inflight entries that cannot be cleared.  Concretely: after the
    injection is cleared (the ghost reports stop), the scheduler inflight
    MUST drain to zero within TTL(30s)+margin.  EXPECTED TO FAIL on the
    current implementation — the failure IS the finding (record the
    resident count and the non-draining ledger as evidence).

    Expected-fail marking: the permanent-resident ghost
    behaviour is the DECLARED finding, so the case is marked
    expected_fail — a failure classifies as finding-confirmed (the
    finding stands, exit 0), an unexpected pass as finding-resolved (the
    activity-clock refresh landed; review the mark).

    Grade: P2 (contract-level finding probe)."""
    env = ctx.env_manager.ensure(_status_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    n_ghosts = 3
    ghost_rids = [base + GHOST_RID_OFFSET + 100 + i for i in range(n_ghosts)]
    try:
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        sched_before = ops.master_scheduler_inflight()

        # Arm the persistent ghost RUNNING reports (one inject per rid; a
        # MERGE-semantics server accumulates them, a replace-semantics
        # server keeps the last — the probe only needs >= 1 resident).
        for rid in ghost_rids:
            inject_type(ops, names[0], "status_fake_task", rid=rid, phase="RUNNING")
        try:
            # Observation window: >= 2x TTL with the reports flowing.
            deadline = time.monotonic() + 2 * STALE_INFLIGHT_TTL_S
            samples = []
            while time.monotonic() < deadline:
                samples.append(ops.master_scheduler_inflight())
                time.sleep(5.0)
            resident = ops.master_scheduler_inflight()
            peak = max(samples) if samples else -1
            bounded = resident <= sched_before + n_ghosts
            master_ok_during = _master_ok(ops)
        finally:
            for rid in ghost_rids:
                clear_type_all(ops, names, "status_fake_task")

        # Contract: once the reports stop, nothing may stay resident.
        drained = _wait_scheduler_zero(ops)
        final = ops.master_scheduler_inflight()
        master_ok = _master_ok(ops)

        passed = (
            clean0
            and bounded
            and master_ok_during
            and drained
            and final == 0
            and master_ok
        )
        return passed, (
            f"baseline_clean={clean0}({clean0_detail}), "
            f"resident_after_2xTTL={resident} (peak={peak}, "
            f"bounded={bounded} <= {sched_before + n_ghosts}), "
            f"drained_after_clear={drained} (final={final}), "
            f"master_200=(during={master_ok_during}, after={master_ok}), "
            f"ghost_rids={n_ghosts} — expected finding: persistent "
            f"activity-clock refresh keeps ghost entries resident"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")
