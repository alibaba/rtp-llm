from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import AssertUtils
from ...registry import case
from ...support.status import (
    GHOST_RID_OFFSET,
    _inflight_fingerprint,
    _master_http,
    _master_ok,
    _prefill_names,
    _run_requests,
    _status_spec,
)


@case(
    "status_foreign_batchid",
    category="status",
    profiles=["batch-window"],
    source=(
        "unknown-id defense matrix: out-of-range batchId "
        "(a foreign dispatch space) with concurrent real traffic"
    ),
)
def status_foreign_batchid(ctx: CaseContext):
    """Scenario: the engine reports a finished fact for a ghost rid under
    a batchId far outside this master's dispatch space (10_000_000 — ids a
    SECOND master sharing the same engines would issue), and the injection
    stays armed while real traffic flows concurrently.

    Behaviour: a shared-engine misconfiguration makes the engine's status
    stream interleave facts the local master never dispatched — both the
    rid and the batchId are unknown in the local dispatch space.

    Expectation (contract): the foreign fact is ignored wholesale — the
    ledger fingerprint is unchanged across the ghost window; the
    concurrent REAL requests all succeed (none is settled by the foreign
    batchId — no cross-space aliasing); the ledger drains and the master
    stays HTTP 200.

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    ghost_rid = base + GHOST_RID_OFFSET + 220
    foreign_batch_id = 10_000_000
    try:
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        before = _inflight_fingerprint(ops)

        # Arm the foreign fact on every prefill (a shared-engine
        # misconfiguration would surface on any of them) and let it ride
        # through several status polls BEFORE the real traffic starts.
        inject_type_all(
            ops,
            names,
            "status_fake_task",
            rid=ghost_rid,
            batch_id=foreign_batch_id,
            phase="finished",
            error_code=8500,
        )
        try:
            time.sleep(3.0)
            ghost_after = _inflight_fingerprint(ops)
            # Real traffic concurrent with the still-armed foreign fact:
            # no real member may be settled by the out-of-range batchId.
            errs = _run_requests(ops, base, 4, concurrency=4)
        finally:
            clear_type_all(ops, names, "status_fake_task")

        ghost_ignored = (
            before is not None and ghost_after is not None and before == ghost_after
        )
        failed = [e for e in errs if e is not None]
        real_unaffected = not failed
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        master_ok = _master_ok(ops)

        passed = (
            clean0 and ghost_ignored and real_unaffected and inflight_ok and master_ok
        )
        return passed, (
            f"baseline_clean={clean0}({clean0_detail}), "
            f"foreign_ignored={ghost_ignored} "
            f"(before={before}, after_ghost={ghost_after}), "
            f"ghost_rid={ghost_rid}, foreign_batchId={foreign_batch_id}, "
            f"real_unaffected={real_unaffected} (failed={len(failed)}/4, "
            f"kinds={sorted({str(e)[:60] for e in failed})[:2]}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")
