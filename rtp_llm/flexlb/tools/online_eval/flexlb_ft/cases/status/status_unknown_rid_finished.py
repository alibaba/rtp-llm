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
)


@case(
    "status_unknown_rid_finished",
    category="status",
    source="P0 status fault family: status_fake_task(finished, unknown rid), one-shot",
)
def status_unknown_rid_finished(ctx: CaseContext):
    """Scenario: an engine reports a TERMINAL (finished, errorCode 8500)
    for a request id the master has never seen (status_fake_task, one-shot
    on the first prefill).

    Behaviour: the master's slot lookup for the ghost rid finds nothing.

    Expectation (contract): the master IGNORES the unknown-rid terminal —
    the inflight ledgers are bit-identical before vs after (fingerprint
    comparison: scheduler count + every endpoint row), no new terminal is
    produced, and the master stays HTTP 200.

    Grade: P0."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    ghost_rid = base + GHOST_RID_OFFSET
    try:
        # Clean baseline so "no mutation" is observable against zero.
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        before = _inflight_fingerprint(ops)

        inject_type(
            ops,
            names[0],
            "status_fake_task",
            rid=ghost_rid,
            phase="finished",
            errorCode=8500,
        )
        try:
            time.sleep(3.0)  # several status poll rounds
        finally:
            clear_type_all(ops, names, "status_fake_task")

        after = _inflight_fingerprint(ops)
        unchanged = before is not None and after is not None and before == after
        master_ok = _master_ok(ops)

        passed = clean0 and unchanged and master_ok
        return passed, (
            f"baseline_clean={clean0}({clean0_detail}), "
            f"ledger_unchanged={unchanged} "
            f"(before={before}, after={after}), "
            f"ghost_rid={ghost_rid}, master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")
