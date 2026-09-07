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
    "status_unbatched_single_request",
    category="status",
    profiles=["batch-window"],
    source=(
        "unknown-id defense matrix: unbatched single-request facts "
        "(batch_id omitted vs 0, RUNNING vs finished)"
    ),
)
def status_unbatched_single_request(ctx: CaseContext):
    """Scenario: the engine reports isolated single-request facts with NO
    batch context — batch_id omitted vs batch_id=0 explicitly, each in
    both a RUNNING and a finished phase (status_fake_task, four
    combinations injected one at a time, each on a fresh never-dispatched
    rid far above the live id space).

    Behaviour: in BATCH dispatch every legitimate status fact carries the
    batch context the master issued; an unbatched fact is malformed input
    from the master's bookkeeping perspective (its rid is unknown too).

    Expectation (contract): EVERY combination is a no-op for the ledgers —
    the fingerprint is bit-identical before vs after each injection window
    (no ghost registration on RUNNING, no phantom settle on finished); the
    master stays HTTP 200 and the ledger is clean at the end.

    Distinction from status_unknown_rid_finished / _running (same channel,
    default batch_id): those pin the single-phase default form; this case
    makes the unbatched semantics EXPLICIT (batch_id=0 declared, omitted
    field declared) and matrixes both phases under it — if the mock's
    default batch_id ever changes, the omitted arms drift away from the
    explicit-0 arms and this case surfaces it.

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)

        combos = []
        for i, (label, params) in enumerate(
            (
                ("omitted/RUNNING", {"phase": "RUNNING"}),
                ("omitted/finished", {"phase": "finished", "error_code": 8500}),
                ("batch0/RUNNING", {"batch_id": 0, "phase": "RUNNING"}),
                (
                    "batch0/finished",
                    {"batch_id": 0, "phase": "finished", "error_code": 8500},
                ),
            )
        ):
            rid = base + GHOST_RID_OFFSET + 210 + i
            before = _inflight_fingerprint(ops)
            inject_type(ops, names[0], "status_fake_task", rid=rid, **params)
            try:
                time.sleep(3.0)  # several status poll rounds
            finally:
                clear_type_all(ops, names, "status_fake_task")
            after = _inflight_fingerprint(ops)
            combos.append(
                (
                    label,
                    before is not None and after is not None and before == after,
                )
            )

        all_noop = all(ok for _, ok in combos)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 20.0
        )
        master_ok = _master_ok(ops)

        passed = clean0 and all_noop and inflight_ok and master_ok
        combo_bits = ", ".join(
            f"{label}={'noop' if ok else 'MUTATED'}" for label, ok in combos
        )
        return passed, (
            f"baseline_clean={clean0}({clean0_detail}), "
            f"all_noop={all_noop} [{combo_bits}], "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")
