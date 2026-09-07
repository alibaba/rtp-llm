from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import AssertUtils
from ...registry import case
from ...support.status import (
    _inflight_fingerprint,
    _master_http,
    _master_ok,
    _prefill_names,
    _run_requests,
    _status_spec,
    _wait_scheduler_zero,
)


@case(
    "status_duplicate_finished",
    category="status",
    source="P1 status fault family: status_duplicate_finished — same terminal reported twice",
)
def status_duplicate_finished(ctx: CaseContext):
    """Scenario: the engine replays its finished reports
    (status_duplicate_finished) while a batch of requests completes.

    Behaviour: the master receives the SAME terminal fact more than once
    per request.

    Expectation (contract): replay is idempotent — no double settle, no
    ledger resurrection: after the batch completes the inflight
    fingerprint stays BIT-IDENTICAL across the replay window and the
    ledger stays clean; master stays HTTP 200.  (Mock-side completed /
    terminal counters are recorded as observables.)

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        inject_type_all(ops, names, "status_duplicate_finished")
        try:
            errs = _run_requests(ops, base, 4, concurrency=4)
            # Fingerprint pair taken INSIDE the replay window (the injection
            # is still armed) — a clear-then-compare pair would only observe
            # the post-injection calm and never the replay itself.  The
            # before baseline must sit on a SETTLED ledger: under the SINGLE
            # decision axis the ledger release trails the client streams by
            # a few seconds, so a snapshot taken straight after
            # _run_requests carries the drain tail and the window would
            # measure the tail settling, not the replay (batch-window
            # settles synchronously, which is why the baseline never
            # needed this gate there).
            if not _wait_scheduler_zero(ops):
                return (
                    False,
                    "ledger did not settle before replay-window baseline sample",
                )
            before = _inflight_fingerprint(ops)
            time.sleep(5.0)  # replay window: terminals re-delivered
            after = _inflight_fingerprint(ops)
        finally:
            clear_type_all(ops, names, "status_duplicate_finished")

        ok = sum(1 for e in errs if e is None)
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        stable = before is not None and after is not None and before == after

        # Mock-side observables (field names defensive — recorded, not
        # asserted; the master-side fingerprint IS the idempotency proof).
        snap = ops.snapshot_by_name()
        mock_obs = {
            n: {
                f: snap.get(n, {}).get(f)
                for f in ("completed", "terminal", "finished")
                if f in snap.get(n, {})
            }
            for n in names
        }
        master_ok = _master_ok(ops)

        passed = ok == 4 and clean_ok and stable and master_ok
        return passed, (
            f"requests_ok={ok}/4, "
            f"inflight_clean={clean_ok}({clean_detail}), "
            f"replay_ledger_stable={stable} (fp_before={before}, "
            f"fp_after={after}), mock_counters={mock_obs}, "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_duplicate_finished")
