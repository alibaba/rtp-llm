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
)


@case(
    "status_cursor_regress",
    category="status",
    profiles=["batch-window"],
    source="P1 status fault family: status_cursor_regress(3) — completion cursor rewinds",
)
def status_cursor_regress(ctx: CaseContext):
    """Scenario: after three requests complete, the engine's completion
    cursor REGRESSES by 3 (status_cursor_regress) — the three old
    completion facts are re-delivered.

    Behaviour: the master sees stale terminals for requests it has already
    settled.

    Expectation (contract): old-completion re-delivery is idempotent — no
    duplicate terminals, no ledger resurrection: the inflight fingerprint
    stays bit-identical across the rewind window, the ledger stays clean,
    master stays HTTP 200, and a fresh request still succeeds.

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        # Build completion history first (the cursor rewinds over these).
        errs0 = _run_requests(ops, base, 3, concurrency=3)
        ok0 = sum(1 for e in errs0 if e is None)
        if ok0 < 3:
            return False, (
                f"history batch failed ({ok0}/3): "
                f"{getattr(_run_requests, 'last_error_types', [])[:2]}"
            )
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)

        inject_type_all(ops, names, "status_cursor_regress", n=3)
        try:
            before = _inflight_fingerprint(ops)
            time.sleep(5.0)  # rewind window: old completions re-delivered
            after = _inflight_fingerprint(ops)
        finally:
            clear_type_all(ops, names, "status_cursor_regress")

        stable = before is not None and after is not None and before == after
        still_clean, still_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        master_ok = _master_ok(ops)
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = clean_ok and stable and still_clean and master_ok and recovery_ok
        return passed, (
            f"history_ok=3/3, baseline_clean={clean_ok}({clean_detail}), "
            f"rewind_ledger_stable={stable} (fp_before={before}, "
            f"fp_after={after}), "
            f"still_clean={still_clean}({still_detail}), "
            f"master_200={master_ok}, recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_cursor_regress")
