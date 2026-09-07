from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...registry import case
from ...support.status import (
    QUEUE_TIMEOUT_S,
    _master_ok,
    _prefill_names,
    _run_requests,
    _stale_inflight_clean,
    _status_spec,
    _timeout_typed,
)


@case(
    "status_prefill_suppress_finished",
    category="status",
    source="P0 status fault family: status_suppress_finished on every prefill",
)
def status_prefill_suppress_finished(ctx: CaseContext):
    """Scenario: every prefill suppresses only the finished facts
    (status_suppress_finished) — the requests keep appearing RUNNING.

    Behaviour: the persistent RUNNING fact refreshes the slot's
    lastWorkerStatusAtMs on every poll, DISARMING the stale-inflight TTL.
    That keep-alive is allowed; consequently the queue/deadline path is
    the ONLY legal exit for the stuck ledger.

    Expectation (contract): master stays HTTP 200; every request ends
    within queueTimeout(10s)+margin with a legal terminal (success if the
    data plane stays up, or a timeout-class terminal via the deadline
    bottom line); AFTER the injection is cleared the whole ledger drains
    (inflight_clean within a TTL+margin window — the keep-alive stops, the
    frozen activity clock finally expires).

    Grade: P0."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        inject_type_all(ops, names, "status_suppress_finished")
        try:
            # queueTimeout(10s) + margin: the deadline bottom line must
            # fire well inside this window when the data plane also hangs.
            errs = _run_requests(
                ops, base, 4, concurrency=4, stream_timeout_s=QUEUE_TIMEOUT_S + 10.0
            )
        finally:
            clear_type_all(ops, names, "status_suppress_finished")

        ok = sum(1 for e in errs if e is None)
        legal_terminal = all(e is None or _timeout_typed(e) for e in errs)
        bad_errs = [
            str(e)[:70] for e in errs if e is not None and not _timeout_typed(e)
        ]
        # Clear -> keep-alive stops -> frozen clock expires -> drain.
        inflight_ok, inflight_detail = _stale_inflight_clean(ops)
        master_ok = _master_ok(ops)

        passed = legal_terminal and inflight_ok and master_ok
        return passed, (
            f"request_terminals: ok={ok}/4 within "
            f"{QUEUE_TIMEOUT_S + 10.0:.0f}s, illegal_errors={bad_errs[:2]}, "
            f"inflight_clean_after_clear={inflight_ok}({inflight_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_suppress_finished")
