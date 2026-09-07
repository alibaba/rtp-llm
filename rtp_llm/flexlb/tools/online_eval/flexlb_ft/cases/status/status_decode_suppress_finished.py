from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import wait_for
from ...registry import case
from ...support.status import (
    LONG_STREAM_TIMEOUT_S,
    STALE_INFLIGHT_TTL_S,
    TTL_MARGIN_S,
    _decode_names,
    _decode_requests_sum,
    _master_ok,
    _prefill_batches_sum,
    _run_requests,
    _status_spec,
    _timeout_typed,
)


@case(
    "status_decode_suppress_finished",
    category="status",
    profiles=["batch-window"],
    source="P1 status fault family: status_suppress_finished on every decode engine",
)
def status_decode_suppress_finished(ctx: CaseContext):
    """Scenario: every DECODE engine suppresses its finished facts
    (status_suppress_finished) while prefills report normally.

    Behaviour: the prefill stage settles normally (prefill inflight_batches
    drain at the usual pace); the decode ledger keeps its entries alive via
    the still-reported ACTIVE facts (TTL disarmed).

    Expectation (contract): every request ends within deadline/TTL
    (success or a timeout-class terminal); the prefill batches drain fast
    (normal settle); AFTER the injection is cleared the decode
    inflight_requests drain to zero within a RELAXED TTL+margin window
    (the fence-exemption margin — decode-side cleanup may lag the strict
    scheduler TTL); master stays HTTP 200.

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    dnames = _decode_names(ops)
    if not dnames:
        return False, "no decode engines found"
    try:
        inject_type_all(ops, dnames, "status_suppress_finished")
        try:
            errs = _run_requests(
                ops, base, 4, concurrency=4, stream_timeout_s=LONG_STREAM_TIMEOUT_S
            )
            # Prefill stage settles normally — fast drain while the
            # injection is still ON.
            p_batches_zero = wait_for(lambda: _prefill_batches_sum(ops) == 0, 20.0, 1.0)
        finally:
            clear_type_all(ops, dnames, "status_suppress_finished")

        ok = sum(1 for e in errs if e is None)
        legal_terminal = all(e is None or _timeout_typed(e) for e in errs)
        bad_errs = [
            str(e)[:70] for e in errs if e is not None and not _timeout_typed(e)
        ]
        # Relaxed window: TTL + margin + the fence-exemption margin.
        d_requests_zero = wait_for(
            lambda: _decode_requests_sum(ops) == 0,
            STALE_INFLIGHT_TTL_S + TTL_MARGIN_S + 60.0,
            2.0,
        )
        final_d = _decode_requests_sum(ops)
        master_ok = _master_ok(ops)

        passed = (
            legal_terminal
            and p_batches_zero
            and d_requests_zero
            and final_d == 0
            and master_ok
        )
        return passed, (
            f"request_terminals: ok={ok}/4, illegal_errors={bad_errs[:2]}, "
            f"prefill_settled={p_batches_zero}, "
            f"decode_drained={d_requests_zero} (final={final_d}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, dnames, "status_suppress_finished")
