from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean, inject_type
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    CODE_RESOURCE_EXHAUSTED,
    STREAM_WAIT_S,
    _all_engine_names,
    _cancel_rpc_total,
    _decode_names,
    _finally_hygiene,
    _fire,
    _master_http,
    _nf_spec,
    _poll_decode_running,
    _poll_engine_finished,
)


@case(
    "atpm_preempt_cancel_not_found",
    category="priority",
    profiles=["single-nonbatch"],
    source="preemption-stages audit (2026-09) — Cancel NOT_FOUND branch",
)
def atpm_preempt_cancel_not_found(ctx: CaseContext):
    """Preemption-chain Cancel NOT_FOUND branch: the victim has FINISHED
    by the time the preemption Cancel reaches its original prefill, and
    the master's consumer side closes per DecodePreemptionCoordinator's
    NOT_FOUND semantics (cleanSingleNotFound → abort → the incoming
    settles 8431, never a hang and never a false success).

    Construction (the engine cancelRequest branch ORDER is decisive):
    a RUNNING victim's Cancel is answered by the downstreamDecodeOwners
    branch (P→D conduction → ACCEPTED) BEFORE the finished-check, so the
    victim must have cleared BOTH ownership directions (decode
    completion runs clearUpstreamOwnership) AND hold a non-running
    prefill-side lifecycle (the prefill marks its lifecycle entry
    finished when the prefill phase ends) — that routes the Cancel to
    alreadyFinished → NOT_FOUND.

    Choreography: victim P30 (input=512, output=200 ≈ 1.5s of decode)
    fires and reaches RUNNING on the single decode engine; the decode
    engine then stops answering the master's status polls
    (status_no_respond) — the master's view freezes on victim RUNNING +
    slot 1/1 (decode maxEngineRequests=1) while the engine-side victim
    runs to completion; the P70 incoming then fires: its decode
    placement BLOCKS on the frozen view → DECODE_ENGINE_OWNED eviction
    → the tokenized Cancel reaches the original prefill with the victim
    already finished → NOT_FOUND → the incoming settles 8431.

    The frozen-view window is bounded by the master's 3-strike health
    demotion (3 consecutive 1s status RPC timeouts ≈ 3s): the victim's
    remaining decode (~0.7s at injection) keeps the whole choreography
    inside it — a slow finish would let the master demote the decode
    engine first and the incoming would surface 8403 instead (recorded
    as a construction miss, not a contract relaxation).

    Contract:
      * the incoming settles exactly 8431 RESOURCE_EXHAUSTED (the
        coordinator's NOT_FOUND semantics); the victim completes
        NORMALLY (full output, never engine-cancelled);
      * the preemption Cancel really went out (Cancel RPC delta >= 1);
      * after the injection clears, the master ledger drains (the
        victim's stale RUNNING settles from the resumed decode status)
        and recovery works.
    """
    env = ctx.env_manager.ensure(_nf_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    decode_name = None
    injected = False
    try:
        decode_name = _decode_names(ops)[0]
        victim = ops.next_request_id(base)
        victim_fire = _fire(ops, victim, priority=30, input_len=512, output_len=200)
        fires.append(victim_fire)
        if not victim_fire.ok:
            return False, f"victim schedule failed: code={victim_fire.code}"
        if not _poll_decode_running(ops, victim):
            return False, "victim never reached decode running"
        time.sleep(0.6)

        baseline_cancel = _cancel_rpc_total(ops)
        inject_type(ops, decode_name, "status_no_respond")
        injected = True
        # Engine-side completion — the MASTER view stays frozen on
        # RUNNING (the stale slot the incoming will block on).
        if not _poll_engine_finished(ops, victim, 3.0):
            return False, "victim never finished engine-side (3s window)"

        inc = ops.next_request_id(base)
        inc_fire = _fire(ops, inc, priority=70, input_len=512, output_len=2)
        fires.append(inc_fire)

        inc_rejected = not inc_fire.ok and inc_fire.code == CODE_RESOURCE_EXHAUSTED
        victim_completed = bool(
            victim_fire.terminal is not None
            and victim_fire.terminal.wait(STREAM_WAIT_S)
            and victim_fire.terminal.completed
        )
        victim_cancelled, victim_cancel_detail = ops.verify_engine_cancelled(victim)
        cancel_delta = _cancel_rpc_total(ops) - baseline_cancel

        # Clear the injection: the master resumes consuming decode
        # status, the victim's stale RUNNING settles, the ledger drains.
        inject_type(ops, decode_name, "status_no_respond", enabled=False)
        injected = False
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 20.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        report.invariant(
            "PR6",
            inc_rejected and victim_completed and not victim_cancelled,
            context="cancel_not_found_settlement",
            detail=(
                f"incoming terminal={inc_fire.code} (expected exactly "
                f"{CODE_RESOURCE_EXHAUSTED} — cleanSingleNotFound abort), "
                f"victim completed={victim_completed} (normal "
                f"completion), victim engine-cancelled={victim_cancelled} "
                f"[{victim_cancel_detail}] (expect False — the Cancel "
                f"arrived AFTER the finish, nothing to cancel)"
            ),
        )
        report.invariant(
            "AT5",
            cancel_delta >= 1,
            context="cancel_rpc_went_out",
            detail=(
                f"Cancel RPC delta={cancel_delta} (>=1 — the preemption "
                f"cancel reached the original prefill and was answered "
                f"NOT_FOUND)"
            ),
        )
        report.invariant(
            "P6",
            clean_ok and engine_clean and recovery_ok,
            detail=(
                f"after injection cleared: "
                f"inflight={'ok' if clean_ok else clean_detail}, "
                f"engine={'ok' if engine_clean else engine_detail}, "
                f"recovery={recovery_msg}"
            ),
        )
        return report.finish(
            f"cancel-not-found: incoming={inc_fire.code}, victim completed"
            f"={victim_completed}, cancel_delta={cancel_delta}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if injected and decode_name is not None:
            try:
                inject_type(ops, decode_name, "status_no_respond", enabled=False)
            except Exception:
                pass
        _finally_hygiene(ops, fires, [])
