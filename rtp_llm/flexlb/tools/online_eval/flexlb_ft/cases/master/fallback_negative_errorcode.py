from __future__ import annotations

import time

from ...context import CaseContext
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import wait_for
from ...registry import case
from ...support.ha import (
    HaRows,
    HaTrafficRunner,
    ha_gate,
    instance_ops,
    restore_masters,
    rows_between,
    tier1_dual_spec,
)
from ...support.master import _check_client_fields, _prefill_engine_names


@case(
    "fallback_negative_errorcode",
    category="master",
    profiles=["batch-window"],
    source="scenario 3 negative (brief p7/p8): business error codes / "
    "DEADLINE never trigger the direct fallback",
)
def fallback_negative_errorcode(ctx: CaseContext):
    """Tier-1 dual standalone, ENABLE_FALLBACK=true (armed but must NOT
    fire).

    Negative contract (brief p8 phase 2 + Tina's field naming):
      * business leg: inject engine-side enqueue_ack_error_code=8431 —
        the master ANSWERS with schedule_error: rows must be
        route_path=master + status=schedule_error + error_kind=business
        + the 8431 code visible in the error text; zero fallback rows.
      * deadline leg: SIGSTOP the sticky master with a short deadline —
        rows must be route_path=failed + error_kind=deadline with
        failover=false (no retry, no switch, no fallback — the
        single-listed DEADLINE_EXCEEDED assertion).
    """
    gate = ha_gate()
    if gate:
        return gate
    env = ctx.env_manager.ensure(tier1_dual_spec(ctx))
    mgr = ctx.env_manager
    ops_a = instance_ops(ctx, env, "A")
    target_a = mgr.master_instance_target(env, "A")
    target_b = mgr.master_instance_target(env, "B")
    case_dir = ctx.case_dir("fallback_negative")
    flow = HaTrafficRunner(
        ctx,
        env,
        case_dir,
        "fallback_negative",
        targets=[target_a, target_b],  # sticky A
        duration_s=75,
        timeout_ms=800,  # short deadline for the DEADLINE leg
        enable_fallback=True,
    )
    injected = False
    frozen = False
    try:
        flow.start()
        time.sleep(10.0)
        # --- deadline leg FIRST: frozen sticky master + 800ms deadline ---
        # Leg order swapped from the original draft (business first, settle
        # between the legs). A mid-case settle can no longer work here: with
        # TIMEOUT_MS=800 the clients drop streams by the hundreds (run
        # 1788359161: 734 client_gone census rows) and the master reconciles
        # those requests ONLY via the 30s endpoint-inflight TTL plus the 60s
        # request-expiry sweep — the last client send at t=75s expires at
        # t=135s, far past the 75s client run, so any inter-leg gate window
        # would either time out (false FAIL, the 20s draft) or outlive the
        # traffic (no rows left for the second leg). Deadline-first removes
        # the original no-resurrection concern structurally — business
        # FAILED rows cannot bleed backwards into an earlier window — and
        # the settle moves to the tail as the clear-resume contract.
        t_freeze = HaTrafficRunner.now()
        mgr.freeze_master_instance(env, "A")
        frozen = True
        time.sleep(8.0)
        t_cont = HaTrafficRunner.now()
        mgr.unfreeze_master_instance(env, "A")
        frozen = False
        time.sleep(8.0)
        # --- business leg: 8431 answers through the master -----------
        names = _prefill_engine_names(ops_a)
        inject_type_all(ops_a, names, "enqueue_ack_error_code", code=8431)
        injected = True
        t_inj = HaTrafficRunner.now()
        time.sleep(10.0)
        t_inj_end = HaTrafficRunner.now()
        clear_type_all(ops_a, names, "enqueue_ack_error_code")
        injected = False
        # The remaining ~39s of the 75s run is post-clear traffic and
        # doubles as recovery evidence that the clear took effect.
        flow.wait_finish()
        rows = flow.rows()
        guard = _check_client_fields(HaRows(rows))
        if guard:
            return guard
        biz = HaRows(rows_between(rows, t_inj, t_inj_end))
        dl = HaRows(rows_between(rows, t_freeze, t_cont))

        biz_sched_err = biz.status("schedule_error")
        biz_code_seen = all("8431" in str(r.get("error", "")) for r in biz_sched_err)
        biz_route_master = (
            len(biz.route("master")) == len(biz.rows) if biz.rows else False
        )
        biz_no_fallback = (
            len(biz.route("fallback")) == 0 and len(biz.route("failed")) == 0
        )
        biz_no_failover = len(biz.failover_rows()) == 0

        dl_rows = dl.error_kind("deadline")
        dl_failed_route = all(r.get("route_path") == "failed" for r in dl_rows)
        dl_no_retry = all(r.get("failover") is False for r in dl_rows)
        dl_no_fallback = len(dl.route("fallback")) == 0

        legs_passed = (
            len(biz_sched_err) >= 5
            and biz_code_seen
            and biz_route_master
            and biz_no_fallback
            and biz_no_failover
            and len(dl_rows) >= 3
            and dl_failed_route
            and dl_no_retry
            and dl_no_fallback
        )
        legs_summary = (
            f"business: schedule_error={len(biz_sched_err)}/{len(biz.rows)}, "
            f"code8431_visible={biz_code_seen}, route_master="
            f"{biz_route_master}, no_fallback={biz_no_fallback}, "
            f"no_failover={biz_no_failover}; "
            f"deadline: rows={len(dl_rows)}, route_failed="
            f"{dl_failed_route}, no_retry={dl_no_retry}, "
            f"no_fallback={dl_no_fallback}"
        )
        if not legs_passed:
            return False, legs_summary
        # --- tail settle: the clear must leave a drainable master -----
        # The mid-run dispatch stall this case provokes (short-deadline
        # client disconnect storm; see the run-1788359161 forensics in the
        # HA case-test notes) is reconciled by the master only through the
        # 30s endpoint-inflight TTL and the 60s request-expiry sweep, so
        # the drain tail runs from the last client send (t=75s) to its
        # expiry (+60s) plus a sweep pass. 150s from wait_finish covers
        # that with margin; a healthy master converges, a true slot leak
        # still times out and fails here.
        residual_tolerance = 8
        settled = wait_for(
            lambda: ops_a.master_scheduler_inflight() <= residual_tolerance,
            150.0,
            2.0,
        )
        if not settled:
            detail = ops_a.master_inflight()
            if isinstance(detail, dict):
                inflight_note = (
                    f"sched={detail.get('scheduler_inflight')}, "
                    f"prefill={[(ep.get('ip_port'), ep.get('inflight_batches', 0)) for ep in (detail.get('prefill_endpoints') or [])]}, "
                    f"decode={[(ep.get('ip_port'), ep.get('inflight_requests', 0) or ep.get('total_load', 0)) for ep in (detail.get('decode_endpoints') or [])]}"
                )
            else:
                inflight_note = str(detail)
            return False, (f"{legs_summary}; SETTLE_FAIL inflight=[{inflight_note}]")
        return True, legs_summary
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            if injected:
                clear_type_all(
                    ops_a, _prefill_engine_names(ops_a), "enqueue_ack_error_code"
                )
        except Exception:
            pass
        try:
            if frozen:
                ctx.env_manager.unfreeze_master_instance(env, "A")
        except Exception:
            pass
        restore_masters(ctx, env)
