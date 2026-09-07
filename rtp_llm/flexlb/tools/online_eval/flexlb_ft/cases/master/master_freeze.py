from __future__ import annotations

import time

from ...context import CaseContext
from ...registry import case
from ...support.ha import (
    HaRows,
    HaTrafficRunner,
    instance_ops,
    restore_masters,
    rows_between,
    tier1_dual_spec,
)
from ...support.master import HA_STEADY_S, _check_client_fields


@case(
    "master_freeze",
    category="master",
    profiles=["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"],
    source="Mode 2 freeze (SIGSTOP→SIGCONT): content-not-lost assertions, "
    "short + long hang tiers (brief p3/p4)",
)
def master_freeze(ctx: CaseContext):
    """Tier-1 dual-standalone, sticky-on-B (brief p4 right lane).

    Two hang tiers against the SAME frozen JVM:
      * short hang (< keepalive ~40s judgement): SIGSTOP 6s → SIGCONT —
        recovery is invisible: rows sent during the hang complete after
        the thaw (no evaporation), no switch happens, pid unchanged.
      * long hang (> 40s judgement): SIGSTOP 46s → the channel keepalive
        (30s time / 10s timeout) marks B dead → UNAVAILABLE → same-
        request retry to A, sticky moves to A; after SIGCONT B is still
        the SAME process (ledger continuity, no cold restart) and the
        post-thaw window shows no error storm.

    Content-not-lost assertion set (brief p3 Mode 2): ledger continuity
    (pid + discovered topology), in-flight/no evaporation, no duplicate
    dispatch (unique rid rows), post-recovery clean traffic.
    """
    env = ctx.env_manager.ensure(tier1_dual_spec(ctx))
    ops_b = instance_ops(ctx, env, "B")
    mgr = ctx.env_manager
    target_a = mgr.master_instance_target(env, "A")
    target_b = mgr.master_instance_target(env, "B")
    case_dir = ctx.case_dir("master_freeze")
    flow = HaTrafficRunner(
        ctx,
        env,
        case_dir,
        "master_freeze",
        targets=[target_b, target_a],  # sticky B first (brief p4)
        duration_s=100,
        timeout_ms=30_000,
    )
    pid_b = None
    frozen = False
    try:
        flow.start()
        pid_b = env.masters["B"].pid
        time.sleep(HA_STEADY_S)
        # --- short-hang tier: SIGSTOP 6s (< 40s keepalive judgement) ---
        t_freeze1 = HaTrafficRunner.now()
        mgr.freeze_master_instance(env, "B")
        frozen = True
        time.sleep(6.0)
        t_cont1 = HaTrafficRunner.now()
        mgr.unfreeze_master_instance(env, "B")
        frozen = False
        time.sleep(8.0)
        t_after1 = HaTrafficRunner.now()
        # --- long-hang tier: SIGSTOP 46s (> 40s keepalive judgement) ---
        # W3 Mode-2 ledger mirror probes (brief p3 assertion face #1):
        # snapshot B's scheduler inflight + discovered counts right
        # before the freeze so the post-thaw snapshots can prove the
        # in-memory ledger was NOT reset to zero (SIGSTOP/SIGCONT keeps
        # the process image; a cold restart would zero both).
        inflight_pre_freeze = ops_b.master_scheduler_inflight()
        pre_info = ops_b.master_info() or {}
        pre_summary = pre_info.get("worker_summary", {}) or {}
        try:
            pre_disc_p = int((pre_summary.get("PREFILL") or {}).get("discovered", -1))
            pre_disc_d = int((pre_summary.get("DECODE") or {}).get("discovered", -1))
        except (TypeError, ValueError):
            pre_disc_p = pre_disc_d = -1
        t_freeze2 = HaTrafficRunner.now()
        mgr.freeze_master_instance(env, "B")
        frozen = True
        time.sleep(46.0)
        t_cont2 = HaTrafficRunner.now()
        mgr.unfreeze_master_instance(env, "B")
        frozen = False
        # Mirror snapshot at the very start of the post-thaw stable
        # window: the frozen in-flight entries must still be on B's
        # ledger (profile-default staleInflightTimeoutMs=300s >> the
        # 46s freeze, so neither the TTL sweep nor anything else may
        # have zeroed it).
        inflight_post_thaw = ops_b.master_scheduler_inflight()
        time.sleep(10.0)
        t_after2 = HaTrafficRunner.now()
        # Same-process + ledger-continuity probes right after the thaw.
        info_b = ops_b.master_info()
        b_ready = bool(info_b and info_b.get("ready"))
        summary = (info_b or {}).get("worker_summary", {}) or {}
        try:
            disc_p = int((summary.get("PREFILL") or {}).get("discovered", -1))
            disc_d = int((summary.get("DECODE") or {}).get("discovered", -1))
        except (TypeError, ValueError):
            disc_p = disc_d = -1
        ledger_kept = disc_p == env.spec.n_prefill and disc_d == env.spec.n_decode
        # W3: Mode-2 inflight ledger "not zeroed" mirror of Mode-1's
        # inflight_clean — two combined probes:
        #  * discovered counts must not regress across the freeze
        #    (monotonic no-rewind; ledger_kept above pins the end value);
        #  * with >=1 entry in flight at freeze time, the immediate
        #    post-thaw snapshot must still see >=1 (a SIGSTOP'd process
        #    retains its ledger; only a cold restart resets it to zero).
        # Unobservable setups (probe failure -1, or zero inflight at the
        # freeze instant) do not block — topology continuity is already
        # covered by ledger_kept.
        disc_monotonic = pre_disc_p < 0 or (
            disc_p >= pre_disc_p and disc_d >= pre_disc_d
        )
        inflight_not_reset = (
            inflight_pre_freeze <= 0
            or inflight_post_thaw < 0
            or inflight_post_thaw >= 1
        )
        inflight_ledger_kept = disc_monotonic and inflight_not_reset
        flow.wait_finish()
        rows = flow.rows()
        guard = _check_client_fields(HaRows(rows))
        if guard:
            return guard
        allr = HaRows(rows)
        hang1 = HaRows(rows_between(rows, t_freeze1, t_cont1))
        post1 = HaRows(rows_between(rows, t_cont1, t_after1))
        # Thaw burst proper: [t_cont1, freeze2-0.5). The last half-second
        # before the long hang is the freeze2 boundary — rows sent there
        # are exactly the frozen in-flight victims that end as deadline
        # rows ~30s later (evidence: 283 burst rows all-ok-on-B plus 8
        # boundary deadline rows landing inside the old 8s post1 window).
        burst1 = HaRows(rows_between(rows, t_cont1, t_freeze2 - 0.5))
        # The long-hang switch is CALLER-DEADLINE driven (flow
        # timeout_ms=30_000), not keepalive-death driven: remote evidence
        # shows the failover rows landing at freeze2+30.0~30.3s — the
        # frozen in-flight calls hit their 30s gRPC deadline, the freed
        # slots same-request-retry to A. Window the judgement around the
        # deadline-driven switch (freeze2+timeout-eps) instead of the old
        # keepalive assumption (freeze2+35s), which started AFTER the
        # switch had already happened.
        judged = HaRows(rows_between(rows, t_freeze2 + 29.0, t_cont2))
        post2 = HaRows(rows_between(rows, t_cont2, t_after2))
        pid_same = pid_b == env.masters["B"].pid

        # Short tier: zero-interference recovery. With MAX_CONCURRENCY=8
        # every slot is held by a frozen in-flight request, so the freeze
        # window itself may legitimately send ZERO rows (t+12~t+16
        # evidence); the honest no-evaporation contract is: whatever WAS
        # sent inside the window is ok-on-B, and the post-thaw burst
        # drains 100% ok-on-B (nothing evaporates into errors either).
        hang1_no_evap = (
            (
                len(hang1.rows) == 0
                or (
                    len(hang1.ok_rows()) == len(hang1.rows)
                    and len(hang1.target(target_b)) == len(hang1.rows)
                )
            )
            and len(burst1.rows) >= 3
            and len(burst1.ok_rows()) == len(burst1.rows)
            and len(burst1.target(target_b)) == len(burst1.rows)
        )
        post1_still_b = (
            len(post1.target(target_b)) == len(post1.rows) if post1.rows else True
        )
        # Long tier: deadline-driven switch → same-request retry to A.
        judged_failover = len(judged.failover_rows()) > 0
        judged_to_a = len(judged.target(target_a)) > 0
        # Mode-2 brief (p3) visibility gap: the pre-freeze in-flight
        # requests must reach a VISIBLE terminal — ok before the freeze,
        # or an explicit error_kind=deadline row when the frozen calls hit
        # their own deadline (remote evidence: 8 deadline rows plus the
        # post-thaw 8510 "generation retired" refusals). No silent
        # evaporation, no rows vanishing mid-flight. Deadline rows carry
        # the ORIGINAL send timestamp, and that timestamp STRADDLES the
        # freeze2 boundary: the frozen victims were already in flight
        # 1-3s before the freeze, so their send lands up to a few seconds
        # EARLY (run-1788363667: the deadline rows sent at rel-s 25.1~25.5
        # while freeze2 lands at ~26s ± jitter — the old [freeze2, cont2)
        # window missed them by pure timing luck). Count deadline
        # terminals over the whole straddle window instead.
        pre_freeze = HaRows(rows_between(rows, t_freeze2 - 2.0, t_freeze2))
        hang_deadline = HaRows(rows_between(rows, t_freeze2 - 4.0, t_cont2))
        inflight_terminal = (
            len(pre_freeze.rows) >= 1
            and all(
                r.get("status") == "ok"
                or r.get("error_kind") in ("deadline", "transport", "business")
                for r in pre_freeze.rows
            )
            and len(hang_deadline.error_kind("deadline")) >= 1
        )
        # Post-thaw: no error storm, traffic healthy (on A).
        post2_ok = post2.ok_rate() >= 0.90 if post2.rows else False
        post2_a_share = (
            len(post2.target(target_a)) / len(post2.rows) if post2.rows else 0.0
        )
        no_dup = not allr.dup_rids()

        passed = (
            hang1_no_evap
            and post1_still_b
            and judged_failover
            and judged_to_a
            and inflight_terminal
            and post2_ok
            and post2_a_share >= 0.80
            and pid_same
            and b_ready
            and ledger_kept
            and inflight_ledger_kept
            and no_dup
        )
        return passed, (
            f"short: hang_no_evap={hang1_no_evap}(hang={len(hang1.rows)} "
            f"rows, burst={len(post1.rows)} rows all-ok-on-B="
            f"{len(post1.ok_rows()) == len(post1.rows)}), "
            f"still_B={post1_still_b}, "
            f"long: judged_failover={judged_failover}, "
            f"to_A={judged_to_a}, "
            f"inflight_terminal={inflight_terminal}(pre_freeze="
            f"{len(pre_freeze.rows)}, deadline="
            f"{len(hang_deadline.error_kind('deadline'))}), "
            f"post2: ok={post2.ok_rate():.0%}, "
            f"A_share={post2_a_share:.0%}, pid_same={pid_same}, "
            f"B_ready={b_ready}, ledger_kept={ledger_kept} "
            f"(discovered P:{disc_p}/D:{disc_d}), "
            f"inflight_ledger_kept={inflight_ledger_kept} "
            f"(pre={inflight_pre_freeze}, post_thaw={inflight_post_thaw}, "
            f"disc_pre P:{pre_disc_p}/D:{pre_disc_d}), "
            f"dup_rids={len(allr.dup_rids())}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            if frozen:
                ctx.env_manager.unfreeze_master_instance(env, "B")
        except Exception:
            pass
        restore_masters(ctx, env)
