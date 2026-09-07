from __future__ import annotations

import re
import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    CODE_INVALID_REQUEST,
    CODE_SLO_EXPIRED,
    PERF_SETTLE_S,
    ROUTE_REJECT_FAMILY,
    _dispatch_order,
    _drain,
    _finally_hygiene,
    _fire,
    _fire_batch,
    _master_http,
    _master_log_text,
    _metric_sum,
    _o1_spec,
    _outcome_map,
    _poll_engine_pending,
    _prefill_names,
    _pv_log_tail,
    _scrape_master_metrics,
)


@case(
    "atpm_observability_integrity",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.4 #15 — AT8 + P6",
)
def atpm_observability_integrity(ctx: CaseContext):
    """Observability integrity (AT8): every signal plane the design's
    assertion ladder names (proto response > auto_tpm.* metrics > pv.log
    > debug log) carries the priority/TPM facts on ONE composite
    choreography.

    ENV-O1: Q2-shaped config (PREFILL_QUEUED preemption, queueTimeout 7s
    — [EV-1-FIXED] flipped from 8s at intake3
    PendingPlacementCoordinator 6ad0315f10: under the pull model the
    wave's third release slot lands at t=9s, which raced the 8s deadline
    of the 4th submitter (70a); 7s puts every non-dispatched deadline
    strictly before the third slot, making the client shape
    deterministic) + master debug log + the auto_tpm family whitelist.
    Implementation-period corrections over the design's env sketch: the
    DEFAULT critical-only whitelist hides auto_tpm.* (the legacy
    flexlb.monitor.mode switch is dead on this line), so the
    FLEXLB_MONITOR_METRIC_WHITELIST entry is required; FLEXLB_PV_LOG is
    a load-client-line knob with no consumer
    on the harness line — the pvLogger writes at INFO by default, so
    the pv.log plane needs no extra knob.  The master_env + debug-log
    differences give O1 its own fingerprint (exclusive env — the metric
    counters start from zero).

    Choreography (the atpm_preempt_prefill_queued wave-1 shape with
    mixed priorities for bucket coverage), [EV-1-FIXED] design-final
    form: a 50 placeholder parks the inflight lease; 30a/30b/50a/50b/
    70a/70b/30c/30d + the 90 ALL park in the pull-based coordinator.
    Prefill is slowed to 3s: ph completes at t=3, the first parker 30a
    takes slot two (t=3-6), the 90 (highest priority) takes slot three
    (t=6-9) — both complete inside their deadlines.  The remaining
    seven (30b, 30c, 30d, 50a, 50b, 70a, 70b) expire at their own 7s
    deadlines as plain 8511 BATCH_SLO_EXPIRED (the low-priority-
    suppression sample; 30d is no longer evicted — no eviction ever
    fires under the pull model, the victim counter stays flat).

    Per-plane assertions:
      * auto_tpm.request.count{priority=30|50|70|90} == the injected
        bucket counts 4/3/2/1 — counted at the schedule RPC entry for
        EVERY request regardless of outcome (FlexlbServiceImpl:723), the
        metric-plane normalization evidence crossing prio_normalize's
        behaviour plane;
      * auto_tpm.schedule.latency_ms{result="success"} present (the
        TIMER family; result is "success" | "error_<code>");
      * auto_tpm.victim.count == 1 — matching the client-side 8400
        count exactly (exclusive env, absolute value);
      * master log contains [priority-scheduler] lines (debug level —
        master_debug_log=True, the analysis-report §7.5.1 pitfall);
      * pv.log tail carries admissionRejectReason fields (channel
        availability; sampled non-null values recorded in the detail).
    """
    env = ctx.env_manager.ensure(_o1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        ph = ops.next_request_id(base)
        ph_fire = _fire(ops, ph, priority=50, input_len=2048, output_len=2)
        fires.append(ph_fire)
        if not ph_fire.ok:
            return False, f"placeholder failed: code={ph_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "placeholder never dispatched"
        # A3 (Mark P1-1): AT6 black-box aggregate, probe 1 — duplicate
        # request_id rejection observed through the schedule RPC response
        # (RequestLifecycleCoordinator.register putIfAbsent, coordinator
        # L262-268 "duplicate request_id: <rid>" →
        # StrategyErrorType.INVALID_REQUEST 8406).  Probed WHILE ph is
        # still inflight: after its terminal the slot ledger may clean
        # up and a re-submit would re-register successfully.  priority=40
        # keeps the probe out of the expected metric buckets
        # {30, 50, 70, 90} (the counter fires at the schedule RPC entry
        # regardless of outcome).
        dup_code = None
        dup_err = None
        try:
            dup_resp = ops.schedule(
                ph, priority=40, input_len=2048, output_len=2, timeout_s=30.0
            )
            dup_code = int(getattr(dup_resp, "code", -1))
        except Exception as exc:
            dup_err = repr(exc)
        dup_rejected = dup_code == CODE_INVALID_REQUEST
        dup_note = (
            f"observed (code={dup_code})"
            if dup_rejected
            else f"MISSING (code={dup_code}, err={dup_err})"
        )
        # The client-shape assertions index rids["ph"] — register the
        # placeholder alongside the ladder tags (first-run KeyError fix).
        rids: dict = {"ph": ph}

        ladder = [
            ("30a", 30),
            ("30b", 30),
            ("50a", 50),
            ("50b", 50),
            ("70a", 70),
            ("70b", 70),
            ("30c", 30),
            ("30d", 30),
            ("90", 90),
        ]
        specs = []
        for tag, prio in ladder:
            rid = ops.next_request_id(base)
            rids[tag] = rid
            specs.append((rid, {"priority": prio, "input_len": 2048, "output_len": 2}))
        wave = _fire_batch(ops, specs)
        fires.extend(wave)

        outcomes = _drain(ops, [ph_fire] + wave)
        m = _outcome_map(outcomes)

        # Client-plane expectations — [EV-1-FIXED] baseline flipped at
        # intake3 PendingPlacementCoordinator (6ad0315f10): every ladder
        # submitter parks; with queueTimeout 7s the deterministic shape
        # is ph + the first parker 30a + the 90 (highest priority, third
        # release slot) completing 200, the remaining seven expiring
        # 8511 at their own deadlines, zero route-reject and zero
        # eviction (no 8400 — the preemption sample retired with the
        # enqueue-failure trigger).
        wave_tags = [t for t, _p in ladder]
        tag_by_rid = {rids[t]: t for t in wave_tags}
        completed = ["ph"] + [t for t in wave_tags if m[rids[t]][0]]
        rejected8402 = [t for t in wave_tags if m[rids[t]][1] in ROUTE_REJECT_FAMILY]
        expired8511 = [t for t in wave_tags if m[rids[t]][1] == CODE_SLO_EXPIRED]
        ph_ok = m[rids["ph"]][0]
        d_order = _dispatch_order(ops, [ph_fire] + wave)
        d_pos = {r: i for i, r in enumerate(d_order)}
        dispatch_pair_ok = d_pos[rids["30a"]] < d_pos[rids["90"]]
        client_shape_ok = (
            completed == ["ph", "30a", "90"]
            and len(expired8511) == 7
            and rejected8402 == []
            and ph_ok
            and dispatch_pair_ok
        )

        # ---- metric plane (management port /prometheus) -----------------
        samples = _scrape_master_metrics(ops)
        buckets = {
            p: _metric_sum(samples, "auto_tpm_request", {"priority": str(p)})
            for p in (30, 50, 70, 90)
        }
        expected_buckets = {30: 4.0, 50: 3.0, 70: 2.0, 90: 1.0}
        buckets_ok = all(
            buckets[p] is not None and buckets[p] == expected_buckets[p]
            for p in expected_buckets
        )
        latency_success = _metric_sum(
            samples, "auto_tpm_schedule", {"result": "success"}
        )
        latency_ok = latency_success is not None
        # [EV-1-FIXED]: no eviction ever fires under the pull model (no
        # failed enqueue feeds the fallback), so the victim counter must
        # stay at zero — matching the client-side zero-8400 count exactly
        # (exclusive env, absolute value).
        victim_total = _metric_sum(samples, "auto_tpm_victim", {})
        victim_ok = (victim_total or 0.0) == 0.0

        # ---- log plane ---------------------------------------------------
        log_text = _master_log_text(env)
        sched_log_ok = "[priority-scheduler]" in log_text

        # ---- pv.log plane ------------------------------------------------
        # A8 (Daniel P2-3): this env's delta only, filtered to this
        # case's rids (the dup probe re-uses ph, so its INVALID_REQUEST
        # row is inside the ph filter as well).
        pv_tail = _pv_log_tail(env, [ph] + [rids[t] for t in wave_tags])
        pv_field_ok = "admissionRejectReason" in pv_tail
        pv_samples = re.findall(r'"admissionRejectReason"\s*:\s*"([A-Z_]+)"', pv_tail)

        report.invariant(
            "AT8",
            client_shape_ok
            and buckets_ok
            and latency_ok
            and victim_ok
            and sched_log_ok
            and pv_field_ok,
            context="observability_integrity_design_final",
            detail=(
                f"[EV-1-FIXED] client shape (design-final): completed="
                f"{completed}, expired8511={len(expired8511)}/7, "
                f"rejected8402={len(rejected8402)}/0, "
                f"30a-before-90 dispatch={dispatch_pair_ok}; "
                f"request.count buckets={ {p: buckets[p] for p in buckets} } "
                f"(expected {expected_buckets}); "
                f"schedule.latency success={'present' if latency_ok else 'MISSING'}; "
                f"victim.count total={victim_total} (expected 0.0 — no "
                f"eviction under the pull model, matches zero client-side "
                f"8400); "
                f"[priority-scheduler] log={'present' if sched_log_ok else 'MISSING'}; "
                f"pv.log admissionRejectReason field="
                f"{'present' if pv_field_ok else 'MISSING'}"
                + (
                    f", samples={pv_samples[:3]}"
                    if pv_samples
                    else " (no non-null sample values)"
                )
            ),
        )
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            client_shape_ok and clean_ok,
            detail=(
                f"[EV-1-FIXED] every request reached a terminal: 3 "
                f"completed [ph, 30a, 90], 7 expired 8511, "
                f"inflight={'ok' if clean_ok else clean_detail}"
            ),
        )
        # A3 (Mark P1-1): AT6 black-box aggregate — the design's white-box
        # observability-closure item, rebuilt as the three black-box-
        # observable facets on one invariant: duplicate request_id
        # rejection through the schedule response (INVALID_REQUEST
        # 8406), P6 request integrity (every request terminal), and the
        # inflight ledger clean.  AT6 was registered in grade.py but had
        # zero call sites (dead entry); this aggregate is its live form.
        report.invariant(
            "AT6",
            dup_rejected and client_shape_ok and clean_ok,
            context="blackbox_aggregate_dup_p6_inflight",
            detail=(
                f"[EV-1-FIXED] duplicate-rid rejection (INVALID_REQUEST "
                f"{CODE_INVALID_REQUEST}): {dup_note}, "
                f"client shape (design-final)={client_shape_ok}, "
                f"inflight={'ok' if clean_ok else clean_detail}"
            ),
        )
        return report.finish(
            f"planes: client(design-final)={client_shape_ok} metrics="
            f"{buckets_ok and latency_ok and victim_ok} log={sched_log_ok} "
            f"pv={pv_field_ok}, grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)
