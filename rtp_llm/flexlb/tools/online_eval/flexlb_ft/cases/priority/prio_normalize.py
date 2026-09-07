from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    PERF_SETTLE_S,
    _all_ok,
    _dispatch_order,
    _drain,
    _finally_hygiene,
    _fire,
    _fire_batch,
    _master_http,
    _metric_lines,
    _metric_sum,
    _n1_spec,
    _outcome_map,
    _poll_engine_pending,
    _prefill_names,
    _q1_spec,
    _q3_spec,
    _scrape_master_metrics,
)


@case(
    "prio_normalize",
    category="priority",
    # profiles=None — the case stays eligible on every one of the four
    # built-in profiles (migration decision: the PRIORITY axis is a
    # case-layer injection, no priority profile exists to branch on, and
    # the source's priority-profile branch collapses to the only branch).
    # requires=["queue"]: the shared-env segment 1 and the Q1/Q3 windows
    # all need the waiting-queue capability.
    profiles=None,
    requires=["queue"],
    source="design §2.2 #3 — PR3 + P6",
)
def prio_normalize(ctx: CaseContext):
    """Three-channel normalization (PR3 invariant, per segment): proto
    field 14 > the DashScope QoS header > defaultPriority, unset →
    default; plus the FIFO-control proof that normalization never
    reorders FIFO arrival.

    Segment 1 (shared env, every profile): no-input and explicit-50
    interleaved — same-weight merge (PrioritySource is observational
    metadata only, design §3.4 row 8: the test must never assume the
    explicit source outranks the default source).  The production config
    has no inflight cap, so this segment is the weak arrival-order form
    (all succeed, dispatch == submit).

    Segment 2 (window env, Q1 — the PRIORITY axis): placeholder(no
    input) + C(no input → 50) + A(proto 70, header 30 — proto must win)
    + B(proto unset, header 70 — header must take effect) + G(explicit
    70).  [EV-1-FIXED] Under the intake3 pull-based coordinator the wave
    parks whole behind the placeholder.  [2026-09 recalibration] the
    post-codex admission release is pure priority-desc: observed
    dispatch [ph, A, B, G, C] — intake3's first-parker special case
    (C, the wave's first submitter, winning the first release slot) is
    gone; the assertion is now STRUCTURAL, not exact-sequence (the
    exact form lagged twice already — intake3, then codex admission):
    the 70-group A/B/G must precede C (either channel failing demotes
    that member to <=50 and behind C), the group must keep submit
    order A→B→G (FIFO survives park→release), and all settle code=200.
    The source's FIFO-profile half
    (F1 control env) is not profile-reachable on this line; the FIFO
    control lives separately in atpm_comparator_frozen_weak's fifo_half.
    Implementation-period note: design §2.2 sketched this segment without
    maxInflightRequestsPerPrefillWorker, but without it there is no
    backlog window and no observable queue-jumping — the window env
    keeps the choreography and makes it observable.

    Segment 3 (ENV-N1 — the defaultPriority=30 variant, built as its own
    case-layer env): [EV-1-FIXED] submit order adapted for the
    design-final baseline: Y(50) leads (first parker), then D(no input),
    then a Z(40) reference, then X(explicit 30).  Expected dispatch
    [ph, Y, Z, D, X] — D (default 30) ties X inside the 30-group FIFO
    behind Z(40); a failed default (D=50) would give [ph, Y, D, Z, X] —
    the two outcomes stay distinguishable, so the assertion really pins
    the third channel.

    Segment 4 (ENV-Q3 — A5, Mark P1-3/PR3 strengthening): channel
    discrimination observed through the METRIC plane.  The behaviour
    plane (dispatch order) is EV-1-blocked for multi-parker waves, but
    auto_tpm.request.count{priority=..} counts at the schedule RPC entry
    regardless of outcome, so proto(70)+header(30) must land bucket 70,
    header-only(30) bucket 30, no-input bucket 50.  Normalization is
    profile-independent (FIFO normalizes identically) — the segment
    runs on every profile, hardening the four-profile gate (Daniel
    P2-1).
    """
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    segments: list = []  # (label, ok, detail)
    p6_flags: list = []
    hygiene: list = []  # (ops, fires, prefill_names)
    try:
        # -- segment 1: default-50 same weight (weak arrival form) ------
        ops0 = ctx.ops()
        s1_rids = [ops0.next_request_id(base) for _ in range(4)]
        s1_specs = []
        for i, rid in enumerate(s1_rids):
            kw: dict = {"input_len": 2048, "output_len": 2}
            if i % 2 == 1:
                kw["priority"] = 50  # explicit-50 alternates with no-input
            s1_specs.append((rid, kw))
        s1_fires = _fire_batch(ops0, s1_specs, gap_s=0.3)
        s1_outcomes = _drain(ops0, s1_fires)
        s1_order = _dispatch_order(ops0, s1_fires)
        s1_ok = s1_order == s1_rids and _all_ok(s1_outcomes)
        segments.append(
            (
                "default50_same_weight",
                s1_ok,
                f"dispatch==submit:{s1_order == s1_rids}, "
                f"all_ok={_all_ok(s1_outcomes)}",
            )
        )
        hygiene.append((ops0, s1_fires, []))
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops0), 30.0)
        p6_flags.append(_all_ok(s1_outcomes) and clean_ok)

        # -- segment 2: proto > header > default (window env) -----------
        env2 = ctx.env_manager.ensure(_q1_spec(ctx))
        ops2 = ctx.engine_ops(env2)
        p2_names = _prefill_names(ops2)
        for name in p2_names:
            ops2.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        ph2 = ops2.next_request_id(base)
        ph2_fire = _fire(ops2, ph2, input_len=2048, output_len=2)
        s2_fires = [ph2_fire]
        if not ph2_fire.ok:
            return False, f"segment2 placeholder failed: code={ph2_fire.code}"
        if not _poll_engine_pending(ops2, p2_names[0], 1):
            return False, "segment2 placeholder never dispatched"

        c_rid = ops2.next_request_id(base)
        a_rid = ops2.next_request_id(base)
        b_rid = ops2.next_request_id(base)
        g_rid = ops2.next_request_id(base)
        s2_specs = [
            (c_rid, {"input_len": 2048, "output_len": 2}),
            (
                a_rid,
                {"priority": 70, "qos_level": 30, "input_len": 2048, "output_len": 2},
            ),
            (b_rid, {"qos_level": 70, "input_len": 2048, "output_len": 2}),
            (g_rid, {"priority": 70, "input_len": 2048, "output_len": 2}),
        ]
        s2_fires.extend(_fire_batch(ops2, s2_specs))
        s2_outcomes = _drain(ops2, s2_fires)
        s2_order = _dispatch_order(ops2, s2_fires)
        # [2026-09 recalibration, post-codex admission] the wave parks
        # whole behind the placeholder and releases pure priority-desc:
        # observed [ph, A, B, G, C] (intake3's first-parker slot-win is
        # gone).  Structural form (sep-anchored like _two_cluster_split,
        # robust to release-order drift that does not cross a priority
        # group): (i) the placeholder and every wave member dispatched;
        # (ii) the placeholder is first (it held the slot before the
        # wave parked); (iii) the 70-group A/B/G strictly precedes C —
        # proto must win for A (a header-loser A lands 30, behind C's
        # 50) and the header must take effect for B (a default-loser B
        # ties C at 50 and FIFO puts submitted-later B behind
        # first-submitter C); (iv) the 70-group keeps submit order
        # A→B→G (FIFO survives park→release); (v) all settle code=200.
        s2_m = _outcome_map(s2_outcomes)
        s2_wave = [c_rid, a_rid, b_rid, g_rid]
        _s2_miss = len(s2_order) + 99  # sentinel: never dispatched
        s2_pos = {
            rid: (s2_order.index(rid) if rid in s2_order else _s2_miss)
            for rid in (ph2, c_rid, a_rid, b_rid, g_rid)
        }
        s2_all = all(p < _s2_miss for p in s2_pos.values())
        s2_ph_first = s2_pos[ph2] == 0
        s2_group_before_c = all(
            s2_pos[r] < s2_pos[c_rid] for r in (a_rid, b_rid, g_rid)
        )
        s2_fifo_in_group = s2_pos[a_rid] < s2_pos[b_rid] < s2_pos[g_rid]
        s2_ok = (
            s2_all
            and s2_ph_first
            and s2_group_before_c
            and s2_fifo_in_group
            and s2_m[ph2][0]
            and all(s2_m[r][0] for r in s2_wave)
        )
        segments.append(
            (
                "proto_header_default_ev1_fixed",
                s2_ok,
                f"[2026-09 recal] dispatch="
                f"{[r % 1_000_000 for r in s2_order]} "
                f"(structural: ph first, 70-group A/B/G before C, "
                f"FIFO in group, all code=200), "
                f"codes={[(r % 1_000_000, s2_m[r][1]) for r in s2_wave]}",
            )
        )
        hygiene.append((ops2, s2_fires, p2_names))
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops2), 30.0)
        p6_flags.append(s2_m[ph2][0] and all(s2_m[r][0] for r in s2_wave) and clean_ok)

        # -- segment 3: defaultPriority=30 (own case-layer env) ---------
        env3 = ctx.env_manager.ensure(_n1_spec(ctx))
        ops3 = ctx.engine_ops(env3)
        p3_names = _prefill_names(ops3)
        for name in p3_names:
            ops3.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        ph3 = ops3.next_request_id(base)
        ph3_fire = _fire(ops3, ph3, priority=10, input_len=2048, output_len=2)
        s3_fires = [ph3_fire]
        if not ph3_fire.ok:
            return False, f"segment3 placeholder failed: code={ph3_fire.code}"
        if not _poll_engine_pending(ops3, p3_names[0], 1):
            return False, "segment3 placeholder never dispatched"

        y_rid = ops3.next_request_id(base)
        d_rid = ops3.next_request_id(base)
        z_rid = ops3.next_request_id(base)
        x_rid = ops3.next_request_id(base)
        # [EV-1-FIXED] submit order adapted for the design-final
        # baseline (intake3 PendingPlacementCoordinator, 6ad0315f10):
        # the wave's FIRST submitter now legitimately wins the first
        # release slot, so the default-channel probe D must NOT sit in
        # first position (there it is order-invariant and the
        # assertion goes vacuous).  Y(50) leads as the first parker;
        # the Z(40) reference between D and X keeps the outcomes
        # distinguishable: default=30 gives [ph, Y, Z, D, X] (D ties X
        # at 30, FIFO inside the group, both behind Z), a failed
        # default (D=50) gives [ph, Y, D, Z, X].
        s3_specs = [
            (y_rid, {"priority": 50, "input_len": 2048, "output_len": 2}),
            (d_rid, {"input_len": 2048, "output_len": 2}),
            (z_rid, {"priority": 40, "input_len": 2048, "output_len": 2}),
            (x_rid, {"priority": 30, "input_len": 2048, "output_len": 2}),
        ]
        s3_fires.extend(_fire_batch(ops3, s3_specs))
        s3_outcomes = _drain(ops3, s3_fires)
        s3_order = _dispatch_order(ops3, s3_fires)
        # [EV-1-FIXED] baseline flipped at intake3
        # PendingPlacementCoordinator (6ad0315f10): the four-request
        # wave parks whole and dispatches [ph, Y (first parker), Z,
        # D, X] — D (no input -> defaultPriority=30) ties X inside the
        # 30-group FIFO behind the Z(40) reference; a failed default
        # (D=50) would give [ph, Y, D, Z, X].  The third channel
        # (defaultPriority) is pinned through D's dispatch position.
        s3_m = _outcome_map(s3_outcomes)
        s3_wave = [y_rid, d_rid, z_rid, x_rid]
        s3_expected = [ph3, y_rid, z_rid, d_rid, x_rid]
        s3_ok = (
            s3_order == s3_expected
            and s3_m[ph3][0]
            and all(s3_m[r][0] for r in s3_wave)
        )
        segments.append(
            (
                "default_priority_30_ev1_fixed",
                s3_ok,
                f"[EV-1-FIXED] dispatch==[ph,Y,Z,D,X]:"
                f"{s3_order == s3_expected} (D=default30 ties X behind "
                f"Z(40); failed default would give [ph,Y,D,Z,X]), "
                f"codes={[(r % 1_000_000, s3_m[r][1]) for r in s3_wave]}",
            )
        )
        hygiene.append((ops3, s3_fires, p3_names))
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops3), 30.0)
        p6_flags.append(s3_m[ph3][0] and all(s3_m[r][0] for r in s3_wave) and clean_ok)

        # -- segment 4 (A5): channel discrimination, metric plane -------
        # Needs its own env (ENV-Q3): the default critical-only metrics
        # whitelist (six link-latency presets) hides auto_tpm.*, and the
        # v2 filter has NO mode switch — the whitelist entry
        # flexlb_auto_tpm_request_count is what exposes the series (see
        # the _q3_spec docstring).  Fresh env means the counters start
        # at zero, so the expected buckets are absolute.  Buckets count
        # at the schedule RPC entry regardless of outcome — EV-1 cannot
        # block this observation plane.
        env4 = ctx.env_manager.ensure(_q3_spec(ctx))
        ops4 = ctx.engine_ops(env4)
        s4_specs = [
            (
                "proto70_over_header30",
                {
                    "priority": 70,
                    "qos_level": 30,
                    "input_len": 2048,
                    "output_len": 2,
                },
            ),
            (
                "header30_only",
                {"qos_level": 30, "input_len": 2048, "output_len": 2},
            ),
            ("no_input_default50", {"input_len": 2048, "output_len": 2}),
        ]
        s4_rids = [ops4.next_request_id(base) for _ in s4_specs]
        s4_fires = _fire_batch(
            ops4, [(rid, kw) for rid, (_l, kw) in zip(s4_rids, s4_specs)]
        )
        _drain(ops4, s4_fires)  # terminals only — buckets count regardless
        s4_samples = _scrape_master_metrics(ops4)
        s4_buckets = {
            p: _metric_sum(
                s4_samples,
                "auto_tpm_request",
                {"priority": str(p)},
            )
            for p in (30, 50, 70)
        }
        # proto(70) beats header(30); header alone lands 30; no input
        # defaults to 50 (segment 1's behaviour-plane default form echoed
        # on the metric plane — a different observation channel, not a
        # duplicate assertion).
        s4_ok = (
            s4_buckets[70] == 1.0 and s4_buckets[30] == 1.0 and s4_buckets[50] == 1.0
        )
        segments.append(
            (
                "channel_discrimination_metric_plane",
                s4_ok,
                f"buckets={ {p: s4_buckets[p] for p in s4_buckets} } "
                f"(expected 70:1, 30:1, 50:1), "
                f"auto_tpm_lines="
                f"{_metric_lines(s4_samples, 'auto_tpm_request')}",
            )
        )
        hygiene.append((ops4, s4_fires, _prefill_names(ops4)))
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops4), 30.0)
        p6_flags.append(s4_ok and clean_ok)

        report.invariant(
            "PR3",
            all(ok for (_label, ok, _detail) in segments),
            context="three_channel_normalization",
            detail="; ".join(
                f"{label}={'ok' if ok else 'FAIL(' + detail + ')'}"
                for label, ok, detail in segments
            ),
        )
        report.invariant(
            "P6",
            all(p6_flags),
            detail=f"per-segment drain+inflight flags={p6_flags}",
        )
        return report.finish(
            f"profile={ctx.profile}, segments="
            f"{sum(1 for _l, ok, _d in segments if ok)}/{len(segments)}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for ops_x, fires_x, names_x in hygiene:
            try:
                _finally_hygiene(ops_x, fires_x, names_x)
            except Exception:
                pass
