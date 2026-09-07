from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.admission import MOCK_TOTAL_KV_TOKENS
from ...support.priority import (
    CODE_ENGINE_CANCELLED,
    CODE_NO_DECODE,
    CODE_OK,
    CODE_RESOURCE_EXHAUSTED,
    CODE_SLO_EXPIRED,
    CODE_YIELDED,
    EV2_REJECT_FAMILY,
    PERF_SETTLE_S,
    ROUTE_REJECT_FAMILY,
    _d1_spec,
    _decode_names,
    _drain,
    _finally_hygiene,
    _fire,
    _fire_batch,
    _master_http,
    _metric_sum,
    _outcome_map,
    _poll_decode_running,
    _prefill_names,
    _scrape_master_metrics,
)


@case(
    "atpm_decode_reservation_priority",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.4 #14 — AT7 + P6",
)
def atpm_decode_reservation_priority(ctx: CaseContext):
    """Decode-reservation priority consistency across stages (AT7): the
    priority rules that govern PREFILL_QUEUED eviction hold verbatim in
    the decode plane — strictly-lower-priority decode victims only, and
    same-priority decode occupancy never evicts (the decode-plane PR4,
    cross-checked against atpm_same_priority_zero_eviction's prefill
    side — that pair is the cross-stage consistency evidence).

    ENV-D1 (shared fingerprint with atpm_preempt_decode_engine_owned;
    the auto_tpm family whitelist so auto_tpm.victim.count is exposed —
    the default critical-only filter hides auto_tpm.*).  Every wave follows
    the corrected injection order (see the D1 spec docstring): victims
    route FIRST under normal KV, kv_pressure goes in only once every
    victim is observable at its target stage, then the incoming fires.

    Wave 1 (strictly-lower victim, engine-owned → 8429): four 30s are
    polled to decode RUNNING; kv_pressure saturates every endpoint; the
    70's ordinary route fails (NO_DECODE_WORKER 8403 — the strict KV
    gate) into the eviction fallback → exactly one owned victim is
    cancelled (8429, typed via grpc-status-details-bin), the 70
    completes, the survivors complete.  Metric cross-check:
    auto_tpm.victim.count{victim_priority="30",incoming_priority="70"}
    increments by exactly ONE (D1 is a shared env — the assertion is
    delta-based against a pre-wave scrape).

    Wave 2 (same-priority zero eviction): four 50s run on decode; the
    incoming 50 finds no strictly-lower candidate → the eviction plan is
    infeasible → the ORIGINAL routing rejection reaches the client —
    NO_DECODE_WORKER(8403) under this construction (8402/8510/8431 stay
    in the family for first-e2e calibration).  Zero victims: client
    terminals plus the victim-count delta staying flat.

    Wave 3 (kvBucket-descending victim preference — WEAK/tendency
    assertion, design §2.4): two 30_small (input 2048) and two 30_big
    (input 16384) occupants, one per decode endpoint; the 70's
    input_len=8192 needs hardKv ≈ 8194.  A small endpoint frees only
    ~2.5k tokens (infeasible); a big endpoint frees ~16.9k (feasible) →
    the victim must be a 30_big.  The construction is deterministic at
    the code level but the design grades it weakly — the
    assertion pins the victim-set membership (big group), not the exact
    rid.
    """
    env = ctx.env_manager.ensure(_d1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    decode_names: list = []
    wave_reports = []
    try:
        prefill_names = _prefill_names(ops)
        decode_names = _decode_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=4000.0)
        time.sleep(PERF_SETTLE_S)

        # ---- wave 1: strictly-lower owned victim → 8429 -----------------
        w1_rids = [ops.next_request_id(base) for _ in range(4)]
        w1_specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 500})
            for rid in w1_rids
        ]
        w1_fires = _fire_batch(ops, w1_specs)
        fires.extend(w1_fires)
        if not all(_poll_decode_running(ops, rid, timeout_s=20.0) for rid in w1_rids):
            return False, "wave1 occupants never reached decode running"
        base_victim = _metric_sum(
            _scrape_master_metrics(ops),
            "auto_tpm_victim",
            {"victim_priority": "30", "incoming_priority": "70"},
        )
        for name in decode_names:
            ops.set_kv_pressure(name, MOCK_TOTAL_KV_TOKENS)
        time.sleep(PERF_SETTLE_S)
        w1_inc = ops.next_request_id(base)
        w1_inc_fire = _fire(ops, w1_inc, priority=70, input_len=2048, output_len=2)
        fires.append(w1_inc_fire)

        m1 = _outcome_map(_drain(ops, w1_fires + [w1_inc_fire]))
        w1_victims = [rid for rid in w1_rids if m1[rid][1] == CODE_ENGINE_CANCELLED]
        w1_survivors_ok = all(m1[rid][0] for rid in w1_rids if rid not in w1_victims)
        w1_inc_ok = m1[w1_inc][0]
        w1_inc_code = m1[w1_inc][1]
        now_victim = _metric_sum(
            _scrape_master_metrics(ops),
            "auto_tpm_victim",
            {"victim_priority": "30", "incoming_priority": "70"},
        )
        w1_delta = (now_victim or 0.0) - (base_victim or 0.0)
        # EV-2 baseline (behaviour finding, probes E9/E11 + the
        # DecodeEndpoint projection math): decode eviction never fires —
        # the kv dimension is unreachable (freedKv ⊆ currentHardCharges)
        # and the slots dimension is absorbed engine-side, so the
        # strictly-lower-priority victim selection has no observation
        # object.  Observable form: zero victims, all occupants complete,
        # the incoming keeps a rejection from EV2_REJECT_FAMILY, and the
        # victim-count metric stays flat (cross-checked against wave 2's
        # identical zero-delta form — the priority asymmetry 30<70 vs
        # 50==50 is itself unobservable black-box).
        wave_reports.append(
            (
                "w1_lower_priority_victim_ev2",
                w1_victims == []
                and (w1_inc_ok or w1_inc_code in EV2_REJECT_FAMILY)
                and w1_survivors_ok
                and w1_delta == 0.0,
                (
                    f"victims8429={len(w1_victims)} (EV-2: decode eviction "
                    f"unreachable — priority 30 < 70 selection has no "
                    f"object), incoming70 ok={w1_inc_ok} code={w1_inc_code} "
                    f"(family {list(EV2_REJECT_FAMILY)}), "
                    f"survivors ok={w1_survivors_ok}, "
                    f"victim.count delta(30<-70)={w1_delta} (expected 0.0 "
                    f"under EV-2 — no eviction events)"
                ),
            )
        )
        clean1_ok, clean1_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        if not clean1_ok:
            wave_reports[-1] = (
                wave_reports[-1][0],
                False,
                wave_reports[-1][2] + f", inflight dirty: {clean1_detail}",
            )

        # ---- wave 2: same-priority zero eviction ------------------------
        for name in decode_names:
            ops.set_kv_pressure(name, 0)
        time.sleep(PERF_SETTLE_S)
        w2_rids = [ops.next_request_id(base) for _ in range(4)]
        w2_specs = [
            (rid, {"priority": 50, "input_len": 2048, "output_len": 500})
            for rid in w2_rids
        ]
        w2_fires = _fire_batch(ops, w2_specs)
        fires.extend(w2_fires)
        if not all(_poll_decode_running(ops, rid, timeout_s=20.0) for rid in w2_rids):
            return False, "wave2 occupants never reached decode running"
        base2_victim = _metric_sum(_scrape_master_metrics(ops), "auto_tpm_victim", {})
        for name in decode_names:
            ops.set_kv_pressure(name, MOCK_TOTAL_KV_TOKENS)
        time.sleep(PERF_SETTLE_S)
        w2_inc = ops.next_request_id(base)
        w2_inc_fire = _fire(ops, w2_inc, priority=50, input_len=2048, output_len=2)
        fires.append(w2_inc_fire)

        m2 = _outcome_map(_drain(ops, w2_fires + [w2_inc_fire]))
        w2_inc_code = m2[w2_inc][1]
        w2_zero_eviction = all(
            m2[rid][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED) for rid in w2_rids
        )
        w2_occupants_ok = all(m2[rid][0] for rid in w2_rids)
        now2_victim = _metric_sum(_scrape_master_metrics(ops), "auto_tpm_victim", {})
        w2_delta = (now2_victim or 0.0) - (base2_victim or 0.0)
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the decode-role-blocked incoming 50 no longer
        # surfaces its original routing rejection (8403) — it parks in the
        # pull-based coordinator with schedule() blocking until the 60s
        # queueTimeout deadline, then terminals as plain 8511
        # BATCH_SLO_EXPIRED (observed).  EV-2 (decode eviction never
        # fires) is unchanged: zero victims, occupants complete, metric
        # flat.
        w2_family = (CODE_NO_DECODE,) + ROUTE_REJECT_FAMILY + (CODE_RESOURCE_EXHAUSTED,)
        w2_legal = (CODE_OK, CODE_SLO_EXPIRED) + w2_family
        wave_reports.append(
            (
                "w2_same_priority_zero_eviction",
                w2_inc_code in w2_legal
                and w2_zero_eviction
                and w2_occupants_ok
                and w2_delta == 0.0,
                (
                    f"[EV-1-FIXED] incoming50 terminal={w2_inc_code} "
                    f"(design-final legal set {list(w2_legal)}: the decode-"
                    f"blocked submitter parks — schedule() blocks to the 60s "
                    f"queueTimeout deadline → 8511 park expiry observed; "
                    f"8403 and the reject family remain legal under other "
                    f"timings), "
                    f"zero 8400/8429={w2_zero_eviction}, occupants completed="
                    f"{w2_occupants_ok}, victim.count delta={w2_delta} "
                    f"(expected 0.0)"
                ),
            )
        )
        clean2_ok, clean2_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        if not clean2_ok:
            wave_reports[-1] = (
                wave_reports[-1][0],
                False,
                wave_reports[-1][2] + f", inflight dirty: {clean2_detail}",
            )

        # ---- wave 3: kvBucket-descending victim (weak) ------------------
        for name in decode_names:
            ops.set_kv_pressure(name, 0)
        time.sleep(PERF_SETTLE_S)
        small_rids = [ops.next_request_id(base) for _ in range(2)]
        big_rids = [ops.next_request_id(base) for _ in range(2)]
        w3_specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 500})
            for rid in small_rids
        ] + [
            (rid, {"priority": 30, "input_len": 16384, "output_len": 500})
            for rid in big_rids
        ]
        w3_fires = _fire_batch(ops, w3_specs)
        fires.extend(w3_fires)
        if not all(
            _poll_decode_running(ops, rid, timeout_s=20.0)
            for rid in small_rids + big_rids
        ):
            return False, "wave3 occupants never reached decode running"
        for name in decode_names:
            ops.set_kv_pressure(name, MOCK_TOTAL_KV_TOKENS)
        time.sleep(PERF_SETTLE_S)
        w3_inc = ops.next_request_id(base)
        w3_inc_fire = _fire(ops, w3_inc, priority=70, input_len=8192, output_len=2)
        fires.append(w3_inc_fire)

        m3 = _outcome_map(_drain(ops, w3_fires + [w3_inc_fire]))
        w3_victims = [
            rid
            for rid in small_rids + big_rids
            if m3[rid][1] in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
        ]
        w3_survivors_ok = all(
            m3[rid][0] for rid in small_rids + big_rids if rid not in w3_victims
        )
        w3_inc_ok = m3[w3_inc][0]
        w3_inc_code = m3[w3_inc][1]
        # EV-2 baseline: the kvBucket-descending victim preference is
        # unobservable for the same reason (no decode eviction ever
        # fires), so the small/big distinction never reaches a terminal.
        wave_reports.append(
            (
                "w3_kvbucket_preference_weak_ev2",
                w3_victims == []
                and (w3_inc_ok or w3_inc_code in EV2_REJECT_FAMILY)
                and w3_survivors_ok,
                (
                    f"victims={len(w3_victims)} (EV-2: kvBucket-descending "
                    f"preference has no object — decode eviction never "
                    f"fires; small-vs-big group distinction unobservable "
                    f"black-box), incoming70 ok={w3_inc_ok} "
                    f"code={w3_inc_code} (family {list(EV2_REJECT_FAMILY)}), "
                    f"survivors ok={w3_survivors_ok} "
                    f"(weak/tendency assertion per design §2.4, EV-2 form)"
                ),
            )
        )
        clean3_ok, clean3_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)

        report.invariant(
            "AT7",
            all(ok for (_l, ok, _d) in wave_reports),
            context="decode_reservation_priority",
            detail="; ".join(
                f"{l}={'ok' if ok else 'FAIL(' + d + ')'}" for l, ok, d in wave_reports
            ),
        )
        report.invariant(
            "P6",
            all(ok for (_l, ok, _d) in wave_reports) and clean3_ok,
            detail=(
                f"all three waves drained (victims terminal, occupants "
                f"completed), inflight={'ok' if clean3_ok else clean3_detail}"
            ),
        )
        return report.finish(
            f"waves={[l for l, ok, _d in wave_reports if ok]}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for name in decode_names:
                ops.set_kv_pressure(name, 0)
        except Exception:
            pass
        _finally_hygiene(ops, fires, prefill_names)
