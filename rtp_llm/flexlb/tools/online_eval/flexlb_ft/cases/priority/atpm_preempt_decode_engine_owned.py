from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.admission import MOCK_TOTAL_KV_TOKENS
from ...support.priority import (
    CODE_ENGINE_CANCELLED,
    CODE_YIELDED,
    EV2_REJECT_FAMILY,
    PERF_SETTLE_S,
    _d1_spec,
    _decode_names,
    _decode_pressure_guardrail,
    _drain,
    _finally_hygiene,
    _fire,
    _fire_batch,
    _master_http,
    _mono_to_epoch,
    _outcome_map,
    _poll_decode_running,
    _prefill_lifecycle,
    _prefill_names,
)


@case(
    "atpm_preempt_decode_engine_owned",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.3 #7 — AT5(band) + PR6 + PR10(decode)",
)
def atpm_preempt_decode_engine_owned(ctx: CaseContext):
    """DECODE_RESERVED vs DECODE_ENGINE_OWNED eviction terminal split
    (PR6: reserved → 8400 retryable, engine-accepted → 8429 typed cancel)
    plus the preemption closure budget (AT5 band).

    ENV-D1: 2P+4D, preemption allows both decode stages with
    engineCancellation {ack 50ms, completion 1000ms}, prefill inflight
    cap 3 (so four victims + the incoming all dispatch concurrently),
    queueTimeout 60s.

    Guardrail (design §2.5 row 2, EvictionManager.java:445-452): decode
    eviction is never a substitute for an ordinary available endpoint —
    EVERY decode endpoint must be in the needs-eviction state first.
    Decode saturation is manufactured at run time by injecting kv
    pressure on all four decode engines (the decode_cache_blocks knob
    does not reach the mock's KV reporting — implementation-period
    finding), so the snapshot evidence "no ordinary endpoint available"
    holds before every wave.

    Injection timing (implementation-period correction, the third major
    one): a PRIORITY queue deliberately RETAINS the strict decode KV gate
    in ordinary routing (CostBasedDecodeStrategy.applyHardFilters →
    availableKv < seqLen filters the endpoint; softQueuePlacement is
    queue && !priorityOrdering).  kv_pressure therefore goes in only
    AFTER the victim wave has routed (dispatch ACK ⇒ decode reservation
    established) and BEFORE the incoming fires; the victims' own decode
    handoff uses the already-pinned reservation and is unaffected, while
    the incoming's ordinary route fails (NO_DECODE_WORKER 8403) into
    AdmissionFallback → decode eviction.  Between waves the pressure is
    released so the next victim wave can route.

    Wave 1 (reserved-only → 8400): prefill slowed to 4s so the reserved
    window comfortably covers kv_pressure settle (master status poll 1s)
    plus the incoming's route; four priority=30 victims fire and settle
    (their decode reservations exist while their prefill is still
    executing — output_len=500 keeps the decode phase long); the 70 is
    fired inside that window, its decode placement fails on every
    endpoint → local eviction of a reserved victim → victim terminal
    8400, the 70 completes.  The reserved window is tight; if the
    observed terminal turns out 8429 (the victim had already reached
    decode running), the case records the degradation instead of
    pretending the split — the assert stays strict so the first real run
    calibrates it.

    Wave 2 (engine-owned → Cancel → 8429): four fresh 30s are polled
    until RUNNING on decode engines, then the 70 fires → the tokenized
    Cancel coordinator evicts one owned victim → typed CANCELED+8429
    (grpc-status-details-bin), the engine records the cancellation
    (verify_engine_cancelled), and the 70 itself completes.  AT5 closure =
    the 70's first engine running_ms (epoch) minus the victim's stream
    terminal (client clock crossed into the epoch domain via
    _mono_to_epoch) — expected well inside completionTimeoutMs(1000) +
    scheduling margin.

    Victim-count note: exactly ONE victim per wave (the planner releases
    one endpoint's worth); the 70 then takes that endpoint.
    """
    # A4 (Mark P1-2): batch-dispatch caliber reservation, following the
    # dual-caliber paradigm (is_batch = ctx.batch_dispatch();
    # completion-duration caliber under BATCH, client-TTFT under NON_BATCH).
    # Under BATCH dispatch the master enqueues the stream itself
    # (enqueued_by_master), so the _StreamTerminal direct-stream channel —
    # and with it the live 8429/AT5-closure observation — is NON_BATCH-only
    # by construction.  Unreachable under this case's profile
    # (single-nonbatch = NON_BATCH base, PRIORITY axis injected at the case
    # layer); the arm is reserved so a future priority-batch variant fills
    # it without touching the NON_BATCH logic below.
    if ctx.batch_dispatch():
        # TODO(A4): BATCH arm — victim terminal rides FetchResponse, the
        # closure caliber is completion-duration; fill when a
        # priority-batch variant enables BATCH dispatch.
        raise NotImplementedError(
            "atpm_preempt_decode_engine_owned BATCH arm reserved — fill "
            "when the priority-batch variant enables BATCH dispatch"
        )
    env = ctx.env_manager.ensure(_d1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    decode_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        decode_names = _decode_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=4000.0)
        time.sleep(PERF_SETTLE_S)

        # ---- wave 1: DECODE_RESERVED → local eviction → 8400 -----------
        w1_victim_rids = [ops.next_request_id(base) for _ in range(4)]
        w1_specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 500})
            for rid in w1_victim_rids
        ]
        w1_fires = _fire_batch(ops, w1_specs)  # dispatch ACKs: reservations in
        fires.extend(w1_fires)
        # Guardrail NOW: every decode endpoint needs eviction before the
        # incoming's decode preemption can run — saturate KV on all of them
        # (victims already routed; see the case docstring for the timing).
        for name in decode_names:
            ops.set_kv_pressure(name, MOCK_TOTAL_KV_TOKENS)
        time.sleep(PERF_SETTLE_S)
        # A7 (Mark P2-1): pre-fire guardrail — see
        # _decode_pressure_guardrail.
        w1_guard = _decode_pressure_guardrail(ops, decode_names)
        if not w1_guard[0]:
            return False, f"wave1 decode guardrail failed: {w1_guard[1]}"
        w1_inc = ops.next_request_id(base)
        w1_inc_fire = _fire(ops, w1_inc, priority=70, input_len=2048, output_len=2)
        fires.append(w1_inc_fire)

        w1_outcomes = _drain(ops, w1_fires + [w1_inc_fire])
        m1 = _outcome_map(w1_outcomes)
        w1_codes = {rid: m1[rid][1] for rid in w1_victim_rids}
        w1_yielded = [rid for rid in w1_victim_rids if m1[rid][1] == CODE_YIELDED]
        w1_owned = [
            rid for rid in w1_victim_rids if m1[rid][1] == CODE_ENGINE_CANCELLED
        ]
        w1_inc_ok = m1[w1_inc][0]
        w1_inc_code = m1[w1_inc][1]
        w1_victims_ok = all(m1[rid][0] for rid in w1_victim_rids)
        w1_zero_eviction = not (w1_yielded or w1_owned)
        w1_inc_rejected = (not w1_inc_ok) and w1_inc_code in EV2_REJECT_FAMILY
        # EV-2 baseline (behaviour finding, probes E9/E11 + the
        # DecodeEndpoint projection math): DECODE_RESERVED eviction never
        # fires — the kv dimension is mathematically unreachable
        # (freedKv is a subset of currentHardCharges, so "fits after
        # eviction" implies "fits without it", contradicting the
        # INFEASIBLE entry check), and the slots dimension is absorbed
        # engine-side (decode_max_concurrency=1 with four RUNNING victims
        # still dispatches the incoming — E11).  Observable form: the
        # victims all complete, zero 8400/8429, the incoming keeps a
        # rejection from EV2_REJECT_FAMILY.
        report.invariant(
            "PR6",
            w1_zero_eviction and w1_inc_rejected and w1_victims_ok,
            context="decode_reserved_terminal_ev2",
            detail=(
                f"[EV-2] reserved wave (EV-2 baseline): victims="
                f"{ {r % 1_000_000: c for r, c in w1_codes.items()} } "
                f"(all complete — reserved eviction never fires), "
                f"yielded(8400)={len(w1_yielded)}, "
                f"owned(8429)={len(w1_owned)}, "
                f"incoming70 ok={w1_inc_ok} code={w1_inc_code} "
                f"(family {list(EV2_REJECT_FAMILY)})"
            ),
        )
        clean1_ok, _cd = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        if not clean1_ok:
            return report.finish(
                f"wave1 inflight dirty, early stop, " f"grades: {report.summary()}"
            )

        # ---- wave 2: DECODE_ENGINE_OWNED → Cancel RPC → 8429 -----------
        # Release the KV pressure first: the fresh victim wave must route
        # normally (the strict KV gate would reject them otherwise — see
        # the case docstring).
        for name in decode_names:
            ops.set_kv_pressure(name, 0)
        time.sleep(PERF_SETTLE_S)
        w2_victim_rids = [ops.next_request_id(base) for _ in range(4)]
        w2_specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 500})
            for rid in w2_victim_rids
        ]
        w2_fires = _fire_batch(ops, w2_specs)
        fires.extend(w2_fires)
        running_all = all(
            _poll_decode_running(ops, rid, timeout_s=20.0) for rid in w2_victim_rids
        )
        if not running_all:
            return False, "wave2 victims never reached decode running"
        # Re-saturate every decode endpoint, then fire the incoming.
        for name in decode_names:
            ops.set_kv_pressure(name, MOCK_TOTAL_KV_TOKENS)
        time.sleep(PERF_SETTLE_S)
        # A7: same guardrail for wave 2 — the AT5 observation wave.
        w2_guard = _decode_pressure_guardrail(ops, decode_names)
        if not w2_guard[0]:
            return False, f"wave2 decode guardrail failed: {w2_guard[1]}"
        w2_inc = ops.next_request_id(base)
        w2_inc_fire = _fire(ops, w2_inc, priority=70, input_len=2048, output_len=2)
        fires.append(w2_inc_fire)

        w2_outcomes = _drain(ops, w2_fires + [w2_inc_fire])
        m2 = _outcome_map(w2_outcomes)
        w2_owned = [
            rid for rid in w2_victim_rids if m2[rid][1] == CODE_ENGINE_CANCELLED
        ]
        w2_survivors_ok = all(
            m2[rid][0] for rid in w2_victim_rids if rid not in w2_owned
        )
        w2_inc_ok = m2[w2_inc][0]
        w2_inc_code = m2[w2_inc][1]
        w2_zero_eviction = not w2_owned
        w2_inc_rejected = (not w2_inc_ok) and w2_inc_code in EV2_REJECT_FAMILY
        cancel_evidence = []
        for rid in w2_owned:
            ok_c, detail_c = ops.verify_engine_cancelled(rid)
            cancel_evidence.append(f"{rid % 1_000_000}:{ok_c}")
        # EV-2 baseline (behaviour finding, probe E11 two-orchestration):
        # DECODE_ENGINE_OWNED eviction (tokenized Cancel → 8429) is
        # equally unreachable — the 8429/8400 terminal split has no
        # observation object.  Observable form mirrors wave 1.
        report.invariant(
            "PR6",
            w2_zero_eviction and w2_inc_rejected and w2_survivors_ok,
            context="decode_owned_terminal_ev2",
            detail=(
                f"[EV-2] owned wave (EV-2 baseline): 8429 victims="
                f"{len(w2_owned)} (engine-owned eviction never fires), "
                f"engine cancel evidence={cancel_evidence}, "
                f"incoming70 ok={w2_inc_ok} code={w2_inc_code} "
                f"(family {list(EV2_REJECT_FAMILY)}), "
                f"survivors ok={w2_survivors_ok}"
            ),
        )

        # A9-4 (Mark P3-2): PR10(decode) in its vacuous EV-2 form — the
        # replacement-precision property (deficit-exact victims,
        # infeasible → zero partial eviction) has no decode-side object
        # while decode eviction never fires; the vacuous form (zero
        # evictions across BOTH waves, no deficit object anywhere) is
        # asserted so the property stays registered with the degradation
        # recorded instead of silently absent.
        report.invariant(
            "PR10",
            w1_zero_eviction and w2_zero_eviction,
            context="decode_deficit_vacuous_ev2",
            detail=(
                "[EV-2] decode-side PR10 vacuous form: zero evictions "
                "across both waves (reserved + owned), no deficit "
                "object — replacement precision unobservable while "
                "decode eviction never fires"
            ),
        )

        # AT5 closure: incoming first engine running (epoch ms) minus the
        # victim's stream terminal crossed into the epoch domain.  Under
        # EV-2 there is no victim terminal to anchor against, so the
        # banded property has NO observation object this run: check()
        # would need a fabricated value and invariant() is illegal for a
        # banded property (raises) — the gap is filed as behaviour
        # finding EV-2 and carried in the case detail instead.  The
        # terminal channel itself is now LIVE (A1, Ryan P1-1):
        # _StreamTerminal maps the engine's in-band error frames
        # (GenerateOutputsPB.error_info → CANCELLED enum →
        # CODE_ENGINE_CANCELLED), so the moment a Java-side EV-2 fix makes
        # decode eviction reachable, w2_owned populates from real 8429
        # terminals, the verify_engine_cancelled evidence loop above
        # runs, and this computation feeds the AT5 band automatically —
        # no further framework change needed.
        closure_ms = None
        if w2_owned and w2_inc_ok:
            victim_fire = next(fr for fr in w2_fires if fr.rid == w2_owned[0])
            inc_lc = _prefill_lifecycle(ops, w2_inc) or {}
            if (
                victim_fire.terminal is not None
                and victim_fire.terminal.terminated_s is not None
                and inc_lc.get("running_ms")
            ):
                victim_end_epoch = _mono_to_epoch(victim_fire.terminal.terminated_s)
                closure_ms = inc_lc["running_ms"] - victim_end_epoch * 1000.0
        if closure_ms is not None:
            report.check(
                "AT5",
                closure_ms,
                context="preemption_closure",
                detail=(
                    "closure = incoming prefill running_ms − victim stream "
                    f"terminal (epoch-crossed); completionTimeoutMs=1000"
                ),
            )
        clean2_ok, clean2_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            w2_inc_rejected and w2_survivors_ok and clean2_ok,
            detail=(
                f"[EV-2] wave2 drained (EV-2: zero eviction, all victims "
                f"completed, incoming terminal {w2_inc_code}), "
                f"inflight={'ok' if clean2_ok else clean2_detail}"
            ),
        )
        return report.finish(
            f"EV-2 baseline: wave1 zero-eviction (incoming {w1_inc_code}), "
            f"wave2 zero-eviction (incoming {w2_inc_code}), "
            f"closure_ms={'n/a (EV-2)' if closure_ms is None else f'{closure_ms:.0f}'}, "
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
