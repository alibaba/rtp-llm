"""Smoke test cases (functional correctness e2e).

Two families:

  * cancel (smoke_cancel_t1..t6) and anomaly (smoke_anomaly_e1..e4) —
    functional-correctness cases ported 1:1 from the legacy scripts
    (cancel_smoke.py / anomaly_smoke.py), unchanged by the task #61 rework.

  * balance/affinity (bal_* / aff_*) — RESULT-PROPERTY cases (task #61
    rework, superseding scheduling_smoke.py S1-S12).  They assert observable
    outcome properties (P-series), not mechanism narratives; every measured
    property is graded against the central band table (grade.GRADE_BANDS) —
    strict=优异 / normal=良好 / loose 地板（超出即不可用）— and each case
    returns its achieved grade for the suite-level verdict
    (all strict=优异 / all ≥normal=良好 / any beyond loose=不可用).
    Hard invariants (P2 no-starvation, P6 completeness) carry no band:
    violation is unusable at every grade.

    Case map (task #61 disposition):

      bal_uniform_serial        <- S1+S6+S8 merged (P1+P2, two variants:
                                    plain / speed-heterogeneous injection)
      bal_concurrent_mix        <- S7 strengthened (P1 relaxed + P2 + P6)
      bal_overload_avoid_prefill<- S4 + new P7 short-request protection
                                    (P5 graded + P6 + P7 dual-caliber)
      bal_overload_avoid_decode <- S11 strengthened (P5 delta-caliber graded
                                    + P6 + takeover assertions)
      bal_decode_spread         <- S3+S10 merged (P2+P1, n=10/50 two tiers)
      aff_prefix_stickiness     <- S2+S5 merged (P9 graded; the legacy
                                    cache_keys>0 assertion demoted to an
                                    observational log — mock-internal cache
                                    accounting belongs to mock unit tests)
      bal_gate_no_starvation    <- S9 rebuilt around a REAL queue-depth
                                    gate (P6 + share-shift observation)
      (S12 deleted — the reserve weight-lowering mechanism it asserted
      does not exist on the Java stack; its collapse guard is subsumed by
      bal_decode_spread's P2.)

    Historical mechanism notes survive as per-case comments only where they
    explain WHY a property band is what it is (e.g. tie-window uniform
    sampling calibrates P1's loose floor).  Result properties are asserted
    profile-agnostically: all bal_/aff_ cases run under every profile (the
    only exception is bal_gate_no_starvation's requires=["enqueue_batch"] —
    the queue-depth gate lives at the engine's EnqueueBatch entry only).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .context import CaseContext, CaseDef, rid_base
from .grade import GradeReport
from .harness import AssertUtils

KV_CACHE_SYNC_WAIT_S = 2.0
STREAM_TIMEOUT_S = 15.0

SMOKE_CASES: list[CaseDef] = []


def case(name: str, profiles=None, requires=None, source: str = ""):
    def deco(fn):
        SMOKE_CASES.append(
            CaseDef(
                name=name,
                suite="smoke",
                fn=fn,
                profiles=profiles,
                requires=requires,
                source=source,
            )
        )
        return fn

    return deco


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


# ===========================================================================
# Cancel cases (cancel_smoke.py T1-T6)
# ===========================================================================


def _engine_side_checks(ops, rid: int, response) -> tuple[str, str, str, str]:
    """Common engine-side verification (recv / cancelled / inflight-clean)."""
    method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
    engine_recv, recv_detail = ops.verify_engine_received(rid, method)
    engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
    if response.enqueued_by_master:
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 10.0
        )
    else:
        inflight_ok, inflight_detail = True, "N/A"
    return (
        f"engine_recv={engine_recv}({recv_detail}), "
        f"engine_cancelled={engine_cancelled}({cancel_detail}), "
        f"inflight_clean={inflight_ok}({inflight_detail}), ",
        recv_detail,
        "",
        "",
    )


@case("smoke_cancel_t1", source="cancel_smoke.py T1")
def t1_basic_cancel(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            handle.cancel()
            return False, "no output received before cancel window"
        cancel_at = time.monotonic()
        ops.cancel(rid, response)
        ended = handle.wait_end(5.0)
        cancel_latency = time.monotonic() - cancel_at
        recovery_ok, recovery_msg = ops.verify_recovery()
        method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
        engine_recv, recv_detail = ops.verify_engine_received(rid, method)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        passed = ended and recovery_ok
        return passed, (
            f"cancel_latency={cancel_latency:.3f}s, stream_terminated={ended}, "
            f"outputs={len(handle.snap.outputs)}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_cancel_t2", source="cancel_smoke.py T2")
def t2_cancel_idempotency(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            handle.cancel()
            return False, "no output received before cancel window"
        ops.cancel(rid, response)
        second_cancel_ok, second_cancel_err = True, ""
        try:
            ops.cancel(rid, response)
        except Exception as exc:
            second_cancel_ok, second_cancel_err = False, repr(exc)
        ended = handle.wait_end(5.0)
        recovery_ok, recovery_msg = ops.verify_recovery()
        method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
        engine_recv, recv_detail = ops.verify_engine_received(rid, method)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        passed = second_cancel_ok and ended and recovery_ok
        return passed, (
            f"second_cancel_ok={second_cancel_ok} {second_cancel_err}, "
            f"stream_terminated={ended}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_cancel_t3", source="cancel_smoke.py T3")
def t3_multi_request_isolation(ctx: CaseContext):
    ops = ctx.ops()
    base = rid_base(ctx, "cancel")
    rids = [ops.next_request_id(base) for _ in range(3)]
    cancel_rid = rids[1]  # B
    # B runs a long decode so the cancel lands while it is still decoding.
    # The default output_len=10 finishes decode in ~200ms and races the
    # cancel: the engine finishes first, the master then (correctly) returns
    # the terminal state idempotently without forwarding an engine cancel,
    # and verify_engine_cancelled flaps.  200 tokens ≈ 4-5s of decode at the
    # default perf step_ms, comfortably spanning the cancel window.
    long_output_len = 200
    try:

        def _schedule(rid: int):
            if rid == cancel_rid:
                return ops.schedule(rid, output_len=long_output_len)
            return ops.schedule(rid)

        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(_schedule, rids))
        for i, resp in enumerate(responses):
            if resp.code != 200 or not resp.success:
                return False, f"schedule failed for rid={rids[i]}: {resp.error_message}"

        handles = []
        for rid, resp in zip(rids, responses):
            input_pb = (
                None if resp.enqueued_by_master else ops.build_generate_input(rid)
            )
            handles.append(ops.start_stream(resp, rid, input_pb=input_pb))

        # Wait for the SHORT requests' (A, C) first output only.  In batch
        # mode the mock engine's FetchResponse surfaces the first message
        # only after decode completes, so waiting for B (output_len=200)
        # would mean B is already terminal when the cancel fires — the
        # master then (correctly) answers REQUEST_STATE_COMPLETED
        # idempotently and never forwards an engine cancel.  A/C finish in
        # ~1s while B still has ~3.5s of decode left, so cancelling right
        # after A/C's first output lands the cancel mid-decode.
        if not all(handles[i].wait_first_output(15.0) for i in (0, 2)):
            for h in handles:
                h.cancel()
            return False, "short requests (A, C) did not receive first output"

        ops.cancel(cancel_rid, responses[1])
        b_ended = handles[1].wait_end(5.0)
        a_complete = handles[0].wait_end(30.0)
        c_complete = handles[2].wait_end(30.0)

        recovery_ok, recovery_msg = ops.verify_recovery()
        method = (
            "enqueue_batch" if responses[1].enqueued_by_master else "generate_stream"
        )
        engine_recv, recv_detail = ops.verify_engine_received(cancel_rid, method)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(cancel_rid)
        if responses[1].enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"

        a_snap, b_snap, c_snap = handles[0].snap, handles[1].snap, handles[2].snap
        passed = (
            b_ended
            and a_complete
            and a_snap.completed
            and c_complete
            and c_snap.completed
            and not b_snap.completed
            and engine_cancelled
            and recovery_ok
        )
        return passed, (
            f"A_completed={a_snap.completed}(outputs={len(a_snap.outputs)}), "
            f"B_cancelled={b_ended}(completed={b_snap.completed}), "
            f"C_completed={c_snap.completed}(outputs={len(c_snap.outputs)}), "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_cancel_t4", source="cancel_smoke.py T4")
def t4_cancel_after_completion(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid, output_len=1)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        deadline = time.monotonic() + 30.0
        while not handle.snap.completed and time.monotonic() < deadline:
            time.sleep(0.05)
        if not handle.snap.completed:
            handle.cancel()
            return False, "request did not complete before timeout"
        cancel_ok, cancel_err = True, ""
        try:
            ops.cancel(rid, response)
        except Exception as exc:
            cancel_ok, cancel_err = False, repr(exc)
        handle.wait_end(2.0)
        recovery_ok, recovery_msg = ops.verify_recovery()
        method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
        engine_recv, recv_detail = ops.verify_engine_received(rid, method)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        passed = cancel_ok and recovery_ok
        return passed, (
            f"cancel_ok={cancel_ok} {cancel_err}, completed={handle.snap.completed}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_cancel_t5", source="cancel_smoke.py T5")
def t5_cancel_nonexistent(ctx: CaseContext):
    ops = ctx.ops()
    try:
        fake_rid = 99999
        cancel_ok, cancel_err = True, ""
        try:
            ops.cancel(fake_rid)
        except Exception as exc:
            cancel_ok, cancel_err = False, repr(exc)
        return cancel_ok, (
            f"cancel(rid={fake_rid}) ok={cancel_ok} {cancel_err}, "
            f"engine_verify=N/A (nonexistent request)"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_cancel_t6", source="cancel_smoke.py T6")
def t6_cancel_at_prefill_vs_decode(ctx: CaseContext):
    ops = ctx.ops()
    base = rid_base(ctx, "cancel")
    try:
        # A: cancel in prefill phase (before any output)
        rid_a = ops.next_request_id(base)
        resp_a = ops.schedule(rid_a)
        if resp_a.code != 200 or not resp_a.success:
            return False, f"schedule A failed: {resp_a.error_message}"
        input_pb_a = (
            None if resp_a.enqueued_by_master else ops.build_generate_input(rid_a)
        )
        handle_a = ops.start_stream(resp_a, rid_a, input_pb=input_pb_a)
        time.sleep(0.1)
        a_in_prefill = not handle_a.snap.first_received
        ops.cancel(rid_a, resp_a)
        a_ended = handle_a.wait_end(5.0)

        # B: cancel in decode phase (after first output)
        rid_b = ops.next_request_id(base)
        resp_b = ops.schedule(rid_b)
        if resp_b.code != 200 or not resp_b.success:
            return False, f"schedule B failed: {resp_b.error_message}"
        input_pb_b = (
            None if resp_b.enqueued_by_master else ops.build_generate_input(rid_b)
        )
        handle_b = ops.start_stream(resp_b, rid_b, input_pb=input_pb_b)
        b_got_first = handle_b.wait_first_output()
        if not b_got_first:
            handle_b.cancel()
            return False, "B never received first output (decode phase)"
        ops.cancel(rid_b, resp_b)
        b_ended = handle_b.wait_end(5.0)

        recovery_ok, recovery_msg = ops.verify_recovery()
        method_a = "enqueue_batch" if resp_a.enqueued_by_master else "generate_stream"
        method_b = "enqueue_batch" if resp_b.enqueued_by_master else "generate_stream"
        engine_recv_a, _ = ops.verify_engine_received(rid_a, method_a)
        engine_cancelled_a, _ = ops.verify_engine_cancelled(rid_a)
        engine_recv_b, _ = ops.verify_engine_received(rid_b, method_b)
        engine_cancelled_b, _ = ops.verify_engine_cancelled(rid_b)
        if resp_a.enqueued_by_master or resp_b.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"

        passed = a_ended and b_ended and a_in_prefill and recovery_ok
        return passed, (
            f"A_prefill_phase={a_in_prefill}, A_terminated={a_ended}, "
            f"A_outputs={len(handle_a.snap.outputs)}, "
            f"B_decode_phase={b_got_first}, B_terminated={b_ended}, "
            f"B_outputs={len(handle_b.snap.outputs)}, "
            f"engine_recv_A={engine_recv_a}, engine_cancel_A={engine_cancelled_a}, "
            f"engine_recv_B={engine_recv_b}, engine_cancel_B={engine_cancelled_b}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


# ===========================================================================
# Balance / affinity cases (result-property graded — task #61 rework of
# scheduling_smoke.py S1-S12)
# ===========================================================================


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot_by_name()
    return sorted(name for name, e in snap.items() if e.get("role") == "prefill")


def _decode_names(ops) -> list[str]:
    snap = ops.snapshot_by_name()
    return sorted(name for name, e in snap.items() if e.get("role") == "decode")


@case(
    "bal_uniform_serial",
    source="scheduling_smoke.py S1+S6+S8 (merged, task #61)",
)
def bal_uniform_serial(ctx: CaseContext):
    """Homogeneous serial traffic spreads evenly across equivalent engines.

    Result properties (graded): P1 request-uniformity max-share + P2
    no-starvation, measured over n=20 serial requests per variant, counted
    from CLIENT landing addresses.

    Two parameterized variants:
      * plain — no injection;
      * speed_hetero — one prefill slowed via set_perf (200ms fixed).  The
        injection is a regression guard, not a routing signal: the prefill
        score is ledgerWaitMs + FORMULA estimate (a pure function of the
        request's token shape) + batcherWaitMs, so engine *speed* never
        enters the score and serial requests leave no backlog — both
        engines stay tied and the tie window is sampled uniformly per
        request.  A skewed split under this variant would mean the score
        model started leaking speed or leaving residue — a real defect the
        P1 property catches at any grade.
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    n = 20
    perf_engine = None
    try:
        for variant, slow in (("plain", False), ("speed_hetero", True)):
            if slow:
                prefill_names = _prefill_names(ops)
                if len(prefill_names) >= 2:
                    ops.set_perf(prefill_names[1], prefill_fixed_ms=200.0)
                    perf_engine = prefill_names[1]
                    time.sleep(1.5)  # master perf sync

            addrs = []
            failure = None
            for _ in range(n):
                rid = ops.next_request_id(rid_base(ctx, "scheduling"))
                keys = [rid * 100 + j for j in range(3)]
                addr, err = ops.run_one_request(
                    rid,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failure = f"{variant}: rid={rid} failed: {err}"
                    break
                addrs.append(addr)
            if failure:
                report.invariant("P6", False, context=variant, detail=failure)
                break

            addr_map = ops.addr_to_name()
            dist = Counter(addr_map.get(a, a) for a in addrs)
            used = len(dist)
            max_share = max(dist.values()) / n
            dist_json = json.dumps(dict(dist), sort_keys=True)
            report.check(
                "P1",
                max_share,
                context=variant,
                detail=f"n={n}, dist={dist_json}",
            )
            report.invariant("P2", used >= 2, context=variant, detail=f"workers={used}")

        slow_note = (
            f", speed_injection={perf_engine} (observational: engine speed "
            f"is not a prefill-score input)"
            if perf_engine
            else ""
        )
        return report.finish(
            f"variants=plain+speed_hetero, n={n}, grades: {report.summary()}"
            f"{slow_note}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if perf_engine:
            try:
                ops.set_perf(perf_engine, prefill_fixed_ms=100.0)
            except Exception:
                pass


@case(
    "bal_concurrent_mix",
    source="scheduling_smoke.py S7 (strengthened, task #61)",
)
def bal_concurrent_mix(ctx: CaseContext):
    """A concurrent mixed burst must not collapse onto a single engine.

    Result properties: P1 (relaxed band inheritance) + P2 no-starvation +
    P6 completeness over a 20-request / 20-way-concurrent burst, counted
    from client landing addresses.

    P1 band note (relax=1): under a concurrent burst the master may process
    the burst in several groups; within a group every request is evaluated
    against the same live ledger snapshot, so the split is a fresh uniform
    draw per group — group-splitting is CORRECT balancing behaviour, not a
    defect.  The effective bands are therefore shifted one tier right
    (strict→0.75, normal/loose→0.85): the calibrated loose floor (0.85,
    false-failure < 1% at 2 engines / 20 samples) is never widened past
    itself.
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "scheduling")
    try:
        rids = [ops.next_request_id(base) for _ in range(20)]

        def run(rid: int):
            keys = [rid * 100 + j for j in range(3)]
            return ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )

        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(run, rids))
        addrs = []
        failures = []
        for rid, (addr, err) in zip(rids, results):
            if err:
                failures.append(f"rid={rid}: {err}")
            else:
                addrs.append(addr)

        addr_map = ops.addr_to_name()
        dist = Counter(addr_map.get(a, a) for a in addrs)
        used = len(dist)
        n_ok = len(addrs)
        max_share = max(dist.values()) / n_ok if n_ok else 1.0
        dist_json = json.dumps(dict(dist), sort_keys=True)

        report.invariant(
            "P6",
            not failures and n_ok == 20,
            detail=f"completed={n_ok}/20, failures={failures[:2]}",
        )
        report.check(
            "P1",
            max_share,
            context="burst20",
            relax=1,
            detail=f"ok={n_ok}, dist={dist_json} (concurrent group-split "
            f"is correct behaviour — bands shifted one tier)",
        )
        report.invariant("P2", used >= 2, detail=f"workers={used}")

        return report.finish(
            f"burst=20x20-way, workers={used}, " f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "bal_overload_avoid_prefill",
    source="scheduling_smoke.py S4 + short-request protection (task #61)",
)
def bal_overload_avoid_prefill(ctx: CaseContext):
    """Single-engine prefill overload: traffic diverts AND short requests stay fast.

    Result properties: P5 overload-avoidance hot share (graded), P6
    completeness, P7 short-request protection (graded, dual caliber).

    Hotspot construction (inherited from the S4 port, Java-true — real
    ledger load, not the legacy fake queue_depth knob):
      1. slow BOTH prefill engines to 5s fixed and let the master sync;
      2. seed one fire-and-forget request with input_len=49152 — the ledger
         predicts ~49s, so whichever engine it lands on stays heavy for the
         whole observation window;
      3. poll the mock snapshot until a prefill engine reports
         waiting+running >= 1 (engine-side proof the seed was dispatched,
         and identification of the hot engine);
      4. restore the cool engine to 100ms (drains instantly, ledger ~0);
      5. baseline: ONE timed request — deterministically lands on the cool
         engine and anchors the P7 denominator;
      6. wave: 5 serial timed requests (0.4s spacing) — each is evaluated
         against live ledger state; every one is consumed to completion so
         per-request timings are client-observed.

    P7 dual caliber (profile-dependent measurement, one band table):
      * NON_BATCH dispatch — client TTFT: schedule-return → first stream
        output (StreamSnapshot.first_received_s);
      * BATCH dispatch — completion duration: schedule-return → stream
        terminal state.  Under BATCH the mock surfaces the first
        FetchResponse message only after decode completes (the smoke T3
        lesson), so FetchResponse "TTFT" cannot observe the prefill phase
        at all; the completion-duration口径 carries the same protection
        signal (a request swallowed by the hot engine pays its ~5s prefill
        either way).

    Drainage (inherited S4 lesson, kept in finally): the seed is
    fire-and-forget, so every fired request is consumed to terminal state
    (cancel as fallback) — otherwise the seed's ~49s ledger prediction
    keeps one engine's wait high for the rest of the suite and poisons
    later balance cases.
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "scheduling")
    is_batch = ctx.batch_dispatch()
    caliber = "completion_duration" if is_batch else "client_ttft"
    prefill_names: list[str] = []
    fired: list[tuple[int, object]] = []  # (rid, response) — drained in finally
    fired_handles: dict[int, object] = (
        {}
    )  # rid -> opened direct stream (NON_BATCH seed)
    try:
        prefill_names = _prefill_names(ops)
        if len(prefill_names) < 2:
            return False, "need >=2 prefill workers"

        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=5000.0)
        time.sleep(1.5)  # master syncs the slowed perf before we seed

        addr_map = ops.addr_to_name()

        def fire(rid: int, **kwargs):
            """Schedule without consuming the stream — keeps it pending."""
            resp = ops.schedule(rid, **kwargs)
            if resp.code != 200 or not resp.success:
                return None, f"schedule failed: {resp.error_message}"
            fired.append((rid, resp))
            if resp.enqueued_by_master:
                return addr_map.get(ops.role_addr(resp, "PREFILL"), ""), None
            # NON_BATCH: the master only published the route decision; the
            # engine sees the seed when the CLIENT opens the stream.  Open
            # it fire-and-forget (never wait) so the engine-side pending
            # the hotspot poll needs really exists.
            input_pb = ops.build_generate_input(rid, **kwargs)
            try:
                fired_handles[rid] = ops.start_stream(resp, rid, input_pb=input_pb)
            except Exception as exc:
                return None, f"seed direct stream failed to open: {exc!r}"
            return addr_map.get(ops.role_addr(resp, "PREFILL"), ""), None

        def timed_request(rid: int, **kwargs):
            """Schedule + consume to completion, capturing client timings.

            Returns (engine_name, ttft_s, duration_s, err); the request is
            fully consumed here (NOT appended to *fired* — only the
            fire-and-forget seed needs the finally-drain).
            """
            t_send = time.monotonic()
            try:
                resp = ops.schedule(rid, **kwargs)
            except Exception as exc:
                return None, None, None, repr(exc)
            if resp.code != 200 or not resp.success:
                return None, None, None, f"schedule failed: {resp.error_message}"
            name = addr_map.get(ops.role_addr(resp, "PREFILL"), "")
            input_pb = (
                None
                if resp.enqueued_by_master
                else ops.build_generate_input(rid, **kwargs)
            )
            try:
                handle = ops.start_stream(resp, rid, input_pb=input_pb)
            except Exception as exc:
                return name, None, None, f"stream failed to open: {exc!r}"
            ended = handle.wait_end(STREAM_TIMEOUT_S)
            snap = handle.snap
            ttft = snap.first_received_s - t_send if snap.first_received_s else None
            dur = snap.terminated_s - t_send if snap.terminated_s else None
            if not ended or snap.error or not snap.completed:
                return name, ttft, dur, (snap.error or "stream did not complete")
            return name, ttft, dur, None

        # -- seed: big ledger footprint, fire-and-forget (~49s predicted wait).
        seed_rid = ops.next_request_id(base)
        seed_name, err = fire(seed_rid, input_len=49152, output_len=2)
        if err:
            return False, f"seed request failed: {err}"
        if seed_name not in prefill_names:
            return False, f"seed request went to unknown worker {seed_name}"

        # -- engine-side proof: poll the snapshot until the seed shows up.
        deadline = time.monotonic() + 6.0
        hot = None
        while time.monotonic() < deadline and hot is None:
            snap = ops.snapshot_by_name()
            for name in prefill_names:
                info = snap.get(name, {})
                if info.get("waiting", 0) + info.get("running", 0) >= 1:
                    hot = name
                    break
            if hot is None:
                time.sleep(0.1)
        if hot is None:
            return False, "seed never appeared on any engine (engine side)"
        if hot != seed_name:
            return False, f"seed routed to {seed_name} but pending showed up on {hot}"
        cool = next(n for n in prefill_names if n != hot)

        # -- cool engine fast again; baseline anchors the P7 denominator
        #    (hot carries the ~49s seed prediction → baseline lands cool).
        ops.set_perf(cool, prefill_fixed_ms=100.0)
        time.sleep(0.3)
        base_rid = ops.next_request_id(base)
        base_name, base_ttft, base_dur, base_err = timed_request(base_rid, output_len=2)
        if base_err:
            report.invariant("P6", False, detail=f"baseline failed: {base_err}")
            return report.finish(f"baseline request failed: {base_err}")

        # -- serial timed wave: every request faces live ledger state.
        wave = []
        for i in range(5):
            rid = ops.next_request_id(base)
            entry = timed_request(rid, output_len=2)
            wave.append(entry)
            if i < 4:
                time.sleep(0.4)

        dist = Counter(w[0] for w in wave if w[3] is None)
        hot_count = dist.get(hot, 0)
        hot_share = hot_count / len(wave) if wave else 1.0
        failures = [f"landing={w[0]}: {w[3]}" for w in wave if w[3] is not None]

        # P6: baseline + every wave request completed (no loss, no hang).
        report.invariant(
            "P6",
            not failures,
            detail=f"failures={failures[:2]}",
        )
        # P5: hot-engine share of the wave (graded; strict=0 = deterministic).
        report.check(
            "P5",
            hot_share,
            context="prefill_overload",
            detail=f"hot={hot}({hot_count}/5), cool={cool}({dist.get(cool, 0)}), "
            f"dist={json.dumps(dict(dist), sort_keys=True)}",
        )
        # P7: short-request protection relative to the unloaded baseline,
        # dual caliber by dispatch mode (see docstring).
        metric_idx = 2 if is_batch else 1  # (name, ttft, dur, err)
        metric_base = (base_dur if is_batch else base_ttft) or 0.0
        wave_metrics = [w[metric_idx] for w in wave if w[3] is None and w[metric_idx]]
        if metric_base > 0 and wave_metrics:
            p7_value = max(wave_metrics) / metric_base
            p7_detail = (
                f"caliber={caliber}, base={metric_base:.3f}s, "
                f"wave_max={max(wave_metrics):.3f}s"
            )
        else:
            p7_value = float("inf")
            p7_detail = f"caliber={caliber}, missing timing (base={metric_base})"
        report.check("P7", p7_value, context=caliber, detail=p7_detail)

        return report.finish(
            f"hot={hot}, cool={cool}, hot_share={hot_share:.2f}, {p7_detail}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for name in prefill_names:
                ops.set_perf(name, prefill_fixed_ms=100.0)
        except Exception:
            pass
        # Drainage (inherited S4 lesson): fire-and-forget requests never
        # consume FetchResponse, so their master-side inflight/ledger entries
        # can linger long after the engine finished and poison later balance
        # cases.  The deterministic cleanup is the normal completion path —
        # consume each fired request's stream to terminal state (the seed's
        # ~5s prefill is the only slow one), with cancel as fallback.
        for rid, resp in fired:
            try:
                if rid in fired_handles:
                    # NON_BATCH: the direct stream opened at fire time IS
                    # the completion path — consume it to terminal state.
                    fired_handles[rid].wait_end(20.0)
                else:
                    ops.start_stream(resp, rid).wait_end(20.0)
            except Exception:
                try:
                    ops.cancel(rid, resp)
                except Exception:
                    pass
        try:
            AssertUtils.inflight_clean(_master_http(ops), 30.0)
        except Exception:
            pass


@case(
    "bal_overload_avoid_decode",
    source="scheduling_smoke.py S11 (strengthened, task #61)",
)
def bal_overload_avoid_decode(ctx: CaseContext):
    """Decode KV exhaustion: the pressured engine stops taking new work and
    the healthy engines absorb the traffic.

    Result properties: P5 overload-avoidance in the *delta caliber* (graded:
    how many of the n requests still complete on the KV-exhausted engine),
    P6 completeness, P2 no-starvation takeover assertions.

    P5 band note (case override, absolute-delta caliber): the global P5
    share bands translate awkwardly to 10 samples (0.05*10 = 0.5); the
    delta bands strict=0 / normal=1 / loose=2 carry the same intent with
    the historical calibration that exactly one straggler request can
    already be in prefill→decode handoff when the pressure snapshot lands
    (the legacy S11 delta<=1 was the stable-pass baseline).

    Takeover strengthening (vs legacy S11, which only bounded the target
    delta): the non-pressured decode engines must actually absorb the
    diverted load — at least two of them take requests, and every one of
    the n requests completes somewhere (no loss).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    injected: str | None = None
    try:
        decode_names = _decode_names(ops)
        if len(decode_names) < 2:
            return False, "need >=2 decode workers"
        target = decode_names[0]
        info = ops.snapshot_by_name()[target]
        total_kv = int(info.get("available_kv_tokens", 0)) + int(
            info.get("active_kv_tokens", 0)
        )
        ops.set_kv_pressure(target, total_kv)  # available -> 0
        injected = target
        time.sleep(1.0)  # master worker-status sync

        snap_sync = ops.snapshot_by_name()
        completed_before = {
            name: snap_sync.get(name, {}).get("completed", 0) for name in decode_names
        }

        n = 10
        failures = []
        for _ in range(n):
            rid = ops.next_request_id(rid_base(ctx, "scheduling"))
            keys = [rid * 100 + j for j in range(3)]
            _, err = ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )
            if err:
                failures.append(f"rid={rid}: {err}")

        snap2 = ops.snapshot_by_name()
        deltas = {
            name: snap2[name].get("completed", 0) - completed_before.get(name, 0)
            for name in decode_names
        }
        target_delta = deltas.get(target, 0)
        others = {name: d for name, d in deltas.items() if name != target}
        others_used = sum(1 for v in others.values() if v > 0)
        others_total = sum(others.values())

        # P6: every request completed somewhere — no loss under pressure.
        report.invariant(
            "P6",
            not failures and others_total + target_delta >= n,
            detail=f"failures={failures[:2]}, total_delta={others_total + target_delta}/{n}",
        )
        # P5: hot-engine delta caliber (graded, case override — see docstring).
        report.check(
            "P5",
            float(target_delta),
            context="decode_kv_pressure",
            bands={"strict": 0.0, "normal": 1.0, "loose": 2.0},
            detail=f"target={target}(delta={target_delta}), "
            f"deltas={json.dumps(deltas, sort_keys=True)}",
        )
        # P2: takeover — the diverted load actually lands on the healthy
        # engines (>=2 of them used), i.e. nobody is starved by the pressure.
        report.invariant(
            "P2",
            others_used >= 2 and others_total >= n - target_delta,
            context="decode_takeover",
            detail=f"others_used={others_used}, others_total={others_total}",
        )

        return report.finish(
            f"target={target}(delta={target_delta}), others_used={others_used}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if injected:
            try:
                ops.set_kv_pressure(injected, 0)
            except Exception:
                pass


@case(
    "bal_decode_spread",
    source="scheduling_smoke.py S3+S10 (merged, task #61)",
)
def bal_decode_spread(ctx: CaseContext):
    """Decode traffic spreads across the decode fleet at both small and
    large sample sizes.

    Result properties: P2 no-starvation (min engines used) + P1 distribution
    bound (case-calibrated 4-engine bands) + P6 completeness, parameterized
    over n=10 and n=50 tiers.  Landing points are engine-completed counts
    (decode deltas via mock snapshots).

    P1 band note (case override, 4-engine caliber): the decode selector is
    KV_USAGE_WEIGHTED_RANDOM, not a uniform draw — weights track per-worker
    KV residue left by earlier cases, so the 2-engine bands do not apply.
    Calibration (task #61, batch-window, engine-completed deltas): n=10
    observed max_share 0.40 (dist 1/4/3/2), n=50 observed 0.40 (dist
    20/10/13/7 — the KV-weighted draw routinely puts ~40% on the residue-
    heaviest engine).  Bands kept at n=10: 0.60/0.70/0.80, n=50:
    0.40/0.50/0.60 — the n=50 strict tier sits ON the observed mode (a
    quality bar for weight convergence, not a statistical guarantee);
    widen from full-suite regression data in task #63 if the
    residue-inheritance across predecessor cases pushes it over.
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    try:
        tiers = (
            # (n, min_used, P1 case bands)
            (10, 2, {"strict": 0.60, "normal": 0.70, "loose": 0.80}),
            (50, 3, {"strict": 0.40, "normal": 0.50, "loose": 0.60}),
        )
        for n, min_used, bands in tiers:
            decode_names = _decode_names(ops)
            if len(decode_names) < 2:
                return False, "need >=2 decode workers"
            snap0 = ops.snapshot_by_name()
            baseline = {name: snap0[name].get("completed", 0) for name in decode_names}

            failures = []
            for _ in range(n):
                rid = ops.next_request_id(rid_base(ctx, "scheduling"))
                keys = [rid * 100 + j for j in range(3)]
                _, err = ops.run_one_request(
                    rid,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"rid={rid}: {err}")

            snap1 = ops.snapshot_by_name()
            deltas = {
                name: snap1[name].get("completed", 0) - baseline.get(name, 0)
                for name in decode_names
            }
            total = sum(deltas.values())
            used = sum(1 for v in deltas.values() if v > 0)
            max_share = max(deltas.values()) / n if n else 1.0
            deltas_json = json.dumps(deltas, sort_keys=True)

            report.invariant(
                "P6",
                not failures and total >= n,
                context=f"n{n}",
                detail=f"failures={failures[:2]}, total_delta={total}/{n}",
            )
            report.invariant(
                "P2",
                used >= min_used,
                context=f"n{n}",
                detail=f"used={used}/{len(decode_names)} (need >= {min_used})",
            )
            report.check(
                "P1",
                max_share,
                context=f"n{n}",
                bands=bands,
                detail=f"dist={deltas_json} (4-engine KV-weighted caliber)",
            )

        return report.finish(f"tiers=n10+n50, grades: {report.summary()}")
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "aff_prefix_stickiness",
    source="scheduling_smoke.py S2+S5 (merged, task #61)",
)
def aff_prefix_stickiness(ctx: CaseContext):
    """Prefix-reuse traffic sticks to the engine that holds the prefix cache.

    Result properties: P9 affinity fidelity (graded, lower band: share of
    prefix-reuse requests landing on the prime engine) + P6 completeness.
    Single-family base form only — the multi-family / free-mixing
    generalization is task #62.

    Construction: one shared key family — a prime request (input_len=2048)
    seeds the block cache on its landing engine; after KV_CACHE_SYNC_WAIT_S
    the master's cache-affinity policy (CacheAffinityPolicy.evaluate in
    CostBasedPrefillStrategy: the cache leader wins when the prefix hit
    clears minPrefixHitPercent) should keep every follower reusing the
    SAME keys on the SAME engine.

    The legacy S5 cache_keys>0 assertion is demoted to an observational
    log: mock-internal cache accounting is the mock's own unit-tested
    behaviour, not an LB contract.

    P9 band note: followers are serial (each completes before the next
    fires), so the prime engine's ledger is ~0 at every draw and a miss
    can only come from a tie-window random pick overriding affinity;
    normal (0.90) tolerates one miss in ten, loose (0.80) two.
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    followers_n = 10
    # Constant key family disjoint from the rid-derived keys used by other
    # cases (rid*100+j stays far below 10^7) — no cross-case cache pollution.
    family_keys = [9900001, 9900002, 9900003]
    try:
        rid_prime = ops.next_request_id(rid_base(ctx, "scheduling"))
        prime_addr, prime_err = ops.run_one_request(
            rid_prime,
            input_len=2048,
            output_len=2,
            block_keys=family_keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if prime_err:
            report.invariant("P6", False, detail=f"prime failed: {prime_err}")
            return report.finish(f"prime request failed: {prime_err}")

        time.sleep(KV_CACHE_SYNC_WAIT_S)  # master cache sync

        addrs = []
        failures = []
        for _ in range(followers_n):
            rid = ops.next_request_id(rid_base(ctx, "scheduling"))
            addr, err = ops.run_one_request(
                rid,
                input_len=2048,
                output_len=2,
                block_keys=family_keys,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                failures.append(f"rid={rid}: {err}")
            else:
                addrs.append(addr)

        addr_map = ops.addr_to_name()
        prime_name = addr_map.get(prime_addr, prime_addr)
        hits = sum(1 for a in addrs if a == prime_addr)
        stick_share = hits / len(addrs) if addrs else 0.0

        # Observational only (legacy S5 demoted to log — mock-internal
        # cache accounting belongs to mock unit tests, not the LB contract).
        cache_keys = ops.snapshot_by_name().get(prime_name, {}).get("cache_keys", -1)

        report.invariant("P6", not failures, detail=f"failures={failures[:2]}")
        report.check(
            "P9",
            stick_share,
            context="single_family",
            detail=f"prime={prime_name}, hits={hits}/{len(addrs)}, "
            f"cache_keys={cache_keys} (observational)",
        )
        return report.finish(
            f"prime={prime_name}, stick={hits}/{len(addrs)}, "
            f"cache_keys={cache_keys}(log), grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "bal_gate_no_starvation",
    requires=["enqueue_batch"],
    source="scheduling_smoke.py S9 (rebuilt around a real gate, task #61)",
)
def bal_gate_no_starvation(ctx: CaseContext):
    """A REAL engine queue-depth gate: rejections are explicit and visible;
    nothing is lost or left hanging behind the gate.

    Result properties: P6 completeness (hard invariant — every wave request
    reaches a visible terminal state: completed, or explicitly rejected by
    the gate with BATCH_DISPATCH_FAILED / "queue depth limit exceeded",
    observed either at the schedule RPC or on the stream) plus
    observational share-shift accounting (where the surviving traffic
    lands while the gate is full).

    Java-truth note (supersedes the legacy S9's 50000-limit no-op form and
    its "rejected requests still complete elsewhere" expectation): the
    Java master does NOT requeue a gate-rejected EnqueueBatch — the
    rejection is fail-fast to a terminal BATCH_DISPATCH_FAILED state
    (PriorityScheduler.reduceDeliveryFailure; e2e-proven by
    FaultInjectionE2ETest.c08).  "No starvation" is therefore the
    black-box completeness contract: a gated request fails fast and
    loudly (never hangs, never vanishes), and the healthy engine keeps
    accepting.

    Construction (all levers real):
      1. slow BOTH prefills to 5s fixed;
      2. decoy: one fire-and-forget input_len=16384 request (~16s ledger
         prediction) — poll the snapshot for its landing engine; that
         engine is priced out for the window (high score), the OTHER
         engine is the gated target;
      3. restore the decoy engine to 100ms; set queue_depth=2 on the
         target — the REAL Java enqueue gate (EnqueueBatch rejects when
         pendingRequests >= limit, JavaMockEngineCluster.enqueueBatch);
      4. two slow fire-and-forget seeds: the decoy's ledger prediction
         keeps the other engine's score high, so both seeds land on the
         target — pending hits 2 and the gate is FULL for ~5s (proven
         engine-side before the wave starts);
      5. wave: 6 serial requests — each either hits the full gate
         (explicit fast rejection) or diverts to the decoy engine once
         the master backs off / the decoy's prediction decays
         (share-shift, observational).

    requires=["enqueue_batch"] (S9 inheritance): the gate is checked ONLY
    at the engine's EnqueueBatch entry — the GenerateStreamCall path
    (NON_BATCH dispatcher) never consults it.
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "scheduling")
    prefill_names: list[str] = []
    fired: list[tuple[int, object]] = []  # (rid, response) — drained in finally
    target: str | None = None
    try:
        prefill_names = _prefill_names(ops)
        if len(prefill_names) < 2:
            return False, "need >=2 prefill workers"

        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=5000.0)
        time.sleep(1.5)  # master perf sync

        addr_map = ops.addr_to_name()

        # -- decoy: prices one engine out of the window (~16s prediction).
        decoy_rid = ops.next_request_id(base)
        resp = ops.schedule(decoy_rid, input_len=16384, output_len=2)
        if resp.code != 200 or not resp.success:
            return False, f"decoy schedule failed: {resp.error_message}"
        fired.append((decoy_rid, resp))
        decoy_name = addr_map.get(ops.role_addr(resp, "PREFILL"), "")
        if decoy_name not in prefill_names:
            return False, f"decoy went to unknown worker {decoy_name}"

        # Engine-side proof the decoy is really parked there.
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            info = ops.snapshot_by_name().get(decoy_name, {})
            if info.get("waiting", 0) + info.get("running", 0) >= 1:
                break
            time.sleep(0.1)

        target = next(n for n in prefill_names if n != decoy_name)
        # -- decoy engine fast again; REAL gate on the target.
        ops.set_perf(decoy_name, prefill_fixed_ms=100.0)
        ops.set_queue_depth(target, 2)

        # -- two slow seeds fill the gate (pending=2), ~5s window.  The
        #    decoy's ~16s ledger prediction keeps its engine's score high,
        #    so the seeds land on the target deterministically.
        for _ in range(2):
            rid = ops.next_request_id(base)
            resp = ops.schedule(rid, input_len=1024, output_len=2)
            if resp.code != 200 or not resp.success:
                return False, f"seed schedule failed: {resp.error_message}"
            fired.append((rid, resp))

        # Engine-side proof the gate is really full before the wave starts.
        deadline = time.monotonic() + 6.0
        gate_full = False
        while time.monotonic() < deadline:
            info = ops.snapshot_by_name().get(target, {})
            if info.get("waiting", 0) + info.get("running", 0) >= 2:
                gate_full = True
                break
            time.sleep(0.1)
        if not gate_full:
            return False, f"seeds never filled the gate on {target} (engine side)"

        # -- wave: 6 serial requests against the full gate.
        wave = []
        for _ in range(6):
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
            )
            wave.append((addr_map.get(addr, addr), err))

        def is_explicit_rejection(err: str) -> bool:
            low = err.lower()
            return "queue depth" in low or "dispatch" in low or "8510" in err

        completed = [w for w in wave if w[1] is None]
        rejected = [w for w in wave if w[1] and is_explicit_rejection(w[1])]
        unmatched = [w for w in wave if w not in completed and w not in rejected]

        # Share-shift observation (not a hard band): where did the traffic
        # that DID get through land, and how many hit the gate?
        shift = Counter(name for name, _ in completed)

        # P6: every request reached a visible terminal state — completed
        # or explicitly gate-rejected; anything else (hang, timeout, silent
        # loss) is a completeness violation at every grade.
        report.invariant(
            "P6",
            not unmatched,
            detail=f"unmatched={[(n, e) for n, e in unmatched][:2]}",
        )
        return report.finish(
            f"target={target}(gate=2), decoy={decoy_name}, "
            f"wave: completed={len(completed)}, gate_rejected={len(rejected)}, "
            f"unmatched={len(unmatched)}, "
            f"shift={json.dumps(dict(shift), sort_keys=True)}, "
            f"rejected_errs={sorted({e for _, e in rejected})[:2]}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        # Restore perf + clear the gate BEFORE draining: once the target is
        # fast again its seeds finish quickly and the drain converges fast.
        try:
            for name in prefill_names:
                ops.set_perf(name, prefill_fixed_ms=100.0)
        except Exception:
            pass
        if target:
            try:
                ops.set_queue_depth(target, 0)
            except Exception:
                pass
        # Drainage (inherited S4 lesson): fire-and-forget requests must be
        # consumed to terminal state or their master-side inflight/ledger
        # entries poison later cases.
        for rid, resp in fired:
            try:
                ops.start_stream(resp, rid).wait_end(20.0)
            except Exception:
                try:
                    ops.cancel(rid, resp)
                except Exception:
                    pass
        try:
            AssertUtils.inflight_clean(_master_http(ops), 30.0)
        except Exception:
            pass


# ===========================================================================
# Anomaly cases (anomaly_smoke.py E1-E3)
# ===========================================================================

TIMEOUT_WAIT_S = 5.0
ANOMALY_STREAM_TIMEOUT_S = 10.0
WORKER_RECOVERY_WAIT_S = 3.0
# E4 probe: short client-side gRPC deadline proving the parked Schedule RPC
# stays pending (the master's own scheduling deadline is queueTimeoutMs,
# default 1h — far beyond any useful probe).
E4_PROBE_DEADLINE_S = 5.0


def _inject_all_prefill(ops, config: dict) -> list[str]:
    snap = ops.snapshot()
    prefill_names = [
        e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"
    ]
    for name in prefill_names:
        ops.inject(name, config)
    return prefill_names


def _clear_all_prefill_inject(ops, names: list[str]) -> None:
    for name in names:
        try:
            ops.clear_inject(name)
        except Exception:
            pass


@case("smoke_anomaly_e1", source="anomaly_smoke.py E1")
def e1_cancel_path(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "anomaly"))
    try:
        response = ops.schedule(rid)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            handle.cancel()
            return False, "no output received before cancel window"
        cancel_at = time.monotonic()
        ops.cancel(rid, response)
        ended = handle.wait_end(5.0)
        cancel_latency = time.monotonic() - cancel_at
        recovery_ok, recovery_msg = ops.verify_recovery()
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            # NON_BATCH: a client Cancel on a delivered request cannot
            # safely release the master ledger entry (fence probe NOT_FOUND
            # is not a safe-release fact), so immediate-zero is not a
            # contract here — see E4's watermark rationale.
            inflight_ok, inflight_detail = True, "N/A (NON_BATCH residue contract)"
        passed = ended and recovery_ok
        return passed, (
            f"cancel_latency={cancel_latency:.3f}s, stream_terminated={ended}, "
            f"outputs={len(handle.snap.outputs)}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


def _anomaly_error_case(
    ctx: CaseContext, inject_config: dict, wait_s: float, require_error_detail: bool
) -> tuple[bool, str]:
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "anomaly"))
    error_observed = False
    error_detail = "no error observed"
    injected_names: list[str] = []
    response = None
    try:
        injected_names = _inject_all_prefill(ops, inject_config)
        try:
            response = ops.schedule(rid)
            if response.code != 200 or not response.success:
                error_observed = True
                error_detail = f"schedule error: {response.error_message}"
            else:
                input_pb = (
                    None
                    if response.enqueued_by_master
                    else ops.build_generate_input(rid)
                )
                handle = ops.start_stream(response, rid, input_pb=input_pb)
                ended = handle.wait_end(wait_s)
                if not ended:
                    error_observed = True
                    error_detail = f"stream timed out (no response within {wait_s}s)"
                if handle.snap.error:
                    error_observed = True
                    error_detail = f"stream error: {handle.snap.error}"
                elif require_error_detail and not handle.snap.completed:
                    error_observed = True
                    error_detail = "stream did not complete"
        except Exception as exc:
            error_observed = True
            error_detail = f"exception: {exc!r}"
        finally:
            _clear_all_prefill_inject(ops, injected_names)

        # Explicitly cancel the failed request to clean up server-side
        # inflight (scheduler keeps the entry until TTL eviction otherwise).
        if response is not None and response.success:
            try:
                ops.cancel(rid, response)
            except Exception:
                pass

        time.sleep(WORKER_RECOVERY_WAIT_S)
        recovery_ok, recovery_msg = ops.verify_recovery()
        if response is not None and response.success and response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            # NON_BATCH: see E1 — client Cancel cannot safely release a
            # delivered ledger entry, so immediate-zero is not asserted.
            inflight_ok, inflight_detail = True, "N/A (NON_BATCH residue contract)"
        passed = error_observed and recovery_ok
        return passed, (
            f"error_observed={error_observed} ({error_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_anomaly_e2", source="anomaly_smoke.py E2")
def e2_timeout(ctx: CaseContext):
    return _anomaly_error_case(ctx, {"no_respond": True}, TIMEOUT_WAIT_S, False)


@case("smoke_anomaly_e3", source="anomaly_smoke.py E3")
def e3_worker_fail(ctx: CaseContext):
    return _anomaly_error_case(
        ctx, {"enqueue_error": True}, ANOMALY_STREAM_TIMEOUT_S, True
    )


@case(
    "smoke_anomaly_e4",
    source="decode-side anomaly gap (G1) — new",
)
def e4_decode_capacity_exhausted(ctx: CaseContext):
    """Decode-side anomaly: every decode engine KV-exhausted -> the request
    is parked undelivered, a master Cancel releases it without residue, and
    clearing the pressure recovers routing.

    Why KV pressure and not the E-series ops.inject faults: a decode engine
    in the Java mock never receives traffic through any gRPC entry point.
    After prefill completes, the request is handed off IN-PROCESS
    (JavaMockEngineCluster.FastRpcService.startDecode ->
    scheduleDecodeCompletion), so enqueue_error / generate_error /
    fetch_error — all checked at the enqueueBatch / generateStreamCall /
    fetchResponse RPC entries — never fire for a decode engine, and
    no_respond on decode only suppresses the "intermediate first-step
    output" which the mock never produces (each request yields exactly one
    finished message).  The one decode-side anomaly observable e2e is KV
    capacity: the delivery-capacity admission hard-filters every decode
    endpoint whose available_kv_tokens < seq_len, so exhausting every
    decode engine's KV must block delivery.

    v2 contract (task #55, source-verified — supersedes the v1 fail-fast
    assertion): the QUEUE scheduler treats decode KV exhaustion as a WAIT
    condition, not a fail-fast rejection:
      * FixedWindowBatcherAlgorithm parks the head when delivery capacity
        cannot be reserved ("Dynamic KV pressure is a wait condition, not a
        rejection"; BatcherContext.admitAndDeliverCapacityFeasiblePrefix
        returns CapacityBlocked and the worker loop waits for the exact
        resource-change event).
      * The scheduling deadline is owned by the queue config
        (QueueSchedulerConfig.queueTimeoutMs, default 1h), not the caller,
        so the Schedule RPC stays pending while parked — the client
        observes its own gRPC DEADLINE_EXCEEDED instead of a rejection
        response.  The pre-v2 fail-fast NO_AVAILABLE_WORKER contract
        belonged to the v1 non-QUEUE flow and does not exist in v2.
      * A client-side RPC deadline/cancellation does NOT release the
        parked entry (it lingered until the stale-inflight TTL eviction in
        the repro); an explicit master Cancel does
        (PriorityScheduler.cancelRequest -> isLocallyReversible -> local
        cleanup).

    Scenario:
      1. set active_kv_tokens = total on every decode engine
      2. probe Schedule with a short client-side deadline: it must stay
         pending (client DEADLINE_EXCEEDED, no rejection response) and the
         parked rid must NOT be delivered to any engine
      3. master Cancel must release the parked request and leave no
         inflight residue
      4. clear the pressure -> a fresh request must complete again

    Profile semantics (v2): the decision and dispatcher axes are invisible
    to the decode-side delivery capacity gate — both delivery modes share
    the per-worker batcher and the same capacity admission — so the case
    runs under all profiles.  The no-residue assertion is a pre-probe
    WATERMARK comparison rather than a global zero check: under NON_BATCH
    dispatch a client-side Cancel cannot safely release a delivered
    request's master ledger entry (the fence probe's NOT_FOUND ack is not
    a safe-release fact — the client connects to the engine
    asynchronously after RouteDecision), so earlier E-series requests on
    the shared env may leave contract-parked entries; this case only owns
    the residue of ITS OWN parked probe.
    """
    ops = ctx.ops()
    base = rid_base(ctx, "anomaly")
    injected: list[str] = []
    try:
        snap = ops.snapshot_by_name()
        decode_names = sorted(
            name for name, e in snap.items() if e.get("role") == "decode"
        )
        if not decode_names:
            return False, "no decode workers found"

        # Exhaust every decode engine: active = total -> available = 0.
        for name in decode_names:
            info = snap[name]
            total_kv = int(info.get("available_kv_tokens", 0)) + int(
                info.get("active_kv_tokens", 0)
            )
            ops.set_kv_pressure(name, total_kv)
            injected.append(name)
        time.sleep(1.5)  # master worker-status sync

        # 1. The probe stays pending: a short client-side deadline fires
        #    instead of the master returning a rejection.
        base_view = ops.master_inflight() or {}

        def _inflight_totals(view: dict) -> tuple[int, int, int]:
            return (
                int(view.get("scheduler_inflight", 0) or 0),
                sum(
                    int(ep.get("inflight_batches", 0) or 0)
                    for ep in view.get("prefill_endpoints", []) or []
                ),
                sum(
                    int(ep.get("inflight_requests", 0) or 0)
                    for ep in view.get("decode_endpoints", []) or []
                ),
            )

        base_sched, base_prefill, base_decode = _inflight_totals(base_view)
        rid = ops.next_request_id(base)
        probe: dict = {}

        def _probe() -> None:
            try:
                resp = ops.schedule(rid, timeout_s=E4_PROBE_DEADLINE_S)
                probe["returned"] = (
                    f"code={resp.code}, success={resp.success}, "
                    f"error={resp.error_message!r}"
                )
            except Exception as exc:  # client deadline while parked
                code_fn = getattr(exc, "code", None)
                probe["grpc_code"] = str(code_fn()) if callable(code_fn) else ""
                probe["exc"] = repr(exc)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_probe).result(timeout=E4_PROBE_DEADLINE_S + 10.0)
        parked_ok = probe.get("grpc_code") == "StatusCode.DEADLINE_EXCEEDED"
        parked_detail = probe.get("returned") or probe.get(
            "grpc_code", probe.get("exc", "no outcome")
        )

        # 2. The parked request must not have been delivered to any engine.
        time.sleep(0.5)
        snap2 = ops.snapshot()
        delivered = [
            engine["name"]
            for engine in snap2.get("engines", [])
            if str(rid) in engine.get("request_lifecycle", {})
        ]
        not_delivered_ok = not delivered

        # 3. An explicit master Cancel releases the parked request: master
        #    inflight must return to the pre-probe watermark (scheduler
        #    entry + decode shadow reservation both released).
        cancel_err = None
        try:
            ops.cancel(rid, None)
        except Exception as exc:
            cancel_err = repr(exc)
        time.sleep(0.5)
        inflight_ok, inflight_detail = False, "no inflight view"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            view = ops.master_inflight()
            if view is not None:
                sched, pre, dec = _inflight_totals(view)
                if sched <= base_sched and pre <= base_prefill and dec <= base_decode:
                    inflight_ok = True
                    inflight_detail = (
                        f"back to pre-probe watermark "
                        f"(scheduler={sched}/{base_sched}, "
                        f"prefill_batches={pre}/{base_prefill}, "
                        f"decode_reservations={dec}/{base_decode})"
                    )
                    break
                inflight_detail = (
                    f"scheduler={sched} (base {base_sched}), "
                    f"prefill_batches={pre} (base {base_prefill}), "
                    f"decode_reservations={dec} (base {base_decode})"
                )
            time.sleep(0.5)

        # 4. Clear the pressure on every decode engine; recovery must be
        #    functional, not just cosmetic: a fresh request must schedule
        #    and complete again.
        for name in injected:
            try:
                ops.set_kv_pressure(name, 0)
            except Exception:
                pass
        time.sleep(2.0)  # master worker-status sync (recovery view)

        rid_rec = ops.next_request_id(base)
        rec_addr, rec_err = ops.run_one_request(
            rid_rec, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            parked_ok
            and not_delivered_ok
            and cancel_err is None
            and inflight_ok
            and rec_err is None
            and recovery_ok
        )
        return passed, (
            f"parked_pending={parked_ok} ({parked_detail}), "
            f"delivered_while_parked={delivered or 'none'}, "
            f"cancel_err={cancel_err}, "
            f"recovered_request_ok={rec_err is None}"
            f"(prefill={rec_addr}, err={rec_err}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for name in injected:
            try:
                ops.set_kv_pressure(name, 0)
            except Exception:
                pass
