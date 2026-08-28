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
                                    accounting belongs to mock unit tests);
                                    task #62 generalization (M1): two seed
                                    families (ledger-forced apart) + free mix
      bal_gate_no_starvation    <- S9 rebuilt around a REAL queue-depth
                                    gate (P6 + share-shift observation)
      (S12 deleted — the reserve weight-lowering mechanism it asserted
      does not exist on the Java stack; its collapse guard is subsumed by
      bal_decode_spread's P2.)

    Task #62 adds the heterogeneity dimensions (design L/M):

      bal_len_mixed            <- L1 bimodal length mix (P3 token-share
                                   graded — first P3 calibration + P2 short
                                   request spread + P6)
      aff_hot_prefix_tension   <- M2 hot-prefix tension (P9 stickiness vs
                                   M2 concentration cap — first M2
                                   calibration + P2 free-flow + P6)
      aff_match_mixed          <- M3 hit-rate tiers (M3 soft contrast
                                   bound: full/half vs zero-hit + P2 zero-hit
                                   spread + P6)

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
    # The default output_len=10 finishes decode in ~80ms and races the
    # cancel: the engine finishes first, the master then (correctly) returns
    # the terminal state idempotently without forwarding an engine cancel,
    # and verify_engine_cancelled flaps.  500 tokens ≈ 3.8s of decode at the
    # default production-fit step pricing (ceil(500/2.6)=193 steps ×
    # (19.5+0.175×running) ms), comfortably spanning the cancel window.
    long_output_len = 500
    try:

        def _schedule(rid: int):
            if rid == cancel_rid:
                return ops.schedule(rid, output_len=long_output_len)
            return ops.schedule(rid)

        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(_schedule, rids))
        for i, resp in enumerate(responses):
            if resp.code != 200 or not resp.success:
                # Drainage discipline (S4 lesson, 2026-08-27 task #63
                # post-mortem): a sibling that was already scheduled must not
                # be left behind as an unconsumed entry — under BATCH dispatch
                # the leaked EnqueueBatch result sits in the engine's fetch
                # queue and the master's inflight/ledger far past the 30s
                # stale TTL (fence-quarantine family), poisoning later cases
                # on the shared env (observed cascade: T3 leak ->
                # aff_prefix_stickiness / bal_len_mixed /
                # bal_gate_no_starvation failures in the batch-window full
                # run, all solo-PASS). Cancel every scheduled sibling before
                # failing the case; the streams were never opened, so the
                # master-side cancel is a clean local release on both
                # dispatch modes.
                for j, sibling in enumerate(responses):
                    if j != i and sibling.code == 200 and sibling.success:
                        try:
                            ops.cancel(rids[j], sibling)
                        except Exception:
                            pass
                return False, f"schedule failed for rid={rids[i]}: {resp.error_message}"

        handles = []
        for rid, resp in zip(rids, responses):
            if resp.enqueued_by_master:
                input_pb = None
            else:
                # Shape fidelity (finding-⑥ family, 2026-08-28 task #63
                # post-mortem): under NON_BATCH dispatch the direct stream's
                # GenerateInputPB must carry the SAME output_len the
                # ScheduleRequest carried.  A default-shape rebuild
                # (output_len=10) finishes B's decode in ~80ms — inside the
                # cancel-path latency (~150-250ms) — turning the docstring's
                # "comfortably spanning" cancel window into a coin flip on
                # every NON_BATCH run (observed: wn full-run + wn solo FAIL
                # with B completed before the engine-side cancel landed,
                # while sn flapped run-dependent).  With output_len=500 the
                # ~3.8s decode dwarfs the cancel path on both dispatch modes.
                kwargs = {"output_len": long_output_len} if rid == cancel_rid else {}
                input_pb = ops.build_generate_input(rid, **kwargs)
            handles.append(ops.start_stream(resp, rid, input_pb=input_pb))

        # Wait for the SHORT requests' (A, C) first output only.  In batch
        # mode the mock engine's FetchResponse surfaces the first message
        # only after decode completes, so waiting for B (output_len=500)
        # would mean B is already terminal when the cancel fires — the
        # master then (correctly) answers REQUEST_STATE_COMPLETED
        # idempotently and never forwards an engine cancel.  A/C finish in
        # ~1s while B still has ~3.3s of decode left, so cancelling right
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

    P6 note (intake2 scheduler contract, task #65 intake2-sync): the
    RequestScheduler/GroupPolicy architecture fails fast with retryable
    NO_PREFILL_WORKER (8402) once the per-engine inflight-batch admission
    ledger is saturated — dispatcher.maxInflightBatchesPerPrefillWorker
    (harness default 4) x 2 smoke prefill workers = 8 concurrent batch
    reservations; each burst request is its own batch, so a 20-way burst
    exceeds the floor and the excess is rejected at select time via
    projection PROJECTION_BLOCKED_DELIVERY_CAPACITY_BATCH_ADMISSION
    (flexlb-sync BatchPrefillAdmission.reserveBatch +
    WorkerBatcher.admissionBlockUnderLock).  The old "master queues the
    entire burst" assumption no longer holds: completeness now means
    (a) no failure other than batch-admission backpressure, and (b) at
    least the 8-reservation floor admitted — the first 8 arrivals always
    find a permit (releases only raise the floor).
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

        admission_denied = [f for f in failures if "NO_PREFILL_WORKER" in f]
        hard_failures = [f for f in failures if "NO_PREFILL_WORKER" not in f]
        report.invariant(
            "P6",
            not hard_failures and n_ok >= 8,
            detail=(
                f"completed={n_ok}/20, batch-admission-denied="
                f"{len(admission_denied)} (retryable 8402, intake2 "
                f"contract), failures={hard_failures[:2]}"
            ),
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
      2. seed one fire-and-forget request with input_len=147456 — the
         production-fit ledger prices it at ~2.06s, so the landing engine
         stays heavy through the routing window (the legacy 1ms/token
         default priced the old 49152 seed at ~49s and could span a serial
         wave; the fit cannot, so the wave below is compressed into the
         seed's ledger lifetime);
      3. poll the mock snapshot until a prefill engine reports
         waiting+running >= 1 (engine-side proof the seed was dispatched,
         and identification of the hot engine);
      4. restore the cool engine to 100ms (drains instantly, ledger ~0);
      5. baseline: ONE timed request — deterministically lands on the cool
         engine and anchors the P7 denominator;
      6. wave: 5 requests fired back-to-back (0.12s spacing) so ALL five
         routing decisions happen while the seed's ~2.06s ledger is still
         live; timings are collected after the last decision — a serial
         consume-and-fire wave would outlive the ledger and re-open the
         tie window mid-wave.

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

        def fire_timed(rid: int, **kwargs):
            """Wave phase 1: schedule + open the stream WITHOUT waiting.

            The routing decision happens here, against the live ledger;
            returns (engine_name, handle, t_send, err).
            """
            t_send = time.monotonic()
            try:
                resp = ops.schedule(rid, **kwargs)
            except Exception as exc:
                return None, None, t_send, repr(exc)
            if resp.code != 200 or not resp.success:
                return None, None, t_send, f"schedule failed: {resp.error_message}"
            name = addr_map.get(ops.role_addr(resp, "PREFILL"), "")
            input_pb = (
                None
                if resp.enqueued_by_master
                else ops.build_generate_input(rid, **kwargs)
            )
            try:
                handle = ops.start_stream(resp, rid, input_pb=input_pb)
            except Exception as exc:
                return name, None, t_send, f"stream failed to open: {exc!r}"
            return name, handle, t_send, None

        def collect_timed(handle, name, t_send):
            """Wave phase 2: consume one fired request, client timings."""
            if handle is None:
                return name, None, None, "stream never opened"
            ended = handle.wait_end(STREAM_TIMEOUT_S)
            snap = handle.snap
            ttft = snap.first_received_s - t_send if snap.first_received_s else None
            dur = snap.terminated_s - t_send if snap.terminated_s else None
            if not ended or snap.error or not snap.completed:
                return name, ttft, dur, (snap.error or "stream did not complete")
            return name, ttft, dur, None

        # -- seed: big ledger footprint, fire-and-forget (~2.06s predicted
        #    ledger under the production fit; the slow mock keeps it in
        #    flight far beyond that, but only the prediction drives routing).
        seed_rid = ops.next_request_id(base)
        seed_name, err = fire(seed_rid, input_len=147456, output_len=2)
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
        #    (hot still carries most of the ~2.06s seed ledger → baseline
        #    deterministically lands cool, well outside the tie window).
        ops.set_perf(cool, prefill_fixed_ms=100.0)
        time.sleep(0.3)
        base_rid = ops.next_request_id(base)
        base_name, base_ttft, base_dur, base_err = timed_request(base_rid, output_len=2)
        if base_err:
            report.invariant("P6", False, detail=f"baseline failed: {base_err}")
            return report.finish(f"baseline request failed: {base_err}")

        # -- timed wave, two-phase: fire all five back-to-back (each
        #    routing decision faces the live seed ledger), then collect
        #    timings once the last decision is made. A serial consume loop
        #    would spend ~0.3s per request and push the final decisions
        #    past the ~2.06s ledger lifetime.
        wave = []
        wave_fired: list[tuple[int, object, object, float]] = []
        for i in range(5):
            rid = ops.next_request_id(base)
            name, handle, t_send, err = fire_timed(rid, output_len=2)
            wave_fired.append((rid, name, handle, t_send))
            wave.append((name, None, None, err) if err else None)
            if i < 4:
                time.sleep(0.12)
        for idx, (rid, name, handle, t_send) in enumerate(wave_fired):
            if wave[idx] is None:
                wave[idx] = collect_timed(handle, name, t_send)

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
        # ~5s mock prefill is the only slow one), with cancel as fallback.
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
    source="scheduling_smoke.py S2+S5 (merged, task #61; M1 generalized, task #62)",
)
def aff_prefix_stickiness(ctx: CaseContext):
    """Prefix-reuse traffic sticks to the engine that holds the prefix cache —
    multi-family + free-mixing generalization (M1).

    Result properties (graded): P9 affinity fidelity (family-A followers
    landing on the family-A seed engine), P2 free-flow multi-engine spread,
    P6 completeness.

    Construction:
      1. seed A (keys 1001-1008, input_len=8192) fired while both prefills
         are slowed to 2s — its ~231ms production-fit estimate keeps the
         landing engine's ledger entry live (tie-window override is
         impossible: the doubled ledger ~463ms vs ~231ms dwarfs the
         ~23ms tie window);
      2. seed B (keys 2001-2008, same shape) scheduled while A is still
         in flight -> deterministically lands on the OTHER engine — the
         family separation the design calls for (a plain serial seeding
         would put both families on the same engine half the time);
      3. after both seeds complete and the master cache syncs
         (KV_CACHE_SYNC_WAIT_S), the main phase runs ~30 serial requests:
         60% family-A continuations (same keys, deterministic stickiness —
         the production-fit estimate prices the hit engine only ~6ms above
         the all-miss engine, but the bounded cache-affinity gate
         (maxExtraTtftMs=20) keeps the cache leader preferred; the legacy
         1ms/token default instead relied on its 0.7*hitTokens discount
         pushing the hit engine ~5s BELOW the tie window) interleaved
         with 40% unique-key free requests (no cache lead on either
         engine -> uniform tie-window spread).

    The legacy S5 cache_keys>0 assertion stays demoted to an observational
    log: mock-internal cache accounting is the mock's own unit-tested
    behaviour, not an LB contract.  Hit-latency benefits are NOT asserted
    (mock execution time is length/cache-blind — framework fact).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "scheduling")
    family_a_keys = list(range(1001, 1009))
    family_b_keys = list(range(2001, 2009))
    prefill_names: list[str] = []
    fired: list[tuple[int, object]] = []
    fired_handles: dict[int, object] = {}
    try:
        prefill_names = _prefill_names(ops)
        if len(prefill_names) < 2:
            return False, "need >=2 prefill workers"
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=2000.0)
        time.sleep(1.5)  # master perf sync

        # -- seed A: fire-and-forget, engine-side proof of its ledger entry.
        rid_a = ops.next_request_id(base)
        seed_a_name, err = _fire_request(
            ops,
            rid_a,
            fired,
            fired_handles,
            input_len=8192,
            output_len=2,
            block_keys=family_a_keys,
        )
        if err:
            report.invariant("P6", False, detail=f"seed A failed: {err}")
            return report.finish(f"seed A failed: {err}")
        if not _poll_engine_pending(ops, seed_a_name, 1):
            report.invariant(
                "P6", False, detail=f"seed A never appeared on {seed_a_name}"
            )
            return report.finish(f"seed A never appeared on {seed_a_name}")

        # -- seed B: deterministic away from seed A's live ledger.
        rid_b = ops.next_request_id(base)
        seed_b_addr, seed_b_err = ops.run_one_request(
            rid_b,
            input_len=8192,
            output_len=2,
            block_keys=family_b_keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        seed_b_name = ops.addr_to_name().get(seed_b_addr, seed_b_addr)
        if seed_b_err:
            report.invariant("P6", False, detail=f"seed B failed: {seed_b_err}")
            return report.finish(f"seed B failed: {seed_b_err}")

        # -- drain seed A, restore fast perf, let the master sync both caches.
        outcomes = _drain_fired(ops, fired, fired_handles)
        fired.clear()
        fired_handles.clear()
        seed_a_ok = outcomes and outcomes[0][2]
        if not seed_a_ok:
            report.invariant(
                "P6", False, detail=f"seed A did not complete: {outcomes[0][3]}"
            )
            return report.finish(f"seed A did not complete: {outcomes[0][3]}")
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=100.0)
        time.sleep(KV_CACHE_SYNC_WAIT_S)  # master cache sync

        if seed_a_name == seed_b_name:
            # The ledger technique makes this practically impossible (the
            # ~8s ledger gap dwarfs the tie window); keep the design's
            # "report it" clause as a loud observation.
            report.invariant(
                "P6",
                False,
                detail=(
                    f"family separation failed: both seeds landed on "
                    f"{seed_a_name} (ledger diversion did not fire)"
                ),
            )
            return report.finish(
                f"family separation failed: both seeds on {seed_a_name}"
            )

        # -- main phase: 60% family-A continuations + 40% unique-key free.
        cont_n, free_n = 18, 12
        addrs_a, addrs_free, failures = [], [], []
        for i in range(cont_n + free_n):
            rid = ops.next_request_id(base)
            if i % 5 < 3:  # 3:2 interleave -> 18 continuations / 12 free
                addr, err = ops.run_one_request(
                    rid,
                    input_len=8192,
                    output_len=2,
                    block_keys=family_a_keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"cont rid={rid}: {err}")
                else:
                    addrs_a.append(addr)
            else:
                keys = [rid * 100 + j for j in range(8)]
                addr, err = ops.run_one_request(
                    rid,
                    input_len=8192,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"free rid={rid}: {err}")
                else:
                    addrs_free.append(addr)

        addr_map = ops.addr_to_name()
        hits = sum(1 for a in addrs_a if addr_map.get(a, a) == seed_a_name)
        stick_share = hits / len(addrs_a) if addrs_a else 0.0
        free_engines = len({addr_map.get(a, a) for a in addrs_free})

        # Observational only (legacy S5 demoted to log).
        cache_keys_a = ops.snapshot_by_name().get(seed_a_name, {}).get("cache_keys", -1)

        report.invariant(
            "P6",
            not failures and len(addrs_a) == cont_n and len(addrs_free) == free_n,
            detail=f"failures={failures[:2]}",
        )
        report.check(
            "P9",
            stick_share,
            context="family_a",
            detail=(
                f"seed_a={seed_a_name}, seed_b={seed_b_name} (ledger-forced "
                f"apart), hits={hits}/{len(addrs_a)}, "
                f"cache_keys={cache_keys_a} (observational)"
            ),
        )
        report.invariant(
            "P2",
            free_engines >= 2,
            context="free_flow",
            detail=f"engines={free_engines}, free_n={len(addrs_free)}",
        )
        return report.finish(
            f"seed_a={seed_a_name}, seed_b={seed_b_name}, "
            f"stick={hits}/{len(addrs_a)}, free_engines={free_engines}, "
            f"cache_keys={cache_keys_a}(log), grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for name in prefill_names:
                ops.set_perf(name, prefill_fixed_ms=100.0)
        except Exception:
            pass
        if fired or fired_handles:
            _drain_fired(ops, fired, fired_handles)
        try:
            AssertUtils.inflight_clean(_master_http(ops), 30.0)
        except Exception:
            pass


# -- shared fire-and-forget helpers (S4 hotspot pattern, task #62 shared) --


def _fire_request(ops, rid: int, fired: list, fired_handles: dict, **kwargs):
    """Schedule without consuming the stream — keeps the request pending
    (ledger entry live) until the wave/case drain.

    Returns (engine_name, error).  Under NON_BATCH dispatch the engine only
    sees the request when the CLIENT opens the stream, so the direct stream
    is opened here fire-and-forget (never waited on).
    """
    try:
        resp = ops.schedule(rid, **kwargs)
    except Exception as exc:
        return None, repr(exc)
    if resp.code != 200 or not resp.success:
        return None, f"schedule failed: {resp.error_message}"
    addr = ops.role_addr(resp, "PREFILL")
    name = ops.addr_to_name().get(addr, addr)
    fired.append((rid, resp))
    if not resp.enqueued_by_master:
        try:
            input_pb = ops.build_generate_input(rid, **kwargs)
            fired_handles[rid] = ops.start_stream(resp, rid, input_pb=input_pb)
        except Exception as exc:
            return name, f"direct stream failed to open: {exc!r}"
    return name, None


def _poll_engine_pending(
    ops, engine_name: str, min_pending: int, timeout_s: float = 6.0
) -> bool:
    """Engine-side proof that a fired request was really dispatched: poll the
    mock snapshot until waiting+running >= min_pending on *engine_name*.

    Reaching the engine implies the master-side ledger entry was registered
    (dispatch precedes engine execution on both dispatch modes).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = ops.snapshot_by_name().get(engine_name, {})
        if info.get("waiting", 0) + info.get("running", 0) >= min_pending:
            return True
        time.sleep(0.1)
    return False


def _drain_fired(ops, fired: list, fired_handles: dict, wait_s: float = 30.0) -> list:
    """Consume every fired request to terminal state (S4 drainage lesson:
    unconsumed fire-and-forget entries linger in master inflight/ledger and
    poison later phases).  Returns [(rid, engine_name, completed, err)]."""
    outcomes = []
    for rid, resp in fired:
        name = ops.addr_to_name().get(ops.role_addr(resp, "PREFILL"), "")
        completed = False
        err = None
        try:
            handle = (
                fired_handles[rid]
                if rid in fired_handles
                else ops.start_stream(resp, rid)
            )
            ended = handle.wait_end(wait_s)
            completed = ended and handle.snap.completed and not handle.snap.error
            if not completed:
                err = handle.snap.error or "stream did not complete"
        except Exception as exc:
            err = repr(exc)
        if not completed:
            try:
                ops.cancel(rid, resp)
            except Exception:
                pass
        outcomes.append((rid, name, completed, err))
    return outcomes


@case(
    "bal_len_mixed",
    source="length-heterogeneity dimension L1 (task #62)",
)
def bal_len_mixed(ctx: CaseContext):
    """Bimodal length mix balances TOKEN footprint, not request count.

    Result properties (graded): P3 token-weighted max-share (first
    calibrated measurement of the P3 band), P2 short-request spread (both
    engines take short work), P6 completeness.

    Construction (5 waves, each 2 long + 6 short fire-and-forget):
      * ONE formula for both sides (task #67): mock execution time and the
        master's ledger prediction share the production DSv4 fit, so a
        long request's ledger entry decays on exactly the clock the mock
        sleeps — the diversion window equals the fitted prefill time;
      * long ladder 131072..147456: the fit predicts ~1.72-2.06 s all-miss,
        wide enough to choreograph a wave well inside the window (the old
        32k ladder predicted ~342 ms — too narrow to orchestrate against,
        which the retired set_perf(3000ms) crutch used to paper over);
      * wave choreography (S4 ledger technique, symmetric — no single hot
        engine is manufactured):
        1. L_a fired on an empty ledger pair -> uniform tie-window pick (X);
        2. poll X pending (engine-side proof the ledger entry exists);
        3. L_b fired immediately after that proof: X carries the FULL
           fitted ledger (~1.72-2.06 s) while Y is still empty — a gap
           ~20x the 10%/20ms tie window, so L_b deterministically lands
           on Y with no timing assumption about when shorter requests
           register their own ledger entries (the t+200ms variant of this
           step raced the shorts' registration and split the longs 7/3);
        4. poll Y pending (both longs now provably in flight);
        5. 6 shorts fired while BOTH longs execute: X has decayed only by
           the polls' overhead (tens of ms — inside the ~190ms tie window
           of the ~1.9s ledgers), so the shorts spread across both
           engines; the exact split does not matter for P3 because the
           shorts carry ~1% of the wave's tokens;
        6. drain the wave to terminal state + master inflight clean,
           so the next wave starts from a settled (double-zero) ledger.
      * per wave: X = L_a + ~3 shorts, Y = L_b + ~3 shorts in tokens; the
        ladder is cyclic (L_a series == L_b series), so the aggregate
        token share is pinned at ~0.50 for ANY short split (see the P3
        calibration note in grade.GRADE_BANDS).

    Why not request-count uniformity (P1): the wave deliberately lands
    BOTH long requests' complements asymmetrically in flight order —
    token balance and request-count balance genuinely conflict in this
    scene, and P3 is the property that matters.
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "scheduling")
    prefill_names: list[str] = []
    fired: list[tuple[int, object]] = []
    fired_handles: dict[int, object] = {}
    try:
        prefill_names = _prefill_names(ops)
        if len(prefill_names) < 2:
            return False, "need >=2 prefill workers"

        # deterministic bimodal ladder 131072..147456 (reproducible reruns):
        # ~1.72-2.06 s fitted all-miss prefill = the ledger window the wave
        # choreography runs inside (same formula on mock and master sides)
        long_lens = [131072 + (i % 5) * 4096 for i in range(10)]
        landed: list[tuple[int, str, int]] = []  # (rid, engine_name, input_len)
        failure = None

        for wave in range(5):
            la, lb = long_lens[2 * wave], long_lens[2 * wave + 1]
            # 1. L_a on the empty ledger pair.
            rid = ops.next_request_id(base)
            name_a, err = _fire_request(
                ops, rid, fired, fired_handles, input_len=la, output_len=2
            )
            if err:
                failure = f"wave{wave} L_a: {err}"
                break
            landed.append((rid, name_a, la))
            # 2. engine-side proof the ledger entry is live.
            if not _poll_engine_pending(ops, name_a, 1):
                failure = f"wave{wave} L_a never appeared on {name_a}"
                break
            # 3. L_b immediately after the L_a dispatch proof: X carries the
            #    full fitted ledger (~1.72-2.06 s) while Y is still empty —
            #    ~20x the tie window, so L_b deterministically lands on Y
            #    (no dependency on the shorts' ledger registration timing).
            rid = ops.next_request_id(base)
            name_b, err = _fire_request(
                ops, rid, fired, fired_handles, input_len=lb, output_len=2
            )
            if err:
                failure = f"wave{wave} L_b: {err}"
                break
            landed.append((rid, name_b, lb))
            if not _poll_engine_pending(ops, name_b, 1):
                failure = f"wave{wave} L_b never appeared on {name_b}"
                break
            # 4. 6 shorts while BOTH longs are in flight: X has decayed only
            #    by the polls' overhead (tens of ms, inside the ~190ms tie
            #    window of the ~1.9s ledgers), so the shorts spread evenly —
            #    both engines take short work (P2) and the wave stays
            #    token-symmetric for any exact split.
            for short_idx in range(6):
                rid = ops.next_request_id(base)
                name, err = _fire_request(
                    ops, rid, fired, fired_handles, input_len=512, output_len=2
                )
                if err:
                    failure = f"wave{wave} short#{short_idx}: {err}"
                    break
                landed.append((rid, name, 512))
            if failure:
                break

            # 6. drain the wave before the next one starts clean.
            outcomes = _drain_fired(ops, fired, fired_handles)
            unfinished = [(r, n, e) for (r, n, ok, e) in outcomes if not ok]
            if unfinished:
                failure = f"wave{wave} drain incomplete: {unfinished[:2]}"
                # drop the undrained tail from landed so P3 counts only
                # completed traffic
                bad_rids = {r for r, _n, _e in unfinished}
                landed = [t for t in landed if t[0] not in bad_rids]
                break
            fired.clear()
            fired_handles.clear()
            clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
            if not clean_ok:
                failure = f"wave{wave} inflight not clean: {clean_detail}"
                break

        if failure:
            report.invariant("P6", False, detail=failure)
        else:
            report.invariant("P6", True, detail="all 40 requests drained")

        if landed:
            token_by_engine: Counter = Counter()
            short_by_engine: Counter = Counter()
            for _rid, name, ln in landed:
                token_by_engine[name] += ln
                if ln == 512:
                    short_by_engine[name] += 1
            total_tokens = sum(token_by_engine.values())
            max_share = (
                max(token_by_engine.values()) / total_tokens if total_tokens else 1.0
            )
            short_engines = len(short_by_engine)
            tokens_json = json.dumps(
                {k: token_by_engine[k] for k in sorted(token_by_engine)},
                sort_keys=True,
            )
            report.check(
                "P3",
                max_share,
                context="bimodal_5waves",
                detail=f"tokens={tokens_json}, shorts={dict(short_by_engine)}",
            )
            report.invariant(
                "P2",
                short_engines >= 2,
                context="short_spread",
                detail=f"engines taking shorts={short_engines}",
            )

        return report.finish(
            f"waves=5, landed={len(landed)}/40, " f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if fired or fired_handles:
            _drain_fired(ops, fired, fired_handles)
        try:
            AssertUtils.inflight_clean(_master_http(ops), 30.0)
        except Exception:
            pass


@case(
    "aff_hot_prefix_tension",
    source="hot-prefix tension M2 (task #62)",
)
def aff_hot_prefix_tension(ctx: CaseContext):
    """A 70%-traffic hot prefix family: stickiness holds AND the holder's
    concentration stays capped.

    Result properties (graded combination): P9 family stickiness (graded —
    the design's tension axis 1), M2 holder total-share cap (graded upper
    bound — tension axis 2, first calibrated measurement of the M2 band),
    P2 free-flow no-starvation (the other engine still takes free traffic),
    P6 completeness.

    Construction: family F shares a 16-block long prefix (keys 3001-3016,
    input_len=16384).  One seed request lands on engine X (uniform initial
    pick); after the master cache sync the main phase runs 40 serial
    requests in a fixed 7:3 interleave — 28 family continuations (every one
    carries the ~10.7s estimate discount on X: est = 16384 - 0.7*15360 =
    5619 vs 16384 elsewhere, a gap ~20x the tie window -> deterministic
    stickiness) and 12 unique-key free requests (no affinity on either
    engine -> uniform tie-window spread).

    On X's accumulating state: each completed family request re-admits the
    same 16 blocks (LRU-refreshed, idempotent) and adds its inputLen to the
    mock's KV accounting — the holder's cache/KV footprint keeps growing
    across the phase (observational), while the routing ledger itself
    resets between serial requests (each completes before the next fires),
    which is what keeps P9 deterministic and pins the M2 model to the free
    flow's binomial spread.

    M2 caliber: family and free requests share input_len=16384, so token
    share and request share coincide; the holder's TOTAL share counts seed,
    family continuations AND the free requests that tie-window scatter onto
    it: (29 + k)/41 with k ~ B(12, 0.5) over the free flow (29 = seed + 28
    continuations deterministically on X when stickiness is perfect), i.e.
    ~0.854 ± 0.042 (1σ) — see the M2 calibration note in grade.GRADE_BANDS
    for the false-fail derivation of the band values.

    Free-flow starvation (P2): if all 12 free requests were swallowed by
    X the other engine would idle — that is the starvation this property
    forbids (probability 0.5**12 under correct uniform spread).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "scheduling")
    family_keys = list(range(3001, 3017))
    input_len = 16384
    try:
        # -- seed: family F prefix lands on X (uniform initial pick).
        rid_seed = ops.next_request_id(base)
        seed_addr, seed_err = ops.run_one_request(
            rid_seed,
            input_len=input_len,
            output_len=2,
            block_keys=family_keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if seed_err:
            report.invariant("P6", False, detail=f"seed failed: {seed_err}")
            return report.finish(f"seed request failed: {seed_err}")
        addr_map = ops.addr_to_name()
        holder = addr_map.get(seed_addr, seed_addr)
        other_names = [n for n in _prefill_names(ops) if n != holder]
        time.sleep(KV_CACHE_SYNC_WAIT_S)  # master cache sync

        # -- main phase: 40 serial, fixed 7:3 interleave (28 family + 12 free).
        cont_n, free_n = 28, 12
        cont_addrs, free_addrs, failures = [], [], []
        for i in range(cont_n + free_n):
            rid = ops.next_request_id(base)
            if i % 10 < 7:  # 7 family : 3 free per decade
                addr, err = ops.run_one_request(
                    rid,
                    input_len=input_len,
                    output_len=2,
                    block_keys=family_keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"cont rid={rid}: {err}")
                else:
                    cont_addrs.append(addr)
            else:
                keys = [rid * 100 + j for j in range(16)]
                addr, err = ops.run_one_request(
                    rid,
                    input_len=input_len,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"free rid={rid}: {err}")
                else:
                    free_addrs.append(addr)

        holder_hits = sum(1 for a in cont_addrs if addr_map.get(a, a) == holder)
        stick_share = holder_hits / len(cont_addrs) if cont_addrs else 0.0
        free_on_other = sum(1 for a in free_addrs if addr_map.get(a, a) != holder)
        # M2 caliber: the holder's TOTAL share — seed + family continuations
        # + free requests scattered onto it by the tie window (token share ==
        # request share by uniform input_len).
        free_on_holder = len(free_addrs) - free_on_other
        holder_total = holder_hits + 1 + free_on_holder  # + seed
        total = 1 + len(cont_addrs) + len(free_addrs)
        holder_share = holder_total / total if total else 1.0
        holder_token_share = (
            holder_total * input_len / (total * input_len) if total else 1.0
        )

        report.invariant(
            "P6",
            not failures and len(cont_addrs) == cont_n and len(free_addrs) == free_n,
            detail=f"failures={failures[:2]}",
        )
        report.check(
            "P9",
            stick_share,
            context="hot_family",
            detail=(
                f"holder={holder}, hits={holder_hits}/{len(cont_addrs)}, "
                f"other={other_names}"
            ),
        )
        report.check(
            "M2",
            holder_share,
            context="holder_total_share",
            detail=(
                f"holder={holder}: {holder_total}/{total} requests "
                f"(token share {holder_token_share:.3f} — equal by uniform "
                f"input_len), free_on_other={free_on_other}/{len(free_addrs)}"
            ),
        )
        report.invariant(
            "P2",
            free_on_other >= 1,
            context="free_flow",
            detail=(
                f"free requests landing off-holder={free_on_other}/"
                f"{len(free_addrs)} (other engine must not be starved)"
            ),
        )
        return report.finish(
            f"holder={holder}, stick={holder_hits}/{len(cont_addrs)}, "
            f"holder_share={holder_share:.3f}, "
            f"free_off_holder={free_on_other}/{len(free_addrs)}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "aff_match_mixed",
    source="hit-rate tier contrast M3 (task #62)",
)
def aff_match_mixed(ctx: CaseContext):
    """Prefix hit-rate tiers: full-hit and half-hit traffic concentrate on
    the holder while zero-hit traffic spreads — a graded contrast.

    Result properties: M3 soft contrast bound (graded lower band on the
    same-engine concentration of the full-hit and half-hit tiers), P2
    zero-hit multi-engine spread, P6 completeness.  Hit-latency benefits
    are NOT asserted (mock execution time is length/cache-blind).

    Construction (fixed input_len=8192, three tiers, all serial):
      * full-hit tier — seed family keys 4001-4008 on engine X1, then 10
        continuations reusing the SAME 8 blocks: hitTokens = 7168 (the last
        partial block is excluded: rawHit >= seqLen -> seqLen - blockSize),
        estimate discount ~5.0s vs tie window ~0.3s -> deterministic
        concentration on X1;
      * half-hit tier — seed keys 5001-5004 (input_len=4096, 4 blocks) on
        X2, then 10 requests carrying [5001-5004 + 4 fresh keys]: the
        continuous prefix match stops at 4 blocks -> hitTokens = 4096,
        discount ~2.9s vs tie window ~0.5s -> deterministic concentration
        on X2 (a 50% hit rate still clears the affinity threshold — the
        contrast with the zero-hit tier is the point, not a partial
        stickiness);
      * zero-hit tier — 10 requests with fresh unique keys on both
        engines: no discount anywhere -> uniform tie-window spread.

    Why P2 covers only the zero-hit tier: P2 forbids starving an engine
    with INDISTINGUISHABLE traffic; full/half-hit requests landing on
    their holder is correct affinity routing, not starvation.  The
    zero-hit tier is exactly the indistinguishable population, so its
    spread carries the P2 contract (probability of a single-engine
    collapse under correct spread: 2 * 0.5**10 ~= 0.2%).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "scheduling")
    full_keys = list(range(4001, 4009))
    half_shared_keys = list(range(5001, 5005))
    try:

        def run_tier_cont(n: int, keys_fn, label: str):
            """Serial run of *n* requests, each keys from keys_fn(rid, i)."""
            addrs, failures = [], []
            for i in range(n):
                rid = ops.next_request_id(base)
                addr, err = ops.run_one_request(
                    rid,
                    input_len=8192,
                    output_len=2,
                    block_keys=keys_fn(rid, i),
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"{label} rid={rid}: {err}")
                else:
                    addrs.append(addr)
            return addrs, failures

        # -- tier 1: full-hit (8-block family).
        rid_seed1 = ops.next_request_id(base)
        seed1_addr, seed1_err = ops.run_one_request(
            rid_seed1,
            input_len=8192,
            output_len=2,
            block_keys=full_keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if seed1_err:
            report.invariant("P6", False, detail=f"seed1 failed: {seed1_err}")
            return report.finish(f"full-hit seed failed: {seed1_err}")
        time.sleep(KV_CACHE_SYNC_WAIT_S)
        full_addrs, full_fail = run_tier_cont(10, lambda rid, i: full_keys, "full")

        # -- tier 2: half-hit (4 shared + 4 fresh per request).
        rid_seed2 = ops.next_request_id(base)
        seed2_addr, seed2_err = ops.run_one_request(
            rid_seed2,
            input_len=4096,
            output_len=2,
            block_keys=half_shared_keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if seed2_err:
            report.invariant("P6", False, detail=f"seed2 failed: {seed2_err}")
            return report.finish(f"half-hit seed failed: {seed2_err}")
        time.sleep(KV_CACHE_SYNC_WAIT_S)
        half_addrs, half_fail = run_tier_cont(
            10,
            lambda rid, i: half_shared_keys + [rid * 100 + 40 + j for j in range(4)],
            "half",
        )

        # -- tier 3: zero-hit (fresh unique keys everywhere).
        zero_addrs, zero_fail = run_tier_cont(
            10, lambda rid, i: [rid * 100 + j for j in range(8)], "zero"
        )

        addr_map = ops.addr_to_name()
        failures = full_fail + half_fail + zero_fail

        def concentration(addrs, anchor_addr) -> float:
            if not addrs:
                return 0.0
            anchor = addr_map.get(anchor_addr, anchor_addr)
            return sum(1 for a in addrs if addr_map.get(a, a) == anchor) / len(addrs)

        full_conc = concentration(full_addrs, seed1_addr)
        half_conc = concentration(half_addrs, seed2_addr)
        zero_dist = Counter(addr_map.get(a, a) for a in zero_addrs)
        zero_engines = len(zero_dist)
        zero_max = max(zero_dist.values()) / len(zero_addrs) if zero_addrs else 1.0

        report.invariant(
            "P6",
            not failures
            and len(full_addrs) == 10
            and len(half_addrs) == 10
            and len(zero_addrs) == 10,
            detail=f"failures={failures[:2]}",
        )
        report.check(
            "M3",
            full_conc,
            context="full_hit",
            detail=(
                f"concentration on full-hit seed engine={full_conc:.2f} "
                f"(vs zero-hit baseline ~0.5)"
            ),
        )
        report.check(
            "M3",
            half_conc,
            context="half_hit",
            detail=(
                f"concentration on half-hit seed engine={half_conc:.2f} "
                f"(50% hit still clears the affinity threshold)"
            ),
        )
        report.invariant(
            "P2",
            zero_engines >= 2,
            context="zero_hit",
            detail=(
                f"zero-hit spread: engines={zero_engines}, "
                f"max_share={zero_max:.2f} (observational, expected ~0.5-0.7)"
            ),
        )
        return report.finish(
            f"full_conc={full_conc:.2f}, half_conc={half_conc:.2f}, "
            f"zero_dist={json.dumps(dict(zero_dist), sort_keys=True)}, "
            f"grades: {report.summary()}"
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
