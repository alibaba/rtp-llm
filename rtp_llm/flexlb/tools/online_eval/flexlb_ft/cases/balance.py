"""Balance-category cases: scheduling result properties.

Theme: RESULT-PROPERTY cases (task #61 rework, superseding
scheduling_smoke.py S1-S12) — they assert observable outcome properties
(P-series), not mechanism narratives; every measured property is graded
against the central band table (grade.GRADE_BANDS) — strict=优异 /
normal=良好 / loose 地板（超出即不可用）— and each case returns its
achieved grade for the run-level verdict (all strict=优异 / all
≥normal=良好 / any beyond loose=不可用).  Hard invariants (P2
no-starvation, P6 completeness) carry no band: violation is unusable at
every grade.

Case map (task #61/#62 disposition):

  balance_uniform_serial        <- S1+S6+S8 merged (P1+P2, two variants:
                                    plain / speed-heterogeneous injection)
  balance_concurrent_mix        <- S7 strengthened (P1 relaxed + P2 + P6)
  balance_overload_avoid_prefill<- S4 + new P7 short-request protection
                                    (P5 graded + P6 + P7 dual-caliber)
  balance_overload_avoid_decode <- S11 strengthened (P5 delta-caliber graded
                                    + P6 + takeover assertions)
  balance_decode_spread         <- S3+S10 merged (P2+P1, n=10/50 two tiers)
  balance_len_mixed             <- L1 bimodal length mix (P3 token-share
                                    graded — first P3 calibration + P2 short
                                    request spread + P6)

Result properties are asserted profile-agnostically: all balance cases
run under every profile.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ..context import CaseContext, CaseDef, rid_base
from ..grade import GradeReport
from ..harness import TTL_DRAIN_TIMEOUT_S, AssertUtils

BALANCE_CASES: list[CaseDef] = []

STREAM_TIMEOUT_S = 15.0


def case(name: str, profiles=None, requires=None, source: str = ""):
    def deco(fn):
        BALANCE_CASES.append(
            CaseDef(
                name=name,
                category="balance",
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


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"]


def _decode_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "decode"]


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


# ===========================================================================
# Balance cases (result-property graded — task #61/#62 rework of
# scheduling_smoke.py S1-S12; rid_base family "scheduling" -> "balance"
# in the task #85 category reorg)
# ===========================================================================


@case(
    "balance_uniform_serial",
    source="scheduling_smoke.py S1+S6+S8 (merged, task #61)",
)
def balance_uniform_serial(ctx: CaseContext):
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
                rid = ops.next_request_id(rid_base(ctx, "balance"))
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
    "balance_concurrent_mix",
    source="scheduling_smoke.py S7 (strengthened, task #61)",
)
def balance_concurrent_mix(ctx: CaseContext):
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
    (harness default 4) x 2 prefill workers = 8 concurrent batch
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
    base = rid_base(ctx, "balance")
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
    "balance_overload_avoid_prefill",
    source="scheduling_smoke.py S4 + short-request protection (task #61)",
)
def balance_overload_avoid_prefill(ctx: CaseContext):
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
        FetchResponse message only after decode completes (the cancel T3
        lesson), so FetchResponse "TTFT" cannot observe the prefill phase
        at all; the completion-duration caliber carries the same protection
        signal (a request swallowed by the hot engine pays its ~5s prefill
        either way).

    Drainage (inherited S4 lesson, kept in finally): the seed is
    fire-and-forget, so every fired request is consumed to terminal state
    (cancel as fallback) — otherwise the seed's ledger prediction keeps one
    engine's wait high for the rest of the suite and poisons later balance
    cases.
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "balance")
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
            # Best-effort residue drain (task #87): a drain-fallback cancel
            # that fails leaves slots settling on the stale-TTL +
            # ExpirationTimer path (worst ~90s) — the legacy 30s window
            # stopped short of it and the residue poisoned later cases on
            # this shared env.  Still not asserted (this finally is
            # hygiene, the case's own contract lives in its verdict).
            AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)
        except Exception:
            pass


@case(
    "balance_overload_avoid_decode",
    source="scheduling_smoke.py S11 (strengthened, task #61)",
)
def balance_overload_avoid_decode(ctx: CaseContext):
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
            rid = ops.next_request_id(rid_base(ctx, "balance"))
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
    "balance_decode_spread",
    source="scheduling_smoke.py S3+S10 (merged, task #61)",
)
def balance_decode_spread(ctx: CaseContext):
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
                rid = ops.next_request_id(rid_base(ctx, "balance"))
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
    "balance_len_mixed",
    source="length-heterogeneity dimension L1 (task #62)",
)
def balance_len_mixed(ctx: CaseContext):
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
    base = rid_base(ctx, "balance")
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
            # Best-effort residue drain with the TTL-aware window (task
            # #87 — same rationale as balance_overload_avoid_prefill: a
            # drain-fallback cancel that fails settles on the stale-TTL +
            # ExpirationTimer path, worst ~90s).
            AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)
        except Exception:
            pass
