"""Routing tests for admission, queue completion and token-sensitive Prefill placement.

BEST_ONLY does not promise uniform request counts across equal-cost Prefill
workers. Distribution assertions are used only with a constructed cost or
admission difference."""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ..context import CaseContext, CaseDef, rid_base
from ..grade import GradeReport
from ..harness import REQUEST_CLEANUP_TIMEOUT_S, AssertUtils

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
    "balance_concurrent_mix",
    source="scheduling_smoke.py S7 (strengthened, task #61)",
)
def balance_concurrent_mix(ctx: CaseContext):
    """All requests in a bounded concurrent burst must complete.

    Batch credits defer QUEUE delivery rather than rejecting publication. The
    landing distribution is diagnostic; equal-cost Prefill ties need not be random."""
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
        dist_json = json.dumps(dict(dist), sort_keys=True)

        report.invariant(
            "P6", not failures and n_ok == len(rids),
            detail=f"completed={n_ok}/{len(rids)}, failures={failures[:2]}, dist={dist_json}",
        )

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
         consume-and-fire wave could outlive the ledger and erase the
         cost difference mid-wave.

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
        #    selects the lower-TTFT cool worker).
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
            AssertUtils.inflight_clean(_master_http(ops), REQUEST_CLEANUP_TIMEOUT_S)
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


@case("balance_decode_rotation", source="Decode rotation over settled admissible workers")
def balance_decode_rotation(ctx: CaseContext):
    """Settled, healthy Decode workers all participate in repeated routing."""
    ops = ctx.ops()
    names = _decode_names(ops)
    if len(names) < 2:
        return False, "need at least two Decode workers"
    try:
        clean, detail = AssertUtils.inflight_clean(_master_http(ops), 15.0)
        if not clean:
            return False, f"fixture has outstanding work: {detail}"
        before = ops.snapshot_by_name()
        baseline = {name: before[name].get("completed", 0) for name in names}
        count = 2 * len(names)
        for _ in range(count):
            rid = ops.next_request_id(rid_base(ctx, "balance"))
            _, error = ops.run_one_request(
                rid, output_len=2, block_keys=[rid * 100],
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if error:
                return False, f"rid={rid}: {error}"
            clean, detail = AssertUtils.inflight_clean(_master_http(ops), 15.0)
            if not clean:
                return False, f"rid={rid} did not settle: {detail}"
        after = ops.snapshot_by_name()
        deltas = {name: after[name].get("completed", 0) - baseline[name] for name in names}
        return all(value > 0 for value in deltas.values()) and sum(deltas.values()) == count, (
            f"Decode completions={deltas}, expected {count} total and every healthy worker used"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "balance_len_mixed",
    source="length-heterogeneity dimension L1 (task #62)",
)
def balance_len_mixed(ctx: CaseContext):
    """Two long requests per wave must select different Prefill workers.

    Wait until the first long request is visible before submitting the second;
    the empty worker then has lower predicted TTFT. Submit six short requests
    while both long requests execute and require every request to complete.
    Token share is measured across five waves; short-request counts need not
    be uniform between equal-cost workers.
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
            #    the lower-TTFT empty worker must receive L_b.
            rid = ops.next_request_id(base)
            name_b, err = _fire_request(
                ops, rid, fired, fired_handles, input_len=lb, output_len=2
            )
            if err:
                failure = f"wave{wave} L_b: {err}"
                break
            if name_b == name_a:
                failure = f"wave{wave}: second long request ignored empty worker"
                break
            landed.append((rid, name_b, lb))
            if not _poll_engine_pending(ops, name_b, 1):
                failure = f"wave{wave} L_b never appeared on {name_b}"
                break
            # 4. 6 shorts while BOTH longs are in flight: X has decayed only
            #    by the polls' overhead (tens of ms, inside the ~190ms tie
            #    window of the ~1.9s ledgers), so the shorts spread evenly —
            #    the wave stays
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
            AssertUtils.inflight_clean(_master_http(ops), REQUEST_CLEANUP_TIMEOUT_S)
        except Exception:
            pass
