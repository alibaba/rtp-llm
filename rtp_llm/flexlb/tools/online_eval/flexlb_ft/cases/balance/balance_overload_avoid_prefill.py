from __future__ import annotations

import json
import time
from collections import Counter

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import TTL_DRAIN_TIMEOUT_S, AssertUtils
from ...registry import case
from ...support.balance import STREAM_TIMEOUT_S, _master_http, _prefill_names


@case(
    "balance_overload_avoid_prefill",
    category="balance",
    source="scheduling_smoke.py S4 + short-request protection",
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
            # Best-effort residue drain (integration-round cascade hygiene): a drain-fallback cancel
            # that fails leaves slots settling on the stale-TTL +
            # ExpirationTimer path (worst ~90s) — the legacy 30s window
            # stopped short of it and the residue poisoned later cases on
            # this shared env.  Still not asserted (this finally is
            # hygiene, the case's own contract lives in its verdict).
            AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)
        except Exception:
            pass
