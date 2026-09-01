"""Engine-fault-category cases: the engine as a fault victim.

Theme: prefill/decode engines dying, flapping, crashing mid-ack, erroring
on enqueue, or stalling (no-respond / deferred enqueue / inflated
execution) — the master must observe the loss, keep serving through the
survivors, keep its ledger bounded, and fully re-converge once the engine
returns.  The category covers both process-level faults (stop/start
oscillation, crash-after) and the /inject fault family (no_respond,
enqueue_error, enqueue_delay, generate_delay).

Case map:

    engine_fault_down_phases      five-phase engine-down assertion set
                                   (S2/S4 merged: master up + takeover +
                                   recovery, TTFT regression gate)
    engine_fault_flap             rapid stop/start oscillation vs the
                                   3-strike eviction / re-discovery race
    engine_fault_crash_after      enqueue-count triggered stop + the
                                   empty-ack uncertain fence contract
    engine_fault_no_respond       prefills stop answering; the failed
                                   request surfaces, env recovers
    engine_fault_enqueue_error    prefills error every enqueue; the failed
                                   request surfaces, env recovers
    engine_fault_enqueue_delay    deferred enqueue ack inflates schedule
                                   latency (delay, not failure)
    engine_fault_generate_delay   inflated prefill execution inflates TTFT
                                   under every profile
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..context import CaseContext, CaseDef, rid_base
from ..engine_ops import (
    _fence_residue_stable,
    clear_type_all,
    engine_inflight_clean,
    inject_type,
    inject_type_all,
)
from ..harness import (
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    _BackgroundFlow,
    _cleanup_dynamic,
    _elastic_env,
    _fault_spec,
    _run_batch,
    _ttft_p50,
    _wait_master_topology,
    http_get_status,
    wait_for,
)

ENGINE_FAULT_CASES: list[CaseDef] = []

STREAM_TIMEOUT_S = 15.0
# 3-strike health demotion + eviction window (fault-family precedent).
MASTER_EVICT_S = 30.0
# Engine restart channel-reconnect settle window.
ENGINE_RECOVERY_WAIT_S = 3.0
# Anomaly-family timing knobs (anomaly_smoke.py E2/E3 calibration).
TIMEOUT_WAIT_S = 5.0
ANOMALY_STREAM_TIMEOUT_S = 10.0
WORKER_RECOVERY_WAIT_S = 3.0


def case(name: str, profiles=None, requires=None, source: str = ""):
    def deco(fn):
        ENGINE_FAULT_CASES.append(
            CaseDef(
                name=name,
                category="engine_fault",
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


def _measure_ttft(
    ops, rid: int, timeout_s: float = 12.0
) -> tuple[float, Optional[str], bool]:
    """schedule + stream one request, measuring first-output latency.

    Returns (ttft, error, enqueued_by_master) — the flag lets callers
    branch master-inflight checks on the actual delivery mode instead of
    inferring it from the profile."""
    response = ops.schedule(rid)
    if response.code != 200 or not response.success:
        return -1.0, f"schedule failed: {response.error_message}", False
    input_pb = None if response.enqueued_by_master else ops.build_generate_input(rid)
    handle = ops.start_stream(response, rid, input_pb=input_pb)
    t0 = time.monotonic()
    got_first = handle.wait_first_output(timeout_s)
    ttft = time.monotonic() - t0
    if not got_first:
        handle.cancel()
        return -1.0, "no first output", response.enqueued_by_master
    ended = handle.wait_end(timeout_s)
    if not ended or handle.snap.error:
        handle.cancel()
        return (
            ttft,
            f"stream error after first output: {handle.snap.error}",
            (response.enqueued_by_master),
        )
    return ttft, None, response.enqueued_by_master


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


def _anomaly_error_case(
    ctx: CaseContext, inject_config: dict, wait_s: float, require_error_detail: bool
) -> tuple[bool, str]:
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "engine_fault"))
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
            # NON_BATCH: see cancel_anomaly_path — client Cancel cannot
            # safely release a delivered ledger entry, so immediate-zero is
            # not asserted.
            inflight_ok, inflight_detail = True, "N/A (NON_BATCH residue contract)"
        passed = error_observed and recovery_ok
        return passed, (
            f"error_observed={error_observed} ({error_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


# ===========================================================================
# Process-level engine faults
# ===========================================================================


@case(
    "engine_fault_down_phases",
    profiles=["batch-window"],  # _elastic_env pins the legacy fault axes
    source="flexlb_behavior_test.sh S2/S4 merged — five-phase engine-down assertion set",
)
def engine_down_http_stop_prefill(ctx: CaseContext):
    """Five phases with the uniform engine-down assertion set (core 3 of 7):
    master stays up + surviving engines take over + recovery rate.
    """
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "engine_fault")
    try:
        _cleanup_dynamic(ops, env)

        # Integration-round cascade hygiene (task #87): residues from the
        # preceding elastic cases on this shared env settle via the
        # stale-TTL + ExpirationTimer path (worst ~90s).  Drain them
        # BEFORE the baseline batch so an earlier case's TTL settle cannot
        # fail this case's Phase-1 gate (best-effort — a true leak is
        # caught by the case's own end-of-run drain assertion below).
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)

        def master_up() -> bool:
            return (
                http_get_status(
                    f"{_master_http(ops)}/rtp_llm/inflight_status", timeout=5
                )
                == 200
            )

        # Phase 1 — baseline: 20 requests, all succeed.  TTFT p50 is
        # recorded for the post-recovery regression gate (Phase 5).
        ok1, err1, _ = _run_batch(ops, base, 20, collect_ttft=True)
        base_ttft_p50 = _ttft_p50(getattr(_run_batch, "last_ttfts", []))
        master_ok1 = master_up()
        if err1:
            return False, f"baseline had {err1} errors (master_up={master_ok1})"

        # Phase 2 — http-stop prefill-0.
        ops.stop_engine("prefill-0")
        # Phase 3 — downtime: wait for the 3-consecutive-failure eviction,
        # then 20 more requests must still succeed (2P redundancy).
        evicted = wait_for(
            lambda: ops.master_alive_count("PREFILL") <= 1,
            MASTER_EVICT_S,
            0.5,
        )
        ok2, err2, _ = _run_batch(ops, base, 20)
        master_ok2 = master_up()
        rate2 = ok2 / 20 if err2 == 0 else ok2 / 20
        takeover_ok = err2 <= 2  # ≥90% success while one prefill is down
        downtime_err_types = list(getattr(_run_batch, "last_error_types", []))[:3]

        # Phase 4 — restart the engine and wait for re-discovery.
        ops.start_engine("prefill-0")
        alive_back = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= 2,
            MASTER_EVICT_S,
            0.5,
        )
        # Channel recovery settle (S2's reconnect window).
        time.sleep(3.0)

        # Phase 5 — recovery: 20 requests ≥95%, and TTFT must fall back to
        # within 1.5x of the baseline p50 (master_recovery_ttft_test.sh
        # semantics: once the fault heals, TTFT returns to baseline — the
        # legacy analyzer tolerates an early 1.5x spike and only degrades
        # the verdict when the *stable* window stays above 1.2x; this batch
        # sits past the 3s channel-settle window, so the 1.5x gate is the
        # conservative bound on steady-state recovery).
        ok5, err5, _ = _run_batch(ops, base, 20, collect_ttft=True)
        recovery_ttft_p50 = _ttft_p50(getattr(_run_batch, "last_ttfts", []))
        ttft_ok, ttft_detail = AssertUtils.ttft_degradation(
            base_ttft_p50, recovery_ttft_p50, threshold_pct=50.0
        )
        master_ok5 = master_up()
        recovery_ok = ok5 >= 19  # ≥95%

        # Drain guard (task #87): the tolerated Phase-3 failures (err2<=2,
        # requests routed onto the stopped engine) settle through the
        # TTL path — wait for the worst-case window so this case does not
        # leak its residue into engine_fault_flap/master_kill on the same
        # env; a slot that never settles still FAILs this case here.
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )

        passed = (
            master_ok1
            and master_ok2
            and master_ok5
            and evicted
            and takeover_ok
            and alive_back
            and recovery_ok
            and ttft_ok
            and inflight_ok
        )
        return passed, (
            f"baseline=20/20, evicted_after_stop={evicted}"
            f"(alive={ops.master_alive_count('PREFILL')}), "
            f"downtime={ok2}/20({rate2:.0%}, err_types={downtime_err_types}), "
            f"alive_restored={alive_back}, "
            f"recovery={ok5}/20({ok5 / 20:.0%}), "
            f"ttft_recovery=[{ttft_detail}], "
            f"master_up=(p1:{master_ok1}, p3:{master_ok2}, p5:{master_ok5}), "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            snap = ops.snapshot_by_name()
            if snap.get("prefill-0", {}).get("stopped"):
                ops.start_engine("prefill-0")
        except Exception:
            pass


@case(
    "engine_fault_flap",
    profiles=["batch-window"],  # _elastic_env pins the legacy fault axes
    source="gap G2: rapid /stop_engine+/start_engine oscillation, 3-strike eviction vs re-discovery race",
)
def engine_flap(ctx: CaseContext):
    """Connection flapping: >=5 rapid stop/start cycles on one prefill.

    Exercises the race window between the master's 3-strike health eviction
    and the engine's re-discovery: each cycle stops prefill-0, holds it down
    long enough for the health poller (20ms interval) to accumulate strikes,
    then brings it back WITHOUT waiting for convergence (the flap).  A
    background flow keeps traffic live throughout.

    Assertions (user-mandated):
      * master stays healthy the whole time — HTTP 200 probe every cycle,
        no hang;
      * after the flapping stops: the engine is re-discovered
        (discovered == alive == initial topology), routing and requests
        recover (>=95% batch), and no inflight leaks (global drain to zero
        within the TTL_DRAIN_TIMEOUT_S cap that covers the 30s
        stale-inflight TTL plus the 60s ExpirationTimer sweep).

    The per-cycle alive count is observational evidence of the eviction vs
    re-discovery race: dipping below 2 means the 3-strike demotion landed,
    staying at 2 means recovery won the race — both are correct; the
    contract is only about final convergence and no leak.
    """
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "engine_fault")
    flow: Optional[_BackgroundFlow] = None
    try:
        _cleanup_dynamic(ops, env)

        flow = _BackgroundFlow(ops, base, interval_s=0.2)
        flow.start()
        time.sleep(1.0)  # let the flow ramp up before the first stop

        cycles = 6
        cycle_log: list[str] = []
        master_200_all = True
        evict_landings = 0
        for i in range(1, cycles + 1):
            ops.stop_engine("prefill-0")
            # Hold down ~0.8s: the health poller runs every 20ms, so the
            # 3-strike counter fires well inside this window, while the
            # EngineSyncRunner endpoint-eviction threshold (max(3*20ms, 1s)
            # from the last successful status) lands at the tail of the
            # window or right after the restart — exactly the race under
            # test.
            time.sleep(0.8)
            alive_mid = ops.master_alive_count("PREFILL")
            if alive_mid < 2:
                evict_landings += 1
            probe = http_get_status(
                f"{_master_http(ops)}/rtp_llm/inflight_status", timeout=5
            )
            if probe != 200:
                master_200_all = False
            ops.start_engine("prefill-0")
            time.sleep(0.4)  # short gap — flap, no convergence wait
            cycle_log.append(f"c{i}[alive={alive_mid}, master={probe}]")

        total, ok = flow.stop()
        rate = ok / total if total else 0.0

        # Post-flap convergence: full re-discovery of the flapped engine
        # (discovered count covers the eviction side of the race — see
        # elastic_rebalance for why alive alone is not a safe signal).
        topology_ok = _wait_master_topology(
            ops, "PREFILL", env.spec.n_prefill, MASTER_EVICT_S
        )
        # Routing/request recovery: 20 requests, >=95%.
        ok_batch, _, _ = _run_batch(ops, base, 20)
        recovery_ok = ok_batch >= 19
        # No inflight leak: global drain to zero (covers the 30s TTL plus
        # the 60s ExpirationTimer sweep — task #87: the legacy 90s cap sat
        # below the worst-phase settle and let residue poison the next
        # case on this shared env).
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )

        passed = (
            master_200_all
            and topology_ok
            and recovery_ok
            and inflight_ok
            and total > 0
            and rate >= 0.5  # availability floor — a total blackout must fail
        )
        return passed, (
            f"cycles={cycles}, evictions_landed={evict_landings}/{cycles}, "
            f"flap=[{'; '.join(cycle_log)}], "
            f"flow_success={ok}/{total}({rate:.0%}), "
            f"topology_converged={topology_ok}, "
            f"post_flap_batch={ok_batch}/20, "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if flow is not None:
            flow.stop()
        try:
            snap = ops.snapshot_by_name()
            if snap.get("prefill-0", {}).get("stopped"):
                ops.start_engine("prefill-0")
        except Exception:
            pass


@case(
    "engine_fault_crash_after",
    profiles=["batch-window"],
    source="gap G6/G7: /inject type=crash_after (enqueue-count triggered stop, not process death)",
)
def inject_crash_after(ctx: CaseContext):
    """crash_after n=1 flips the landing engine to stopped at its first
    enqueue and answers that enqueue with an EMPTY ack (no errors, no
    successes).  Single-JVM semantics: a stopped flag, not a dead process
    — getWorkerStatus keeps answering alive=false, so the master marks it
    down and routes around it; /start_engine clears the fault config,
    resets the enqueue counter and rebinds the port.

    Assertions: exactly one engine crashes, the master observes the loss
    (alive drops), traffic is served by the surviving engine (>=60% of a
    5-request burst — the engine_down err2 <= 2 tolerance: the alive drop
    and the routable-set update are not one atomic step), and after
    /start_engine the topology fully recovers.  Master-ledger residue from the empty-ack
    batch(s) is asserted to be BOUNDED and NON-GROWING, not fully clean:
    with the production UnsupportedEngineCancelChannel the uncertain-entry
    engine fence parks in quarantine forever (no cancel channel, and the
    engine never saw the request so no WorkerStatus terminal ever
    settles it) — verified in PriorityScheduler.handleEngineFenceOutcome /
    cleanupInflight.  The engine side, which never registered the request,
    must be fully clean.

    Profile semantics (v2, task #55): the fault fires at the engine's
    EnqueueBatch entry (BATCH dispatcher only) and _fault_spec pins the
    legacy fault axes (PRIORITY + FIXED_WINDOW + BATCH) via FLEXLB_CONFIG,
    so the declaration stays batch-window — re-running under another
    --profile would execute the identical configuration.
    """
    ops = ctx.engine_ops(ctx.env_manager.ensure(_fault_spec(ctx)))
    base = rid_base(ctx, "engine_fault")
    names = _prefill_names(ops)
    if len(names) < 2:
        return False, "need >=2 prefill engines"
    try:
        inject_type_all(ops, names, "crash_after", n=1)

        # R1 triggers the crash wherever it lands; its own fate is the
        # uncertain-reconcile path (empty ack) — reported, not asserted.
        rid1 = ops.next_request_id(base)
        _, err1 = ops.run_one_request(rid1, stream_timeout_s=12.0)

        snap = ops.snapshot_by_name()
        stopped = [n for n in names if snap.get(n, {}).get("stopped")]
        # Disarm the fault on the engines that did NOT fire: with n=1 the
        # NEXT request landing on a still-armed engine would crash it too
        # (first-round evidence: takeover went 0/5 with both prefills
        # stopped and the master rejecting "Worker scheduling queue").
        for n in names:
            if n not in stopped:
                inject_type(ops, n, "crash_after", enabled=False)

        alive_dropped = wait_for(
            lambda: ops.master_alive_count("PREFILL") <= len(names) - 1,
            MASTER_EVICT_S,
            0.5,
        )
        # Settle window: the alive drop and the router's routable-set update
        # are separate steps (engine_down precedent); a burst fired on the
        # boundary can still be admitted towards the stopped engine and fail
        # through "Worker scheduling queue rejected" (observed round 3:
        # takeover 3/5 with exactly that error plus a transient admission
        # capacity rejection — both master-side scheduling states, not a
        # surviving-engine service failure).
        time.sleep(2.0)

        # Takeover: a 5-request burst on the surviving engine (>=60%, the
        # engine_down tolerance err2 <= 2: the alive drop precedes the
        # routable-set update, so the first request(s) may still hit the
        # stopped engine and fail through the empty-ack uncertain path).
        takeover_rids = [ops.next_request_id(base) for _ in range(5)]

        def run(rid: int):
            return ops.run_one_request(rid, stream_timeout_s=12.0)[1]

        with ThreadPoolExecutor(max_workers=5) as pool:
            takeover_errs = list(pool.map(run, takeover_rids))
        takeover_ok = sum(1 for e in takeover_errs if e is None)
        takeover_types = sorted({str(e)[:60] for e in takeover_errs if e})[:3]

        # Restart the crashed engine: clears fault config + enqueue counter.
        for n in stopped:
            ops.start_engine(n)
        alive_back = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= len(names),
            MASTER_EVICT_S,
            0.5,
        )
        time.sleep(ENGINE_RECOVERY_WAIT_S)  # channel reconnect settle

        rid3 = ops.next_request_id(base)
        _, err3 = ops.run_one_request(rid3, stream_timeout_s=12.0)

        # Bounded residue: R1's uncertain entry always parks; each FAILED
        # takeover request (routed to the stopped engine before the
        # routable-set caught up) adds at most one more empty-ack entry.
        failed_takeover = 5 - takeover_ok
        residue_ok, residue_detail = _fence_residue_stable(ops, 1 + failed_takeover)
        engine_clean, engine_detail = engine_inflight_clean(ops, names)

        passed = (
            len(stopped) == 1
            and alive_dropped
            and takeover_ok >= 3
            and alive_back
            and err3 is None
            and residue_ok
            and engine_clean
        )
        return passed, (
            f"crashed={stopped}, "
            f"r1_fate={'error: ' + str(err1)[:60] if err1 else 'ok'}, "
            f"master_saw_loss={alive_dropped}, "
            f"takeover={takeover_ok}/5, err_types={takeover_types}, "
            f"alive_restored={alive_back}, "
            f"after_restart={err3 is None}"
            f"{'' if err3 is None else ' err=' + str(err3)[:60]}, "
            f"master_fence_residue={residue_ok}({residue_detail}), "
            f"engine_inflight_clean={engine_clean}({engine_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            snap = ops.snapshot_by_name()
            for n in names:
                if snap.get(n, {}).get("stopped"):
                    ops.start_engine(n)
                else:
                    inject_type(ops, n, "crash_after", enabled=False)
        except Exception:
            pass


# ===========================================================================
# Injected engine faults: error / stall shapes (anomaly E2/E3 + delays)
# ===========================================================================


@case("engine_fault_no_respond", source="anomaly_smoke.py E2")
def e2_timeout(ctx: CaseContext):
    return _anomaly_error_case(ctx, {"no_respond": True}, TIMEOUT_WAIT_S, False)


@case("engine_fault_enqueue_error", source="anomaly_smoke.py E3")
def e3_worker_fail(ctx: CaseContext):
    return _anomaly_error_case(
        ctx, {"enqueue_error": True}, ANOMALY_STREAM_TIMEOUT_S, True
    )


@case(
    "engine_fault_enqueue_delay",
    requires=["enqueue_batch"],
    source="gap G6/G7: /inject type=enqueue_delay (deferred enqueue ack, BATCH dispatch)",
)
def inject_enqueue_delay(ctx: CaseContext):
    """enqueue_delay defers the whole enqueue runnable (admission + ack) by
    delay_ms, so the BATCH-dispatch schedule() — which waits for the enqueue
    ack — grows by roughly delay_ms.  delay_ms must stay well below
    dispatcher.enqueueRpcTimeoutMs (default 5000) or the RPC deadline
    fires first.

    Assertions: end-to-end latency delta >= 1.2s at delay_ms=1500, the
    request still SUCCEEDS (delay, not failure), and latency recovers once
    the injection is cleared.

    Profile semantics (v2, task #55): the deferred runnable is the
    engine's EnqueueBatch processing, which exists only under the BATCH
    dispatcher — requires=["enqueue_batch"] keeps the case to the
    BATCH-dispatch profiles (batch-window, single-batch).  Unlike the
    fault_spec cases this one runs on the shared smoke env
    (real per-profile config), so single-batch exercises the SINGLE
    decision axis on the same enqueue path.
    """
    ops = ctx.ops()
    base = rid_base(ctx, "engine_fault")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        rid0 = ops.next_request_id(base)
        t0 = time.monotonic()
        _, err0 = ops.run_one_request(rid0, stream_timeout_s=STREAM_TIMEOUT_S)
        baseline_total = time.monotonic() - t0
        if err0:
            return False, f"baseline request failed: {err0}"

        inject_type_all(ops, names, "enqueue_delay", delay_ms=1500)
        rid1 = ops.next_request_id(base)
        t1 = time.monotonic()
        _, err1 = ops.run_one_request(rid1, stream_timeout_s=STREAM_TIMEOUT_S)
        delayed_total = time.monotonic() - t1

        clear_type_all(ops, names, "enqueue_delay")
        rid2 = ops.next_request_id(base)
        t2 = time.monotonic()
        _, err2 = ops.run_one_request(rid2, stream_timeout_s=STREAM_TIMEOUT_S)
        recovered_total = time.monotonic() - t2

        delta = delayed_total - baseline_total
        # Integration-round cascade hygiene (task #87): residue from
        # earlier cases on this shared env settles via the stale-TTL +
        # ExpirationTimer path (worst ~90s); drain it BEFORE the clean
        # assertion so another case's TTL settle cannot fail this one.
        # Best-effort on purpose — a true leak never drains and the 10s
        # assertion below still catches it.
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 10.0
        )

        passed = (
            err1 is None
            and err2 is None
            and delta >= 1.2
            and recovered_total <= baseline_total + 1.0
            and inflight_ok
        )
        return passed, (
            f"baseline={baseline_total:.2f}s, delayed={delayed_total:.2f}s "
            f"(delta={delta:.2f}s >= 1.2), recovered={recovered_total:.2f}s, "
            f"delayed_ok={err1 is None}, recovered_ok={err2 is None}, "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "enqueue_delay")


@case(
    "engine_fault_generate_delay",
    source="gap G6/G7: /inject type=generate_delay (prefill execution inflation, all profiles)",
)
def inject_generate_delay(ctx: CaseContext):
    """generate_delay adds delay_ms to the prefill execution estimate
    (runPrefillBatch), so the first-output latency grows by roughly
    delay_ms under EVERY profile (unlike enqueue_delay, schedule() stays
    fast and only TTFT inflates).

    Assertions: TTFT delta >= 1.2s at delay_ms=1500, the request still
    SUCCEEDS, and TTFT recovers after the injection is cleared."""
    ops = ctx.ops()
    base = rid_base(ctx, "engine_fault")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        rid0 = ops.next_request_id(base)
        ttft_base, err0, enq0 = _measure_ttft(ops, rid0)
        if err0:
            return False, f"baseline request failed: {err0}"

        inject_type_all(ops, names, "generate_delay", delay_ms=1500)
        rid1 = ops.next_request_id(base)
        ttft_delayed, err1, _ = _measure_ttft(ops, rid1)

        clear_type_all(ops, names, "generate_delay")
        rid2 = ops.next_request_id(base)
        ttft_recovered, err2, _ = _measure_ttft(ops, rid2)

        delta = ttft_delayed - ttft_base
        if enq0:
            # Integration-round cascade hygiene (task #87): drain earlier
            # cases' TTL-settling residue before the clean assertion (see
            # inject_enqueue_delay) — best-effort, the 10s assertion below
            # keeps the real leak detection.
            AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A (non-batch path)"

        passed = (
            err1 is None
            and err2 is None
            and delta >= 1.2
            and ttft_recovered <= ttft_base + 1.0
            and inflight_ok
        )
        return passed, (
            f"ttft_baseline={ttft_base:.2f}s, ttft_delayed={ttft_delayed:.2f}s "
            f"(delta={delta:.2f}s >= 1.2), ttft_recovered={ttft_recovered:.2f}s, "
            f"delayed_ok={err1 is None}, recovered_ok={err2 is None}, "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "generate_delay")
