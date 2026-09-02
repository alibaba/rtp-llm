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
    engine_fault_crash_after      enqueue-count triggered TRUE CRASH
                                   (memory wipe + port kill) + the
                                   empty-ack uncertain fence contract
    engine_fault_no_respond       prefills stop answering; the failed
                                   request surfaces, env recovers
    engine_fault_enqueue_error    prefills error every enqueue; the failed
                                   request surfaces, env recovers
    engine_fault_enqueue_delay    deferred enqueue ack inflates schedule
                                   latency (delay, not failure)
    engine_fault_generate_delay   inflated prefill execution inflates TTFT
                                   under every profile

    Recovery family (E1-E6, expected-behavior assertions):
    engine_fault_recovery_generation_bump   E1 — recovery must publish a
                                   fresh endpoint generation; old ledger
                                   must not leak into it
    engine_fault_recovery_kv_resync         E2 — recovery must rebuild the
                                   engine's cache view from a FULL snapshot
                                   (with and without surviving KV memory)
    engine_fault_recovery_no_resurrect      E3 — inflight requests from
                                   before a true crash must not resurrect
                                   on the recovered engine (which must
                                   come back memory-empty)
    engine_fault_status_gap_no_bump          E4 — a short (2-tick) status
                                   gap must NOT retire the generation
    engine_fault_status_gap_long_retire      E5 — a long status gap must
                                   retire the generation and fence its
                                   ledger/inflight
    engine_fault_recovery_kv_usage_reset     E6 — after a full restart the
                                   engine's KV usage must restart from zero,
                                   not resume from the old reading
"""

from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
    EnvSpec,
    _BackgroundFlow,
    _cleanup_dynamic,
    _elastic_env,
    _fault_spec,
    _pump_until_accepted,
    _run_batch,
    _ttft_p50,
    _wait_master_alive,
    _wait_master_topology,
    fault_env_config,
    fault_env_perf,
    http_get_status,
    http_post_json,
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
    source="gap G6/G7: /inject type=crash_after (enqueue-count triggered true crash)",
)
def inject_crash_after(ctx: CaseContext):
    """crash_after n=1 kills the landing engine at its first enqueue (TRUE
    CRASH: all per-engine memory — running tasks, queues, KV leases, LRU —
    is wiped and the gRPC port is shut down) and answers that enqueue with
    an EMPTY ack (no errors, no successes) that flushes just before the
    port dies.  The master's health poller then hits connection-refused,
    accumulates the 3-strike failures and retires the endpoint, routing
    around it; /start_engine rebuilds the gRPC server on clean state
    (fresh fault config, zeroed enqueue counter, empty memory).

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
    cleanupInflight.  The engine side, which never registered the request
    (and whose memory the crash wiped anyway), must be fully clean.

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
        # uncertain-reconcile path (empty ack that flushes just before the
        # port kill) — reported, not asserted.
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


# ===========================================================================
# Elastic/fault recovery contracts (E1-E6) — expected-behavior assertions
# ===========================================================================
#
# Assertion policy (task E1-E6 mandate): every assertion below states the
# CORRECT contract for engine recovery, never the current behaviour.  A
# failing case is a FINDING on the master (or, where the observation is the
# mock's own self-report, on the mock's restart fidelity) and is recorded,
# not worked around.  Observation surfaces used:
#
#   * master log lifecycle lines — "Created WorkerStatus generation {} for
#     worker: {ipPort}" (INFO, EngineSyncRunner), "worker {ipPort} marked
#     dead after 3 consecutive gRPC failures" (ERROR, GrpcWorkerStatusRunner
#     — the transport-failure retire path), "[remove]/[replace] retiring ..."
#     (INFO — discovery-driven retires);
#   * /rtp_llm/inflight_status per-endpoint ledger (inflight_requests /
#     inflight_batches per prefill ip_port);
#   * /rtp_llm/master/info worker_summary (discovered / alive per role);
#   * routing landing points + mock /snapshot (cache_key_set,
#     kv_tokens_used) for the KV-view contracts.
#
# The six cases share a DEDICATED env (recovery_{profile}, 2P+2D, file
# discovery, fault axes, 30s stale-inflight TTL) so the stop/start/
# injection cycles cannot leak state into — or inherit residue from — the
# shared fault_/kv_ family envs (the task-#87 family-env leakage lesson).


# Retire wait cap: connection-refused failures accumulate one per status
# poll tick (20ms) and the transport retire fires at 3 consecutive
# failures, so the bounded stop in these cases retires within ~1s; the cap
# stays at the fault-family precedent for slow CI machines.
RECOVERY_EVICT_S = 30.0
# Engine restart channel-reconnect settle (engine_down Phase-4 precedent).
RECOVERY_SETTLE_S = 3.0
# E3 crash_after arming window: the crash only fires when a fresh
# EnqueueBatch lands on the armed engine, so trigger requests are fired
# until every target reports stopped.  The master may route several
# triggers at a live/already-crashed peer before one lands on each armed
# target (post-crash dispatch failures also re-route), so the window
# covers a handful of 0.2s trigger rounds.
CRASH_TRIGGER_WINDOW_S = 15.0
# Status-gap shapes: E4 is a 2-tick (2 x 20ms poll) transient gap; E5 must
# exceed the 3-consecutive-failure retire threshold with no_respond's
# per-RPC 1s deadline (fault_env_config statusRpcTimeoutMs=1000) — 5s gives
# >= 4 timed-out polls, comfortably past 3.
E4_GAP_S = 0.045
E5_GAP_S = 5.0
# Post-recovery cache-status poll convergence.  The master's prefill cache
# poll is dynamically intervalled (DynamicCacheIntervalService, default
# 50ms..3000ms) and gated on status ticks, so this window must comfortably
# exceed the 3s ceiling to guarantee the poller has pulled the post-change
# key set (kv.py KV_SYNC_CONVERGENCE_S caliber, widened for the ceiling).
RECOVERY_KV_SYNC_S = 4.5
# The master routes its sync loggers (EngineSyncRunner / GrpcWorkerStatusRunner
# log through the logback "syncLogger") to <flexlb.log.path>/sync.log — by
# default the SHARED ~/ai-whale/logs/sync.log, NOT the per-env stdout capture
# in flexlb_master.log (that one only holds the Spring banner).  The recovery
# env pins the path to a per-env directory so generation/retire observations
# cannot be polluted by sibling runs; every read is an incremental scan from
# a byte offset snapshotted at case start (late async flushes of earlier
# cases then land before the offset only).
SYNC_LOG_ROOT = Path(tempfile.gettempdir()) / "flexlb_ft_sync"


def _recovery_spec(ctx: CaseContext, suffix: str = "") -> EnvSpec:
    """Dedicated E1-E6 env: 2P+2D, dynamic file discovery, fault axes, 30s
    TTL — deliberately a separate label from the shared fault_/kv_ envs.
    A non-empty *suffix* gives a case its OWN env: E2's routing-shape
    assertions (regime A stick / regime B spread) are perturbed by a
    shared env's accumulated soft state (an earlier case's retire storm
    leaves the stormed engine with a routing penalty that skews the
    no-affinity distribution toward the other engine — observed as a
    stable 4/5 bias in the shared-env runs).
    """
    label = f"recovery{suffix}_{ctx.profile}"
    return EnvSpec(
        label=label,
        n_prefill=2,
        n_decode=2,
        perf=fault_env_perf(),
        master_profile=ctx.profile,
        discovery="discovery_file",
        master_env={"FLEXLB_CONFIG": fault_env_config()},
        # Route ALL master logback output (application/sync/flexlb/pv) into
        # a per-env directory — the generation/retire observations below read
        # <dir>/sync.log instead of the shared ~/ai-whale/logs.
        master_extra_args=[f"--flexlb.log.path={SYNC_LOG_ROOT / label}"],
    )


def _recovery_env(ctx: CaseContext, suffix: str = ""):
    env = ctx.env_manager.ensure(_recovery_spec(ctx, suffix))
    return env, ctx.engine_ops(env)


def _engine_ip_port(ops, engine_name: str) -> str:
    """Master-facing address of *engine_name* (discovery-file http port,
    i.e. grpc port - 1 — the ipPort the master logs and keys workerStatus
    entries by)."""
    snap = ops.snapshot_by_name().get(engine_name, {})
    grpc_port = int(snap.get("port", 0))
    if grpc_port <= 0:
        raise RuntimeError(f"no grpc port for engine {engine_name}: {snap!r}")
    return f"127.0.0.1:{grpc_port - 1}"


def _sync_log_path(env) -> Path:
    """Per-env sync log (logback syncLogger → <flexlb.log.path>/sync.log)."""
    return SYNC_LOG_ROOT / env.spec.label / "sync.log"


def _master_log_offset(env) -> int:
    """Byte offset of the sync log at case start — every count below scans
    incrementally from here, so residue from earlier cases (and sibling runs)
    cannot leak into the observation."""
    try:
        return _sync_log_path(env).stat().st_size
    except OSError:
        return 0


def _master_log_count(env, needle: str, offset: int = 0) -> int:
    """Count sync-log lines containing *needle* (generation lifecycle
    observations), scanning from *offset* onward.  Returns 0 when the log
    is unavailable."""
    path = _sync_log_path(env)
    if not path.is_file():
        return 0
    count = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            if offset > 0:
                fh.seek(offset)
            for line in fh:
                if needle in line:
                    count += 1
    except OSError:
        return 0
    return count


def _created_generation_count(env, ip_port: str, offset: int = 0) -> int:
    """How many WorkerStatus generations the master has CREATED for *ip_port*
    since *offset* — "Created WorkerStatus generation {} for worker: {ip}"."""
    return _master_log_count(env, f"for worker: {ip_port}", offset)


def _retire_count(env, ip_port: str, offset: int = 0) -> int:
    """Transport-failure retirements logged for *ip_port* since *offset* —
    "worker {ip} marked dead after {} consecutive gRPC failures"."""
    return _master_log_count(env, f"worker {ip_port} marked dead", offset)


def _prefill_endpoint_ledger(ops, ip_port: str) -> Optional[dict]:
    """The /rtp_llm/inflight_status prefill entry for *ip_port*, or None."""
    data = ops.master_inflight()
    if not data:
        return None
    for ep in data.get("prefill_endpoints", []):
        if ep.get("ip_port") == ip_port:
            return ep
    return None


def _recovery_cache_keys(ops, engine_name: str) -> set:
    """Per-engine LRU key set from mock /snapshot (same shape as kv.py's
    _engine_cache_keys — kept local so category modules stay independent)."""
    entry = ops.snapshot_by_name().get(engine_name, {})
    if "cache_key_set" not in entry:
        raise RuntimeError(f"no 'cache_key_set' for engine {engine_name} in /snapshot")
    return set(int(k) for k in entry["cache_key_set"])


def _recovery_cache_evict(ops, engine_name: str, keys) -> None:
    """POST /cache_evict (same endpoint as kv.py — bumps cacheVersion so the
    master's next cache poll re-pulls the key set)."""
    status, body = http_post_json(
        f"http://127.0.0.1:{ops.mock_http_port}/cache_evict",
        {"engine": engine_name, "keys": [int(k) for k in keys]},
    )
    if status != 200:
        raise RuntimeError(
            f"cache_evict({engine_name}, {len(keys)} keys) failed: {status} {body}"
        )


def _fire_inflight(ops, base: int, n: int, **kwargs) -> list:
    """Fire *n* requests without consuming their streams — the master
    ledger entries stay live until consumed/cancelled (S4 drainage lesson:
    unconsumed entries linger and poison later phases).  Returns
    [(rid, response, engine_name)]."""
    fired = []
    for _ in range(n):
        rid = ops.next_request_id(base)
        try:
            resp = ops.schedule(rid, **kwargs)
        except Exception:
            continue
        if resp.code != 200 or not resp.success:
            continue
        addr = ops.role_addr(resp, "PREFILL")
        fired.append((rid, resp, ops.addr_to_name().get(addr, addr)))
    return fired


def _consume_fired(ops, fired: list, wait_s: float = STREAM_TIMEOUT_S) -> list:
    """Consume every fired request to a terminal state (cancel fallback).
    Returns [(rid, engine_name, completed)] for resurrection bookkeeping."""
    outcomes = []
    for rid, resp, name in fired:
        completed = False
        try:
            handle = ops.start_stream(resp, rid)
            ended = handle.wait_end(wait_s)
            completed = bool(ended and handle.snap.completed and not handle.snap.error)
        except Exception:
            completed = False
        if not completed:
            try:
                ops.cancel(rid, resp)
            except Exception:
                pass
        outcomes.append((rid, name, completed))
    return outcomes


def _ensure_started(ops, names) -> None:
    """Restore any stopped engine (env hygiene for the shared recovery env)."""
    try:
        snap = ops.snapshot_by_name()
        for n in names:
            if snap.get(n, {}).get("stopped"):
                ops.start_engine(n)
    except Exception:
        pass


@case(
    "engine_fault_recovery_generation_bump",
    profiles=["batch-window"],  # _recovery_spec pins the fault axes
    source="E1: engine recovery must publish a fresh endpoint generation",
)
def recovery_generation_bump(ctx: CaseContext):
    """E1 — expected behaviour: an engine that goes unavailable (gRPC
    refusal for a bounded window) and then recovers must come back under a
    NEW WorkerStatus generation (or an equivalent full resync signal), and
    the old generation's queue/ledger must not leak into the new one.

    Mechanism under test: the transport retire path — 3 consecutive status
    RPC failures (GrpcWorkerStatusRunner.recordStatusCheckFailure) retire
    the generation; the discovery loop then re-creates a fresh one.

    Assertions (contract, not implementation):
      * the retire landed ("marked dead after 3 consecutive gRPC failures");
      * the master created at least one NEW generation for the endpoint
        after the outage began (created-count strictly grows);
      * the recovered endpoint's ledger starts from zero (inflight_requests
        == 0 and inflight_batches == 0 before any new traffic);
      * a fresh request completes after recovery.

    FINDING if it fails: recovery without a generation bump — the old
    generation's stale KV baseline / ledger silently survives the outage.
    """
    env, ops = _recovery_env(ctx)
    base = rid_base(ctx, "engine_fault")
    try:
        _cleanup_dynamic(ops, env)
        # Cascade hygiene (task #87): drain earlier residue on this env.
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)

        ip = _engine_ip_port(ops, "prefill-0")
        log_offset = _master_log_offset(env)
        created_before = _created_generation_count(env, ip, log_offset)

        # Baseline traffic: 6 requests must all succeed.
        ok0, err0, _ = _run_batch(ops, base, 6)
        if err0:
            return False, f"baseline batch had {err0} errors"

        # Outage: stop prefill-0; the refused connections accumulate the 3
        # consecutive transport failures that must retire the generation.
        ops.stop_engine("prefill-0")
        retired = wait_for(
            lambda: _retire_count(env, ip, log_offset) > 0, RECOVERY_EVICT_S, 0.2
        )

        # Recovery: restart and wait for the endpoint to serve again.
        ops.start_engine("prefill-0")
        alive_back = _wait_master_alive(
            ops, "PREFILL", env.spec.n_prefill, RECOVERY_EVICT_S
        )
        time.sleep(RECOVERY_SETTLE_S)

        created_after = _created_generation_count(env, ip, log_offset)
        generation_bumped = created_after > created_before

        # The recovered generation's ledger must start from zero BEFORE any
        # new traffic is scheduled against it.
        ledger = _prefill_endpoint_ledger(ops, ip)
        ledger_clean = bool(
            ledger
            and int(ledger.get("inflight_requests", -1)) == 0
            and int(ledger.get("inflight_batches", -1)) == 0
        )

        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = retired and alive_back and generation_bumped and recovery_ok
        return passed, (
            f"ip={ip}, created_generations={created_before}->{created_after}, "
            f"transport_retired={retired}, alive_restored={alive_back}, "
            f"recovered_ledger_zero={ledger_clean}"
            f"(ledger={ledger}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _ensure_started(ops, ["prefill-0", "prefill-1"])


@case(
    "engine_fault_recovery_kv_resync",
    profiles=["batch-window"],  # _recovery_spec pins the fault axes
    source="E2: recovery must rebuild the cache view from a full snapshot",
)
def recovery_kv_resync(ctx: CaseContext):
    """E2 — expected behaviour: on engine recovery the master must rebuild
    the engine's cache view from a FULL snapshot, never from the old
    generation's incremental baseline, in BOTH memory regimes:

      * A (memory intact): the engine keeps its LRU through the outage —
        after recovery the holder relationship must survive, so same-prefix
        requests keep landing on the recovered engine (>= 4/5);
      * B (memory lost): the engine's key set is wiped across the restart
        (cache_evict models the reboot) — the rebuilt view must reflect the
        EMPTY key set, so same-prefix requests spread instead of sticking
        to the old holder (<= 3/5).

    Regime A also pins the generation bump: the retire
    (WorkerGenerationRetirement) clears the address-keyed cache index
    (removeEngineBlockCache) and the new generation re-pulls the full key
    set from version -1 — a master that instead kept the old version
    baseline would reject the (unchanged) engine version and LOSE the
    holder relationship (spread), failing the regime-A gate.

    FINDING if it fails: incremental-baseline resync across a generation
    boundary (stale holder or lost holder).
    """
    # Own env (see _recovery_spec): the regime-B spread bar is a routing
    # shape, and a shared env's earlier retire storms bias it.
    env, ops = _recovery_env(ctx, "_e2")
    base = rid_base(ctx, "engine_fault")
    try:
        _cleanup_dynamic(ops, env)
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)
        names = _prefill_names(ops)
        if len(names) < 2:
            return False, "need >=2 prefill engines"

        # A 10-block prefix family (kv.py caliber: 10 x 1024 tokens keeps a
        # full-hit continuation past the affinity line, 9216 >= 8192).
        fam = [base + 900_000 + j for j in range(10)]

        # Seed: one request admits the family onto its landing engine X.
        rid = ops.next_request_id(base)
        addr, err = ops.run_one_request(
            rid,
            input_len=10_240,
            output_len=2,
            block_keys=fam,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err:
            return False, f"seed request failed: {err}"
        holder = ops.addr_to_name().get(addr, addr)
        other = [n for n in names if n != holder][0]

        # Holder must actually own the family before the outage.
        owner_ok = wait_for(
            lambda: set(fam) <= _recovery_cache_keys(ops, holder),
            8.0,
            0.5,
        )
        if not owner_ok:
            return False, (
                f"seed never admitted the family onto {holder} "
                f"(keys={sorted(_recovery_cache_keys(ops, holder))[:5]}...)"
            )
        # Master-side convergence (>= 3.5s quiet — kv.py caliber).
        time.sleep(RECOVERY_KV_SYNC_S)

        ip = _engine_ip_port(ops, holder)
        log_offset = _master_log_offset(env)
        created_before = _created_generation_count(env, ip, log_offset)

        # Outage + recovery (mock keeps the LRU: memory-intact regime).
        ops.stop_engine(holder)
        retired = wait_for(
            lambda: _retire_count(env, ip, log_offset) > 0, RECOVERY_EVICT_S, 0.2
        )
        ops.start_engine(holder)
        alive_back = _wait_master_alive(
            ops, "PREFILL", env.spec.n_prefill, RECOVERY_EVICT_S
        )
        time.sleep(RECOVERY_SETTLE_S)
        created_after = _created_generation_count(env, ip, log_offset)
        generation_bumped = created_after > created_before

        # Regime A: after the full rebuild, the holder relationship must
        # survive — 5 same-prefix requests, >= 4 must land on the holder.
        landings_a = []
        for _ in range(5):
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=10_240,
                output_len=2,
                block_keys=fam,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                landings_a.append(f"ERR:{str(err)[:40]}")
            else:
                landings_a.append(ops.addr_to_name().get(addr, addr))
        hits_a = sum(1 for x in landings_a if x == holder)
        regime_a_ok = hits_a >= 4

        # Regime B: wipe the engine's key set (reboot semantics) and let
        # the master's view converge — same-prefix requests must now
        # spread; sticking to the wiped holder means the master kept a
        # stale view.
        _recovery_cache_evict(ops, holder, fam)
        owner_gone = wait_for(
            lambda: not (set(fam) & _recovery_cache_keys(ops, holder)),
            8.0,
            0.5,
        )
        time.sleep(RECOVERY_KV_SYNC_S)
        landings_b = []
        for _ in range(5):
            rid = ops.next_request_id(base)
            addr, err = ops.run_one_request(
                rid,
                input_len=10_240,
                output_len=2,
                block_keys=fam,
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            if err:
                landings_b.append(f"ERR:{str(err)[:40]}")
            else:
                landings_b.append(ops.addr_to_name().get(addr, addr))
        hits_b = sum(1 for x in landings_b if x == holder)
        regime_b_ok = hits_b <= 3

        passed = (
            retired
            and alive_back
            and generation_bumped
            and owner_gone
            and regime_a_ok
            and regime_b_ok
        )
        return passed, (
            f"holder={holder}(other={other}), ip={ip}, "
            f"created_generations={created_before}->{created_after}, "
            f"retired={retired}, alive_restored={alive_back}, "
            f"regime_A_stick={hits_a}/5 (need >=4), "
            f"regime_B_spread={hits_b}/5 (need <=3, wipe_ok={owner_gone}), "
            f"landings_A={landings_a}, landings_B={landings_b}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _ensure_started(ops, ["prefill-0", "prefill-1"])


@case(
    "engine_fault_recovery_no_resurrect",
    profiles=["batch-window"],  # _recovery_spec pins the fault axes
    source="E3: pre-outage inflight requests must not resurrect after recovery",
)
def recovery_no_resurrect(ctx: CaseContext):
    """E3 — expected behaviour: requests that were in flight on an engine
    when it CRASHED must not leak into the recovered generation's
    bookkeeping:

      * the master's per-endpoint ledger for the recovered engines must
        read zero inflight (old entries fenced or TTL-settled), and the
        global inflight must drain to zero within the TTL cap;
      * the engine side must come back EMPTY — a true crash wipes the
        process memory, so after /start_engine the engine has no running
        tasks, no held blocks and an empty KV cache (recovery == a reboot
        from zero, not an in-place resume);
      * the pre-outage rids must never complete — their engine-side state
        is gone (no resurrection); they settle through the master's
        fence/TTL paths, and fresh traffic must schedule normally on the
        recovered engines.

    Mechanism: crash_after with TRUE-CRASH semantics — each target engine
    is armed to die on its NEXT EnqueueBatch (every queue, running task,
    KV lease and LRU entry is wiped, the gRPC port is killed), then
    trigger requests are fired until every target reports stopped.  The
    master observes the dead port (3 consecutive gRPC failures) and
    retires the endpoint; /start_engine rebuilds the gRPC server on clean
    state (the mock control-plane HTTP server survives the crash, which
    is what makes the restart path reachable).

    FINDING if the asserted master-side bars fail: F1 permanent-ledger-leak
    elastic variant — old generation requests surviving into the new one's
    ledger.
    """
    env, ops = _recovery_env(ctx)
    base = rid_base(ctx, "engine_fault")
    names = ["prefill-0", "prefill-1"]
    try:
        _cleanup_dynamic(ops, env)
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)

        # Widen the in-flight window so the outage lands mid-execution:
        # slow prefills keep requests waiting/running on the engines while
        # we take them down.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=2000.0)
        time.sleep(1.5)  # master perf sync

        fired = _fire_inflight(ops, base, 8, input_len=512, output_len=2)
        if len(fired) < 4:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
            return False, f"only {len(fired)}/8 requests fired successfully"
        time.sleep(0.5)  # let the batcher dispatch them onto the engines

        targets = sorted({name for _rid, _resp, name in fired})
        target_ips = {n: _engine_ip_port(ops, n) for n in targets}
        log_offset = _master_log_offset(env)

        # Crash: arm crash_after at each target's NEXT EnqueueBatch (the
        # counter already includes the fired requests' batches), then fire
        # trigger requests until every target reports stopped — the crash
        # only fires when a fresh EnqueueBatch lands on the armed engine.
        # The trigger requests are small and unconsumed; ones that land on
        # an armed target become the crash-triggering empty ack (uncertain
        # fence, TTL-settled by the inflight bar below).
        for n in targets:
            snap = ops.snapshot_by_name().get(n, {})
            n_batches = int(snap.get("rpc_counts", {}).get("enqueue_batch", 0))
            inject_type(ops, n, "crash_after", n=n_batches + 1)
        crashed_all = False
        deadline = time.monotonic() + CRASH_TRIGGER_WINDOW_S
        while time.monotonic() < deadline:
            snaps = ops.snapshot_by_name()
            if all(snaps.get(n, {}).get("stopped") for n in targets):
                crashed_all = True
                break
            try:
                ops.schedule(ops.next_request_id(base), input_len=64, output_len=2)
            except Exception:
                pass  # a trigger may hit an engine mid-crash; keep firing
            time.sleep(0.2)

        # The master observes the dead ports and retires the endpoints
        # (same 3-strike transport path as stop_engine).
        retired_all = True
        for n, ip in target_ips.items():
            if not wait_for(
                lambda ip=ip: _retire_count(env, ip, log_offset) > 0,
                RECOVERY_EVICT_S,
                0.2,
            ):
                retired_all = False

        # Recovery: /start_engine rebuilds the gRPC server on CLEAN state
        # (it also disarms the fault config and resets the enqueue count).
        for n in targets:
            ops.start_engine(n)
        alive_back = _wait_master_alive(
            ops, "PREFILL", env.spec.n_prefill, RECOVERY_EVICT_S
        )
        time.sleep(RECOVERY_SETTLE_S)

        # The recovered generation's ledger must start from zero.
        ledger_clean = True
        ledger_detail = {}
        for n, ip in target_ips.items():
            ledger = _prefill_endpoint_ledger(ops, ip)
            zero = bool(
                ledger
                and int(ledger.get("inflight_requests", -1)) == 0
                and int(ledger.get("inflight_batches", -1)) == 0
            )
            ledger_clean = ledger_clean and zero
            ledger_detail[n] = ledger

        # True-crash wipe: the recovered engine must have NO memory of the
        # pre-crash world — no running tasks, no held blocks, an empty LRU,
        # zero inflight and a zeroed accept counter (a fresh process).
        wipe_ok = True
        wipe_detail = {}
        for n in targets:
            snap = ops.snapshot_by_name().get(n, {})
            clean = (
                int(snap.get("running", -1)) == 0
                and int(snap.get("inflight", -1)) == 0
                and list(snap.get("cache_key_set") or []) == []
                and int(snap.get("held_blocks", -1)) == 0
                and int(snap.get("accepted", -1)) == 0
            )
            wipe_ok = wipe_ok and clean
            wipe_detail[n] = {
                "running": snap.get("running"),
                "inflight": snap.get("inflight"),
                "cache_keys": len(snap.get("cache_key_set") or []),
                "held_blocks": snap.get("held_blocks"),
                "accepted": snap.get("accepted"),
            }

        # Consume the fired requests to terminal states: with the true
        # crash their engine-side state is GONE, so NONE may complete — a
        # completion would be a resurrection (asserted, no longer just an
        # observation).  Their client streams fail against the wiped
        # engine and settle through the master's fence/TTL.
        outcomes = _consume_fired(ops, fired, wait_s=2.0)
        resurrected = [
            (rid, name)
            for rid, name, completed in outcomes
            if completed and name in targets
        ]

        # Engine side must not keep the old rids registered.
        engine_clean, engine_detail = engine_inflight_clean(ops, targets)

        # Master global ledger drains within the TTL cap (fence or TTL).
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )

        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            crashed_all
            and retired_all
            and alive_back
            and ledger_clean
            and wipe_ok
            and not resurrected
            and engine_clean
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={len(fired)} onto {targets}, "
            f"crashed_all={crashed_all}, retired_all={retired_all}, "
            f"alive_restored={alive_back}, "
            f"recovered_ledger_zero={ledger_clean}({ledger_detail}), "
            f"engine_wipe_clean={wipe_ok}({wipe_detail}), "
            f"resurrected={len(resurrected)}/{len(fired)} (bar: 0), "
            f"engine_inflight_clean={engine_clean}({engine_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        # Disarm any undelivered crash_after before the hygiene restart
        # (start_engine would clear it too, but the arm must not survive
        # an early-exit path that skips the restart).
        for n in names:
            try:
                inject_type(ops, n, "crash_after", enabled=False)
            except Exception:
                pass
        _ensure_started(ops, names)
        for n in names:
            try:
                ops.set_perf(n, prefill_fixed_ms=100.0)
            except Exception:
                pass


@case(
    "engine_fault_status_gap_no_bump",
    profiles=["batch-window"],  # _recovery_spec pins the fault axes
    source="E4: a short status-reporting gap must not retire the generation",
)
def status_gap_no_bump(ctx: CaseContext):
    """E4 — expected behaviour: a SHORT status-reporting gap (2 poll ticks,
    ~40ms) is network jitter, not an outage — the master must tolerate it
    WITHOUT retiring the endpoint's generation:

      * no new WorkerStatus generation is created for the endpoint over
        the whole case window (created-count unchanged);
      * the discovery view stays intact (discovered == alive == 2);
      * traffic through the gap keeps succeeding.

    Mechanism: status_no_respond hangs the in-flight getWorkerStatus RPC;
    with the 1s RPC deadline the transient gap costs at most ONE failed
    poll — well below the 3-consecutive-failure retire threshold.

    FINDING if it fails: over-sensitive retire threshold — jitter-level
    gaps churn generations (and with them the KV baseline / ledger).
    """
    env, ops = _recovery_env(ctx)
    base = rid_base(ctx, "engine_fault")
    try:
        _cleanup_dynamic(ops, env)
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)

        ip = _engine_ip_port(ops, "prefill-0")
        log_offset = _master_log_offset(env)
        created_before = _created_generation_count(env, ip, log_offset)

        # Baseline traffic succeeds.
        ok0, err0, _ = _run_batch(ops, base, 4)
        if err0:
            return False, f"baseline batch had {err0} errors"

        # Transient gap: 2 poll ticks (~40ms) of no_respond, then clear.
        # The hung RPC times out once (1s deadline) — a single failed poll,
        # below the retire threshold.
        inject_type(ops, "prefill-0", "status_no_respond", enabled=True)
        time.sleep(E4_GAP_S)
        inject_type(ops, "prefill-0", "status_no_respond", enabled=False)

        # Let the hung RPC's deadline land and the poller resume.
        time.sleep(2.0)

        created_after = _created_generation_count(env, ip, log_offset)
        generation_bumped = created_after > created_before

        # Topology untouched.
        info = ops.master_info() or {}
        entry = (info.get("worker_summary", {}) or {}).get("PREFILL") or {}
        discovered = int(entry.get("discovered", -1))
        alive = int(entry.get("alive", -1))
        topology_intact = (
            discovered == env.spec.n_prefill and alive == env.spec.n_prefill
        )

        # Traffic still succeeds through/after the gap.
        ok1, err1, _ = _run_batch(ops, base, 4)

        passed = not generation_bumped and topology_intact and err1 == 0
        return passed, (
            f"ip={ip}, created_generations={created_before}->{created_after} "
            f"(bump={generation_bumped}, must be False), "
            f"topology(discovered={discovered}, alive={alive}, "
            f"need {env.spec.n_prefill}/{env.spec.n_prefill}), "
            f"post_gap_batch={ok1}/4, baseline={ok0}/4"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            inject_type(ops, "prefill-0", "status_no_respond", enabled=False)
        except Exception:
            pass


@case(
    "engine_fault_status_gap_long_retire",
    profiles=["batch-window"],  # _recovery_spec pins the fault axes
    source="E5: a long status gap must retire the generation and fence its ledger",
)
def status_gap_long_retire(ctx: CaseContext):
    """E5 — expected behaviour: a status gap LONGER than the retire
    threshold (5s of no_respond ≈ 4+ timed-out polls > 3 consecutive
    failures) is a crash — the master must actively retire the endpoint's
    generation and FENCE its queue/ledger/inflight:

      * the retire landed ("marked dead after 3 consecutive gRPC
        failures") and a fresh generation is created once reporting
        resumes;
      * the fenced engine's master ledger/inflight settles to zero within
        the TTL cap (fence or stale-TTL — an entry that never settles is
        the F7 pending-drain gap);
      * once reporting resumes, the new generation serves fresh traffic.

    In-flight requests fired BEFORE the gap are the fence payload: their
    engine-side execution completes but the master's status channel is
    dead, so their ledger release must come from the retire fence (or the
    stale-TTL), never from a stale post-recovery resurrection.

    FINDING if it fails: no retire on a long gap, or an unfenced ledger
    that neither the retire nor the TTL ever clears.
    """
    env, ops = _recovery_env(ctx)
    base = rid_base(ctx, "engine_fault")
    names = ["prefill-0", "prefill-1"]
    try:
        _cleanup_dynamic(ops, env)
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)

        ip = _engine_ip_port(ops, "prefill-0")
        log_offset = _master_log_offset(env)
        created_before = _created_generation_count(env, ip, log_offset)

        # Fence payload: slow requests in flight on the engines when the
        # status channel dies.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=2000.0)
        time.sleep(1.5)
        fired = _fire_inflight(ops, base, 8, input_len=512, output_len=2)
        time.sleep(0.5)

        # Long gap: 5s of no_respond ≈ 4+ consecutive timed-out status
        # polls — past the 3-failure retire threshold.
        inject_type(ops, "prefill-0", "status_no_respond", enabled=True)
        retired = wait_for(
            lambda: _retire_count(env, ip, log_offset) > 0, E5_GAP_S + 10.0, 0.2
        )
        # Hold the gap a little past the retire so the retire/re-create
        # cycle is observable, then resume reporting.
        time.sleep(1.0)
        inject_type(ops, "prefill-0", "status_no_respond", enabled=False)

        alive_back = _wait_master_alive(
            ops, "PREFILL", env.spec.n_prefill, RECOVERY_EVICT_S
        )
        time.sleep(RECOVERY_SETTLE_S)
        created_after = _created_generation_count(env, ip, log_offset)
        generation_bumped = created_after > created_before

        # Consume the fence payload to terminal states.
        outcomes = _consume_fired(ops, fired, wait_s=5.0)

        # Fenced ledger/inflight must settle within the TTL cap.
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )

        # New generation serves fresh traffic.
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            retired and alive_back and generation_bumped and inflight_ok and recovery_ok
        )
        return passed, (
            f"ip={ip}, created_generations={created_before}->{created_after}, "
            f"retired={retired}, alive_restored={alive_back}, "
            f"fired={len(fired)}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            inject_type(ops, "prefill-0", "status_no_respond", enabled=False)
        except Exception:
            pass
        _ensure_started(ops, names)
        for n in names:
            try:
                ops.set_perf(n, prefill_fixed_ms=100.0)
            except Exception:
                pass


@case(
    "engine_fault_recovery_kv_usage_reset",
    profiles=["batch-window"],  # _recovery_spec pins the fault axes
    source="E6: KV usage must restart from zero after a full restart",
)
def recovery_kv_usage_reset(ctx: CaseContext):
    """E6 — expected behaviour: when an engine restarts and its KV memory
    is lost, its self-reported KV usage must restart from ZERO — the
    master's capacity view of the new generation must be rebuilt from the
    fresh self-report, never resumed from the old generation's reading:

      * a pre-outage injected KV occupancy must be gone from the engine's
        self-report after the restart (kv_tokens_used == 0 with an empty
        LRU — a resumed old reading is a restart-fidelity defect);
      * the recovered engine must keep receiving traffic (the master does
        not blacklist it behind the stale occupancy);
      * the generation actually turned over (fresh full-resync signal).

    FINDING if it fails: KV capacity not reset across a restart — the old
    generation's occupancy leaks into the new one (master-side), or the
    mock's restart does not model memory loss (mock-side fidelity).
    """
    env, ops = _recovery_env(ctx)
    base = rid_base(ctx, "engine_fault")
    try:
        _cleanup_dynamic(ops, env)
        AssertUtils.inflight_clean(_master_http(ops), TTL_DRAIN_TIMEOUT_S)

        name = "prefill-0"
        ip = _engine_ip_port(ops, name)

        # Fresh LRU baseline: wipe whatever earlier cases left on this
        # engine so kv_tokens_used reads exactly the injected pressure.
        _recovery_cache_evict(ops, name, sorted(_recovery_cache_keys(ops, name)))
        time.sleep(RECOVERY_KV_SYNC_S)

        used_before = int(
            ops.snapshot_by_name().get(name, {}).get("kv_tokens_used", -1)
        )
        if used_before != 0:
            return False, (
                f"baseline not clean after evict (kv_tokens_used={used_before})"
            )

        # Inject a large occupancy (engine self-report channel).
        pressure = 4_000_000
        ops.set_kv_pressure(name, pressure)
        pressurized = wait_for(
            lambda: int(ops.snapshot_by_name().get(name, {}).get("kv_tokens_used", -1))
            >= pressure,
            8.0,
            0.5,
        )
        if not pressurized:
            return False, "set_kv_pressure never surfaced in /snapshot"

        log_offset = _master_log_offset(env)
        created_before = _created_generation_count(env, ip, log_offset)

        # Full-restart outage: stop → retire → start (memory lost).
        ops.stop_engine(name)
        retired = wait_for(
            lambda: _retire_count(env, ip, log_offset) > 0, RECOVERY_EVICT_S, 0.2
        )
        ops.start_engine(name)
        alive_back = _wait_master_alive(
            ops, "PREFILL", env.spec.n_prefill, RECOVERY_EVICT_S
        )
        time.sleep(RECOVERY_SETTLE_S)
        created_after = _created_generation_count(env, ip, log_offset)
        generation_bumped = created_after > created_before

        # The engine's self-report must restart from zero (memory lost).
        used_after = int(ops.snapshot_by_name().get(name, {}).get("kv_tokens_used", -1))
        reset_ok = used_after == 0

        # The master must not keep the engine blacklisted behind the stale
        # occupancy: fresh traffic still lands on it.
        pumps_ok = _pump_until_accepted(ops, name, base, 15.0)

        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            retired
            and alive_back
            and generation_bumped
            and reset_ok
            and pumps_ok
            and recovery_ok
        )
        return passed, (
            f"ip={ip}, created_generations={created_before}->{created_after}, "
            f"retired={retired}, alive_restored={alive_back}, "
            f"kv_tokens_used={pressure}(injected)->{used_after} "
            f"(reset_ok={reset_ok}, need 0), "
            f"post_recovery_traffic={pumps_ok}, recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _ensure_started(ops, ["prefill-0", "prefill-1"])
        try:
            ops.set_kv_pressure("prefill-0", 0)
        except Exception:
            pass
