"""Cross-process fault-injection coverage (gaps G6/G7) + admission gates
(gaps G8/G10/G11).

Task #51.  The mock control plane (/inject) supports NINE injection types
(MockControlServer.handleInject, Java "type" format, MERGE semantics):

    enqueue_error | generate_error | fetch_error | no_respond | kv_pressure
    | queue_depth | crash_after | enqueue_delay | generate_delay

Before this file the cross-process functional suite consumed only TWO of
them — no_respond (smoke_cases E2) and enqueue_error (smoke_cases E3),
both via the Python "config" format which REPLACES the whole config and
supports only the four boolean flags.  The remaining SEVEN are covered
here via the Java "type" format:

    generate_error  -> chaos_inject_generate_error (client-direct
                       GenerateStreamCall path: the fault fires ONLY in
                       generateStreamCall, and master-routed traffic — all
                       three modes deliver via enqueueBatch + FetchResponse
                       (direct-run evidence: generate_stream_rpcs=0 with
                       enqueue_rpcs=3/fetch_response_rpcs=3) — never
                       reaches it; the case drives the client-direct
                       contract instead of imagining a master path)
    fetch_error     -> chaos_inject_fetch_error (batch only: FetchResponse
                       is the batch-mode client stream)
    crash_after     -> chaos_inject_crash_after (batch: Nth enqueue flips
                       the engine to stopped — NOT a process exit, a
                       stopped flag; /start_engine clears the fault config
                       and resets the enqueue counter)
    enqueue_delay   -> chaos_inject_enqueue_delay (batch only: the enqueue
                       ack is deferred by the scheduler, so the batch-mode
                       schedule() latency grows by delay_ms)
    generate_delay  -> chaos_inject_generate_delay (all modes: delay_ms is
                       added to the prefill execution time, visible in TTFT)
    kv_pressure     -> consumed by gate_slo_queue_deadline (G11)
    queue_depth     -> consumed by gate_queue_depth_reject (G8)

Admission-gate cases (suite="smoke", functional taxonomy per the 2026-08
rework — gates are functional, injections are chaos):

    gate_queue_depth_reject    (G8)  engine-side queue_depth limit: fast
                                      per-request rejection with the
                                      "queue depth limit exceeded" error
                                      (EnqueueBatch errors list ->
                                      DefaultBatchDispatcher.handleResponse
                                      -> EngineRejectedException ->
                                      BATCH_DISPATCH_FAILED), not a
                                      silent pile-up; recovery after the
                                      gate is lifted.
    gate_lru_eviction_affinity (G10) small prefill_cache_blocks capacity:
                                      prefix reuse routes back to the
                                      primed engine (S2-style affinity),
                                      a 5-key admit into the capacity-4
                                      LRU evicts exactly the eldest block
                                      (snapshot cache_keys/cache_evictions).
    gate_slo_queue_deadline    (G11a) tiny scheduler.queueTimeoutMs +
                                      kv_pressure-squeezed prefill KV:
                                      the batcher's KV gate is a WAIT
                                      condition (BatcherContext.
                                      admitAndDeliverCapacityFeasiblePrefix
                                      -> CapacityBlocked), so the request
                                      sits in the active queue until the
                                      SLO deadline fires.
    gate_master_capacity_reject(G11b) capacity.maxOutstandingRequestsGlobal=2
                                      under PRIORITY ordering: the submit
                                      path fast-rejects the excess with
                                      RESOURCE_EXHAUSTED "master outstanding
                                      capacity exhausted".

Contracts verified against the Java implementation (do not "imagine"
semantics — see the smoke_cases.py docstring for the six-case lesson):

* crash_after N: enqueueTotal >= N sets stopped=true and answers the
  triggering enqueue with an EMPTY ack (no errors, no successes) — the
  master classifies it dispatch-uncertain and installs a BATCH_ACK_
  UNCERTAIN engine fence; under the cross-process production wiring
  (UnsupportedEngineCancelChannel) the UNSUPPORTED probe ack is not a
  safe release fact, so the entry parks in the quarantined fence sweep
  indefinitely (bounded residue is the EXPECTED contract); the batch
  itself never enters the engine ledger.  getWorkerStatus keeps
  answering alive=false (stopped flag, NOT a dead process).
* enqueue_delay defers the WHOLE process runnable (admission + ack):
  batch-mode schedule() blocks for delay_ms (< enqueueRpcTimeoutMs default
  5000 or the RPC deadline fires first).
* generate_delay adds to the prefill execution estimate only (TTFT).
* kv_pressure: WorkerStatus.availableKvCache = totalKv - (activeKv +
  kvPressureTokens), capped at 0.
* queue_depth: pendingRequests >= limit rejects EVERY request of the
  enqueue batch with "queue depth limit exceeded"; the gate frees itself
  once the occupier finishes (decrement at completion).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .context import CaseContext, CaseDef, rid_base
from .engine_ops import StreamHandle, StreamSnapshot
from .harness import (
    AssertUtils,
    EnvSpec,
    build_flexlb_config,
    default_perf,
    http_get_json,
    http_post_json,
    wait_for,
)

INJECTION_GATE_CASES: list[CaseDef] = []

STREAM_TIMEOUT_S = 15.0
# Master cache-status sync (GrpcCacheStatusCheckRunner poll, smoke S2 uses 2s).
KV_CACHE_SYNC_WAIT_S = 2.0
# 3-strike health marking + eviction window (chaos MASTER_EVICT_S precedent).
MASTER_EVICT_S = 30.0
# Mock DEFAULT_TOTAL_KV_TOKENS — squeezing with this value drives
# availableKvCache to 0.
MOCK_TOTAL_KV_TOKENS = 6_291_456
ENGINE_RECOVERY_WAIT_S = 3.0


def case(name: str, modes=None, source: str = "", suite: str = "chaos"):
    """Register into INJECTION_GATE_CASES; *suite* drives the runner
    grouping (injections -> chaos, gates -> smoke), following the
    elastic_* precedent in chaos_cases.py."""

    def deco(fn):
        INJECTION_GATE_CASES.append(
            CaseDef(name=name, suite=suite, fn=fn, modes=modes, source=source)
        )
        return fn

    return deco


# ===========================================================================
# Shared helpers
# ===========================================================================


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"]


def inject_type(
    ops, engine_name: str, fault_type: str, enabled: bool = True, **params
) -> dict:
    """POST /inject with the ORIGINAL Java "type" format (MERGE semantics,
    supports all nine fault types plus their parameters).

    The EngineOps.inject() helper uses the Python "config" format which
    REPLACES the whole config and only knows the four boolean flags, so it
    cannot express kv_pressure / queue_depth / crash_after / delays.
    """
    payload = {"engine": engine_name, "type": fault_type, "enabled": enabled}
    payload.update(params)
    status, body = http_post_json(
        f"http://127.0.0.1:{ops.mock_http_port}/inject", payload
    )
    if status != 200:
        raise RuntimeError(
            f"inject_type({engine_name}, {fault_type}, {params}) "
            f"failed: {status} {body}"
        )
    return body or {}


def inject_type_all(ops, names: list[str], fault_type: str, **params) -> None:
    for name in names:
        inject_type(ops, name, fault_type, **params)


def clear_type_all(ops, names: list[str], fault_type: str) -> None:
    for name in names:
        try:
            inject_type(ops, name, fault_type, enabled=False)
        except Exception:
            pass


def engine_inflight_clean(
    ops, names: list[str], timeout_s: float = 10.0
) -> tuple[bool, str]:
    """Engine-side leak check: every named engine reports inflight == 0 and
    leak_detected == false in /snapshot."""

    def clean() -> bool:
        snap = ops.snapshot_by_name()
        return all(
            snap.get(n, {}).get("inflight", 0) == 0
            and not snap.get(n, {}).get("leak_detected", False)
            for n in names
        )

    ok = wait_for(clean, timeout_s, 0.5)
    snap = ops.snapshot_by_name()
    detail = {
        n: (
            snap.get(n, {}).get("inflight", -1),
            snap.get(n, {}).get("leak_detected", None),
        )
        for n in names
    }
    return ok, f"{json.dumps(detail, sort_keys=True)}"


def _measure_ttft(
    ops, rid: int, timeout_s: float = 12.0
) -> tuple[float, Optional[str]]:
    """schedule + stream one request, measuring first-output latency."""
    response = ops.schedule(rid)
    if response.code != 200 or not response.success:
        return -1.0, f"schedule failed: {response.error_message}"
    input_pb = None if response.enqueued_by_master else ops.build_generate_input(rid)
    handle = ops.start_stream(response, rid, input_pb=input_pb)
    t0 = time.monotonic()
    got_first = handle.wait_first_output(timeout_s)
    ttft = time.monotonic() - t0
    if not got_first:
        handle.cancel()
        return -1.0, "no first output"
    ended = handle.wait_end(timeout_s)
    if not ended or handle.snap.error:
        handle.cancel()
        return ttft, f"stream error after first output: {handle.snap.error}"
    return ttft, None


def _any_engine_busy(ops, names: list[str]) -> bool:
    snap = ops.snapshot_by_name()
    return any(
        snap.get(n, {}).get("waiting", 0) + snap.get(n, {}).get("running", 0) >= 1
        for n in names
    )


def _all_engines_busy(ops, names: list[str]) -> bool:
    snap = ops.snapshot_by_name()
    return all(
        snap.get(n, {}).get("waiting", 0) + snap.get(n, {}).get("running", 0) >= 1
        for n in names
    )


def _gate_config(
    queue_timeout_ms: int = 60_000,
    max_outstanding: int = 5_000,
    stale_inflight_ms: int = 30_000,
) -> str:
    """FLEXLB_CONFIG for the gate cases: the legacy chaos axes (QUEUE +
    PRIORITY + FIXED_WINDOW + BATCH) via the unified
    harness.build_flexlb_config template, with the admission knobs
    parameterised."""
    return build_flexlb_config(
        ordering="priority",
        decision="fixed_window",
        dispatcher="batch",
        queue_timeout_ms=queue_timeout_ms,
        max_outstanding=max_outstanding,
        stale_inflight_ms=stale_inflight_ms,
    )


# ===========================================================================
# Injection cases — chaos suite (gap G6/G7: the 5 previously
# uncovered cross-process injection types)
# ===========================================================================


@case(
    "chaos_inject_generate_error",
    modes=["direct"],
    source="gap G6/G7: /inject type=generate_error (GenerateStreamCall entry, client-direct path)",
)
def inject_generate_error(ctx: CaseContext):
    """generate_error is checked ONLY at the engine's GenerateStreamCall
    entry (JavaMockEngineCluster.generateStreamCall: onError before any
    request state is registered).  Master-routed traffic structurally never
    reaches that check: ALL three master modes (direct/queue/batch) deliver
    through enqueueBatch + FetchResponse — round-4 evidence from a direct
    run: generate_stream_rpcs=0 while enqueue_rpcs=3 and
    fetch_response_rpcs=3 — so instead of imagining a master path, this
    case pins the CROSS-PROCESS contract on the client-direct
    GenerateStreamCall path (the load-client direct deployment shape;
    same direct-stub sequence EngineOps already uses for worker_cancel):

    inject -> the direct stream fails immediately with the injected
    error and registers no engine-side inflight; clear -> a fresh direct
    request completes normally."""
    ops = ctx.ops()
    base = rid_base(ctx, "chaos")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    snap = ops.snapshot_by_name()
    target = None
    for n in names:
        entry = snap.get(n) or {}
        addr = entry.get("grpc_addr")
        if addr:
            target = str(addr)
            break
    if not target:
        return False, "no prefill engine address in snapshot"

    def direct_request(rid: int) -> tuple[Optional[str], object]:
        input_pb = ops.build_generate_input(rid)
        stub = ops.pb2_grpc.RpcServiceStub(ops._channel(target))
        call = stub.GenerateStreamCall(input_pb, timeout=30.0)
        handle = StreamHandle(call, StreamSnapshot())
        handle.wait_end(STREAM_TIMEOUT_S)
        if handle.snap.error:
            return str(handle.snap.error), handle
        if not handle.snap.completed:
            return "stream did not complete", handle
        return None, handle

    try:
        rid0 = ops.next_request_id(base)
        err0, _ = direct_request(rid0)
        if err0:
            return False, f"baseline direct request failed: {err0}"

        inject_type_all(ops, names, "generate_error")
        try:
            rid1 = ops.next_request_id(base)
            err1, _ = direct_request(rid1)
            # Cross-process the engine's onError(RuntimeException("injected
            # generate_error")) reaches the client as grpc status 2
            # (UNKNOWN) with an EMPTY message — the text is not transmitted
            # (verified round 5: grpc_message:"", grpc_status:2) — so the
            # assertion is error-arrived, same contract as the fetch_error
            # case; causality comes from the inject/clear sandwich.
            error_ok = err1 is not None
        finally:
            clear_type_all(ops, names, "generate_error")

        rid2 = ops.next_request_id(base)
        err2, _ = direct_request(rid2)
        engine_clean, engine_detail = engine_inflight_clean(ops, names)

        passed = error_ok and err2 is None and engine_clean
        return passed, (
            f"direct_target={target}, "
            f"error_surfaced={error_ok} ({err1}), "
            f"recovered={err2 is None}"
            f"{'' if err2 is None else ' err=' + err2[:60]}, "
            f"engine_inflight_clean={engine_clean}({engine_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "generate_error")


def _fault_spec(ctx: CaseContext) -> EnvSpec:
    """Env for fault cases whose requests die mid-flight (fetch_error,
    crash_after): short staleInflightTimeoutMs (30s vs the na130 default
    300s) because the VERIFIED contract is that a request already
    accepted by an engine but whose client stream dies is cleaned by the
    stale-inflight TTL, not by an immediate terminal (engine-side inflight
    DOES drain immediately; the master ledger entry lingers).  With the
    smoke env's 300s TTL the case would have to wait 5 minutes."""
    return EnvSpec(
        label=f"inject_fault_{ctx.mode}",
        n_prefill=2,
        n_decode=2,
        perf=default_perf(),
        master_mode=ctx.mode,
        master_env={"FLEXLB_CONFIG": _gate_config(stale_inflight_ms=30_000)},
    )


def _stale_inflight_clean(ops, timeout_s: float = 95.0) -> tuple[bool, str]:
    """Master inflight drain with the TTL-aware window (30s TTL + margin)."""
    return AssertUtils.inflight_clean(_master_http(ops), timeout_s)


def _fence_residue_stable(
    ops, max_residue: int, settle_s: float = 20.0
) -> tuple[bool, str]:
    """Cross-process contract for an empty-ack (uncertain) enqueue batch.

    The master installs a BATCH_ACK_UNCERTAIN engine fence
    (PriorityScheduler.fenceEntryForUncertainBatchDelivery).  In the
    cross-process production wiring the cancel channel is
    UnsupportedEngineCancelChannel, whose UNSUPPORTED ack is NOT a safe
    release fact (handleEngineFenceOutcome groups it with FAILED /
    NOT_FOUND), so the entry parks in the 60s quarantined-fence sweep
    indefinitely; cleanupInflight explicitly skips engineFence entries
    from the stale TTL.  A bounded, non-growing scheduler-ledger residue
    is therefore the EXPECTED production behaviour, not a leak: assert
    residue <= max_residue (the uncertain batches themselves) and that a
    later sample does not grow (no amplification).
    """
    http = _master_http(ops)
    first = None
    deadline = time.monotonic() + settle_s
    while time.monotonic() < deadline:
        data = http_get_json(f"{http}/rtp_llm/inflight_status", timeout=5)
        if data is not None:
            first = data.get("scheduler_inflight", 0)
            if first <= max_residue:
                break
        time.sleep(1.0)
    if first is None:
        return False, "no inflight_status response"
    if first > max_residue:
        return False, f"residue {first} > bound {max_residue}"
    time.sleep(8.0)
    data = http_get_json(f"{http}/rtp_llm/inflight_status", timeout=5)
    second = -1 if data is None else data.get("scheduler_inflight", 0)
    if second < 0:
        return False, "no inflight_status response (second sample)"
    if second > first:
        return False, f"residue grew {first} -> {second} (leak amplification)"
    return True, f"quarantined residue bounded and stable: {first} -> {second}"


@case(
    "chaos_inject_fetch_error",
    modes=["batch"],
    source="gap G6/G7: /inject type=fetch_error (cross-process, batch FetchResponse path)",
)
def inject_fetch_error(ctx: CaseContext):
    """fetch_error makes the batch-mode FetchResponse stream fail after
    emitting one unfinished output.  The client must observe the error;
    the engine-side inflight drains immediately; the master-side ledger
    entry is cleaned by the 30s stale-inflight TTL (verified contract);
    a fresh request succeeds once the injection is cleared."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_fault_spec(ctx)))
    base = rid_base(ctx, "chaos")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    rid = ops.next_request_id(base)
    try:
        inject_type_all(ops, names, "fetch_error")
        try:
            response = ops.schedule(rid)
            if response.code != 200 or not response.success:
                surfaced, detail = True, (f"schedule failed: {response.error_message}")
            else:
                handle = ops.start_stream(response, rid, input_pb=None)
                handle.wait_end(10.0)
                if handle.snap.error:
                    surfaced, detail = True, f"stream error: {handle.snap.error}"
                elif not handle.snap.completed:
                    surfaced, detail = True, "stream did not complete"
                else:
                    surfaced, detail = False, "request completed despite fetch_error"
                # NOTE: no explicit master cancel here.  The stream already
                # terminated with the engine's error, and a cancel would set
                # cancellationReason, which the TTL cleaner SKIPS (it waits
                # for an authoritative engine terminal through the cancel
                # fence instead) — verified on the Java side
                # (PriorityScheduler.cleanupInflight).
        finally:
            clear_type_all(ops, names, "fetch_error")

        rid2 = ops.next_request_id(base)
        _, err2 = ops.run_one_request(rid2, stream_timeout_s=STREAM_TIMEOUT_S)
        inflight_ok, inflight_detail = _stale_inflight_clean(ops)
        engine_clean, engine_detail = engine_inflight_clean(ops, names)
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            surfaced and err2 is None and inflight_ok and engine_clean and recovery_ok
        )
        return passed, (
            f"error_surfaced={surfaced} ({detail}), "
            f"recovered={err2 is None}, "
            f"master_inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_inflight_clean={engine_clean}({engine_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "fetch_error")


@case(
    "chaos_inject_crash_after",
    modes=["batch"],
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
    must be fully clean."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_fault_spec(ctx)))
    base = rid_base(ctx, "chaos")
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


@case(
    "chaos_inject_enqueue_delay",
    modes=["batch"],
    source="gap G6/G7: /inject type=enqueue_delay (deferred enqueue ack, batch mode)",
)
def inject_enqueue_delay(ctx: CaseContext):
    """enqueue_delay defers the whole enqueue runnable (admission + ack) by
    delay_ms, so the batch-mode schedule() — which waits for the enqueue
    ack — grows by roughly delay_ms.  delay_ms must stay well below
    dispatcher.enqueueRpcTimeoutMs (default 5000) or the RPC deadline
    fires first.

    Assertions: end-to-end latency delta >= 1.2s at delay_ms=1500, the
    request still SUCCEEDS (delay, not failure), and latency recovers once
    the injection is cleared."""
    ops = ctx.ops()
    base = rid_base(ctx, "chaos")
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
    "chaos_inject_generate_delay",
    source="gap G6/G7: /inject type=generate_delay (prefill execution inflation, all modes)",
)
def inject_generate_delay(ctx: CaseContext):
    """generate_delay adds delay_ms to the prefill execution estimate
    (runPrefillBatch), so the first-output latency grows by roughly
    delay_ms in EVERY mode (unlike enqueue_delay, schedule() stays fast
    and only TTFT inflates).

    Assertions: TTFT delta >= 1.2s at delay_ms=1500, the request still
    SUCCEEDS, and TTFT recovers after the injection is cleared."""
    ops = ctx.ops()
    base = rid_base(ctx, "chaos")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        rid0 = ops.next_request_id(base)
        ttft_base, err0 = _measure_ttft(ops, rid0)
        if err0:
            return False, f"baseline request failed: {err0}"

        inject_type_all(ops, names, "generate_delay", delay_ms=1500)
        rid1 = ops.next_request_id(base)
        ttft_delayed, err1 = _measure_ttft(ops, rid1)

        clear_type_all(ops, names, "generate_delay")
        rid2 = ops.next_request_id(base)
        ttft_recovered, err2 = _measure_ttft(ops, rid2)

        delta = ttft_delayed - ttft_base
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 10.0
        )
        if ctx.mode != "batch":
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
# Admission-gate cases — smoke suite (functional taxonomy)
# ===========================================================================


@case(
    "gate_queue_depth_reject",
    modes=["batch"],
    source="gap G8: engine queue_depth admission gate (fast reject + recovery)",
    suite="smoke",
)
def gate_queue_depth(ctx: CaseContext):
    """Engine-side queue_depth gate: once every prefill holds >=1 slow
    pending request, the next enqueue is rejected FAST with "queue depth
    limit exceeded" (-> BATCH_DISPATCH_FAILED schedule response), NOT an
    unbounded pile-up; after the gate is lifted the occupiers finish and
    a fresh request succeeds with no inflight leak."""
    ops = ctx.ops()
    base = rid_base(ctx, "chaos")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"

    seed_pool: Optional[ThreadPoolExecutor] = None
    futures = []
    try:
        # Slow prefills keep the seed requests pending long enough.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=4000.0)
            inject_type(ops, n, "queue_depth", depth=1)

        # Fire seeds until EVERY prefill holds >=1 pending request, so the
        # probe request cannot dodge the gate by landing on a free engine.
        # With depth=1 some seeds may themselves be rejected — their errors
        # are expected and not part of the verdict.
        seed_pool = ThreadPoolExecutor(max_workers=8)
        deadline = time.monotonic() + 15.0
        while not _all_engines_busy(ops, names) and time.monotonic() < deadline:
            rid = ops.next_request_id(base)
            futures.append(
                seed_pool.submit(
                    ops.run_one_request,
                    rid,
                    input_len=512,
                    output_len=2,
                    stream_timeout_s=20.0,
                )
            )
            time.sleep(0.3)
        occupied = _all_engines_busy(ops, names)
        if not occupied:
            return False, "seeds never occupied every prefill engine"

        # Probe: must be fast-rejected with the queue-depth error.
        rid_probe = ops.next_request_id(base)
        t0 = time.monotonic()
        _, err_probe = ops.run_one_request(
            rid_probe, input_len=512, output_len=2, stream_timeout_s=20.0
        )
        reject_latency = time.monotonic() - t0

        # Lift the gate, let the seeds drain, then verify recovery.
        clear_type_all(ops, names, "queue_depth")
        for fut in futures:
            try:
                fut.result(timeout=30.0)
            except Exception:
                pass

        rid_after = ops.next_request_id(base)
        _, err_after = ops.run_one_request(
            rid_after, input_len=512, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
        )

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(ops, names)

        rejected = (
            err_probe is not None
            and "queue depth" in str(err_probe)
            and reject_latency < 3.0
        )
        passed = (
            occupied and rejected and err_after is None and inflight_ok and engine_clean
        )
        return passed, (
            f"all_engines_occupied={occupied}, "
            f"probe_rejected={rejected} "
            f"(latency={reject_latency:.2f}s, err={str(err_probe)[:80]}), "
            f"recovered={err_after is None}, "
            f"master_inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_inflight_clean={engine_clean}({engine_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "queue_depth")
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass
        if seed_pool is not None:
            seed_pool.shutdown(wait=True)


def _lru_spec(ctx: CaseContext) -> EnvSpec:
    """G10 env: 2P+2D with a tiny per-engine prefill LRU (4 blocks)."""
    return EnvSpec(
        label=f"gate_lru_{ctx.mode}",
        n_prefill=2,
        n_decode=2,
        perf=default_perf(),
        master_mode=ctx.mode,
        prefill_cache_blocks=4,
        decode_cache_blocks=4,
    )


@case(
    "gate_lru_eviction_affinity",
    source="gap G10: LRU prefix reuse + capacity eviction + affinity routing e2e",
    suite="smoke",
)
def gate_lru(ctx: CaseContext):
    """Drive the mock's per-engine MockLruBlockCache end to end:

    1. R1 primes [k1,k2] on its landing engine X — snapshot proves
       cache_keys >= 2 with zero evictions.
    2. R2 replays the SAME keys: master-side cache-status sync must route
       it back to X (S2-style affinity, one retry for sync lag).
    3. R3 replays the prefix [k1,k2] plus three fresh keys: five keys
       admitted into the capacity-4 LRU evict exactly the eldest block —
       snapshot proves evictions >= 1 and cache_keys capped at 4, and the
       prefix hit keeps R3 on X as well.
    """
    env = ctx.env_manager.ensure(_lru_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "chaos")

    def run(rid, keys, input_len):
        return ops.run_one_request(
            rid,
            input_len=input_len,
            output_len=2,
            block_keys=keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )

    try:
        keys_a = [base + 1, base + 2]
        rid1 = ops.next_request_id(base)
        addr1, err1 = run(rid1, keys_a, 2048)
        if err1:
            return False, f"R1 (prime) failed: {err1}"
        time.sleep(KV_CACHE_SYNC_WAIT_S)

        # Affinity: same keys must return to the priming engine.
        rid2 = ops.next_request_id(base)
        addr2, err2 = run(rid2, keys_a, 2048)
        if err2:
            return False, f"R2 (replay) failed: {err2}"
        affinity = addr1 == addr2
        if not affinity:
            # S2-style retry: the cache-status sync may lag one poll.
            time.sleep(KV_CACHE_SYNC_WAIT_S)
            rid2b = ops.next_request_id(base)
            addr2b, err2b = run(rid2b, keys_a, 2048)
            if err2b:
                return False, f"R2 retry failed: {err2b}"
            affinity = addr1 == addr2b

        addr_map = ops.addr_to_name()
        engine_x = addr_map.get(addr1, "?")
        snap = ops.snapshot_by_name()
        keys_after_prime = snap.get(engine_x, {}).get("cache_keys", 0)
        evictions_after_prime = snap.get(engine_x, {}).get("cache_evictions", 0)

        # Capacity pressure: prefix [k1,k2] + 3 fresh keys -> 5 admits into
        # a capacity-4 LRU -> exactly the eldest block evicted.
        keys_ext = keys_a + [base + 3, base + 4, base + 5]
        rid3 = ops.next_request_id(base)
        addr3, err3 = run(rid3, keys_ext, 4096)
        if err3:
            return False, f"R3 (pressure) failed: {err3}"
        time.sleep(0.5)  # admit lands at prefill completion
        snap = ops.snapshot_by_name()
        engine_z = addr_map.get(addr3, "?")
        keys_after_pressure = snap.get(engine_z, {}).get("cache_keys", 0)
        evictions_after_pressure = snap.get(engine_z, {}).get("cache_evictions", 0)
        prefix_affinity = addr3 == addr1

        prime_ok = keys_after_prime >= 2 and evictions_after_prime == 0
        eviction_ok = evictions_after_pressure >= 1 and keys_after_pressure <= 4
        passed = affinity and prime_ok and eviction_ok and prefix_affinity
        return passed, (
            f"engine_x={engine_x}, affinity_r2={affinity}, "
            f"after_prime: keys={keys_after_prime}, evictions={evictions_after_prime}, "
            f"pressure_landed_on={engine_z}, prefix_affinity_r3={prefix_affinity}, "
            f"after_pressure: keys={keys_after_pressure} (<=4), "
            f"evictions={evictions_after_pressure} (>=1)"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


def _slo_spec(ctx: CaseContext) -> EnvSpec:
    """G11a env: tight SLO deadline (scheduler.queueTimeoutMs=1500)."""
    return EnvSpec(
        label=f"gate_slo_{ctx.mode}",
        n_prefill=2,
        n_decode=2,
        perf=default_perf(),
        master_mode=ctx.mode,
        master_env={"FLEXLB_CONFIG": _gate_config(queue_timeout_ms=1500)},
    )


@case(
    "gate_slo_queue_deadline",
    modes=["batch"],
    source="gap G11: SLO queue deadline + kv_pressure admission (wait-then-expire)",
    suite="smoke",
)
def gate_slo_deadline(ctx: CaseContext):
    """SLO/KV admission: kv_pressure squeezes every prefill's
    availableKvCache to 0; the batcher's KV gate is a WAIT condition
    (admitAndDeliverCapacityFeasiblePrefix -> CapacityBlocked), so with
    scheduler.queueTimeoutMs=1500 the request FAILS with the deadline
    error around 1.5s — fast, terminal, surfaced to the client.  Also
    covers the kv_pressure injection type cross-process (gap G6/G7).

    Recovery: clear kv_pressure and a fresh request must succeed."""
    env = ctx.env_manager.ensure(_slo_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "chaos")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        inject_type_all(ops, names, "kv_pressure", tokens=MOCK_TOTAL_KV_TOKENS)
        time.sleep(1.0)  # master polls the squeezed worker status

        rid1 = ops.next_request_id(base)
        t0 = time.monotonic()
        _, err1 = ops.run_one_request(
            rid1, input_len=512, output_len=2, stream_timeout_s=12.0
        )
        fail_latency = time.monotonic() - t0

        clear_type_all(ops, names, "kv_pressure")
        time.sleep(1.0)  # status poll refreshes the KV budget

        rid2 = ops.next_request_id(base)
        _, err2 = ops.run_one_request(
            rid2, input_len=512, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
        )

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 20.0
        )
        err_text = str(err1 or "")
        deadline_typed = any(
            kw in err_text.lower()
            for kw in ("deadline", "expired", "exhaust", "8400", "8511", "8431")
        )
        rejected = (
            err1 is not None
            and fail_latency <= 8.0
            and fail_latency >= 1.0
            and deadline_typed
        )
        passed = rejected and err2 is None and inflight_ok
        return passed, (
            f"deadline_rejected={rejected} "
            f"(latency={fail_latency:.2f}s in [1.0, 8.0], err={err_text[:100]}), "
            f"recovered={err2 is None}, "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "kv_pressure")


def _capacity_spec(ctx: CaseContext) -> EnvSpec:
    """G11b env: global outstanding capacity of 2 under PRIORITY ordering."""
    return EnvSpec(
        label=f"gate_cap_{ctx.mode}",
        n_prefill=2,
        n_decode=2,
        perf=default_perf(),
        master_mode=ctx.mode,
        master_env={"FLEXLB_CONFIG": _gate_config(max_outstanding=2)},
    )


@case(
    "gate_master_capacity_reject",
    modes=["batch"],
    source="gap G11: master outstanding-capacity admission (RESOURCE_EXHAUSTED fast reject)",
    suite="smoke",
)
def gate_capacity(ctx: CaseContext):
    """Master-side unified admission: with
    capacity.maxOutstandingRequestsGlobal=2 and PRIORITY ordering, the
    submit path (PriorityScheduler.submit -> tryAcquireOutstandingPermit)
    fast-rejects every request beyond the global budget with
    RESOURCE_EXHAUSTED "master outstanding capacity exhausted" — a
    synchronous rejection, no queueing and no leak.  Once the in-flight
    occupants terminate, a sequential request must succeed again."""
    env = ctx.env_manager.ensure(_capacity_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "chaos")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        # Slow prefills keep the two admitted occupants inside the budget.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=4000.0)

        def run(rid: int):
            t0 = time.monotonic()
            _, err = ops.run_one_request(
                rid, input_len=512, output_len=2, stream_timeout_s=15.0
            )
            return err, time.monotonic() - t0

        rids = [ops.next_request_id(base) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, rids))
        rejected = [(e, t) for e, t in results if e is not None]
        served = [t for e, t in results if e is None]
        reject_types = sorted({str(e)[:70] for e, _ in rejected})
        reject_fast = all(t < 3.0 for _, t in rejected)
        reject_typed = all(
            any(
                kw in str(e).lower()
                for kw in ("outstanding", "exhaust", "resource", "8431")
            )
            for e, _ in rejected
        )

        for n in names:
            ops.set_perf(n, prefill_fixed_ms=100.0)
        rid5 = ops.next_request_id(base)
        _, err5 = ops.run_one_request(
            rid5, input_len=512, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
        )

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        passed = (
            1 <= len(rejected) <= 2
            and len(served) >= 2
            and reject_fast
            and reject_typed
            and err5 is None
            and inflight_ok
        )
        return passed, (
            f"served={len(served)}, rejected={len(rejected)} "
            f"(fast={reject_fast}, typed={reject_typed}, "
            f"types={reject_types[:2]}), "
            f"sequential_recovery={err5 is None}, "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass
