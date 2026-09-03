"""Admission-category cases: the admission-gate contract.

Theme: requests the gates REFUSE must fail fast, loudly and typed — never
hang, never vanish, never leak inflight state — and once the pressure is
lifted the system must recover.  Conversely, gates that are WAIT
conditions must park (never silently drop or reject under queue
pressure).  One gate per case (admission wave-2, 2026-09):

  admission_queue_depth_reject    engine-side queue_depth gate
                                  (EnqueueBatch entry): fast per-request
                                  "queue depth limit exceeded" rejection ->
                                  BATCH_DISPATCH_FAILED, not a silent
                                  pile-up; recovery after the gate lifts.
  admission_slo_queue_deadline    SLO queue deadline under kv_pressure:
                                  the batcher's KV gate is a WAIT condition,
                                  so a squeezed budget holds the request
                                  until scheduler.queueTimeoutMs expires it
                                  with a typed deadline error.
  admission_master_capacity_reject master outstanding-capacity permit
                                  (capacity.maxOutstandingRequestsGlobal):
                                  typed QUEUE_FULL fast reject (8502
                                  TooManyRequests / detail QUEUE_FULL) on
                                  the submit path; recovery once the
                                  occupants terminate.
  engine_prefill_concurrency_gate_park
                                  engine prefill-concurrency gate
                                  (maxPrefillConcurrency=1): a full batch
                                  window parks whole batches in the
                                  engine's prefillPendingQueue — no request
                                  is rejected; batches complete in order.
  engine_decode_hard_gate_unbounded_park
                                  engine decode hard gate
                                  (decodeMaxConcurrency=128): overflow
                                  parks unboundedly in decodePendingQueue —
                                  zero queue-pressure rejections; drains
                                  fully after the wave.
  admission_priority_incomer_reject
                                  PRIORITY incomer without preemption:
                                  when the admission capacity permit is
                                  exhausted the higher-priority incomer is
                                  fast-rejected typed 8431
                                  ("temporarily exhausted"); the lower-
                                  priority occupants finish unmolested.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..context import CaseContext, CaseDef, rid_base
from ..engine_ops import (
    clear_type_all,
    engine_inflight_clean,
    inject_type,
    inject_type_all,
)
from ..harness import AssertUtils, EnvSpec, admission_config, default_perf, wait_for

ADMISSION_CASES: list[CaseDef] = []

STREAM_TIMEOUT_S = 15.0
# Mock DEFAULT_TOTAL_KV_TOKENS — squeezing with this value drives
# availableKvCache to 0.
MOCK_TOTAL_KV_TOKENS = 6_291_456


def case(name: str, profiles=None, requires=None, source: str = ""):
    """Register into ADMISSION_CASES (category is always "admission")."""

    def deco(fn):
        ADMISSION_CASES.append(
            CaseDef(
                name=name,
                category="admission",
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


def _all_engines_busy(ops, names: list[str]) -> bool:
    snap = ops.snapshot_by_name()
    return all(
        snap.get(n, {}).get("waiting", 0) + snap.get(n, {}).get("running", 0) >= 1
        for n in names
    )


# ===========================================================================
# Engine-side queue-depth gate (gap G8)
# ===========================================================================


@case(
    "admission_queue_depth_reject",
    requires=["enqueue_batch"],
    source="gap G8: engine queue_depth admission gate (fast reject + recovery)",
)
def admission_queue_depth(ctx: CaseContext):
    """Engine-side queue_depth gate: once every prefill holds >=1 slow
    pending request, the next enqueue is rejected FAST with "queue depth
    limit exceeded" (-> BATCH_DISPATCH_FAILED schedule response), NOT an
    unbounded pile-up; after the gate is lifted the occupiers finish and
    a fresh request succeeds with no inflight leak.

    Profile semantics (v2, task #55): the gate is checked only at the
    engine's EnqueueBatch entry (BATCH dispatcher) — the
    GenerateStreamCall path never consults it — so
    requires=["enqueue_batch"] keeps the case to the BATCH-dispatch
    profiles (batch-window, single-batch).  The case runs on the shared
    default env (real per-profile config), so single-batch exercises the
    SINGLE decision axis on the same gate path.
    """
    ops = ctx.ops()
    base = rid_base(ctx, "admission")
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


# ===========================================================================
# SLO queue deadline under KV pressure (gap G11a)
# ===========================================================================


def _slo_spec(ctx: CaseContext) -> EnvSpec:
    """G11a env: tight SLO deadline (scheduler.queueTimeoutMs=1500)."""
    return EnvSpec(
        label=f"admission_slo_{ctx.profile}",
        n_prefill=2,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={"FLEXLB_CONFIG": admission_config(queue_timeout_ms=1500)},
    )


@case(
    "admission_slo_queue_deadline",
    profiles=["batch-window"],
    source="gap G11: SLO queue deadline + kv_pressure admission (wait-then-expire)",
)
def admission_slo_deadline(ctx: CaseContext):
    """SLO/KV admission: kv_pressure squeezes every prefill's
    availableKvCache to 0; the batcher's KV gate is a WAIT condition
    (admitAndDeliverCapacityFeasiblePrefix -> CapacityBlocked), so with
    scheduler.queueTimeoutMs=1500 the request FAILS with the deadline
    error around 1.5s — fast, terminal, surfaced to the client.  Also
    covers the kv_pressure injection type cross-process (gap G6/G7).

    Recovery: clear kv_pressure and a fresh request must succeed.

    Profile semantics (v2, task #55): the KV gate + queue deadline apply
    to the scheduler queue regardless of the decision/dispatcher axes,
    but _slo_spec pins the legacy fault axes (PRIORITY + FIXED_WINDOW +
    BATCH) via FLEXLB_CONFIG — re-running under another --profile would
    execute the identical configuration, so the declaration stays
    batch-window (label honesty + regression efficiency).
    """
    env = ctx.env_manager.ensure(_slo_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
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


# ===========================================================================
# Master outstanding-capacity permit (gap G11b)
# ===========================================================================


def _capacity_spec(ctx: CaseContext) -> EnvSpec:
    """G11b env: global outstanding capacity of 2 under PRIORITY ordering."""
    return EnvSpec(
        label=f"admission_cap_{ctx.profile}",
        n_prefill=2,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={"FLEXLB_CONFIG": admission_config(max_outstanding=2)},
    )


@case(
    "admission_master_capacity_reject",
    profiles=["batch-window"],
    source=(
        "gap G11: master outstanding-capacity admission — typed QUEUE_FULL "
        "(8502 TooManyRequests) fast reject.  F6 verdict overturned: the "
        "reject IS typed (dedicated code + status_name + detail); the legacy "
        "assertion family ('outstanding'/'exhaust'/'resource'/'8431') "
        "matched zero tokens of the actual response payload."
    ),
)
def admission_master_capacity(ctx: CaseContext):
    """Master-side unified admission: with
    capacity.maxOutstandingRequestsGlobal=2 and PRIORITY ordering, the
    submit path (RequestRegistry.register -> tryAcquireOutstandingPermit)
    fast-rejects every request beyond the global budget with the typed
    QUEUE_FULL error — code 8502, error_message JSON
    {"status_name":"TooManyRequests","detail":"QUEUE_FULL"} — a
    synchronous, typed rejection, no queueing and no leak.  Once the
    in-flight occupants terminate, a sequential request must succeed
    again.

    F6 note (admission wave-2): the old docstring claimed
    RESOURCE_EXHAUSTED "master outstanding capacity exhausted" — that
    string family never matched the real payload (the 8431
    RESOURCE_EXHAUSTED path is the acceptance-limit / eviction gate, not
    the outstanding permit).  The master behaviour was always typed; the
    defect was the test's assertion family.

    Profile semantics (v2, task #55): the outstanding-capacity permit is
    taken on the master submit path for every delivery mode, but
    _capacity_spec pins the legacy fault axes (PRIORITY + FIXED_WINDOW +
    BATCH) via FLEXLB_CONFIG — re-running under another --profile would
    execute the identical configuration, so the declaration stays
    batch-window (label honesty + regression efficiency).
    """
    env = ctx.env_manager.ensure(_capacity_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        # Slow prefills keep the two admitted occupants inside the budget.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=4000.0)

        def run(rid: int):
            # Direct Schedule so the typed reject is asserted on the raw
            # response (code + error_message), not on run_one_request's
            # flattened text (which drops the code).
            t0 = time.monotonic()
            resp = ops.schedule(rid, input_len=512, output_len=2)
            if resp.code != 200 or not resp.success:
                return (
                    "reject",
                    resp.code,
                    str(resp.error_message),
                    (time.monotonic() - t0),
                )
            # batch-window profile: BATCH dispatch -> FetchResponse stream.
            handle = ops.start_stream(resp, rid)
            handle.wait_end(15.0)
            snap = handle.snap
            if snap.error or not snap.completed:
                return (
                    "serve_err",
                    resp.code,
                    (snap.error or "stream did not complete"),
                    (time.monotonic() - t0),
                )
            return "served", resp.code, None, time.monotonic() - t0

        rids = [ops.next_request_id(base) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, rids))
        rejected = [
            (code, msg, t) for kind, code, msg, t in results if kind == "reject"
        ]
        serve_failures = [
            (code, msg) for kind, code, msg, _ in results if kind == "serve_err"
        ]
        served = [t for kind, _, _, t in results if kind == "served"]
        reject_types = sorted({f"{code}:{msg[:60]}" for code, msg, _ in rejected})
        reject_fast = all(t < 3.0 for _, _, t in rejected)
        reject_typed = bool(rejected) and all(
            code == 8502
            and "toomanyrequests" in msg.lower()
            and "queue_full" in msg.lower()
            for code, msg, _ in rejected
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
            and not serve_failures
            and reject_fast
            and reject_typed
            and err5 is None
            and inflight_ok
        )
        return passed, (
            f"served={len(served)}, rejected={len(rejected)} "
            f"(fast={reject_fast}, typed_8502_queue_full={reject_typed}, "
            f"types={reject_types[:2]}), "
            f"serve_failures={serve_failures[:1]}, "
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


# ===========================================================================
# Engine prefill-concurrency gate (admission wave-2 W1)
# ===========================================================================


def _fire_request(ops, rid: int, fired: list, **kwargs) -> Optional[str]:
    """Schedule without consuming the stream — under BATCH dispatch the
    master-enqueued ledger entry stays live until the case drains it
    (the kv.py fire-and-forget pattern, kept local for parallel-edit
    decoupling like cancel.py's _schedule_with_priority)."""
    try:
        resp = ops.schedule(rid, **kwargs)
    except Exception as exc:
        return repr(exc)
    if resp.code != 200 or not resp.success:
        return f"schedule failed ({resp.code}): {resp.error_message}"
    fired.append((rid, resp))
    return None


def _drain_fired(ops, fired: list, wait_s: float = 60.0) -> list:
    """Consume every fired request to terminal state (cancel fallback).
    Returns [(rid, completed, err)] in fire order."""
    outcomes = []
    for rid, resp in fired:
        completed = False
        err = None
        try:
            handle = ops.start_stream(resp, rid)
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
        outcomes.append((rid, completed, err))
    return outcomes


def _prefill_park_spec(ctx: CaseContext) -> EnvSpec:
    """W1 env: 1 prefill (all batches land on one engine), default
    admission axes; dispatcher maxInflightBatchesPerPrefillWorker=4 (the
    build_flexlb_config default) lets several batches reach the engine
    while the engine's maxPrefillConcurrency=1 keeps only one running."""
    return EnvSpec(
        label=f"admit_prefill_park_{ctx.profile}",
        n_prefill=1,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={"FLEXLB_CONFIG": admission_config()},
    )


@case(
    "engine_prefill_concurrency_gate_park",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "admission wave-2 W1: engine prefill-concurrency gate "
        "(maxPrefillConcurrency park — wait condition, no reject)"
    ),
)
def engine_prefill_concurrency_gate_park(ctx: CaseContext):
    """Engine prefill-concurrency gate: the gate is a WAIT condition.

    Scenario: dedicated 1P+2D env.  The single prefill engine runs with
    the production-locked maxPrefillConcurrency=1 (no /set_perf override
    — the lock is the point); prefill_fixed_ms=3000 stretches each
    batch's execution window.  Four requests are fired ~0.4s apart —
    far beyond maxCollectionWaitMs=10, so each arrives as its OWN batch,
    and the dispatcher's maxInflightBatchesPerPrefillWorker=4 window
    lets all four EnqueueBatch deliveries reach the engine while batch
    #1 is still running.

    Behaviour: JavaMockEngineCluster.schedulePrefillCompletion admits
    batch #1 (activePrefillBatches 0 -> 1) and parks batches #2-4 whole
    in the unbounded prefillPendingQueue (max_waiting_batches default
    0 = unlimited) — a WAIT, not a rejection: no request-level error is
    surfaced.  As each running batch goes terminal the parked queue
    head is admitted FIFO.

    Expected (contract): every fired request schedules successfully
    (zero rejections — the park absorbs the pressure); a snapshot poll
    observes prefill_waiting_batches >= 1 (the park is real); after the
    drain every fired request reached its terminal as a COMPLETED
    stream (FIFO order), the park is empty (waiting == 0 and
    prefill_waiting_batches == 0), the master inflight ledger is clean
    and a fresh request succeeds (recovery).

    Prediction: expected to pass — engine-side park semantics were
    ported with the v2 mock (schedulePrefillCompletion) and the batch
    lease window (4) strictly exceeds the concurrency window (1), so
    the park is deterministic.  Risk: FIXED_WINDOW coalescing —
    mitigated by the 0.4s inter-fire gap (40x the collection window).
    """
    env = ctx.env_manager.ensure(_prefill_park_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    fired: list = []
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)

        fire_errors = []
        rids = []
        for _ in range(4):
            rid = ops.next_request_id(base)
            rids.append(rid)
            err = _fire_request(ops, rid, fired, input_len=512, output_len=2)
            if err is not None:
                fire_errors.append((rid, err))
            time.sleep(0.4)  # >> maxCollectionWaitMs: one batch per fire

        # Observe the park: batch #1 runs 3s, batches #2-4 sit in
        # prefillPendingQueue — poll before the first terminal.
        park_max = 0
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            snap = ops.snapshot_by_name()
            for n in names:
                info = snap.get(n, {})
                park_max = max(
                    park_max,
                    int(info.get("prefill_waiting_batches", 0)),
                    int(info.get("waiting", 0)),
                )
            if park_max >= 1:
                break
            time.sleep(0.2)

        outcomes = _drain_fired(ops, fired, wait_s=45.0)
        completed = [rid for rid, ok, _ in outcomes if ok]
        drain_errors = [(rid, err) for rid, ok, err in outcomes if not ok]

        # Park must settle back to empty after the drain.
        def park_empty() -> bool:
            snap = ops.snapshot_by_name()
            return all(
                int(snap.get(n, {}).get("prefill_waiting_batches", 0)) == 0
                and int(snap.get(n, {}).get("waiting", 0)) == 0
                for n in names
            )

        settled = wait_for(park_empty, 10.0, 0.2)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            not fire_errors
            and park_max >= 1
            and len(completed) == len(rids)
            and not drain_errors
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={len(rids)} (fire_errors={fire_errors[:1]}), "
            f"park_observed_max={park_max}, "
            f"completed={len(completed)}/{len(rids)} "
            f"(drain_errors={drain_errors[:1]}), "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass


# ===========================================================================
# Engine decode hard gate (admission wave-2 W2)
# ===========================================================================


DECODE_WAVE_REQUESTS = 150  # > 132 (master cap) > 128 (engine hard gate)


def _decode_park_spec(ctx: CaseContext) -> EnvSpec:
    """W2 env: 2P+1D (all decode traffic concentrates on one engine).

    The master-side decode routing cap is raised far above the engine's
    128 hard gate (maxEngineRequests 132 -> 5000) so the ENGINE gate is
    the only admission edge in play: every fired request is routed and
    delivered, and whatever overflows 128 running slots must park in the
    engine's decodePendingQueue instead of being bounced anywhere."""
    config = json.loads(admission_config())
    config["router"]["roles"]["decode"]["availability"]["maxEngineRequests"] = 5000
    return EnvSpec(
        label=f"admit_decode_park_{ctx.profile}",
        n_prefill=2,
        n_decode=1,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={"FLEXLB_CONFIG": json.dumps(config, separators=(",", ":"))},
    )


def _decode_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "decode"]


@case(
    "engine_decode_hard_gate_unbounded_park",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "admission wave-2 W2: engine decode hard gate "
        "(decodeMaxConcurrency=128 unbounded park — no queue-pressure reject)"
    ),
)
def engine_decode_hard_gate_unbounded_park(ctx: CaseContext):
    """Engine decode hard gate: unbounded park, never a queue-pressure
    rejection.

    Scenario: dedicated 2P+1D env with the master decode routing cap
    raised to 5000, so the ONLY admission edge is the engine's
    decodeMaxConcurrency=128 hard gate (the production waiting_streams_
    semantics).  decode_scale=10 stretches each decode step
    ((19.5 + 0.175 x running) ms x 10 ≈ 0.42s at running=128) so the
    backlog window is comfortably observable, and 150 requests are
    fired-and-forgotten (output_len=8 → 4 steps ≈ 1.7s per wave).

    Behaviour: scheduleDecodeCompletion admits the first 128
    TransferToDecode arrivals as running and parks every overflow in
    the UNBOUNDED decodePendingQueue (no cap, no rejection — unlike
    the prefill gate this queue never bounces a request under queue
    pressure).  As running slots free up, parked requests are admitted
    wave by wave.

    Expected (contract): all 150 Schedule calls succeed (zero rejections
    — the engine-side form of a waitable gate); a snapshot poll observes
    decode waiting >= 1 (the park is real); after the drain >= 95% of
    the fired requests completed their streams, the decode park is empty
    (waiting == 0), the master inflight ledger is clean and a fresh
    request succeeds (recovery).

    Prediction: expected to pass — the 128 gate and unbounded pending
    queue are direct v2 mock ports (scheduleDecodeCompletion) and the
    150 > 132 > 128 overshoot makes the park inevitable.  Drain budget:
    two decode waves ≈ 3.5s plus prefill ≈ 2s, well under the 45s
    per-stream cap; if a slow machine stretches waves the completion
    bar is the 95% ratio, not perfection.
    """
    env = ctx.env_manager.ensure(_decode_park_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    decode_engines = _decode_names(ops)
    if not decode_engines:
        return False, "no decode engines found"
    fired: list = []
    try:
        for n in decode_engines:
            ops.set_perf(n, decode_scale=10.0)

        fire_errors = []
        for i in range(DECODE_WAVE_REQUESTS):
            rid = ops.next_request_id(base)
            err = _fire_request(ops, rid, fired, input_len=512, output_len=8)
            if err is not None:
                fire_errors.append((rid, err))
            if (i + 1) % 25 == 0:
                time.sleep(0.05)  # tiny pacing, keeps batches flowing

        # Observe the park: running climbs to 128 while the remaining
        # arrivals accumulate in decodePendingQueue.
        waiting_max = 0
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            snap = ops.snapshot_by_name()
            for n in decode_engines:
                info = snap.get(n, {})
                waiting_max = max(waiting_max, int(info.get("waiting", 0)))
            if waiting_max >= 1:
                break
            time.sleep(0.2)

        outcomes = _drain_fired(ops, fired, wait_s=45.0)
        completed = sum(1 for _, ok, _ in outcomes if ok)
        drain_errors = [(rid, err) for rid, ok, err in outcomes if not ok]

        def decode_park_empty() -> bool:
            snap = ops.snapshot_by_name()
            return all(
                int(snap.get(n, {}).get("waiting", 0)) == 0 for n in decode_engines
            )

        settled = wait_for(decode_park_empty, 15.0, 0.3)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 60.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        completion_ratio = completed / DECODE_WAVE_REQUESTS
        passed = (
            not fire_errors
            and waiting_max >= 1
            and completion_ratio >= 0.95
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={DECODE_WAVE_REQUESTS} "
            f"(fire_errors={len(fire_errors)}, first={fire_errors[:1]}), "
            f"decode_waiting_max={waiting_max}, "
            f"completed={completed}/{DECODE_WAVE_REQUESTS} "
            f"({completion_ratio:.0%}, drain_errors={drain_errors[:2]}), "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for n in decode_engines:
                ops.set_perf(n, decode_scale=1.0)
        except Exception:
            pass


# ===========================================================================
# PRIORITY incomer fast reject without preemption (admission wave-2 W3)
# ===========================================================================


def _schedule_with_priority(ops, request_id: int, priority: int, **kwargs):
    """Schedule RPC carrying an explicit priority (proto field 14) — the
    same local copy as cancel.py's (protobuf messages are mutable;
    engine_ops is owned by another agent in this wave)."""
    req = ops.build_schedule_request(request_id, **kwargs)
    req.priority = priority
    stub = ops.schedule_pb2_grpc.FlexlbServiceStub(ops._channel(ops.master_target()))
    return stub.Schedule(req, timeout=30.0)


def _incomer_spec(ctx: CaseContext) -> EnvSpec:
    """W3 env: 1P+1D, PRIORITY ordering, NO preemption block and the
    acceptance-limit door tightened to ONE permit
    (scheduler.lifecycle.maxDeliveredNotAcceptedRequestsGlobal=1).

    The decode routing cap stays at the template default (132) so the
    incomer's route comes back ACQUIRED and tryPublish reaches the
    acceptance-limit gate — where the typed 8431 fast reject lives
    (RequestScheduler.completeAcceptanceLimit).  No preemption block is
    emitted (build_flexlb_config never writes one), so
    EvictionManager.tryAdmit is a no-op — the no-preemption complement
    of cancel_preemption_victim."""
    config = json.loads(admission_config(max_delivered_not_accepted=1))
    return EnvSpec(
        label=f"admit_incomer_{ctx.profile}",
        n_prefill=1,
        n_decode=1,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={"FLEXLB_CONFIG": json.dumps(config, separators=(",", ":"))},
    )


@case(
    "admission_priority_incomer_reject",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "admission wave-2 W3: PRIORITY incomer fast reject — acceptance "
        "permit exhausted, no preemption fallback (8431 typed reject)"
    ),
)
def admission_priority_incomer_reject(ctx: CaseContext):
    """PRIORITY incomer without preemption: typed 8431 fast reject.

    Scenario: dedicated 1P+1D env, PRIORITY ordering, NO preemption
    block (allowedVictimStages unset — EvictionManager.tryAdmit is a
    no-op) and lifecycle.maxDeliveredNotAcceptedRequestsGlobal=1, so
    exactly one admission permit exists.  A low-priority victim
    (priority 30, output_len=200 — decode runs ~1.5s) is scheduled
    first and holds the single permit; once it is RUNNING on decode a
    higher-priority incomer (priority 70, output_len=2) arrives.

    Behaviour: the incomer's route selection SUCCEEDS (decode routing
    capacity is far from exhausted at 132), so QueueRouteAdmission
    .tryPublish reaches the decode-acceptance permit acquisition —
    which fails against the exhausted global limit
    (AcceptanceLimitReached) and RequestScheduler.completeAcceptanceLimit
    completes the future synchronously with
    AdmissionFailure.resourceExhausted().  With no preemption block
    there is no eviction path to steal the victim's slot: capacity
    trouble for a higher priority cannot be solved by force.

    Expected (contract): the incomer's Schedule RPC returns FAST (< 3s)
    with code 8431 and an error_message containing "admission capacity
    is temporarily exhausted" — typed, synchronous, no hang; the victim
    is NOT preempted (its stream completes normally, no 8429 anywhere);
    after the victim terminates the permit is released, so a fresh
    request succeeds again (recovery); master inflight and engine
    ledgers drain clean.

    Prediction: expected to pass — the acceptance-limit path is the
    same completeAcceptanceLimit producer the fault family already
    exercises; the only novel wiring is the single-permit limit and
    the priority-carrying Schedule.  Complement of
    cancel_preemption_victim: preemption ON there (victim 8429,
    incomer wins) vs OFF here (victim lives, incomer 8431).
    """
    env = ctx.env_manager.ensure(_incomer_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    decode_engines = _decode_names(ops)
    if not decode_engines:
        return False, "no decode engines found"
    victim_handle = None
    try:
        # Victim takes the single admission permit (held until terminal).
        victim_rid = ops.next_request_id(base)
        victim_keys = [victim_rid * 100 + 1]
        victim_resp = _schedule_with_priority(
            ops,
            victim_rid,
            30,
            input_len=512,
            output_len=200,
            block_keys=victim_keys,
        )
        if victim_resp.code != 200 or not victim_resp.success:
            return False, (f"victim schedule failed: {victim_resp.error_message}")
        victim_input = (
            None
            if victim_resp.enqueued_by_master
            else ops.build_generate_input(
                victim_rid,
                input_len=512,
                output_len=200,
                block_keys=victim_keys,
            )
        )
        victim_handle = ops.start_stream(victim_resp, victim_rid, input_pb=victim_input)

        # Victim must be past prefill and RUNNING on decode before the
        # incomer arrives — otherwise the probe would race the transfer.
        def victim_running() -> bool:
            snap = ops.snapshot_by_name()
            return any(
                int(snap.get(n, {}).get("running", 0)) >= 1 for n in decode_engines
            )

        running = wait_for(victim_running, 10.0, 0.1)
        if not running:
            return False, "victim never reached RUNNING on decode"

        # Incomer: fast typed reject on the acceptance-limit gate.
        incomer_rid = ops.next_request_id(base)
        t0 = time.monotonic()
        incomer_resp = _schedule_with_priority(
            ops,
            incomer_rid,
            70,
            input_len=512,
            output_len=2,
            block_keys=[incomer_rid * 100 + 1],
        )
        reject_latency = time.monotonic() - t0
        incomer_msg = str(incomer_resp.error_message)

        # Victim finishes unmolested (no preemption -> no 8429).
        victim_ended = victim_handle.wait_end(30.0)
        victim_completed = (
            victim_ended
            and victim_handle.snap.completed
            and not victim_handle.snap.error
        )

        # Permit released on terminal: a fresh request must succeed.
        recovery_ok, recovery_msg = ops.verify_recovery()
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _prefill_names(ops) + decode_engines, 15.0
        )

        rejected_typed = (
            incomer_resp.code == 8431 and "temporarily exhausted" in incomer_msg.lower()
        )
        rejected_fast = reject_latency < 3.0
        passed = (
            rejected_typed
            and rejected_fast
            and victim_completed
            and recovery_ok
            and inflight_ok
            and engine_clean
        )
        return passed, (
            f"incomer_rejected={rejected_typed} "
            f"(code={incomer_resp.code}, latency={reject_latency:.2f}s, "
            f"msg={incomer_msg[:80]}), "
            f"victim_completed={victim_completed} "
            f"(outputs={len(victim_handle.snap.outputs)}), "
            f"recovery={recovery_msg}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if victim_handle is not None:
            victim_handle.cancel()
