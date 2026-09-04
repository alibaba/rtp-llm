"""Admission-category cases: the admission-gate contract.

Theme: requests the gates REFUSE must fail fast, loudly and typed — never
hang, never vanish, never leak inflight state — and once the pressure is
lifted the system must recover.  Conversely, gates that are WAIT
conditions must park (never silently drop or reject under queue
pressure).  One gate per case (admission wave-2 + wave-3, 2026-09):

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
  admission_batcher_queue_capacity_park
                                  master batcher-queue capacity gate
                                  (scheduler.capacity
                                  maxWaitingRequestsPerPrefillWorker=2):
                                  overflow past the dispatcher lease window
                                  fills the queue, then parks in
                                  placementWaiters — waitable, never a fast
                                  reject; FIFO drain once seats release.
  admission_batcher_queue_deadline
                                  the same batcher-queue gate under
                                  scheduler.queueTimeoutMs=1500: parked
                                  requests expire with the typed 8511
                                  BATCH_SLO_EXPIRED deadline — the same
                                  code, different trigger source than
                                  admission_slo_queue_deadline's KV gate.
  admission_placement_pool_wait
                                  prefill placement pool gate (router
                                  availability maxPendingRequests=1): the
                                  second arrival parks until the pool
                                  occupant terminates, then retries
                                  successfully — strictly after it.
  admission_engine_waiting_batch_cap_reject
                                  engine waiting-batch cap gate
                                  (prefill.max_waiting_batches=1 via
                                  /set_perf): with the cap saturated
                                  (1 running + 1 queued) the next batch
                                  is whole-batch REJECTED with the
                                  backpressure error — fast reject, not
                                  a park; relaxing the cap admits the
                                  next batch under the same pressure.
  admission_engine_kv_lack_mem_fast_reject
                                  engine prefill KV block-pool gate
                                  (KV v2 BlockLease admission): on a
                                  17-block pool two 8-block leases
                                  saturate it and the third 8-block
                                  request is synchronously rejected
                                  602 LACK_MEM in the EnqueueBatch ack
                                  — no park; lease hand-back on
                                  completion restores the pool.
  engine_prefill_token_budget_split
                                  engine-internal dual-budget prefill
                                  regroup (#8): a master batch whose
                                  total logical tokens (sum of
                                  computeTokens + hitTokens) exceed
                                  prefill.max_batch_tokens is split
                                  prefix/tail — every member completes,
                                  the ledger closes per request and the
                                  executed-batch counters reflect the
                                  regrouped shape (2 batches / 4 reqs).
  engine_prefill_token_budget_split_fifo
                                  the same split pinning ORDER: tail
                                  members finish strictly after the
                                  prefix (engine lifecycle end_ms,
                                  >1s apart) — arrival order survives
                                  the regroup.
  engine_prefill_token_budget_boundary
                                  a batch exactly AT the token budget
                                  (==) executes verbatim — one batch,
                                  no split, no park.
  engine_prefill_regroup_disabled_verbatim
                                  prefill.max_batch_tokens=0 AND
                                  prefill.max_batch_requests=0 disable
                                  the regroup entirely: the master
                                  batch executes as-is (legacy pre-#8
                                  behaviour).
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
from ..harness import (
    AssertUtils,
    EnvSpec,
    admission_config,
    build_flexlb_config,
    default_perf,
    http_get_json,
    wait_for,
)

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
        # task #107 fix (#20): the recovery verdict used to swallow err5 —
        # a failed recovery had NO visible cause.  Surface the raw error
        # (resp code + message) inside the detail so the failure is
        # diagnosable from the report alone.
        err5_detail = f" err5={str(err5)[:120]!r}" if err5 else ""

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
            f"sequential_recovery={err5 is None}{err5_detail}, "
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
    ((19.5 + 0.175 x running) ms x 10 ≈ 0.42s at running=128) and
    output_len=32 keeps every request ~13 decode steps (≈ 3.1s) in the
    engine, so the arrival rate (≈ 25 requests per ~0.22s prefill batch)
    sustains a decode concurrency far above 128 — the hard gate fills
    and the overflow parks.  (task #107 fix (#15): with output_len=8 the
    ~0.96s decode residency balanced the ~0.22s batch cadence at ≈ 109
    concurrent — the gate never filled and decode_waiting stayed 0;
    lengthening the residency makes the breakthrough deterministic.)

    Behaviour: scheduleDecodeCompletion admits the first 128
    TransferToDecode arrivals as running and parks every overflow in
    the UNBOUNDED decodePendingQueue (no cap, no rejection — unlike
    the prefill gate this queue never bounces a request under queue
    pressure).  As running slots free up, parked requests are admitted
    wave by wave.

    Expected (contract): all 150 Schedule calls succeed (zero rejections
    — the engine-side form of a waitable gate); a snapshot poll observes
    decode waiting >= 1 (the park is real — running_max is recorded
    alongside as the breakthrough diagnostic); after the drain >= 95% of
    the fired requests completed their streams, the decode park is empty
    (waiting == 0), the master inflight ledger is clean and a fresh
    request succeeds (recovery).

    Prediction: expected to pass — the 128 gate and unbounded pending
    queue are direct v2 mock ports (scheduleDecodeCompletion) and the
    150 > 5000(cap) > 128(gate) overshoot with the stretched residency
    makes the park inevitable.  Drain budget: ~4800 decode tokens at the
    ≈ 794 tok/s full-gate rate ≈ 6s plus ramp and the 22-request backlog
    tail, well under the 45s per-stream cap; if a slow machine stretches
    waves the completion bar is the 95% ratio, not perfection.
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
            err = _fire_request(ops, rid, fired, input_len=512, output_len=32)
            if err is not None:
                fire_errors.append((rid, err))
            if (i + 1) % 25 == 0:
                time.sleep(0.05)  # tiny pacing, keeps batches flowing

        # Observe the park across the full fill window (no early exit:
        # waiting_max and running_max are both peak gauges — the pair
        # proves the gate filled and overflowed, which a waiting-only
        # early break would truncate).
        waiting_max = 0
        running_max = 0
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            snap = ops.snapshot_by_name()
            for n in decode_engines:
                info = snap.get(n, {})
                waiting_max = max(waiting_max, int(info.get("waiting", 0)))
                running_max = max(running_max, int(info.get("running", 0)))
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
            f"decode_waiting_max={waiting_max}, decode_running_max={running_max} "
            f"(gate=128 breakthrough diagnostic), "
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


# ===========================================================================
# Master batcher-queue capacity gate (admission wave-2 A5)
# ===========================================================================


BQ_PARK_REQUESTS = 7  # > lease window (4) + queue capacity (2): the 7th parks
BQ_DEADLINE_REQUESTS = 8  # 4 leases + 2 queue seats + 2 placementWaiters
BQ_DEADLINE_MS = 1500


def _master_side_parked(ops, prefill_names, decode_names) -> tuple:
    """Requests live on the master ledger but absent from every engine.

    The A5-vs-W1 discriminator: a request parked on the ENGINE shows up
    in the engine snapshot's waiting/running, while a request parked on
    the MASTER (batcher queue depth / placementWaiters) only inflates
    the scheduler ledger — nothing else can see it.
    """
    data = http_get_json(f"{_master_http(ops)}/rtp_llm/inflight_status", timeout=5)
    if data is None:
        return -1, "inflight_status unavailable"
    sched = int(data.get("scheduler_inflight", 0))
    snap = ops.snapshot_by_name()

    def live(n: str) -> int:
        info = snap.get(n, {})
        return int(info.get("waiting", 0)) + int(info.get("running", 0))

    engine_live = sum(live(n) for n in prefill_names)
    engine_live += sum(live(n) for n in decode_names)
    return sched - engine_live, f"sched={sched}, engine_live={engine_live}"


def _fire_tracked(ops, rid: int, fired: list, **kwargs) -> Optional[str]:
    """Schedule AND immediately open the response stream, recording the
    fire instant so per-request terminal timing (completion order,
    deadline expiry) is measurable — the W3 victim-handle pattern,
    batched.  Fire errors are reported like _fire_request's."""
    try:
        resp = ops.schedule(rid, **kwargs)
    except Exception as exc:
        return repr(exc)
    if resp.code != 200 or not resp.success:
        return f"schedule failed ({resp.code}): {resp.error_message}"
    try:
        handle = ops.start_stream(resp, rid)
    except Exception as exc:
        return repr(exc)
    fired.append((rid, handle, time.monotonic()))
    return None


def _await_tracked(fired: list, wait_s: float = 45.0) -> list:
    """Concurrently await every tracked stream.

    Returns [(rid, fire_t, end_t, completed, err)] in fire order, with
    fire/end as monotonic instants so callers can assert per-request
    timing.  Unfinished streams are cancelled (drain hygiene)."""

    def _one(item):
        rid, handle, t0 = item
        try:
            ended = handle.wait_end(wait_s)
            completed = ended and handle.snap.completed and not handle.snap.error
            err = (
                None if completed else (handle.snap.error or "stream did not complete")
            )
        except Exception as exc:
            completed, err = False, repr(exc)
        if not completed:
            try:
                handle.cancel()
            except Exception:
                pass
        return (rid, t0, time.monotonic(), completed, err)

    if not fired:
        return []
    with ThreadPoolExecutor(max_workers=len(fired)) as pool:
        return list(pool.map(_one, fired))


def _batcher_queue_spec(ctx: CaseContext, queue_timeout_ms: int) -> EnvSpec:
    """A5 env: 1 prefill (a single batcher queue), the legacy fault axes
    (PRIORITY + FIXED_WINDOW + BATCH), with the batcher waiting-queue
    capacity tightened to TWO (scheduler.capacity
    maxWaitingRequestsPerPrefillWorker=2 — the Java default is 1024).

    The dispatcher lease window stays at the template default
    (maxInflightBatchesPerPrefillWorker=4), so under slow prefills the
    first four fires occupy engine-side batch leases, the next two fill
    the master batcher queue to its capacity ceiling and every later
    fire meets the capacity gate (Blocked -> placementWaiters)."""
    suffix = "deadline" if queue_timeout_ms < 60_000 else "park"
    return EnvSpec(
        label=f"admit_bq_{suffix}_{ctx.profile}",
        n_prefill=1,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={
            "FLEXLB_CONFIG": admission_config(
                queue_timeout_ms=queue_timeout_ms,
                max_waiting_requests_per_prefill_worker=2,
            )
        },
    )


@case(
    "admission_batcher_queue_capacity_park",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "admission wave-2 A5: master batcher-queue capacity gate "
        "(maxWaitingRequestsPerPrefillWorker park — waitable, no fast reject)"
    ),
)
def admission_batcher_queue_capacity_park(ctx: CaseContext):
    """Master batcher-queue capacity gate: the gate is a WAIT condition.

    Scenario: dedicated 1P+2D env with the batcher waiting-queue capacity
    tightened to 2 (scheduler.capacity.maxWaitingRequestsPerPrefillWorker
    — the Java default is 1024); prefill_fixed_ms=3000 stretches each
    batch.  Seven requests are fired 0.4s apart (each its own batch, 40x
    the 10ms collection window): the dispatcher lease window
    (maxInflightBatchesPerPrefillWorker=4) carries fires 1-4 onto the
    engine (1 running + 3 engine-side pending), fires 5-6 fill the master
    batcher queue to its capacity-2 ceiling, and fire 7 finds the queue
    full.

    Behaviour: WorkerBatcher.offer reports FULL at queue depth >= 2, so
    QueueRouteAdmission.tryPublish returns Blocked and RequestScheduler
    parks the request in PlacementWaitRegistry — a WAIT, never a
    rejection.  Each engine-side batch terminal releases a lease seat;
    the committed queue-head delivery then fires
    signalPlacementCapacityChanged, which wakes the parked waiter to
    retry successfully.  The queue drains FIFO.

    Expected (contract): all seven schedules succeed (zero fast
    rejects); during the pressure window the master-side parked count
    (scheduler ledger minus engine-live requests) is >= 1 — the A5
    discriminator: the park lives on the MASTER side while W1's
    engine_prefill_concurrency_gate_park observes it in the ENGINE's
    prefillPendingQueue; every request reaches its terminal as a
    COMPLETED stream, in fire order with end times NON-DECREASING and a
    serialized drain span (max-min) >= 12s across the wave (task #107
    fix (#16): the engine executes one 3s batch at a time across ~6
    batches, so strictly-increasing per-request ends with >= 1.2s gaps
    is the wrong invariant — when a lease seat frees, FIXED_WINDOW
    coalesces the queued waiters (fires 5-6) into ONE batch whose
    members terminate together; the correct serialization proof is the
    non-decreasing order plus the total span, with min_gap kept as an
    observation only); after the drain the engine park is empty, the
    master inflight ledger is clean and a fresh request succeeds
    (recovery).

    Prediction: expected to pass — the FULL branch of enqueueUnderLock
    and the PlacementWaitRegistry retry loop are the same wait-
    condition machinery the KV gate already exercises, and the ~6x3s
    FIFO drain (~18s worst) stays well under the 60s default
    queueTimeoutMs.  Risk: FIXED_WINDOW coalescing — now part of the
    contract (same-batch members terminate together), no longer a
    flake source.
    """
    env = ctx.env_manager.ensure(_batcher_queue_spec(ctx, queue_timeout_ms=60_000))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    decode_names = _decode_names(ops)
    if not names:
        return False, "no prefill engines found"
    fired: list = []
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)

        fire_errors = []
        for _ in range(BQ_PARK_REQUESTS):
            rid = ops.next_request_id(base)
            err = _fire_tracked(ops, rid, fired, input_len=512, output_len=2)
            if err is not None:
                fire_errors.append((rid, err))
            time.sleep(0.4)  # >> maxCollectionWaitMs: one batch per fire

        # Observe the master-side park while the pressure holds: ledger
        # entries beyond the engine-live set are parked on the master
        # (batcher queue / placementWaiters), never on the engine.
        parked_max = -1
        parked_detail = "not observed"
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            parked, detail = _master_side_parked(ops, names, decode_names)
            if parked > parked_max:
                parked_max, parked_detail = parked, detail
            if parked_max >= 1:
                break
            time.sleep(0.2)

        outcomes = _await_tracked(fired, wait_s=45.0)
        completed = [rid for rid, _, _, ok, _ in outcomes if ok]
        failures = [(rid, err) for rid, _, _, ok, err in outcomes if not ok]
        ends = [end for _, _, end, _, _ in outcomes]
        # Batch-aware FIFO (task #107 fix #16): same-batch members (the
        # coalesced fires 5-6) terminate together — order is non-
        # decreasing, not strictly increasing; serialization is proven
        # by the drain span (~6 batches x 3s), not per-request gaps.
        fifo_ordered = all(ends[i] <= ends[i + 1] for i in range(len(ends) - 1))
        fifo_serialized = (max(ends) - min(ends)) >= 12.0 if ends else False

        def engine_park_empty() -> bool:
            snap = ops.snapshot_by_name()
            return all(
                int(snap.get(n, {}).get("prefill_waiting_batches", 0)) == 0
                and int(snap.get(n, {}).get("waiting", 0)) == 0
                for n in names
            )

        settled = wait_for(engine_park_empty, 10.0, 0.2)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            not fire_errors
            and parked_max >= 1
            and len(completed) == BQ_PARK_REQUESTS
            and not failures
            and fifo_ordered
            and fifo_serialized
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={BQ_PARK_REQUESTS} (fire_errors={fire_errors[:1]}), "
            f"master_side_parked_max={parked_max} ({parked_detail}), "
            f"completed={len(completed)}/{BQ_PARK_REQUESTS} "
            f"(failures={failures[:1]}), "
            f"fifo_ordered={fifo_ordered}, fifo_serialized={fifo_serialized} "
            f"(span={max(ends) - min(ends):.2f}s, "
            f"min_gap={min((ends[i + 1] - ends[i] for i in range(len(ends) - 1)), default=0.0):.2f}s), "
            f"engine_park_settled_empty={settled}, "
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


@case(
    "admission_batcher_queue_deadline",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "admission wave-2 A5: batcher-queue gate deadline — the same "
        "BATCH_SLO_EXPIRED (8511) terminal as admission_slo_queue_deadline, "
        "with the trigger source moved from the KV gate to the batcher "
        "queue capacity gate (same code, different source — the deadline "
        "classification must stay uniform)"
    ),
)
def admission_batcher_queue_deadline(ctx: CaseContext):
    """Batcher-queue gate under an SLO deadline: park, then typed 8511.

    Scenario: the A5 env (batcher queue capacity 2, dispatcher lease
    window 4, prefill_fixed_ms=3000) with
    scheduler.queueTimeoutMs=1500.  Eight requests are fired 0.15s apart
    — fires 1-4 reach the engine and are delivery-confirmed before their
    deadline could fire (the request deadline detaches at
    markDeliveryConfirmed); fires 5-6 sit in the batcher queue, fires
    7-8 park in placementWaiters behind the capacity gate.

    Behaviour: every parked/queued request's absolute expiration
    (admissionTimeMs + queueTimeoutMs) fires while it still waits,
    completing its future with the typed BATCH_SLO_EXPIRED error (8511)
    via RequestSlot.cancelForDeadline — the same producer
    admission_slo_queue_deadline exercises from the KV gate.  Expired
    requests are removed from the batcher queue / placementWaiters and
    the scheduler ledger synchronously, so nothing dangles; the four
    delivered requests finish their 3s batches unmolested.

    Dual-form contract (task #107 fix #17): the deadline terminal
    surfaces on EITHER of two channels, and both are the SAME correct
    arrival —  (1) rpc_reject: a waiter parked in placementWaiters
    whose Schedule RPC is still waiting for its placement answer when
    the expiration fires gets the typed reject ON THE RPC (code != 200,
    8511-family text — the same synchronous-reject surface Tara
    accepted in admission_engine_waiting_batch_cap_reject);  (2)
    stream_terminal: a request whose RPC already returned (it sits in
    the batcher queue) gets the typed error as its stream's terminal.
    Which fires land in which form is a timing race the test must NOT
    pin — every one of fires 5-8 must arrive on one of the two forms,
    correctly typed, within the window.  Fires 1-4 are delivered and
    must complete normally on both channels.

    Expected (contract): fires 1-4 fire successfully, open their
    streams and complete normally; during the pre-expiry window the
    master-side parked count is >= 1 (the overflow wave is parked, not
    rejected); every one of fires 5-8 terminates with the deadline
    error family ("deadline"/"expired"/"exhaust"/"8400"/"8511"/"8431"
    — the same assertion family as the KV-gate deadline case, asserting
    the classification uniformity) within 1.0-5.0s of its fire, on
    either form; after the wave the master inflight ledger is clean and
    a fresh request on the relieved gate succeeds (recovery).

    Prediction: expected to pass — the deadline path is deadline-error
    type BATCH_SLO_EXPIRED installed at register and only detached by
    delivery confirmation, and the 0.15s fire cadence parks the whole
    overflow wave ~0.9s before the earliest expiry (fire 5 dies at
    2.10s, the first engine terminal is 3.0s — no wake-up race).  Risk:
    a late delivery acknowledgement could rescue a queued request —
    the 4-seat lease window plus 3s prefills make that a 0.9s-margin
    impossibility.
    """
    env = ctx.env_manager.ensure(
        _batcher_queue_spec(ctx, queue_timeout_ms=BQ_DEADLINE_MS)
    )
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    decode_names = _decode_names(ops)
    if not names:
        return False, "no prefill engines found"
    fired: list = []
    rpc_rejects: list = []  # form (1): deadline typed on the Schedule RPC
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)

        # Dual-form fire loop: fires 1-4 must DELIVER (fire + stream
        # opened); fires 5-8 may surface the deadline on either form —
        # the RPC reject (park waiter expires while its Schedule RPC
        # still waits for a placement answer) or the stream terminal
        # (queued request expires later).  fire_errors now records only
        # real failures (RPC exception / stream-open failure), never a
        # typed deadline reject.
        fire_errors = []
        for _ in range(BQ_DEADLINE_REQUESTS):
            rid = ops.next_request_id(base)
            t_call = time.monotonic()
            try:
                resp = ops.schedule(rid, input_len=512, output_len=2)
            except Exception as exc:
                fire_errors.append((rid, repr(exc)))
                time.sleep(0.15)
                continue
            if resp.code != 200 or not resp.success:
                rpc_rejects.append(
                    (
                        rid,
                        t_call,
                        time.monotonic(),
                        resp.code,
                        str(resp.error_message),
                    )
                )
            else:
                try:
                    handle = ops.start_stream(resp, rid)
                except Exception as exc:
                    fire_errors.append((rid, repr(exc)))
                    time.sleep(0.15)
                    continue
                fired.append((rid, handle, t_call))
            time.sleep(0.15)  # parks the whole wave before any expiry

        # Pre-expiry observation: the overflow wave (fires 5-8) is parked
        # on the MASTER — ledger-live but engine-absent — and NOT rejected.
        # Both forms count: an RPC-rejected waiter parks in
        # placementWaiters with its Schedule RPC still outstanding.
        parked_max = -1
        parked_detail = "not observed"
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            parked, detail = _master_side_parked(ops, names, decode_names)
            if parked > parked_max:
                parked_max, parked_detail = parked, detail
            if parked_max >= 1:
                break
            time.sleep(0.2)

        outcomes = _await_tracked(fired, wait_s=30.0)
        delivered = outcomes[:4]
        delivered_ok = len(delivered) == 4 and all(
            ok and err is None for _, _, _, ok, err in delivered
        )
        # Form (2): stream-terminal deadline errors among the overflow
        # wave that opened a stream (queued in the batcher queue).
        stream_wave = outcomes[4:]

        def _deadline_typed(text: str) -> bool:
            lowered = text.lower()
            return any(
                kw in lowered
                for kw in (
                    "deadline",
                    "expired",
                    "exhaust",
                    "8400",
                    "8511",
                    "8431",
                )
            )

        wave_ok = []
        wave_details = []
        for rid, t_call, t_end, code, msg in rpc_rejects:
            typed = code == 8511 or _deadline_typed(msg)
            in_window = 1.0 <= (t_end - t_call) <= 5.0
            wave_ok.append(typed and in_window)
            wave_details.append(f"rpc:{code}:{msg[:50]}@{t_end - t_call:.2f}s")
        for rid, t0, end, ok, err in stream_wave:
            text = str(err or "")
            typed = _deadline_typed(text)
            in_window = 1.0 <= (end - t0) <= 5.0
            wave_ok.append(typed and in_window and not ok)
            wave_details.append(f"stream:{text[:50]}@{end - t0:.2f}s")
        all_deadline = len(wave_ok) == 4 and all(wave_ok)

        # Deadline death removes the queue/waiter/ledger entries; the
        # four delivered batches drain normally.  Relieve the gate and
        # verify a fresh request succeeds.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=100.0)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            not fire_errors
            and parked_max >= 1
            and delivered_ok
            and all_deadline
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={BQ_DEADLINE_REQUESTS} (fire_errors={fire_errors[:1]}), "
            f"deadline_forms=rpc_reject:{len(rpc_rejects)}"
            f"/stream_terminal:{len(stream_wave)}, "
            f"master_side_parked_max={parked_max} ({parked_detail}), "
            f"delivered_completed={delivered_ok}, "
            f"deadline_typed={all_deadline} (details={wave_details}), "
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
# Master placement pool gate (admission wave-2 A4)
# ===========================================================================


def _pool_wait_spec(ctx: CaseContext) -> EnvSpec:
    """A4 env: 1 prefill, the placement pool tightened to ONE seat
    (router.roles.prefill.availability.maxPendingRequests=1 — the Java
    default is 64; the harness template default of 100000 is overridden
    here explicitly).

    The batcher queue capacity stays at the Java default (1024) and the
    dispatcher lease window at the template default (4): the placement
    pool is the ONLY admission edge in play, so a second arrival while
    the single seat is owned parks in placementWaiters instead of
    stacking anywhere else."""
    return EnvSpec(
        label=f"admit_pool_{ctx.profile}",
        n_prefill=1,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={"FLEXLB_CONFIG": admission_config(prefill_max_pending_requests=1)},
    )


@case(
    "admission_placement_pool_wait",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "admission wave-2 A4: prefill placement pool gate "
        "(router availability maxPendingRequests park — waitable, "
        "pool-release wakeup)"
    ),
)
def admission_placement_pool_wait(ctx: CaseContext):
    """Prefill placement pool gate: wait for the pool, never fast-reject.

    Scenario: dedicated 1P+2D env with the prefill placement pool
    capped at a single seat (router.roles.prefill.availability
    maxPendingRequests=1) and prefill_fixed_ms=5000.  Request A is fired
    first and takes the only pool seat; once A is OBSERVABLY RUNNING on
    the engine AND still live on the master ledger (both asserted — see
    the precondition below), request B arrives.

    Gate semantics (task #107 #18 verification): the pool counts, per
    engine, every request routed to it but not yet terminal — ACTIVE
    (still in the batcher queue) and COMMITTED (running on the engine)
    alike (PrefillState.pendingRequestCount).  Two enforcement points
    consume the same cap: the ESTIMATED_TTFT selector's availability
    filter (CostBasedPrefillStrategy -> PrefillResourceMeasure:
    pending < maxPendingRequests) refuses the route -> routeForQueue
    returns Blocked -> B parks in placementWaiters, and the offer path
    (WorkerBatcher.offerForPlacement) re-checks pending >= max.  With
    one prefill engine the per-engine pool IS the whole pool.

    Behaviour: a pool-full condition is a WAIT.  B's Schedule RPC stays
    open while B parks (the same placement-answer synchronization the
    deadline case sees from the other side), so the RPC's own duration
    is a direct park measurement.  When A terminates, its PrefillState
    entry retires (pending 1 -> 0) and the next capacity-changed
    publication (PlacementAvailability event) wakes the parked waiter,
    which retries the placement successfully and runs to completion.

    Expected (contract): both schedules succeed (zero fast rejects);
    the pre-B precondition holds (A running on the engine AND live on
    the ledger — a construction that fires B after A already settled
    is vacuous and fails loudly with the snapshot evidence); B's
    Schedule RPC itself takes >= 1.0s (the park: the RPC waits for the
    placement answer until A's release — an immediate 200 would prove
    B never waited for the pool); while A holds the pool the
    master-side parked count is >= 1 — B is ledger-live but absent
    from every engine (the A4 discriminator); both requests complete;
    B terminates strictly AFTER A (>= 1.0s later — the time-order
    proof that B parked and retried on pool release rather than
    running concurrently) and B's end-to-end latency reflects the park
    (>= 4.0s vs the 5s batch); no leakage (master + engine ledgers
    clean) and a fresh request succeeds (recovery).

    Prediction: expected to pass — the pool gate is the same
    offerForPlacement refusal the queue-capacity case exercises one
    layer up, and the wakeup rides the periodic worker-status
    publication (status_rpc_ms=1000), so B's retry lands within ~1-2s
    of A's terminal.  Risk: the pre-B precondition is the fragile
    link — under a compressed schedule (slow snapshot polling) A
    could settle before B's fire; the 5s prefill plus the explicit
    ledger+running check close that window, and a breach now fails
    loudly instead of silently not parking.
    """
    env = ctx.env_manager.ensure(_pool_wait_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    decode_names = _decode_names(ops)
    if not names:
        return False, "no prefill engines found"
    fired: list = []
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=5000.0)

        # A takes the single placement-pool seat and runs.
        rid_a = ops.next_request_id(base)
        fire_err_a = _fire_tracked(ops, rid_a, fired, input_len=512, output_len=2)
        if fire_err_a is not None:
            return False, f"request A fire failed: {fire_err_a}"

        def a_running() -> bool:
            snap = ops.snapshot_by_name()
            return any(int(snap.get(n, {}).get("running", 0)) >= 1 for n in names)

        if not wait_for(a_running, 10.0, 0.1):
            return False, "request A never reached RUNNING on prefill"

        # PRECONDITION (fail loudly, task #107 #18): B must arrive while
        # A REALLY still owns the pool — A running on the engine AND live
        # on the master ledger.  A fast snapshot that observed a stale
        # running fact (A already settled, pending 1 -> 0) would let B
        # route straight through — the old silent not-parking failure.
        ledger_before_b = ops.master_scheduler_inflight()
        snap_before_b = ops.snapshot_by_name()
        a_running_now = any(
            int(snap_before_b.get(n, {}).get("running", 0)) >= 1 for n in names
        )
        if ledger_before_b < 1 or not a_running_now:
            engine_state = {
                n: (
                    snap_before_b.get(n, {}).get("running", -1),
                    snap_before_b.get(n, {}).get("waiting", -1),
                )
                for n in names
            }
            return False, (
                f"precondition failed before B fire: A must still hold the "
                f"pool (ledger={ledger_before_b}, engine_running="
                f"{a_running_now}, engines={engine_state}) — firing B now "
                f"would be vacuous (the pool seat is free)"
            )

        # B arrives while A owns the pool: placement blocked -> park.
        # The Schedule RPC stays open for the placement answer, so its
        # duration measures the park directly.
        rid_b = ops.next_request_id(base)
        t_b_call = time.monotonic()
        fire_err_b = _fire_tracked(ops, rid_b, fired, input_len=512, output_len=2)
        b_fire_rpc_s = time.monotonic() - t_b_call

        # B is parked on the MASTER: ledger-live, engine-absent.  The
        # window is bound to A's own residency (once A's running slot
        # empties, B's retry begins and the park is over) with a hard
        # 10s ceiling.
        parked_max = -1
        parked_detail = "not observed"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            parked, detail = _master_side_parked(ops, names, decode_names)
            if parked > parked_max:
                parked_max, parked_detail = parked, detail
            if parked_max >= 1:
                break
            snap = ops.snapshot_by_name()
            if not any(int(snap.get(n, {}).get("running", 0)) >= 1 for n in names):
                break  # A's residency ended; the park window is over
            time.sleep(0.2)

        outcomes = _await_tracked(fired, wait_s=30.0)
        if len(outcomes) != 2:
            return False, f"expected 2 tracked outcomes, got {len(outcomes)}"
        (_, t0a, end_a, ok_a, err_a) = outcomes[0]
        (_, t0b, end_b, ok_b, err_b) = outcomes[1]
        both_completed = ok_a and err_a is None and ok_b and err_b is None
        b_after_a = (end_b - end_a) >= 1.0
        b_parked_long = (end_b - t0b) >= 4.0
        b_fire_waited = fire_err_b is None and b_fire_rpc_s >= 1.0

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(
            ops, names + decode_names, 15.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            fire_err_b is None
            and b_fire_waited
            and parked_max >= 1
            and both_completed
            and b_after_a
            and b_parked_long
            and inflight_ok
            and engine_clean
            and recovery_ok
        )
        return passed, (
            f"a_completed={ok_a and err_a is None} "
            f"(e2e={end_a - t0a:.2f}s), "
            f"b_completed={ok_b and err_b is None} "
            f"(e2e={end_b - t0b:.2f}s, fire_err={fire_err_b}, "
            f"fire_rpc={b_fire_rpc_s:.2f}s), "
            f"b_after_a={b_after_a} (gap={end_b - end_a:.2f}s), "
            f"b_parked_long={b_parked_long}, "
            f"master_side_parked_max={parked_max} ({parked_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_detail}), "
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
# Engine waiting-batch cap gate (admission wave-3 B3)
# ============================================================================


def _waiting_cap_spec(ctx: CaseContext) -> EnvSpec:
    """B3 env: 1 prefill (every batch lands on one engine, so the cap
    pressure is concentrated), the legacy fault axes and default
    admission knobs — the waiting-queue cap itself is applied at RUNTIME
    via /set_perf max_waiting_batches (ef76751553), so the env shape is
    the plain W1 one."""
    return EnvSpec(
        label=f"admit_wcap_{ctx.profile}",
        n_prefill=1,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={"FLEXLB_CONFIG": admission_config()},
    )


@case(
    "admission_engine_waiting_batch_cap_reject",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "admission wave-3 B3: engine prefill waiting-queue cap "
        "(max_waiting_batches whole-batch backpressure reject — the "
        "non-waitable complement of W1's unbounded park)"
    ),
)
def admission_engine_waiting_batch_cap_reject(ctx: CaseContext):
    """Engine waiting-batch cap gate: the cap is NOT a wait condition.

    Scenario: dedicated 1P+2D env (the W1 shape); prefill_fixed_ms=3000
    stretches each batch's execution window and /set_perf applies the
    runtime cap max_waiting_batches=1 (>0 = queued-batch ceiling, 0 =
    unbounded — the default).  Three requests are fired 0.4s apart (each
    its own batch, 40x the 10ms collection window): batch #1 is admitted
    running, batch #2 parks in prefillPendingQueue (waiting = 1 = cap —
    a snapshot poll proves the saturated state BEFORE the probe fires),
    and batch #3 finds waiting >= cap at schedulePrefillCompletion.

    Behaviour: the cap hit is a WHOLE-BATCH backpressure reject, not a
    park — schedulePrefillCompletion returns false before claiming any
    counter, every member of batch #3 is rolled back and the
    EnqueueBatch ack carries the batch-level error "prefill waiting
    queue full (backpressure): waiting=1 cap=1"; DefaultBatchDispatcher
    wraps it as EngineRejectedException ("EnqueueBatch rejected request
    N: prefill waiting queue full ...") and the master completes the
    request terminal — synchronous, typed, no queueing.  The cap counts
    QUEUED batches only (running is not charged), so the two in-flight
    occupants are untouched.

    Expected (contract): the probe terminates FAST (< 3s from fire,
    no park residence) with the backpressure error family ("prefill
    waiting queue full" + "backpressure") in its terminal error; the
    occupants (1 running + 1 queued) complete normally; the gate
    RECOVERS UNDER THE SAME PRESSURE — with batches #1/#2 still
    occupying the engine, /set_perf max_waiting_batches=0 (unbounded)
    lets a fourth fire park in the queue and run to completion; after
    the drain the engine park is empty, the master inflight ledger is
    clean and a fresh request succeeds (recovery).

    Prediction: expected to pass — the runtime override, the cap check
    (waiting >= cap rejects before claiming any counter) and the
    ack-error surface are covered by SetPerfMaxWaitingBatchesTest (4/4
    green); the only novel wiring is the master's
    EngineRejectedException-to-terminal path, already exercised by the
    enqueue-ack fault family.  Risk: batch coalescing collapsing the
    fires into fewer batches — mitigated by the 0.4s inter-fire gap and
    the pre-probe waiting>=1 observation (the case fails loudly rather
    than probing an unsaturated cap).
    """
    env = ctx.env_manager.ensure(_waiting_cap_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    fired: list = []
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0, max_waiting_batches=1)

        # Occupants: batch #1 running (3s window), batch #2 queued —
        # together they saturate waiting=1=cap.
        fire_errors = []
        for _ in range(2):
            rid = ops.next_request_id(base)
            err = _fire_tracked(ops, rid, fired, input_len=512, output_len=2)
            if err is not None:
                fire_errors.append((rid, err))
            time.sleep(0.4)  # >> maxCollectionWaitMs: one batch per fire
        if fire_errors:
            return False, f"occupant fire failed: {fire_errors[:1]}"

        # Cap-state proof BEFORE the probe: batch #2 sits in
        # prefillPendingQueue (waiting=1=cap).
        def cap_saturated() -> bool:
            snap = ops.snapshot_by_name()
            return all(
                int(snap.get(n, {}).get("prefill_waiting_batches", 0)) >= 1
                for n in names
            )

        cap_observed = wait_for(cap_saturated, 8.0, 0.1)
        if not cap_observed:
            return (
                False,
                "batch #2 never reached the waiting queue (cap never saturated)",
            )

        # Probe batch #3: whole-batch backpressure reject.  The master
        # surfaces the engine's EnqueueBatch reject SYNCHRONOUSLY on the
        # Schedule RPC — code 8510, "Delivery failed: EnqueueBatch rejected
        # request N: prefill waiting queue full (backpressure): waiting=1
        # cap=1" (DefaultBatchDispatcher wraps the ack error and the RPC
        # returns it to the caller) — so the probe accepts EITHER surface:
        # the RPC-level typed reject, or a successful fire whose stream
        # then terminates with the backpressure family.
        rid3 = ops.next_request_id(base)
        r3_t0 = time.monotonic()
        r3_err = ""
        try:
            resp3 = ops.schedule(rid3, input_len=512, output_len=2)
            if resp3.code != 200 or not resp3.success:
                r3_err = f"schedule failed ({resp3.code}): " f"{resp3.error_message}"
            else:
                handle3 = ops.start_stream(resp3, rid3)
                ended3 = handle3.wait_end(10.0)
                r3_err = str(handle3.snap.error or "") if ended3 else "no terminal"
        except Exception as exc:
            r3_err = repr(exc)
        reject_latency = time.monotonic() - r3_t0
        rejected = (
            "prefill waiting queue full" in r3_err.lower()
            and "backpressure" in r3_err.lower()
            and reject_latency < 3.0
        )

        # Gate recovery UNDER THE SAME PRESSURE: relax the cap while
        # batch #1 still runs and batch #2 still queues — the next fire
        # must park in the (now unbounded) queue and complete.
        for n in names:
            ops.set_perf(n, max_waiting_batches=0)
        snap_at_r4 = ops.snapshot_by_name()
        waiting_at_r4 = max(
            int(snap_at_r4.get(n, {}).get("prefill_waiting_batches", 0)) for n in names
        )
        rid4 = ops.next_request_id(base)
        fire_err4 = _fire_tracked(ops, rid4, fired, input_len=512, output_len=2)

        outcomes = _await_tracked(fired, wait_s=45.0)
        occupant_ok = len(outcomes) >= 2 and all(
            ok and err is None for _, _, _, ok, err in outcomes[:2]
        )
        # rid4 is fired[-1] whenever its fire succeeded; a synchronously
        # rejected probe (rid3) never enters `fired`, so index by identity,
        # not by ordinal.
        recovers_ok = (
            fire_err4 is None
            and len(fired) >= 3
            and outcomes[-1][0] == rid4
            and outcomes[-1][3]
            and outcomes[-1][4] is None
        )

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
            cap_observed
            and rejected
            and occupant_ok
            and recovers_ok
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"cap_observed={cap_observed}, "
            f"probe_rejected={rejected} "
            f"(latency={reject_latency:.2f}s, err={r3_err[:80]}), "
            f"occupants_completed={occupant_ok}, "
            f"gate_recovers_under_pressure={recovers_ok} "
            f"(waiting_at_r4={waiting_at_r4}, fire_err4={fire_err4}), "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0, max_waiting_batches=0)
        except Exception:
            pass


# ===========================================================================
# Engine prefill KV block-pool gate (admission wave-3 B2)
# ============================================================================


LACKMEM_POOL_BLOCKS = 17  # reserve=ceil(5% x 17)=1: two 8-block leases fit
LACKMEM_KEYS_PER_REQUEST = 8  # per-request block_cache_keys count (= need)


def _lack_mem_spec(ctx: CaseContext) -> EnvSpec:
    """B2 env: 1 prefill with a 17-block KV pool (KV v2 block pool:
    reserve = ceil(5% x 17) = 1 block).  Two 8-block leases fill the
    pool (8+8 = 16 held of 17 — the remaining free block sits below the
    reserve margin), so a THIRD 8-block request fails the
    TOTAL_AND_AVAILABLE gate.  The decode pool stays at the harness
    default: decode-side admission (ceil(512/1024) = 1 block) must
    never interfere with the prefill gate."""
    return EnvSpec(
        label=f"admit_lackmem_{ctx.profile}",
        n_prefill=1,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={"FLEXLB_CONFIG": admission_config()},
        prefill_cache_blocks=LACKMEM_POOL_BLOCKS,
    )


def _lease_keys(rid: int) -> list:
    """Per-request block keys derived from the rid — disjoint key spaces
    (no shared prefix), so every lease is a fresh LACKMEM_KEYS_PER_REQUEST
    block allocation with zero prefix reuse."""
    return [rid * 100 + i for i in range(1, LACKMEM_KEYS_PER_REQUEST + 1)]


@case(
    "admission_engine_kv_lack_mem_fast_reject",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "admission wave-3 B2: engine prefill KV block-pool gate "
        "(KV v2 BlockLease admission — 602 LACK_MEM synchronous fast "
        "reject, the non-waitable engine-side complement of the "
        "master KV squeeze in admission_slo_queue_deadline)"
    ),
)
def admission_engine_kv_lack_mem_fast_reject(ctx: CaseContext):
    """Engine prefill KV block-pool gate: 602 LACK_MEM fast reject.

    Scenario: dedicated 1P+2D env with a 17-block prefill KV pool
    (EnvSpec prefill_cache_blocks=17; reserve = ceil(5% x 17) = 1 block
    — the KV v2 TOTAL_AND_AVAILABLE gate).  Every request carries 8
    per-request block_cache_keys with input_len=512 — the engine-side
    need caliber is the KEY COUNT (8 blocks), while the master-side KV
    gate compares the request's seqLen (512 tokens) against the
    engine-reported available tokens (>= 1 block = 1024 even at peak
    occupancy), so the master gate never intercepts: the ENGINE gate
    is the only admission edge in play.  prefill_fixed_ms=3000 holds
    batch #1 running while batch #2 queues; both leases are
    provisioned at enqueue (the KV v2 admission-lease semantics), so
    after two fires held_blocks = 16 of 17 — a snapshot poll proves
    the pool-full state BEFORE the probe fires.

    Behaviour: the third 8-block request fails acquireBlockLease at
    EnqueueBatch Phase-1.5 — the ack carries the per-request error
    code 602 (MALLOC_FAILED, never the master's 8431) with "LACK_MEM:
    insufficient KV cache blocks (need=8, avail=1, spb=1024)";
    DefaultBatchDispatcher wraps it as EngineRejectedException
    ("EnqueueBatch rejected request N: LACK_MEM: ...") and the master
    completes the request terminal — synchronous, typed, no park, no
    queue (the non-waitable complement of admission_slo_queue_deadline,
    where the SAME master KV surface parks because that squeeze is a
    WAIT condition).  The rejected request leaves no residue (lease
    acquisition rolled back, requestStates -> "rejected").

    Expected (contract): the probe terminates FAST (< 3s from fire,
    no park residence) with the LACK_MEM family in its terminal error
    ("lack_mem" + "insufficient kv cache" + the master's
    "enqueuebatch rejected" wrapper — the actual ack-to-terminal
    transparent path); the two occupants complete normally and their
    leases hand back to the LRU on completion (pool recovery —
    pure-LRU blocks count as available again); a fresh 8-block
    request on the recovered pool succeeds; the master inflight and
    engine ledgers drain clean and recovery holds.

    Prediction: expected to pass — the 602 ack surface is
    BlockPoolCapacityTest's master-visible contract (11 hash-channel
    blocks vs a 10-block pool) and the EngineRejectedException
    terminal path is the enqueue-ack fault family's.  The 17-block
    sizing makes the arithmetic exact: 8+8 admitted with the reserve
    to spare, the third 8-block need denied outright by the available
    check (8 > 1) — no borderline rounding.  Risk: the two occupancy
    fires coalescing into ONE batch — both leases still provision at
    enqueue, so the pool still saturates (only the "third request"
    ordinal shifts); the 0.4s inter-fire gap keeps them separate
    anyway.
    """
    env = ctx.env_manager.ensure(_lack_mem_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    fired: list = []
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)

        # Occupants: batch #1 running (lease taken at enqueue), batch #2
        # queued (lease ALSO taken at enqueue — KV v2 admission leases).
        fire_errors = []
        for _ in range(2):
            rid = ops.next_request_id(base)
            err = _fire_tracked(
                ops,
                rid,
                fired,
                input_len=512,
                output_len=2,
                block_keys=_lease_keys(rid),
            )
            if err is not None:
                fire_errors.append((rid, err))
            time.sleep(0.4)  # >> maxCollectionWaitMs: one batch per fire
        if fire_errors:
            return False, f"occupant fire failed: {fire_errors[:1]}"

        # Pool-full proof BEFORE the probe: held = 16 of 17 blocks.
        held_seen = 0

        def pool_full() -> bool:
            snap = ops.snapshot_by_name()
            nonlocal held_seen
            held_seen = max(
                held_seen,
                max(int(snap.get(n, {}).get("held_blocks", 0)) for n in names),
            )
            return all(
                int(snap.get(n, {}).get("held_blocks", 0))
                >= 2 * LACKMEM_KEYS_PER_REQUEST
                for n in names
            )

        pool_observed = wait_for(pool_full, 8.0, 0.1)
        if not pool_observed:
            return False, (
                f"occupancy leases never saturated the pool "
                f"(held_max={held_seen}/{LACKMEM_POOL_BLOCKS})"
            )

        # Probe: 602 LACK_MEM synchronous fast reject (no park, no queue).
        rid3 = ops.next_request_id(base)
        fire_err3 = _fire_tracked(
            ops,
            rid3,
            fired,
            input_len=512,
            output_len=2,
            block_keys=_lease_keys(rid3),
        )
        if fire_err3 is not None:
            return False, f"probe fire failed: {fire_err3}"
        _, r3_handle, r3_t0 = fired[-1]
        r3_ended = r3_handle.wait_end(10.0)
        r3_err = str(r3_handle.snap.error or "") if r3_ended else "no terminal"
        reject_latency = time.monotonic() - r3_t0
        err_low = r3_err.lower()
        rejected = (
            "lack_mem" in err_low
            and "insufficient kv cache" in err_low
            and "enqueuebatch rejected" in err_low
            and reject_latency < 3.0
        )

        # Occupants complete; their leases hand back to the LRU on
        # completion — the pool recovers (pure-LRU counts as available).
        outcomes = _await_tracked(fired, wait_s=45.0)
        occupant_ok = len(outcomes) >= 2 and all(
            ok and err is None for _, _, _, ok, err in outcomes[:2]
        )

        avail_seen = 0

        def pool_recovered() -> bool:
            snap = ops.snapshot_by_name()
            nonlocal avail_seen
            avail_seen = max(
                avail_seen,
                max(int(snap.get(n, {}).get("available_blocks", 0)) for n in names),
            )
            return all(
                int(snap.get(n, {}).get("available_blocks", 0))
                >= LACKMEM_KEYS_PER_REQUEST
                for n in names
            )

        recovered_seen = wait_for(pool_recovered, 10.0, 0.2)

        # Post-release probe: a fresh 8-block lease on the recovered pool.
        fired4: list = []
        rid4 = ops.next_request_id(base)
        fire_err4 = _fire_tracked(
            ops,
            rid4,
            fired4,
            input_len=512,
            output_len=2,
            block_keys=_lease_keys(rid4),
        )
        outcomes4 = _await_tracked(fired4, wait_s=STREAM_TIMEOUT_S)
        lease4_ok = (
            fire_err4 is None
            and len(outcomes4) == 1
            and outcomes4[0][3]
            and outcomes4[0][4] is None
        )

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(
            ops, names + _decode_names(ops), 15.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            pool_observed
            and rejected
            and occupant_ok
            and recovered_seen
            and lease4_ok
            and inflight_ok
            and engine_clean
            and recovery_ok
        )
        return passed, (
            f"pool_observed={pool_observed} "
            f"(held={held_seen}/{LACKMEM_POOL_BLOCKS}), "
            f"probe_rejected={rejected} "
            f"(latency={reject_latency:.2f}s, err={r3_err[:80]}), "
            f"occupants_completed={occupant_ok}, "
            f"pool_recovers={recovered_seen} "
            f"(avail={avail_seen}/{LACKMEM_POOL_BLOCKS}), "
            f"fresh_lease_succeeds={lease4_ok} (fire_err4={fire_err4}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_detail}), "
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
# Engine-internal dual-budget prefill regroup (#8, 2026-09)
# ===========================================================================
#
# The regroup axes arrive via the performance JSON (the sanctioned
# explicit channel, same as fault_env_perf): prefill.max_batch_tokens /
# prefill.max_batch_requests with production-aligned engine defaults
# (WorkerStatus.max_batch_tokens_size = 1_048_576, FIXED_WINDOW
# maxRequests = 32).  An explicit 0 disables the dimension; 0/0
# disables the regroup entirely (verbatim master batches — the legacy
# pre-#8 behaviour).  The four cases pin the contract: over-budget
# split with a closed ledger, arrival-order preservation across the
# split, the == boundary (no split) and the 0/0 legacy switch.


def _regroup_spec(
    ctx: CaseContext, max_batch_tokens: int, max_batch_requests: int
) -> EnvSpec:
    """#8 env: one prefill engine (the whole master batch lands there),
    flat prefill.fixed_ms=3000 stretching each execution batch past the
    observation windows, explicit regroup budget axes.  The master runs
    a wide 100ms FIXED_WINDOW collection window so four 10ms-spaced
    fires coalesce into ONE master batch — the split then happens
    engine-side, never master-side."""
    perf = default_perf()
    perf["prefill"] = {
        "fixed_ms": 3000.0,
        "scale": 1.0,
        "max_batch_tokens": max_batch_tokens,
        "max_batch_requests": max_batch_requests,
    }
    return EnvSpec(
        label=f"admit_regroup_{ctx.profile}",
        n_prefill=1,
        n_decode=2,
        perf=perf,
        master_profile=ctx.profile,
        master_env={
            "FLEXLB_CONFIG": build_flexlb_config(
                ordering="priority",
                decision="fixed_window",
                dispatcher="batch",
                max_collection_wait_ms=100,
                queue_timeout_ms=60_000,
            )
        },
    )


def _fire_regroup_wave(ops, base: int, n: int = 4):
    """Fire *n* 512-token requests 10ms apart — all inside the master's
    100ms collection window, so they coalesce into ONE master batch.
    Returns (rids, fired, fire_errors)."""
    fired: list = []
    rids: list = []
    fire_errors: list = []
    for _ in range(n):
        rid = ops.next_request_id(base)
        rids.append(rid)
        err = _fire_request(ops, rid, fired, input_len=512, output_len=2)
        if err is not None:
            fire_errors.append((rid, err))
        time.sleep(0.01)
    return rids, fired, fire_errors


def _prefill_batch_counters(ops, name: str):
    """#8 observation surface — per-engine executed-prefill-batch
    counters: (prefill_batches, prefill_batch_requests,
    max_prefill_batch_size) from the engine snapshot."""
    snap = ops.snapshot_by_name()
    info = snap.get(name, {})
    return (
        int(info.get("prefill_batches", -1)),
        int(info.get("prefill_batch_requests", -1)),
        int(info.get("max_prefill_batch_size", -1)),
    )


def _park_settled(ops, names: list) -> bool:
    snap = ops.snapshot_by_name()
    return all(
        int(snap.get(n, {}).get("prefill_waiting_batches", 0)) == 0
        and int(snap.get(n, {}).get("waiting", 0)) == 0
        for n in names
    )


def _lifecycle_rows(ops, name: str, rids: list) -> dict:
    """Engine request_lifecycle rows for *rids* (keyed by rid)."""
    snap = ops.snapshot_by_name()
    lifecycle = snap.get(name, {}).get("request_lifecycle", {})
    return {rid: lifecycle.get(str(rid)) for rid in rids}


@case(
    "engine_prefill_token_budget_split",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "engine regroup #8: in-engine dual-budget prefill regroup "
        "(over-budget master batch split + closed ledger)"
    ),
)
def engine_prefill_token_budget_split(ctx: CaseContext):
    """Engine-internal token-budget regroup: an over-budget master batch
    splits prefix/tail and every member still closes its ledger.

    Scenario: dedicated 1P+2D env, prefill.max_batch_tokens=1024 (the
    request dimension off), flat prefill.fixed_ms=3000.  Four
    512-token requests fire 10ms apart — all inside the master's 100ms
    collection window — so the engine receives ONE four-member master
    batch whose total logical tokens (4 x 512 = 2048, sum of
    computeTokens + hitTokens) is 2x the budget.

    Behaviour: the engine-side regroup composer (production
    FIFOScheduler.cc:371-481 semantics — the budget is a STOP, members
    join while admitted < budget) fills the execution batch with the
    first two arrivals, parks the tail members as one PrefillPendingBatch
    in prefillPendingQueue and admits them FIFO when the running batch
    drains.

    Expected (contract): all four schedules succeed; a snapshot poll
    observes prefill_waiting_batches >= 1 while the prefix executes;
    every fired request drains to terminal COMPLETED; the executed-
    batch counters (delta over the pre-fire baseline, captured before
    the recovery probe adds its own batch) grow by exactly 2 batches /
    4 requests with max size 2 — the regrouped shape, not the master's
    verbatim 1x4; every member's request_lifecycle row still carries
    the SAME master batch_id (ledger identity survives the split);
    the park settles empty, the master inflight ledger is clean and a
    fresh request succeeds (recovery).

    Prediction: expected to pass — the split path is the primary #8
    deliverable; PrefillBudgetRegroupTest pins the same shape in-JVM
    (2 batches / 4 requests / max 2).  Risk: FIXED_WINDOW coalescing
    flake would reshape the master batch — mitigated by the 10x margin
    between the 10ms inter-fire gap and the 100ms collection window.
    """
    env = ctx.env_manager.ensure(_regroup_spec(ctx, 1024, 0))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        base_counters = _prefill_batch_counters(ops, names[0])
        rids, fired, fire_errors = _fire_regroup_wave(ops, base)

        # Park: while the prefix [r1, r2] executes its 3s window the
        # tail [r3, r4] sits in prefillPendingQueue.
        park_max = 0
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            snap = ops.snapshot_by_name()
            for n in names:
                park_max = max(
                    park_max,
                    int(snap.get(n, {}).get("prefill_waiting_batches", 0)),
                    int(snap.get(n, {}).get("waiting", 0)),
                )
            if park_max >= 1:
                break
            time.sleep(0.2)

        outcomes = _drain_fired(ops, fired, wait_s=45.0)
        completed = [rid for rid, ok, _ in outcomes if ok]
        drain_errors = [(rid, err) for rid, ok, err in outcomes if not ok]

        # Counters AFTER the drain, BEFORE verify_recovery's probe.
        after_counters = _prefill_batch_counters(ops, names[0])
        delta_batches = after_counters[0] - base_counters[0]
        delta_requests = after_counters[1] - base_counters[1]
        counters_ok = (
            delta_batches == 2 and delta_requests == 4 and after_counters[2] == 2
        )

        # Ledger identity: all four members still attribute to the ONE
        # master batch (EnqueueBatch-time batch_id is request-level —
        # the split never rewrites it).
        rows = _lifecycle_rows(ops, names[0], rids)
        member_batch_ids = {
            row.get("batch_id") if row else None for row in rows.values()
        }
        batch_id_ok = len(member_batch_ids) == 1 and next(
            iter(member_batch_ids), None
        ) not in (None, 0, -1)

        settled = wait_for(lambda: _park_settled(ops, names), 10.0, 0.2)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            not fire_errors
            and park_max >= 1
            and len(completed) == len(rids)
            and not drain_errors
            and counters_ok
            and batch_id_ok
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={len(rids)} (fire_errors={fire_errors[:1]}), "
            f"park_observed_max={park_max}, "
            f"completed={len(completed)}/{len(rids)} "
            f"(drain_errors={drain_errors[:1]}), "
            f"executed_delta={delta_batches}b/{delta_requests}r "
            f"max_size={after_counters[2]} (expect 2b/4r/2), "
            f"member_batch_ids={member_batch_ids}, "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "engine_prefill_token_budget_split_fifo",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "engine regroup #8: in-engine dual-budget prefill regroup "
        "(split preserves arrival order across execution batches)"
    ),
)
def engine_prefill_token_budget_split_fifo(ctx: CaseContext):
    """The split preserves arrival order across execution batches.

    Scenario: identical config to engine_prefill_token_budget_split
    (1024-token budget, flat 3000ms prefill, four 512-token requests
    in one master batch) — the spec fingerprint matches, so ensure()
    reuses the very same env; this case pins the ORDER contract on a
    fresh wave of rids.

    Behaviour: the composer admits members while admitted < budget —
    prefix [r1, r2] forms execution batch #1, tail [r3, r4] parks
    until the prefix drains, so execution batches run in arrival
    order.

    Expected (contract): all four requests complete; the executed-
    batch counters grow by exactly 2 batches / 4 requests with max
    size 2 (delta over the shared env's pre-fire baseline); each
    fired request's engine lifecycle end_ms — written at prefill
    completion — shows the tail finishing STRICTLY after the prefix
    with >1s separation (each execution batch runs 3000ms; a
    reshuffled or interleaved composition collapses the gap); the
    master inflight ledger is clean and a fresh request succeeds.

    Prediction: expected to pass — the FIFO contract is structural
    (the composer consumes the master batch head-first, the tail
    parks behind it).  The >1s threshold sits at 1/3 of the execution
    window.
    """
    env = ctx.env_manager.ensure(_regroup_spec(ctx, 1024, 0))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        base_counters = _prefill_batch_counters(ops, names[0])
        rids, fired, fire_errors = _fire_regroup_wave(ops, base)

        outcomes = _drain_fired(ops, fired, wait_s=45.0)
        completed = [rid for rid, ok, _ in outcomes if ok]
        drain_errors = [(rid, err) for rid, ok, err in outcomes if not ok]

        after_counters = _prefill_batch_counters(ops, names[0])
        delta_batches = after_counters[0] - base_counters[0]
        delta_requests = after_counters[1] - base_counters[1]
        counters_ok = (
            delta_batches == 2 and delta_requests == 4 and after_counters[2] == 2
        )

        # ORDER: prefix members' prefill-completion timestamps must
        # sit strictly before the tail's, with a >1s gap (the parked
        # tail only starts after the prefix's 3s batch drains).
        rows = _lifecycle_rows(ops, names[0], rids)
        end_ms = {
            rid: int(row.get("end_ms", 0)) if row else 0 for rid, row in rows.items()
        }
        prefix_done = max(end_ms[rids[0]], end_ms[rids[1]])
        tail_done = min(end_ms[rids[2]], end_ms[rids[3]])
        fifo_gap_ms = tail_done - prefix_done
        fifo_ok = (
            all(v > 0 for v in end_ms.values())
            and prefix_done < tail_done
            and fifo_gap_ms > 1000
        )

        settled = wait_for(lambda: _park_settled(ops, names), 10.0, 0.2)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            not fire_errors
            and len(completed) == len(rids)
            and not drain_errors
            and counters_ok
            and fifo_ok
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={len(rids)} (fire_errors={fire_errors[:1]}), "
            f"completed={len(completed)}/{len(rids)} "
            f"(drain_errors={drain_errors[:1]}), "
            f"executed_delta={delta_batches}b/{delta_requests}r "
            f"max_size={after_counters[2]} (expect 2b/4r/2), "
            f"fifo_prefix_done={prefix_done} tail_done={tail_done} "
            f"gap_ms={fifo_gap_ms} (expect >1000), "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "engine_prefill_token_budget_boundary",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "engine regroup #8: in-engine dual-budget prefill regroup "
        "(boundary — a batch exactly at the token budget is NOT split)"
    ),
)
def engine_prefill_token_budget_boundary(ctx: CaseContext):
    """Boundary: a batch exactly AT the token budget executes verbatim.

    Scenario: prefill.max_batch_tokens=2048 — exactly the total of
    four 512-token requests coalesced into one master batch.

    Behaviour: production admission semantics (FIFOScheduler.cc:
    371-481) — members join while admitted < budget (strict), so the
    fourth member (admitted 1536 < 2048) still fits and the whole
    batch executes as ONE.

    Expected (contract): all four requests complete; a snapshot poll
    through the 3s execution window observes prefill_waiting_batches
    == 0 throughout (no park — nothing is left over); the executed-
    batch counters grow by exactly 1 batch / 4 requests with max
    size 4 (verbatim, the boundary is inclusive); the master inflight
    ledger is clean and a fresh request succeeds.

    Prediction: expected to pass — the boundary condition is pinned
    in-JVM by PrefillBudgetRegroupTest.
    boundaryExactBudgetBatchDoesNotSplit.
    """
    env = ctx.env_manager.ensure(_regroup_spec(ctx, 2048, 0))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        base_counters = _prefill_batch_counters(ops, names[0])
        rids, fired, fire_errors = _fire_regroup_wave(ops, base)

        # No park: the single verbatim batch admits immediately —
        # poll through its 3s execution window.
        no_park = True
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            snap = ops.snapshot_by_name()
            if any(
                int(snap.get(n, {}).get("prefill_waiting_batches", 0)) > 0
                for n in names
            ):
                no_park = False
                break
            time.sleep(0.2)

        outcomes = _drain_fired(ops, fired, wait_s=45.0)
        completed = [rid for rid, ok, _ in outcomes if ok]
        drain_errors = [(rid, err) for rid, ok, err in outcomes if not ok]

        after_counters = _prefill_batch_counters(ops, names[0])
        delta_batches = after_counters[0] - base_counters[0]
        delta_requests = after_counters[1] - base_counters[1]
        counters_ok = (
            delta_batches == 1 and delta_requests == 4 and after_counters[2] == 4
        )

        settled = wait_for(lambda: _park_settled(ops, names), 10.0, 0.2)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            not fire_errors
            and no_park
            and len(completed) == len(rids)
            and not drain_errors
            and counters_ok
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={len(rids)} (fire_errors={fire_errors[:1]}), "
            f"no_park_through_window={no_park}, "
            f"completed={len(completed)}/{len(rids)} "
            f"(drain_errors={drain_errors[:1]}), "
            f"executed_delta={delta_batches}b/{delta_requests}r "
            f"max_size={after_counters[2]} (expect 1b/4r/4), "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "engine_prefill_regroup_disabled_verbatim",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "engine regroup #8: in-engine dual-budget prefill regroup "
        "(0/0 disables regroup — legacy verbatim master batches)"
    ),
)
def engine_prefill_regroup_disabled_verbatim(ctx: CaseContext):
    """The 0/0 switch: regroup off, the master batch executes as-is.

    Scenario: prefill.max_batch_tokens=0 AND
    prefill.max_batch_requests=0 — the documented off switch: both
    dimensions zero disables the in-engine regroup entirely and the
    master batch executes verbatim (the pre-#8 behaviour).

    Expected (contract): all four requests complete; no park is
    observed (the single master batch admits immediately); the
    executed-batch counters grow by exactly 1 batch / 4 requests with
    max size 4 — the verbatim master shape (4x the tokens a 1024
    budget would have split); every member's lifecycle row attributes
    to the SAME master batch_id; the master inflight ledger is clean
    and a fresh request succeeds.

    Prediction: expected to pass — the disabled path is the preserved
    legacy code (prefillRegroupEnabled() == false), pinned in-JVM by
    PrefillBudgetRegroupTest.regroupOffReproducesVerbatimMasterBatch.
    """
    env = ctx.env_manager.ensure(_regroup_spec(ctx, 0, 0))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        base_counters = _prefill_batch_counters(ops, names[0])
        rids, fired, fire_errors = _fire_regroup_wave(ops, base)

        no_park = True
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            snap = ops.snapshot_by_name()
            if any(
                int(snap.get(n, {}).get("prefill_waiting_batches", 0)) > 0
                for n in names
            ):
                no_park = False
                break
            time.sleep(0.2)

        outcomes = _drain_fired(ops, fired, wait_s=45.0)
        completed = [rid for rid, ok, _ in outcomes if ok]
        drain_errors = [(rid, err) for rid, ok, err in outcomes if not ok]

        after_counters = _prefill_batch_counters(ops, names[0])
        delta_batches = after_counters[0] - base_counters[0]
        delta_requests = after_counters[1] - base_counters[1]
        counters_ok = (
            delta_batches == 1 and delta_requests == 4 and after_counters[2] == 4
        )

        rows = _lifecycle_rows(ops, names[0], rids)
        member_batch_ids = {
            row.get("batch_id") if row else None for row in rows.values()
        }
        batch_id_ok = len(member_batch_ids) == 1 and next(
            iter(member_batch_ids), None
        ) not in (None, 0, -1)

        settled = wait_for(lambda: _park_settled(ops, names), 10.0, 0.2)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            not fire_errors
            and no_park
            and len(completed) == len(rids)
            and not drain_errors
            and counters_ok
            and batch_id_ok
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={len(rids)} (fire_errors={fire_errors[:1]}), "
            f"no_park_through_window={no_park}, "
            f"completed={len(completed)}/{len(rids)} "
            f"(drain_errors={drain_errors[:1]}), "
            f"executed_delta={delta_batches}b/{delta_requests}r "
            f"max_size={after_counters[2]} (expect 1b/4r/4), "
            f"member_batch_ids={member_batch_ids}, "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
