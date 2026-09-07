"""Shared admission scenario components. Resource and timing semantics live here."""

from __future__ import annotations

import json
import threading
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
    ConfigOverride,
    EnvSpec,
    _ttft_p50,
    default_perf,
    http_get_json,
    wait_for,
)

STREAM_TIMEOUT_S = 15.0
# Mock DEFAULT_TOTAL_KV_TOKENS — squeezing with this value drives
# availableKvCache to 0.
MOCK_TOTAL_KV_TOKENS = 6_291_456


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
        config_overrides=ConfigOverride(
            ordering="priority",
            queue_timeout_ms=1500,
        ),
    )


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
        config_overrides=ConfigOverride(
            ordering="priority",
            queue_timeout_ms=60_000,
            max_outstanding=2,
        ),
    )


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
        config_overrides=ConfigOverride(
            ordering="priority",
            queue_timeout_ms=60_000,
        ),
    )


# ===========================================================================
# Engine decode hard gate (admission wave-2 W2)
# ===========================================================================


DECODE_WAVE_REQUESTS = 280  # gate-reachability anchor — see case docstring
DECODE_HARD_GATE = 128  # mock engine decodeMaxConcurrency (CLI default)


def _decode_park_spec(ctx: CaseContext) -> EnvSpec:
    """W2 env: 2P+1D (all decode traffic concentrates on one engine).

    The master-side decode routing cap is raised far above the engine's
    128 hard gate (maxEngineRequests 132 -> 5000) so the ENGINE gate is
    the only admission edge in play: every fired request is routed and
    delivered, and whatever overflows 128 running slots must park in the
    engine's decodePendingQueue instead of being bounced anywhere."""
    return EnvSpec(
        label=f"admit_decode_park_{ctx.profile}",
        n_prefill=2,
        n_decode=1,
        perf=default_perf(),
        master_profile=ctx.profile,
        config_overrides=ConfigOverride(
            ordering="priority",
            queue_timeout_ms=60_000,
            decode_max_engine_requests=5000,
        ),
    )


def _decode_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "decode"]


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
    incomer's route comes back ACQUIRED.  Under the new-B semantics the
    single acceptance permit is RELEASED at the DecodeAccepted event —
    not held to the occupant's terminal — so once the victim is RUNNING
    the permit is back in the pool and the incomer acquires it (the old
    completeAcceptanceLimit 8431 reject path is no longer reachable at
    this probe point).  No preemption block is emitted
    (the generator never writes one), so EvictionManager.tryAdmit
    is a no-op — the no-preemption complement of
    cancel_preemption_victim."""
    return EnvSpec(
        label=f"admit_incomer_{ctx.profile}",
        n_prefill=1,
        n_decode=1,
        perf=default_perf(),
        master_profile=ctx.profile,
        config_overrides=ConfigOverride(
            ordering="priority",
            queue_timeout_ms=60_000,
            max_delivered_not_accepted=1,
        ),
    )


# ===========================================================================
# Master batcher-queue capacity gate (admission wave-2 A5)
# ===========================================================================


BQ_PARK_REQUESTS = 7  # > lease window (4) + queue capacity (2): the 7th parks
BQ_DEADLINE_REQUESTS = 8  # 4 leases + 2 queue seats + 2 placementWaiters
BQ_DEADLINE_MS = 1500
# New-B terminal split for the deadline case: the 4 dispatcher lease
# seats + 2 batcher-queue seats admit and COMPLETE (a queued entry's
# request deadline detaches at DELIVERY CONFIRMATION — RequestSlot.
# confirmDeliveryForPublication clears requestDeadline when the
# EnqueueBatch delivery is confirmed, verdict §1.5 — so the queue
# absorbs fires 5-6 past the old expiry boundary); the 2 fires parked
# behind the capacity gate expire on their still-open Schedule RPC
# (8511).
BQ_DEADLINE_ADMITTED = 6
BQ_DEADLINE_OVERFLOW = 2


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


def _fire_tracked(
    ops, rid: int, fired: list, timeout_s: float = 30.0, **kwargs
) -> Optional[str]:
    """Schedule AND immediately open the response stream, recording the
    fire instant so per-request terminal timing (completion order,
    deadline expiry) is measurable — the W3 victim-handle pattern,
    batched.  Fire errors are reported like _fire_request's.

    ``timeout_s`` is the client gRPC deadline for the Schedule RPC: a
    capacity-blocked submitter parks as a Blocked request with the RPC
    still OPEN (verdict §1.1), so callers that expect parking pass a
    generous deadline.  NON_BATCH responses (enqueued_by_master=False)
    open the direct GenerateStreamCall with an input_pb rebuilt from
    the SAME kwargs — start_stream's default-shape fallback would
    desynchronize the engine's view from the master's schedule (the
    trap engine_ops.verify_recovery documents)."""
    try:
        resp = ops.schedule(rid, timeout_s=timeout_s, **kwargs)
    except Exception as exc:
        return repr(exc)
    if resp.code != 200 or not resp.success:
        return f"schedule failed ({resp.code}): {resp.error_message}"
    input_pb = None
    if not resp.enqueued_by_master:
        input_pb = ops.build_generate_input(rid, **kwargs)
    try:
        handle = ops.start_stream(resp, rid, input_pb=input_pb)
    except Exception as exc:
        return repr(exc)
    fired.append((rid, handle, time.monotonic()))
    return None


class _ParkedSampler:
    """Background master-side-parked time-series sampler (verdict §3 fix).

    The parked window lives DURING the concurrent fire: a Blocked park
    holds the Schedule RPC open and the park drains the moment capacity
    frees, so a sampler that only starts AFTER the fires settle
    structurally misses the window — the 2026-09-04 run measured
    parked_max=0 on exactly that defect while the wave demonstrably
    queued.  This thread samples :func:`_master_side_parked` every
    ``interval_s`` while the fires are in flight (the decode_hard_gate
    L863-872 poll-loop cadence on the master.py coldstart-sampler
    thread shape), recording ``(monotonic_t, parked, detail)`` samples;
    the caller asserts on the series AFTER joining the fires.  HTTP
    failures (parked == -1) are counted, never recorded as samples.
    """

    def __init__(self, ops, prefill_names, decode_names, interval_s: float = 0.2):
        self._ops = ops
        self._prefill = list(prefill_names)
        self._decode = list(decode_names)
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: list[tuple[float, int, str]] = []
        self.http_failures = 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            parked, detail = _master_side_parked(self._ops, self._prefill, self._decode)
            if parked < 0:
                self.http_failures += 1
            else:
                self.samples.append((time.monotonic(), parked, detail))
            self._stop.wait(self._interval)

    def start(self) -> "_ParkedSampler":
        self._thread = threading.Thread(
            target=self._loop, name="parked-sampler", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)

    @property
    def max_parked(self) -> int:
        return max((p for _, p, _ in self.samples), default=-1)

    @property
    def max_detail(self) -> str:
        best = max(self.samples, key=lambda s: s[1], default=None)
        return best[2] if best is not None else "no samples"

    def parked_in_window(self, t_lo: float, t_hi: float) -> bool:
        """True when any sample inside [t_lo, t_hi] observed parked >= 1.

        Both bounds and the sample stamps are time.monotonic() instants
        from the same clock, so callers can align the park evidence with
        independently recorded event windows (e.g. the 8511 expiry
        window of the deadline case)."""
        return any(t_lo <= t <= t_hi and p >= 1 for t, p, _ in self.samples)


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
    """A5 env: 1 prefill (a single batcher queue), PRIORITY ordering over
    the profile's own decision/dispatcher axes (profile-aware since the
    tier2 spec unpick), with the batcher waiting-queue
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
        config_overrides=ConfigOverride(
            ordering="priority",
            queue_timeout_ms=queue_timeout_ms,
            max_waiting_requests_per_prefill_worker=2,
        ),
    )


# ===========================================================================
# Master placement pool gate (admission wave-2 A4)
# ===========================================================================


def _pool_wait_spec(ctx: CaseContext) -> EnvSpec:
    """A4 env (verdict §4.1 rebuild): 1P+2D on the NON_BATCH dispatcher
    axes of the ctx profile (single-nonbatch / window-nonbatch lanes;
    profile-aware since the tier2 spec unpick) with the two LIVE capacity
    knobs — dispatcher
    maxInflightRequestsPerPrefillWorker=1 (RoutePrefillAdmission leases
    one in-flight delivery per dispatch; priority.py's verified backlog
    window — without the cap every request dispatches immediately, no
    queueing is observable) plus scheduler.capacity
    maxWaitingRequestsPerPrefillWorker=2 (the WorkerBatcher ACTIVE-queue
    ceiling and the planning-frontier credits source).

    The retired prefill_max_pending_requests no-op is GONE (the codex
    schema removed router.roles.prefill.availability, commit
    3a6bb84000; admission_config emits no key for it): the old
    construction — 1P + lease 4 + queue 1024 — had NO admission edge in
    play and could never observe capacity behaviour (the 2026-09-04
    false-green root cause).  With the delivery lease at 1, a second
    arrival while the first runs finds the lease held and parks as a
    Blocked request with its Schedule RPC open."""
    return EnvSpec(
        label=f"admit_pool_{ctx.profile}",
        n_prefill=1,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        config_overrides=ConfigOverride(
            ordering="fifo",
            queue_timeout_ms=60_000,
            max_inflight_requests_per_worker=1,
            max_waiting_requests_per_prefill_worker=2,
        ),
    )


# ===========================================================================
# Engine waiting-batch cap gate (admission wave-3 B3)
# ============================================================================


def _waiting_cap_spec(ctx: CaseContext) -> EnvSpec:
    """B3 env: 1 prefill (every batch lands on one engine, so the cap
    pressure is concentrated), the profile's own axes and default
    admission knobs (profile-aware since the tier2 spec unpick) — the
    waiting-queue cap itself is applied at RUNTIME
    via /set_perf max_waiting_batches (ef76751553), so the env shape is
    the plain W1 one."""
    return EnvSpec(
        label=f"admit_wcap_{ctx.profile}",
        n_prefill=1,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        config_overrides=ConfigOverride(
            ordering="priority",
            queue_timeout_ms=60_000,
        ),
    )


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
        config_overrides=ConfigOverride(
            ordering="priority",
            queue_timeout_ms=60_000,
        ),
        prefill_cache_blocks=LACKMEM_POOL_BLOCKS,
    )


def _lease_keys(rid: int) -> list:
    """Per-request block keys derived from the rid — disjoint key spaces
    (no shared prefix), so every lease is a fresh LACKMEM_KEYS_PER_REQUEST
    block allocation with zero prefix reuse."""
    return [rid * 100 + i for i in range(1, LACKMEM_KEYS_PER_REQUEST + 1)]


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
        config_overrides=ConfigOverride(
            ordering="priority",
            max_collection_wait_ms=100,
            queue_timeout_ms=60_000,
        ),
    )


def _fire_regroup_wave(ops, base: int, n: int = 4):
    """Fire *n* 512-token requests 10ms apart — all inside the master's
    100ms collection window, so they coalesce into ONE master batch.

    Concurrent submission (priority._fire_batch precedent, one submit
    per worker on a ThreadPoolExecutor): _fire_request BLOCKS on the
    schedule RPC — under BATCH dispatch the Schedule response settles
    only after the EnqueueBatch ACK — so a serial fire could never land
    four submits inside the 100ms window and the master answered each
    fire with its own singleton batch, leaving the engine-side regroup
    nothing to work on.  The 10ms inter-SUBMIT gap (main thread) keeps
    the deterministic arrival order; futures are collected afterwards.
    Returns (rids, fired, fire_errors)."""
    fired: list = []
    rids: list = []
    fire_errors: list = []
    pool = ThreadPoolExecutor(max_workers=n)
    try:
        futures = []
        for _ in range(n):
            rid = ops.next_request_id(base)
            rids.append(rid)
            futures.append(
                pool.submit(_fire_request, ops, rid, fired, input_len=512, output_len=2)
            )
            time.sleep(0.01)
        for rid, future in zip(rids, futures):
            err = future.result()
            if err is not None:
                fire_errors.append((rid, err))
    finally:
        pool.shutdown(wait=True)
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


def _ledger_series_ok(
    samples: list, peak_batches: int, n_requests: int, expect_intermediate: bool
) -> tuple[bool, str]:
    """Master-side ledger linkage assertion for the regroup cases (the
    VERDICT leg — engine-side batch shape is only a construction gate).

    Caliber (source-verified, HttpLoadBalanceServer.inflightStatus ->
    PrefillState.stats()): one EnqueueBatch's bookkeeping stays counted as
    ONE inflight batch until every member settles — the ENGINE-side split
    never multiplies the master's batch count, and the master digests the
    engine's per-execution-batch completion events event-driven (20ms
    WorkerStatus reconcile):

      * peak inflight_batches == 1 — the master books the one batch IT
        dispatched, regardless of how the engine regrouped it;
      * peak prefill inflight_requests == n_requests — every fired member
        entered the ledger;
      * when *expect_intermediate* (engine split the batch), the member
        accounting must show at least one intermediate plateau (0 < v <
        n_requests) — an event-driven step-down, not a single settle-or-
        never-release jump; when not (verbatim batch), NO intermediate
        value may appear — the batch settles atomically;
      * scheduler_inflight must never climb mid-series (a re-admission of
        parked work back into the scheduler would be a master-side
        anomaly: the tail executes inside the engine, the master must not
        re-dispatch anything).

    The end-of-run zero is owned by AssertUtils.inflight_clean (its TTL
    drain window tolerates slow CI); the sampler only pins the linkage.
    """
    if not samples:
        return False, "no inflight samples (endpoint unreachable in window)"
    requests_series = [s[3] for s in samples]
    sched_series = [s[1] for s in samples]
    peak_requests = max(requests_series)
    intermediates = sorted({v for v in requests_series if 0 < v < peak_requests})
    sched_monotonic = all(
        later <= earlier for earlier, later in zip(sched_series, sched_series[1:])
    )
    ok = (
        peak_batches == 1
        and peak_requests == n_requests
        and (bool(intermediates) if expect_intermediate else not intermediates)
        and sched_monotonic
    )
    return ok, (
        f"peak_batches={peak_batches}/1, peak_requests={peak_requests}/"
        f"{n_requests}, intermediate_steps={intermediates or 'none'}, "
        f"sched_series={sched_series[:8]}"
        f"{'...' if len(sched_series) > 8 else ''} "
        f"(monotonic_non_increasing={sched_monotonic})"
    )


def _timed_request(ops, rid: int, **kwargs) -> tuple:
    """schedule + consume to terminal, returning (err, duration_s).

    Client-side completion-duration caliber (schedule -> stream end):
    under BATCH dispatch the mock surfaces the first streamed output only
    at fetch completion, so this is the TTFT observable the client sees.
    """
    t0 = time.monotonic()
    try:
        resp = ops.schedule(rid, **kwargs)
        if resp.code != 200 or not resp.success:
            return f"schedule failed ({resp.code}): {resp.error_message}", None
        input_pb = None if resp.enqueued_by_master else ops.build_generate_input(rid)
        handle = ops.start_stream(resp, rid, input_pb=input_pb)
        ended = handle.wait_end(STREAM_TIMEOUT_S)
        if not ended or not handle.snap.completed or handle.snap.error:
            return handle.snap.error or "stream did not complete", None
        return None, time.monotonic() - t0
    except Exception as exc:
        return repr(exc), None


def _shape_gate(
    delta_batches: int, delta_requests: int, max_size: int, expected: tuple
) -> tuple:
    """Engine-side executed-batch shape gate (construction verification,
    NOT a verdict): returns (ok, detail) so the case can record the
    achieved vs expected split shape while the verdict stays on the
    master linkage — a gate miss means the constructed scenario differs
    from the intended one, the master assertions still run against the
    ACTUAL shape (that is the tested value)."""
    exp_b, exp_r, exp_max = expected
    ok = delta_batches == exp_b and delta_requests == exp_r and max_size == exp_max
    return ok, (
        f"shape_gate={'ok' if ok else 'MISMATCH'} "
        f"(construct executed={delta_batches}b/{delta_requests}r/max{max_size}, "
        f"intended {exp_b}b/{exp_r}r/max{exp_max})"
    )


def _drain_one(ops, rid, resp, wait_s: float):
    """Consume ONE fired request to terminal, recording its completion
    timestamp the moment the stream ends.  Returns (rid, completed, err,
    done_monotonic); done_monotonic is None when it never completed."""
    done = None
    err = None
    try:
        handle = ops.start_stream(resp, rid)
        ended = handle.wait_end(wait_s)
        if ended and handle.snap.completed and not handle.snap.error:
            done = time.monotonic()
        else:
            err = handle.snap.error or "stream did not complete"
    except Exception as exc:
        err = repr(exc)
    if done is None:
        try:
            ops.cancel(rid, resp)
        except Exception:
            pass
    return rid, done is not None, err, done


def _drain_fired_start(ops, fired: list, wait_s: float = 60.0):
    """Start CONCURRENT consumption of every fired request WITHOUT
    waiting — each worker's completion timestamp must reflect the ACTUAL
    completion event, so the streams have to be opened while the requests
    are still executing (opening them after an observation window would
    stamp already-finished streams with the caller's consume order —
    measuring nothing).  Returns (pool, futures); collect with
    _drain_fired_collect, then pool.shutdown(wait=True)."""
    pool = ThreadPoolExecutor(max_workers=max(1, len(fired)))
    futures = [pool.submit(_drain_one, ops, rid, resp, wait_s) for rid, resp in fired]
    return pool, futures


def _drain_fired_collect(futures: list) -> list:
    """Collect _drain_fired_start results as [(rid, completed, err,
    done_monotonic)] in fire order (a crashed worker surfaces as a
    failed outcome, never an exception past this boundary)."""
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except Exception as exc:
            outcomes.append((None, False, repr(exc), None))
    return outcomes


def _timed_wave_start(ops, base: int, n: int = 4, **kwargs):
    """Fire *n* requests 10ms apart (the _fire_regroup_wave coalescing
    shape) with each worker measuring its own schedule->end duration
    CONCURRENTLY — the TTFT-shape observable for the boundary case.
    Workers block on their streams the whole execution window; the
    caller runs its observation windows in the meantime and collects
    afterwards.  Returns (rids, pool, futures)."""
    rids = [ops.next_request_id(base) for _ in range(n)]
    pool = ThreadPoolExecutor(max_workers=n)
    futures = []
    for rid in rids:
        futures.append(pool.submit(_timed_request, ops, rid, **kwargs))
        time.sleep(0.01)
    return rids, pool, futures


def _two_cluster_split(values: list, sep: float = 1000.0) -> tuple:
    """Two-execution-batch separation check (arrival-order robust).

    The engine composes execution batches in ARRIVAL order — with the
    concurrent wave the arrival order is nondeterministic, so rids[:2]
    vs rids[2:] is NOT the batch split (observed in a live run: r3,r4
    composed batch #1 and finished 3s BEFORE the rids[:2] members).
    The ORDER contract the split actually pins: the two batches run
    SERIALLY — the four completion stamps cluster into two pairs, the
    pairs separated by > *sep* (each execution batch runs 3000ms) with
    the members INSIDE a pair settling together (an execution batch
    settles atomically on the engine; client-side the pair gaps are
    poll-granularity, orders of magnitude under sep).  *values* may be
    seconds (client done stamps) or milliseconds (engine end_ms) —
    *sep* is in the caller's unit.  Returns (ok, detail).
    """
    vals = sorted(v for v in values if v is not None and v > 0)
    if len(vals) != 4:
        return False, f"need 4 stamps, got {len(vals)}"
    early_pair_gap = vals[1] - vals[0]
    batch_gap = vals[2] - vals[1]
    late_pair_gap = vals[3] - vals[2]
    ok = batch_gap > sep and early_pair_gap <= sep and late_pair_gap <= sep
    return ok, (
        f"clusters=[[{vals[0]:.3f},{vals[1]:.3f}],"
        f"[{vals[2]:.3f},{vals[3]:.3f}]] "
        f"(intra {early_pair_gap:.3f}/{late_pair_gap:.3f} <= {sep}, "
        f"inter {batch_gap:.3f} > {sep})"
    )
