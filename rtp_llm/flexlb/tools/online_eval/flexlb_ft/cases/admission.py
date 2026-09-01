"""Admission-category cases: the admission-gate contract.

Theme: requests the gates REFUSE must fail fast, loudly and typed — never
hang, never vanish, never leak inflight state — and once the pressure is
lifted the system must recover.  One refusal path per case:

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
                                  RESOURCE_EXHAUSTED fast reject on the
                                  submit path; recovery once the occupants
                                  terminate.
  admission_gate_no_starvation    no-starvation completeness: a gated
                                  request reaches a visible terminal state
                                  (explicit rejection — the Java master does
                                  NOT requeue gate rejections) and the
                                  healthy engine keeps accepting.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..context import CaseContext, CaseDef, rid_base
from ..engine_ops import (
    clear_type_all,
    engine_inflight_clean,
    inject_type,
    inject_type_all,
)
from ..grade import GradeReport
from ..harness import AssertUtils, EnvSpec, admission_config, default_perf

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
    source="gap G11: master outstanding-capacity admission (RESOURCE_EXHAUSTED fast reject)",
)
def admission_master_capacity(ctx: CaseContext):
    """Master-side unified admission: with
    capacity.maxOutstandingRequestsGlobal=2 and PRIORITY ordering, the
    submit path (PriorityScheduler.submit -> tryAcquireOutstandingPermit)
    fast-rejects every request beyond the global budget with
    RESOURCE_EXHAUSTED "master outstanding capacity exhausted" — a
    synchronous rejection, no queueing and no leak.  Once the in-flight
    occupants terminate, a sequential request must succeed again.

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


# ===========================================================================
# No-starvation completeness (scheduling_smoke.py S9, task #61 rebuild)
# ===========================================================================


@case(
    "admission_gate_no_starvation",
    requires=["enqueue_batch"],
    source="scheduling_smoke.py S9 (rebuilt around a real gate, task #61)",
)
def admission_gate_no_starvation(ctx: CaseContext):
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
    (PriorityScheduler.reduceDeliveryFailure; proven end-to-end by
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
    base = rid_base(ctx, "admission")
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
