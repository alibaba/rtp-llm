from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ...context import CaseContext, rid_base
from ...engine_ops import (
    _fence_residue_stable,
    engine_inflight_clean,
    inject_type,
    inject_type_all,
)
from ...harness import wait_for
from ...registry import case
from ...support.engine_fault import (
    ENGINE_RECOVERY_WAIT_S,
    MASTER_EVICT_S,
    _fault_spec,
    _prefill_names,
)


@case(
    "engine_fault_crash_after",
    category="engine_fault",
    profiles=[
        "batch-window",
        "single-batch",
    ],  # crash fires at the EnqueueBatch entry (BATCH dispatcher only)
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

    Profile semantics (v2): the fault fires at the engine's
    EnqueueBatch entry (BATCH dispatcher only) and _fault_spec layers
    PRIORITY ordering on the ctx profile's own decision/dispatcher axes
    (profile-aware since the tier2 spec unpick) — so the declaration
    covers the BATCH-dispatch profiles (batch-window, single-batch);
    the NON_BATCH dispatch channel never reaches the EnqueueBatch entry,
    mechanically excluding single-nonbatch / window-nonbatch.
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
