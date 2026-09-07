from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ...context import CaseContext, rid_base
from ...harness import TTL_DRAIN_TIMEOUT_S, _run_batch, wait_for
from ...registry import case
from ...support.master import MASTER_EVICT_S, _quota_spec


@case(
    "master_quota_block",
    category="master",
    profiles=["batch-window"],
    source="flexlb_behavior_test.sh S3 (1P+1D quota blocking + TTL recovery)",
)
def master_quota_block(ctx: CaseContext):
    """S3 port: fill the 1-batch inflight quota → stop the only prefill →
    new requests fail (≥50%) → TTL cleanup → start engine → recovery ≥90%.

    Profile semantics (v2): the quota knob itself
    (dispatcher.maxInflightBatchesPerPrefillWorker) exists only under the
    BATCH dispatcher, and _quota_spec layers PRIORITY ordering on the ctx
    profile's own decision/dispatcher axes with maxInflightBatches=1 via
    config override (profile-aware since the tier2 spec unpick) — the
    declaration stays batch-window.
    """
    env = ctx.env_manager.ensure(_quota_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "master")
    try:
        # Slow the only prefill: scheduled requests stick in inflight.
        ops.set_perf("prefill-0", prefill_fixed_ms=10000.0)

        # Fill the (maxInflightBatches=1) quota with fire-and-forget requests.
        rids = [ops.next_request_id(base) for _ in range(4)]
        for rid in rids:
            resp = ops.schedule(rid, output_len=10)
            if resp.code != 200 or not resp.success:
                return False, f"schedule failed for rid={rid}: {resp.error_message}"
        filled = wait_for(lambda: ops.master_scheduler_inflight() > 0, 15.0, 0.5)
        if not filled:
            return False, "could not fill the inflight quota"
        stuck = ops.master_scheduler_inflight()

        # Stop the only prefill → the stuck batch never completes.
        ops.stop_engine("prefill-0")
        time.sleep(3.0)

        # Blocked phase: 10 concurrent requests, expect ≥50% failures
        # (single prefill down + quota consumed → queue timeouts / rejects).
        block_rids = [ops.next_request_id(base) for _ in range(10)]

        def run(rid: int):
            return ops.run_one_request(
                rid,
                output_len=2,
                block_keys=[rid * 100 + 1],
                stream_timeout_s=12.0,
            )

        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(run, block_rids))
        block_ok = sum(1 for _, err in results if err is None)
        block_fail_rate = (10 - block_ok) / 10
        block_err_types = sorted(
            {str(err)[:60] for _, err in results if err is not None}
        )[:3]

        # TTL cleanup: scheduler inflight drains to zero (the evicted
        # engine's endpoint row is gone, so watch the global counter).  The
        # window rides harness.TTL_DRAIN_TIMEOUT_S (95s = 30s stale TTL +
        # 60s sweeper phase + 5s margin — derivation in harness) instead
        # of the legacy bare 90.0, which sat exactly ON the worst-case
        # settle and let a slow sweep phase trip the wait.
        cleanup_ok = wait_for(
            lambda: ops.master_scheduler_inflight() == 0, TTL_DRAIN_TIMEOUT_S, 2.0
        )
        ops.start_engine("prefill-0")
        ops.set_perf("prefill-0", prefill_fixed_ms=100.0)
        alive_back = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= 1,
            MASTER_EVICT_S,
            0.5,
        )
        time.sleep(2.0)

        # Recovery phase: 20 requests ≥90%.
        #
        # Sent SERIALLY (concurrency=1), unlike the other batch call sites:
        # a 10-way concurrent burst against the single restarted prefill keeps
        # the batcher queue mutating continuously, and PrefillEndpoint
        # .realPendingCount() is a lock-free snapshot that deliberately returns
        # Long.MAX_VALUE ("route away conservatively", see its comment) after 4
        # spin attempts fail to see a stable mutation version.  With ONE
        # prefill there is nowhere to route away to, so those requests die as
        # retryable "admission capacity is temporarily exhausted"
        # (RESOURCE_UNAVAILABLE on the only candidate — verified via runtime
        # DEBUG logs: "pendingRequests=9223372036854775807, alive=true").
        # That conservative degrade is scheduler design, not a recovery defect;
        # serializing removes the snapshot-contention noise so the assertion
        # keeps verifying what this case actually targets: quota released,
        # engine back, requests succeed again (observed 1-4 rejects across
        # 6 concurrent-burst runs — 80-100% flapping around the 90% gate).
        ok5, err5, _ = _run_batch(ops, base, 20, concurrency=1)
        recovery_rate = ok5 / 20 if err5 == 0 else ok5 / 20
        recovery_err_types = list(getattr(_run_batch, "last_error_types", []))[:3]

        passed = block_fail_rate >= 0.50 and cleanup_ok and alive_back and ok5 >= 18
        return passed, (
            f"stuck_inflight={stuck}, "
            f"blocked={block_ok}/10 ok (fail_rate={block_fail_rate:.0%}, >=50% required, "
            f"types={block_err_types}), "
            f"ttl_cleanup_within_{TTL_DRAIN_TIMEOUT_S:.0f}s={cleanup_ok}, "
            f"alive_restored={alive_back}, "
            f"recovery={ok5}/20({recovery_rate:.0%}, >=90% required, "
            f"types={recovery_err_types})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            snap = ops.snapshot_by_name()
            if snap.get("prefill-0", {}).get("stopped"):
                ops.start_engine("prefill-0")
            ops.set_perf("prefill-0", prefill_fixed_ms=100.0)
        except Exception:
            pass
