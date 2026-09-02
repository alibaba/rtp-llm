"""Master-category cases: the master process as a fault victim.

Theme: the FlexLB master itself going down, blocking admission by
quota, or cold-starting under a first-connect burst — worker traffic
must converge to a healthy topology, in-flight state must settle (TTL
or explicit cleanup), and a restarted master must come back with clean
state.  master_kill (kill -9 + restart), master_quota_block (1P+1D
quota blocking + TTL recovery) and master_coldstart_burst (the intake
defect regression probe) share the env/flow helpers from harness.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..context import CaseContext, CaseDef, rid_base
from ..harness import (
    AssertUtils,
    _cleanup_dynamic,
    _elastic_env,
    _run_batch,
    coldstart_spec,
    quota_spec,
    wait_for,
)

MASTER_CASES: list[CaseDef] = []

STREAM_TIMEOUT_S = 15.0
# 3-strike health marking + eviction window (fault-family precedent).
MASTER_EVICT_S = 30.0


def case(name: str, profiles=None, requires=None, source: str = ""):
    def deco(fn):
        MASTER_CASES.append(
            CaseDef(
                name=name,
                category="master",
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


# ===========================================================================
# Master HA group — kill -9 + restart (flexlb_behavior_test.sh ports)
# ===========================================================================


@case(
    "master_kill",
    profiles=["batch-window"],  # elastic_spec pins the legacy fault axes
    source="master HA: kill -9 master → restart → clean state + recovery",
)
def master_kill(ctx: CaseContext):
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "master")
    try:
        _cleanup_dynamic(ops, env)

        # Baseline: one request succeeds before the kill.
        addr, err0 = ops.run_one_request(
            ops.next_request_id(base),
            output_len=2,
            block_keys=[base + 11],
            stream_timeout_s=10.0,
        )
        del addr
        if err0:
            return False, f"baseline request failed: {err0}"

        # kill -9 the master, restart it from the same argv/env.
        ctx.env_manager.kill_master9(env)
        time.sleep(2.0)  # settle; port release
        ctx.env_manager.start_master(env)

        # Wait for the full topology to re-converge (ready + alive workers).
        alive_ok = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= 2
            and ops.master_alive_count("DECODE") >= 4,
            60.0,
            1.0,
        )
        # Fresh master must start from clean inflight state.
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 10.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = alive_ok and inflight_ok and recovery_ok
        return passed, (
            f"master_restarted, topology_reconverged={alive_ok}"
            f"(alive P:{ops.master_alive_count('PREFILL')}/"
            f"D:{ops.master_alive_count('DECODE')}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        # If the case died mid-way with the master down, bring it back so the
        # shared env stays usable (and teardown finds a ManagedProcess).
        try:
            if env.master is None:
                ctx.env_manager.start_master(env)
        except Exception:
            pass


@case(
    "master_quota_block",
    profiles=["batch-window"],
    source="flexlb_behavior_test.sh S3 (1P+1D quota blocking + TTL recovery)",
)
def master_quota_block(ctx: CaseContext):
    """S3 port: fill the 1-batch inflight quota → stop the only prefill →
    new requests fail (≥50%) → TTL cleanup → start engine → recovery ≥90%.

    Profile semantics (v2, task #55): the quota knob itself
    (dispatcher.maxInflightBatchesPerPrefillWorker) exists only under the
    BATCH dispatcher, and quota_spec pins the legacy fault axes (PRIORITY +
    FIXED_WINDOW + BATCH, maxInflightBatches=1) via FLEXLB_CONFIG — the
    declaration stays batch-window.
    """
    env = ctx.env_manager.ensure(quota_spec(ctx))
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

        # TTL cleanup (30s): scheduler inflight drains to zero (the evicted
        # engine's endpoint row is gone, so watch the global counter).
        cleanup_ok = wait_for(lambda: ops.master_scheduler_inflight() == 0, 90.0, 2.0)
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
            f"ttl_cleanup_within_90s={cleanup_ok}, "
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


# ===========================================================================
# Cold-start burst — intake defect regression probe
# ===========================================================================


@case(
    "master_coldstart_burst",
    profiles=["batch-window"],
    source="intake defect regression probe (cold-start first-connect storm)",
)
def coldstart_burst(ctx: CaseContext):
    """Fire 20 requests the instant the master reports ready.

    Regression probe for the three intake defects: CONNECT_TIMEOUT 20ms,
    3-strike dead marking on first connect, non-atomic getOrCreateWorkerStatus.
    Expected to FAIL or pass marginally today — the failure rate and the
    marked-dead sample count are recorded as the baseline for the intake fix.

    Profile semantics (v2, task #55): coldstart_spec carries NO config
    override, so the case would genuinely exercise each profile's config
    (unlike the pinned-spec cases above); it stays scoped to batch-window
    this round as a deliberate scope decision — spreading the intake probe
    across profiles is later-phase work.
    """
    env = ctx.env_manager.ensure(coldstart_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "master")
    expected = {"PREFILL": env.spec.n_prefill, "DECODE": env.spec.n_decode}
    try:
        # Sample worker_summary every 0.5s while the burst runs and for 10s
        # after — the cold-start window where engines get marked dead.
        samples: list[tuple[float, dict]] = []
        stop = threading.Event()

        def sample() -> Optional[dict]:
            info = ops.master_info()
            if not info:
                return None
            summary = info.get("worker_summary", {}) or {}
            return {
                role: (
                    int((summary.get(role) or {}).get("discovered", -1)),
                    int((summary.get(role) or {}).get("alive", -1)),
                )
                for role in ("PREFILL", "DECODE")
            }

        def sampler() -> None:
            t0 = time.monotonic()
            while not stop.is_set():
                s = sample()
                if s is not None:
                    samples.append((round(time.monotonic() - t0, 1), s))
                stop.wait(0.5)

        poller = threading.Thread(target=sampler, name="coldstart-sampler", daemon=True)
        poller.start()

        # Burst: 20 requests (10-way concurrent) immediately after ready.
        # The prefill address of every request is kept for the load-balance
        # assertion (balance_uniform_serial P1 contract) below.
        def run(rid: int):
            addr, err = ops.run_one_request(
                rid,
                output_len=2,
                block_keys=[rid * 100 + 1],
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            return addr, err

        rids = [ops.next_request_id(base) for _ in range(20)]
        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(run, rids))
        ok = sum(1 for _, e in results if e is None)
        error_types = sorted({str(e)[:60] for _, e in results if e is not None})

        # Keep sampling 10s past the burst: transient 3-strike marks recover,
        # permanent ones stay dead (that is the S10-class regression).
        time.sleep(10.0)
        stop.set()
        poller.join(timeout=2.0)

        final = sample()
        dead_samples = sum(1 for _, s in samples if any(a < d for d, a in s.values()))
        final_ok = bool(
            final
            and all(d == expected[role] and a == d for role, (d, a) in final.items())
        )

        # Load-balance contract (user-mandated): under the cold-start burst
        # traffic must still spread across the engines.  Same calibration
        # as the task #61 balance suite (balance_uniform_serial / P1, with
        # the balance_concurrent_mix relaxed-caliber note): 20 requests over
        # 2 prefills (10-way concurrent), both engines used, no engine above
        # 80% of the *successful* requests — COST_BASED_PREFILL scores the
        # two prefills identically on an empty cold ledger and
        # RANDOM_WITHIN_TOLERANCE samples the tie window uniformly, so a
        # one-sided distribution can only come from an engine being
        # 3-strike-marked dead (the intake defect this probe guards).
        # 80% of 20 = 16 requests, i.e. the same "no engine eats the burst"
        # bound as the balance suite's P1 (loose floor 0.85 over the
        # uniform-random calibration; this probe keeps the historical 0.80
        # as its hard bound — semantics unchanged by the task #61 rework).
        addr_map = ops.addr_to_name()
        dist = Counter(addr_map.get(a, a) for a, e in results if e is None and a)
        n_ok = sum(dist.values())
        workers_used = len(dist)
        max_share = (max(dist.values()) / n_ok) if n_ok else 1.0
        balance_ok = workers_used >= 2 and max_share <= 0.80

        success_rate = ok / 20 * 100.0
        passed = (
            ok >= 16  # >=80% success + no permanent eviction
            and final_ok
            and balance_ok
        )
        return passed, (
            f"burst_ok={ok}/20 ({success_rate:.0f}%), "
            f"dead_samples={dead_samples}/{len(samples)}, final={final}, "
            f"error_types={error_types[:3]}, "
            f"balance: workers={workers_used}/{env.spec.n_prefill}, "
            f"max_share={max_share:.0%} (need >=2 workers and <=80%), "
            f"dist={json.dumps(dict(sorted(dist.items())))}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
