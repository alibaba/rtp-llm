from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import (
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    _cleanup_dynamic,
    _run_batch,
    _wait_master_alive,
    wait_for,
)
from ...registry import case
from ...support.engine_fault import (
    RECOVERY_EVICT_S,
    RECOVERY_SETTLE_S,
    _created_generation_count,
    _engine_ip_port,
    _ensure_started,
    _master_http,
    _master_log_offset,
    _prefill_endpoint_ledger,
    _recovery_env,
    _retire_count,
)


@case(
    "engine_fault_recovery_generation_bump",
    category="engine_fault",
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
        # Cascade hygiene: drain earlier residue on this env.
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

        # 修复（eval batch A）：docstring 四条承诺之一"recovered ledger
        # starts from zero"进 passed——E3 同项已断，族内对齐。
        passed = (
            retired
            and alive_back
            and generation_bumped
            and ledger_clean
            and recovery_ok
        )
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
