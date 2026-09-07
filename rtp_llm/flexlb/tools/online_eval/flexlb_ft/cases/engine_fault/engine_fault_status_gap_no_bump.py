from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import inject_type
from ...harness import TTL_DRAIN_TIMEOUT_S, AssertUtils, _cleanup_dynamic, _run_batch
from ...registry import case
from ...support.engine_fault import (
    E4_GAP_S,
    _created_generation_count,
    _engine_ip_port,
    _master_http,
    _master_log_offset,
    _recovery_env,
)


@case(
    "engine_fault_status_gap_no_bump",
    category="engine_fault",
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
