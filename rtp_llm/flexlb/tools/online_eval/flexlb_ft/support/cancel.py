"""Shared cancel scenario components. Resource and timing semantics live here."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import grpc

from ..context import CaseContext, CaseDef, rid_base
from ..engine_ops import (
    _fence_residue_stable,
    clear_type_all,
    engine_inflight_clean,
    inject_type,
    inject_type_all,
)
from ..harness import (
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    ConfigOverride,
    EnvSpec,
    default_perf,
    wait_for,
    wait_for_port,
)

# Typed preemption-terminal codes (StrategyErrorType): 8429
# PRIORITY_PREEMPTED is the DECODE_ENGINE_OWNED victim's terminal.  The
# client stream's in-band error frame carries the proto enum CANCELLED (2),
# never the 8429 numeric (that rides the TaskInfoPB master channel only) —
# the mapping mirrors priority.py's _StreamTerminal (A1, Ryan P1-1).
_PB_ERROR_CANCELLED = 2
CODE_ENGINE_CANCELLED = 8429


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"]


def _all_engine_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", [])]


def _schedule_with_priority(ops, request_id: int, priority: int, **kwargs):
    """Schedule RPC carrying an explicit priority (proto field 14).

    EngineOps.build_schedule_request does not expose the priority kwarg
    yet; the legacy smoke client's schedule builder (since removed with the
    rest of the smoke family) proved the proto carries it ("Priority must be
    carried by the schedule protocol;
    embedding it only in unique_key metadata does not reach Auto-TPM
    admission").  Rather than widening engine_ops.py from the cancel
    category (other agents own the neighbouring modules), set the field
    on the built message here — protobuf messages are mutable.
    """
    req = ops.build_schedule_request(request_id, **kwargs)
    req.priority = priority
    stub = ops.schedule_pb2_grpc.FlexlbServiceStub(ops._channel(ops.master_target()))
    return stub.Schedule(req, timeout=30.0)


def _schedule_future(ops, request_id: int, **kwargs):
    """Fire-and-forget Schedule: a grpc Future whose cancel() aborts the
    in-flight Schedule RPC itself.

    Under BATCH dispatch the master completes the Schedule response only
    after the EnqueueBatch ACK (RequestRegistry.deliveryPublication runs
    on the ACK path), so an injected prefill ``enqueue_delay`` keeps the
    Schedule RPC in flight while the batch has already been claimed —
    exactly the window in which cancelling the client-side RPC triggers
    the master's inbound-context CancellationListener
    (FlexlbServiceImpl: Context.addListener → cancelUndeliveredRoute).
    """
    req = ops.build_schedule_request(request_id, **kwargs)
    stub = ops.schedule_pb2_grpc.FlexlbServiceStub(ops._channel(ops.master_target()))
    return stub.Schedule.future(req, timeout=30.0)


def _cancel_rpc_total(ops) -> int:
    """Sum of per-engine Cancel RPC counters from /snapshot."""
    snap = ops.snapshot()
    return sum(
        int(e.get("rpc_counts", {}).get("cancel", 0)) for e in snap.get("engines", [])
    )


def _engine_cancel_receipt_within(
    ops, rid: int, timeout_s: float = 5.0, since: Optional[float] = None
) -> tuple[bool, str]:
    """Engine-side cancel receipt within a tight propagation bound.

    The master→engine cancel channel is a real gRPC wiring
    (GrpcEngineCancelChannel), so the engine recording the rid in
    cancelled_rids is a second-scale expectation.  The 95s TTL drain
    window is a leak safety net, NOT an acceptable propagation path:
    waiting "eventually" conflates a correct cancel with TTL-swept
    cleanup — exactly where F1/F8-class findings hide (2026-09 eval
    batch A timing-contract upgrade).  *since* anchors the clock at the
    ops.cancel() issuance instant for callers whose poll starts after
    intermediate waits; the measured receipt latency lands in the
    detail.
    """
    t0 = time.monotonic() if since is None else since
    detail = "no poll yet"
    while True:
        ok, detail = ops.verify_engine_cancelled(rid)
        elapsed = time.monotonic() - t0
        if ok:
            if elapsed <= timeout_s:
                return True, (
                    f"{detail}, receipt={elapsed:.3f}s "
                    f"(within {timeout_s:.0f}s propagation contract)"
                )
            # Receipt exists but landed outside the window — the
            # TTL/fence sweep, not a timely cancel.
            return False, (
                f"{detail} but receipt={elapsed:.3f}s exceeds the "
                f"{timeout_s:.0f}s propagation contract (TTL sweep is NOT "
                "an acceptable cancel path)"
            )
        if elapsed >= timeout_s:
            break
        time.sleep(0.05)
    return False, (
        f"{detail}; no engine receipt within {timeout_s:.0f}s of cancel "
        f"issuance ({time.monotonic() - t0:.3f}s elapsed) — TTL sweep is "
        "NOT an acceptable cancel path"
    )


def _inflight_fingerprint(ops):
    """Master inflight fingerprint: scheduler count + per-endpoint
    (ip_port, inflight_batches, inflight_requests) rows.

    Same construction as the status family's homonym (equal
    fingerprints mean "no ledger mutation"); copied locally to keep the
    cancel category decoupled from status.py (parallel edits, 2026-09
    eval batch A).
    """
    data = ops.master_inflight()
    if data is None:
        return None

    def ep_rows(eps) -> tuple:
        rows = []
        for ep in eps or []:
            batches = ep.get("inflight_batches", 0)
            counted = len(batches) if isinstance(batches, list) else int(batches)
            rows.append(
                (
                    ep.get("ip_port", "?"),
                    counted,
                    int(ep.get("inflight_requests", 0) or 0),
                )
            )
        return tuple(rows)

    return (
        int(data.get("scheduler_inflight", 0)),
        ep_rows(data.get("prefill_endpoints")),
        ep_rows(data.get("decode_endpoints")),
    )


# ===========================================================================
# Cancel cases (cancel_smoke.py T1-T6, ported 1:1)
# ===========================================================================


# ===========================================================================
# Cancel-path anomaly case (anomaly_smoke.py E1 — the same contract seen
# from the client side of a failed request; rid_base family "anomaly" ->
# "cancel" in the category reorg)
# ===========================================================================


# ===========================================================================
# Git-session gap-analysis cases: the cancel contract around the
# deliveryClaimKind boundary.  Assertions pin the CONTRACT behaviour; cases
# predicted to fail before a parallel mock-engine capability lands carry an
# explicit finding note (docstring Prediction).
# ===========================================================================


# ===========================================================================
# HA cancel family (2026-09): the cancel contract across engine
# restarts, dead-prefill windows, decode retirement and transport-layer
# faults.  Production ground truth (code-audited):
#   * the master cancel is ONE-SHOT — EngineFenceCoordinator "never
#     retries and never owns a timer"; a TOMBSTONED ack settles the slot
#     immediately (resumeTombstoned) while ACCEPTED / NOT_FOUND / FAILED /
#     exceptions park in awaitAuthoritativeTerminal;
#   * a TRUE engine crash (crash_after) wipes all per-engine memory — a
#     restarted instance has NEVER SEEN pre-restart rids, so the master's
#     cancel answers TOMBSTONED and installs the ABSENT_FENCE tombstone
#     (later same-rid enqueues are 8429-rejected pre-admission);
#   * decode WorkerStatus terminals and decode generation retire close
#     open cancellation fences; prefill retire never closes fenced slots;
#     there is no fallback sweeper for cancellation first-cause slots (a
#     known production gap — the windows below are sized to the REAL
#     settle paths instead of relying on a nonexistent safety net);
#   * fencing is in-engine memory only: a second crash drops the tombstone
#     and a late Enqueue of the settled rid is ACCEPTED by the fresh
#     instance (documented design trade-off: the master ledger stays
#     settled — no resurrection — and the engine-side orphan computation
#     is bounded).
# ===========================================================================

# 3-strike health demotion + eviction window (engine_fault precedent).
MASTER_EVICT_S = 30.0
# Engine restart channel-reconnect settle window (engine_fault precedent).
ENGINE_RECOVERY_WAIT_S = 3.0
# A TOMBSTONED cancel settles the slot immediately — the client stream must
# close well inside this bound, far away from the 95s TTL drain net.
CANCEL_SETTLE_BOUND_S = 5.0


def _direct_enqueue(ops, addr: str, input_pb, batch_id: int):
    """Client-side EnqueueBatch probe straight at one engine's gRPC port.

    Bypasses the master entirely — the late-Enqueue / fence probes below
    must observe the ENGINE's admission decision, not the master's
    already-settled ledger view.
    """
    stub = ops.pb2_grpc.RpcServiceStub(ops._channel(addr))
    request = ops.pb2.EnqueueBatchRequestPB(
        batch_id=batch_id,
        dp_slots=[
            ops.pb2.EnqueueBatchDpSlotPB(
                dp_rank=0,
                requests=[ops.pb2.EnqueueBatchExternalInputPB(input=input_pb)],
            )
        ],
        fetch_attach_timeout_ms=30_000,
    )
    return stub.EnqueueBatch(request, timeout=10.0)


def _fence_rejected_8429(ack, rid: int) -> tuple:
    """True when the direct-enqueue ack carries exactly the typed 8429
    absent-fence rejection for rid (no successes, no admission)."""
    errors = list(ack.errors)
    rejected = (
        not ack.successes
        and len(errors) == 1
        and errors[0].request_id == rid
        and errors[0].error_info.error_code == 8429
    )
    detail = (
        f"successes={len(ack.successes)}, errors="
        f"{[(e.request_id, e.error_info.error_code) for e in errors]}"
    )
    return rejected, detail


def _crash_and_restart(ops, engine_name: str) -> tuple:
    """True-crash + restart cycle on one engine (crash_after n=1).

    The sacrificial request's own fate is the empty-ack uncertain path and
    is deliberately not asserted (engine_fault_crash_after precedent);
    what matters here is that ALL per-engine memory — running tasks,
    cancel tombstones, absent-fence records, RPC counters — is wiped, so
    the restarted instance has never seen any pre-restart rid.  Returns
    (alive_dropped, alive_restored).
    """
    inject_type(ops, engine_name, "crash_after", n=1)
    try:
        sacrificial = ops.next_request_id()
        ops.schedule(sacrificial, timeout_s=8.0)
    except Exception:
        # The crash may cut the RPC mid-flight — either way the port dies.
        pass
    dropped = wait_for(
        lambda: ops.master_alive_count("PREFILL") <= 0, MASTER_EVICT_S, 0.5
    )
    ops.start_engine(engine_name)  # clears fault config + enqueue counter
    restored = wait_for(
        lambda: ops.master_alive_count("PREFILL") >= 1, MASTER_EVICT_S, 0.5
    )
    time.sleep(ENGINE_RECOVERY_WAIT_S)
    return dropped, restored


def _restore_engines(ops) -> None:
    """Best-effort topology + fault restore for finally blocks."""
    try:
        for name, engine in ops.snapshot_by_name().items():
            if engine.get("stopped"):
                try:
                    ops.start_engine(name)
                except Exception:
                    pass
            try:
                inject_type(ops, name, "crash_after", enabled=False)
            except Exception:
                pass
    except Exception:
        pass


def _ha_env(ctx: CaseContext, label_suffix: str) -> tuple:
    """1P/1D dedicated env for the HA cancel family.

    One prefill keeps the crash trigger deterministic (every enqueue lands
    on prefill-0); one decode keeps the handoff target unambiguous.  The
    label embeds the profile so each profile gets its own env instance.
    """
    spec = EnvSpec(
        label=f"cancel_{label_suffix}_{ctx.profile}",
        n_prefill=1,
        n_decode=1,
        perf=default_perf(),
        master_profile=ctx.profile,
    )
    env = ctx.env_manager.ensure(spec)
    return ctx.engine_ops(env), env
