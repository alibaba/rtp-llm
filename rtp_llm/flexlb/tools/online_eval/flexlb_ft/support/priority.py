"""Shared priority scenario components. Resource and timing semantics live here."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import grpc

from ..context import CaseContext, CaseDef, rid_base
from ..engine_ops import (
    _fence_residue_stable,
    engine_inflight_clean,
    inject_type,
    parse_prometheus_samples,
)
from ..grade import GradeReport
from ..harness import (
    OMIT,
    AssertUtils,
    ConfigOverride,
    EnvSpec,
    default_perf,
    render_env,
    wait_for,
)
from .admission import MOCK_TOTAL_KV_TOKENS

# ===========================================================================
# Constants: error codes (StrategyErrorType), proto enum field 9, timings
# ===========================================================================

CODE_OK = 200
# NO_AVAILABLE_WORKER — PREFILL_QUEUED/DECODE_RESERVED victim "yielded to
# higher-priority request", retryable, engine never saw (part of) it.
CODE_YIELDED = 8400
# NO_PREFILL_WORKER — route/commit failure family (retryable).
CODE_NO_PREFILL = 8402
# DECODE_ENGINE_OWNED victim typed CANCELED, rides grpc-status-details-bin.
CODE_ENGINE_CANCELLED = 8429
CODE_ADMISSION_TIMEOUT = 8430
CODE_RESOURCE_EXHAUSTED = 8431
CODE_QUEUE_FULL = 8502
CODE_QUEUE_TIMEOUT = 8503  # dead code in the master (never emitted)
# BATCH_DISPATCH_FAILED — queue full + eviction fallback DECLINED (the
# code-level correction to the design's "8402 family" for that path).
CODE_QUEUE_REJECTED = 8510

#: Actual plain queue-timeout terminal.  Implementation-period finding
#: (code-level): StrategyErrorType.QUEUE_TIMEOUT (8503) is never referenced
#: by the master — an ordinary (non-priority-admission) queued request that
#: expires goes through RequestLifecycleCoordinator.timeoutEntry's fallback
#: buildErrorResponse(entry.timeoutErrorType()) where deadlineErrorType is
#: BATCH_SLO_EXPIRED (RequestSlot.java:43, configured at registration,
#: RequestLifecycleCoordinator.java:281).  The queue-head dropHead path
#: (SingleRequestGroupPolicy → onExpired → DEADLINE_EXCEEDED cancellation)
#: resolves to the same deadlineErrorType.  Design §2.2 listed the family as
#: {8503, 8402, 8430}; the first e2e run records the observed distribution.
CODE_SLO_EXPIRED = 8511

# A4 (Mark P1-2, PR3): BATCH dispatcher family reservation — the group
# token-capacity / plan-conflict codes only fire on the BATCH dispatcher,
# which the current priority family base (SINGLE + NON_BATCH) never enters;
# constants reserved so atpm_error_code_family's segment-4 skeleton has
# them in place for a future priority-batch variant.
CODE_BATCH_TOKEN_CAPACITY = 8514
CODE_SCHEDULER_PLAN_CONFLICT = 8515
# A3 (Mark P1-1): INVALID_REQUEST — RequestLifecycleCoordinator.register's
# duplicate request_id rejection (putIfAbsent, coordinator L262-268:
# "duplicate request_id: <rid>"), observable black-box through the
# schedule RPC response.
CODE_INVALID_REQUEST = 8406
# A1 (Ryan P1-1): ErrorCodePB.CANCELLED enum value — the ONLY code the mock
# engine ever puts on a client-stream error frame (RpcErrorPB; both the
# client-cancel and preemption-cancel paths emit this enum, NOT the 8429
# numeric — 8429 rides the TaskInfoPB master channel only).
_PB_ERROR_CANCELLED = 2

# ScheduleFailureReasonPB (proto field 9)
REASON_UNSPECIFIED = 0
REASON_HIGHER = 1
REASON_SAME = 2
REASON_RESOURCE = 3
REASON_NAMES = {
    0: "UNSPECIFIED",
    1: "HIGHER_PRIORITY_AHEAD",
    2: "SAME_PRIORITY_AHEAD",
    3: "RESOURCE_EXHAUSTED",
}

# The route/commit rejection family (queue full, eviction infeasible or
# declined) surfaces as 8402 or 8510 depending on which master stage
# rejected — both are asserted as one family, actual codes recorded for
# first-e2e calibration (design §2.4 atpm_error_code_family note).
ROUTE_REJECT_FAMILY = (CODE_NO_PREFILL, CODE_QUEUE_REJECTED)

# The observed terminal family for a decode-role-blocked incoming under
# EV-2 (decode eviction never fires — see _design_final_pattern's EV-1
# history note and the E9/E11 probes): the ordinary route fails on the
# strict decode KV gate (8403 NO_DECODE_WORKER) or the route/commit
# family, and the admission fallback cannot repair it, so the client
# keeps a rejection from this family.  Actual codes are recorded in the
# case details for calibration.
EV2_REJECT_FAMILY = (8403,) + ROUTE_REJECT_FAMILY + (CODE_RESOURCE_EXHAUSTED,)

# A6 (Daniel P2-2): flip-runbook anchor for the two empirically
# established behaviour baselines.  Every EV-1/EV-2 assertion detail
# carries its key ("[EV-1]" / "[EV-2]"); once the Java side fixes the
# underlying behaviour, grep these keys to enumerate every assertion
# point that must be flipped back to the designed shape.
# [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
# (commit 6ad0315f10, 2026-08-31): capacity blocking now parks EVERY
# submitter in the pull-based coordinator (priority desc + FIFO
# tiebreak) — the design-final assertions were restored from the EV-1
# downgrades; grep [EV-1-FIXED] for every flip point.  EV-2 remains the
# live downgrade baseline (decode eviction still unreachable).
EXPECTED_BASELINES = {
    "EV-1": (
        "FIXED (flipped at intake3 PendingPlacementCoordinator, commit "
        "6ad0315f10): capacity blocking parks every submitter (pull-based, "
        "priority desc + FIFO); all-200 design-final dispatch shape — "
        "zero route-reject (was: single park slot, later submitters "
        "route-reject {8402, 8510})"
    ),
    "EV-2": (
        "decode eviction never fires — zero 8400/8429 victims, the "
        "incoming keeps an EV2_REJECT_FAMILY rejection"
    ),
}

STREAM_WAIT_S = 35.0
# Deterministic arrival ordering: sequential fire submits with a small
# inter-fire gap so enqueuedAtMs order == submit order (victim selection
# and same-priority FIFO depend on it; design §2.5 pitfall 5/12).
FIRE_GAP_S = 0.15
# Master status-poll period is 1s (status_rpc_ms) — injection settle.
PERF_SETTLE_S = 1.5
# ===========================================================================
# Fire infrastructure: tracked schedule + NON_BATCH direct-stream terminal
# ===========================================================================
#
# Why not reuse the shared per-category fire helpers (cases/kv.py /
# cases/admission.py _fire_request & friends): those collapse a failed
# schedule into a "schedule failed: <message>" string and keep repr(exc) —
# the StrategyErrorType code and the grpc-status-details-bin raw code
# (8400/8429 typed terminals) would both be unobservable, and every atpm_*
# assertion needs exactly those (design §4.5 allows "import or copy" — the
# copy exists because the observation surface differs).


@dataclass
class _Fire:
    """One fire-and-forget request, tracked from submit to terminal."""

    rid: int
    kwargs: dict
    submitted_s: float = 0.0  # monotonic at submit
    settled_s: float = 0.0  # monotonic when schedule() returned
    resp: object = None  # FlexlbScheduleResponsePB (None on RPC error)
    rpc_error: Optional[str] = None
    terminal: Optional[object] = None  # _StreamTerminal (NON_BATCH direct)

    @property
    def code(self):
        if self.rpc_error or self.resp is None:
            return None
        return int(self.resp.code)

    @property
    def reason(self):
        if self.rpc_error or self.resp is None:
            return None
        return int(self.resp.admission_reject_reason)

    @property
    def ok(self) -> bool:
        return (
            self.rpc_error is None
            and self.resp is not None
            and self.resp.code == CODE_OK
            and self.resp.success
        )


class _StreamTerminal:
    """Background NON_BATCH direct-stream consumer capturing the typed
    terminal: completion, gRPC status, the trailing-metadata raw error
    code (grpc-status-details-bin) AND the in-band error frame
    (GenerateOutputsPB.error_info — see _consume; the trailing-metadata-
    only paradigm inherited from priority_preemption_smoke.py was
    incomplete: the mock engine terminates error streams IN-BAND, so the
    RpcError branch alone never sees engine-side cancellations).

    The shared per-category stream paths cannot be reused: they store
    ``repr(exc)`` and drop trailing_metadata, which is where the raw code
    lives.
    """

    def __init__(self, ops, resp, rid: int, kwargs: dict):
        target = ops.prefill_addr(resp)
        if not target:
            raise RuntimeError("schedule response has no PREFILL address")
        stub = ops.pb2_grpc.RpcServiceStub(ops._channel(target))
        input_pb = ops.build_generate_input(rid, **kwargs)
        ops._copy_role_addrs(input_pb, resp)
        self.call = stub.GenerateStreamCall(input_pb, timeout=120.0)
        self.rid = rid
        self.completed = False
        self.grpc_code = None
        self.raw_error_code = None
        self.error_text = None
        self.inband_error_code = None
        self.inband_error_message = None
        self.terminated_s = None
        self._pb2 = ops.pb2
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def _consume(self) -> None:
        try:
            for output in self.call:
                # A1 (Ryan P1-1): the mock engine terminates error streams
                # IN-BAND — a GenerateOutputsPB carrying error_info (proto
                # field 9, RpcErrorPB) is the last frame, then
                # observer.onCompleted() (JavaMockEngineCluster.java
                # L988-1010), so the client stream ends with gRPC status
                # OK and the RpcError/trailing-metadata branch below never
                # fires for engine-side cancellations.  Record the frame
                # here; the typed mapping happens after the loop.
                try:
                    if output.HasField("error_info"):
                        self.inband_error_code = int(output.error_info.error_code)
                        self.inband_error_message = output.error_info.error_message
                except AttributeError:
                    pass
            if self.inband_error_code is None:
                # No error frame observed — the stream genuinely completed
                # (flatten finished flags), so the terminal stays CODE_OK.
                self.completed = True
            else:
                # In-band typed terminal.  The engine only ever puts
                # ErrorCodePB.CANCELLED (2) on the stream frame — for BOTH
                # the client-cancel and the preemption-cancel paths
                # (JavaMockEngineCluster L1232-1238 / L1375-1385; the 8429
                # numeric rides the TaskInfoPB master channel only), so
                # the enum maps onto CODE_ENGINE_CANCELLED (8429), the
                # preemption-family terminal.  Any other value records
                # as-is for first-run calibration.
                # TODO(A9-5, EV-2 flip): once the Java side makes decode
                # eviction reachable, this branch becomes the FIRST real
                # 8429 observation point (w2_owned in
                # atpm_preempt_decode_engine_owned) — do not remove.
                if self.inband_error_code == _PB_ERROR_CANCELLED:
                    self.raw_error_code = CODE_ENGINE_CANCELLED
                else:
                    self.raw_error_code = self.inband_error_code
                self.error_text = (
                    f"in-band error frame: code={self.inband_error_code} "
                    f"msg={self.inband_error_message}"
                )
        except grpc.RpcError as exc:
            # Client-side cancellation is not a typed terminal.
            if exc.code() != grpc.StatusCode.CANCELLED:
                self.grpc_code = exc.code()
                self.raw_error_code = _extract_raw_error_code(
                    self._pb2, _trailing_metadata(exc)
                )
                self.error_text = str(exc)
        except Exception as exc:
            self.error_text = repr(exc)
        finally:
            self.terminated_s = time.monotonic()

    def wait(self, timeout_s: float = STREAM_WAIT_S) -> bool:
        self.thread.join(timeout_s)
        if self.thread.is_alive():
            try:
                self.call.cancel()
            except Exception:
                pass
            self.thread.join(5.0)
            return False
        return True

    def cancel(self) -> None:
        try:
            self.call.cancel()
        except Exception:
            pass


def _trailing_metadata(exc):
    try:
        meta = exc.trailing_metadata()
        return meta if meta else None
    except Exception:
        return None


def _extract_raw_error_code(pb2, metadata):
    """grpc-status-details-bin -> ErrorDetailsPB.error_code (raw typed code)."""
    if not metadata:
        return None
    for key, value in metadata:
        if key == "grpc-status-details-bin":
            try:
                details = pb2.ErrorDetailsPB.FromString(value)
                return int(details.error_code)
            except Exception:
                return None
    return None


def _fire(ops, rid: int, **kwargs) -> _Fire:
    """Blocking fire: schedule() (settles at decision/terminal time —
    NON_BATCH parks capacity-blocked requests) + open the direct stream on
    success.  Designed to run on an executor thread."""
    fr = _Fire(rid=rid, kwargs=kwargs, submitted_s=time.monotonic())
    sched_kwargs = dict(kwargs)
    # Deep-backlog choreographies settle late (serial prefill chains);
    # the gRPC deadline must not fire before the queueTimeout does.
    sched_kwargs["timeout_s"] = 90.0
    try:
        resp = ops.schedule(rid, **sched_kwargs)
    except Exception as exc:
        fr.rpc_error = repr(exc)
        fr.settled_s = time.monotonic()
        return fr
    fr.resp = resp
    fr.settled_s = time.monotonic()
    if fr.ok and not resp.enqueued_by_master:
        try:
            fr.terminal = _StreamTerminal(ops, resp, rid, kwargs)
        except Exception as exc:
            fr.rpc_error = f"direct stream failed to open: {exc!r}"
    return fr


def _fire_batch(ops, specs, gap_s: float = FIRE_GAP_S) -> list:
    """Sequential fire with inter-submit gap (deterministic arrival order);
    returns the _Fire list after every schedule has settled."""
    pool = ThreadPoolExecutor(max_workers=max(1, len(specs)))
    try:
        futures = []
        for rid, kwargs in specs:
            futures.append(pool.submit(_fire, ops, rid, **kwargs))
            time.sleep(gap_s)
        return [f.result() for f in futures]
    finally:
        pool.shutdown(wait=True)


def _drain(ops, fires: list, wait_s: float = STREAM_WAIT_S) -> list:
    """Wait every fired request to its terminal; returns
    [(rid, ok, code, detail)] — schedule failures carry their proto code,
    stream failures the raw typed code."""
    outcomes = []
    for fr in fires:
        if fr.rpc_error or fr.resp is None:
            outcomes.append((fr.rid, False, None, f"rpc: {fr.rpc_error}"))
            continue
        if not fr.ok:
            outcomes.append(
                (
                    fr.rid,
                    False,
                    fr.code,
                    f"reason={REASON_NAMES.get(fr.reason, fr.reason)} "
                    f"msg={str(fr.resp.error_message)[:90]}",
                )
            )
            continue
        term = fr.terminal
        if term is None:  # enqueued_by_master path (BATCH) — not used here
            outcomes.append((fr.rid, True, CODE_OK, "enqueued"))
            continue
        ended = term.wait(wait_s)
        if ended and term.completed:
            outcomes.append((fr.rid, True, CODE_OK, "completed"))
        else:
            outcomes.append(
                (
                    fr.rid,
                    False,
                    term.raw_error_code,
                    f"grpc={term.grpc_code} raw={term.raw_error_code} "
                    f"err={str(term.error_text)[:90]}",
                )
            )
    return outcomes


def _cancel_all(ops, fires: list) -> None:
    """Best-effort cleanup for fire entries that never settled."""
    for fr in fires:
        if fr.terminal is not None and fr.terminal.thread.is_alive():
            fr.terminal.cancel()
        elif fr.ok:
            try:
                ops.cancel(fr.rid, fr.resp)
            except Exception:
                pass


def _poll_engine_pending(
    ops, engine_name: str, min_pending: int, timeout_s: float = 6.0
) -> bool:
    """Engine-side proof that a fired request was really dispatched (the
    wave-wheel synchronization point, design §3.3 element 4)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = ops.snapshot_by_name().get(engine_name, {})
        if info.get("waiting", 0) + info.get("running", 0) >= min_pending:
            return True
        time.sleep(0.1)
    return False


def _poll_decode_running(ops, rid: int, timeout_s: float = 10.0) -> bool:
    """Wait until *rid* is RUNNING on a decode engine (engine-accepted
    observation for DECODE_ENGINE_OWNED choreographies)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for engine in ops.snapshot().get("engines", []):
            if engine.get("role") != "decode":
                continue
            lc = engine.get("request_lifecycle", {}).get(str(rid), {})
            if lc.get("end_state") == "running" or (
                lc.get("running_ms") and not lc.get("end_state")
            ):
                return True
        time.sleep(0.1)
    return False


def _prefill_names(ops) -> list:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"]


def _decode_names(ops) -> list:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "decode"]


def _decode_pressure_guardrail(ops, decode_names: list) -> tuple:
    """A7 (Mark P2-1): assert every decode endpoint is actually in the
    needs-eviction state — available_kv_tokens drained to <= 0 in the
    engine snapshot (JavaMockEngineCluster getSnapshot reports
    max(0, totalKv - effectiveActiveKv)) — BEFORE the incoming fires.
    An endpoint that silently escaped saturation lets the ordinary route
    succeed and the choreography degrades to a plain 8402 route-reject
    (or a normal completion) with zero eviction evidence, which is
    indistinguishable from the EV-2 baseline.  Returns (ok, evidence)."""
    evidence = []
    ok = True
    try:
        snap = ops.snapshot_by_name()
    except Exception as exc:
        return False, f"snapshot failed: {exc!r}"
    for name in decode_names:
        entry = snap.get(name) or {}
        try:
            avail = int(entry.get("available_kv_tokens", -1))
        except (TypeError, ValueError):
            avail = -1
        evidence.append(f"{name}:{avail}")
        if avail > 0:
            ok = False
    return ok, ", ".join(evidence)


def _single_prefill(ops):
    names = _prefill_names(ops)
    return names[0] if len(names) == 1 else None


# ===========================================================================
# Dispatch-order observation (design §3.3 arbitration chain)
# ===========================================================================


def _prefill_lifecycle(ops, rid: int):
    """The rid's request_lifecycle entry on a PREFILL engine (dispatch-order
    anchor).  The mock cluster is one JVM — running_ms is comparable across
    engines."""
    for engine in ops.snapshot().get("engines", []):
        if engine.get("role") != "prefill":
            continue
        lc = engine.get("request_lifecycle", {}).get(str(rid))
        if lc:
            return lc
    return None


def _decode_lifecycle(ops, rid: int):
    for engine in ops.snapshot().get("engines", []):
        if engine.get("role") != "decode":
            continue
        lc = engine.get("request_lifecycle", {}).get(str(rid))
        if lc:
            return lc
    return None


def _dispatch_order(ops, fires: list) -> list:
    """Prefill running_ms ascending == master dispatch order, with the client
    schedule() settle order arbitrating same-millisecond conflicts (design
    §3.3: SINGLE dispatches one head per cycle; under a
    maxInflightRequestsPerPrefillWorker=1 window the running_ms gaps equal
    the prefill execution time, so conflicts are rare — the arbitration is
    the fallback).  Returns the ordered rid list; queue-timeout terminals
    (never dispatched) sort last."""
    rows = _dispatch_rows(ops, fires)
    return [r[2] for r in rows]


def _dispatch_rows(ops, fires: list) -> list:
    """(running_ms, settle_rank, rid) tuples, sorted — for detail reporting."""
    settled = sorted(fires, key=lambda f: (f.settled_s, f.rid))
    settle_rank = {fr.rid: i for i, fr in enumerate(settled)}
    rows = []
    for fr in fires:
        lc = _prefill_lifecycle(ops, fr.rid) or {}
        running_ms = lc.get("running_ms")
        rows.append(
            (
                running_ms if running_ms is not None else -1,
                settle_rank.get(fr.rid, 1 << 30),
                fr.rid,
            )
        )
    rows.sort(key=lambda r: (r[0] if r[0] >= 0 else 1 << 60, r[1], r[2]))
    return rows


def _inversion_ratio(order: list, priorities: dict, exclude=None) -> float:
    """PR1 calibre: (high, low) pairs dispatched inverted / total
    cross-priority pairs, computed on the REAL dispatch order (engine
    running_ms asc + settle-rank arbitration).  ``exclude`` removes the
    design-final first parker (and any pre-wave running placeholder)
    from the scoring set — under the intake3 pull model the wave's first
    submitter legitimately wins the first release slot ahead of higher
    priorities, so counting it would blur the inversion signal
    ([EV-1-FIXED]).  0.0 under a deterministic choreography."""
    skip = set(exclude or ())
    pos = {rid: i for i, rid in enumerate(order)}
    rids = [r for r in priorities if r in pos and r not in skip]
    inversions = 0
    total = 0
    for i, a in enumerate(rids):
        for b in rids[i + 1 :]:
            pa, pb = priorities[a], priorities[b]
            if pa == pb:
                continue
            hi, lo = (a, b) if pa > pb else (b, a)
            total += 1
            if pos[hi] > pos[lo]:
                inversions += 1
    return inversions / total if total else 0.0


def _group_order_ok(order: list, rids: list) -> bool:
    """True when *rids* appear in the given relative order within *order*
    (same-priority FIFO inside a group)."""
    pos = {rid: i for i, rid in enumerate(order)}
    seq = [pos[r] for r in rids if r in pos]
    return seq == sorted(seq)


# EV-1 history — FIXED at intake3 (2026-08-31): the former single-park
# slot baseline (only a wave's first submitter parked; every later
# submitter route-rejected {8402, 8510} regardless of priority — probes
# E8/E8b/E8c/E10, CostBasedPrefillStrategy evaluateCandidates dropping
# BLOCKED projections) was superseded by the pull-based
# PendingPlacementCoordinator (commit 6ad0315f10; park path
# RequestScheduler.java L95-112): capacity-blocked submitters park in a
# WaitBucket (TreeSet, ORDER = priority desc + sequence asc) and are
# re-pulled on every capacity release.  [EV-1-FIXED] probe evidence (ph +
# 30a/30b/50a/50b/70a/70b wave, prefill_fixed_ms=3000): all code=200,
# dispatch order [ph, 30a (first parker), 70a, 70b, 50a, 50b, 30b],
# gaps ~3015ms — the wave's FIRST submitter legitimately wins the FIRST
# release slot; every later release follows strict priority desc +
# same-level FIFO.
def _design_final_pattern(
    ops, fires: list, ordered_rids: list, priorities: dict, fifo: bool = False
) -> tuple:
    """Design-final (intake3 PendingPlacementCoordinator) shape classifier
    for one wave's dispatch order.

    Returns (first_parker_rid, shape_ok, wave_dispatch_order): shape_ok is
    True when the wave's dispatch order (engine running_ms asc +
    settle-rank arbitration) equals [first submitter — the wave's first
    parker, by design] + remaining rids sorted by (priority desc, submit
    order) — or pure submit order under ``fifo``.  The first-parker
    precedence is the DESIGNED pull-model behaviour, not an inversion;
    the flip contract lives in EXPECTED_BASELINES ("[EV-1-FIXED]")."""
    order = _dispatch_order(ops, fires)
    wave_set = set(ordered_rids)
    wave_order = [r for r in order if r in wave_set]
    first_parker = ordered_rids[0] if ordered_rids else None
    if len(wave_order) != len(ordered_rids):
        return first_parker, False, wave_order
    rest = ordered_rids[1:]
    if fifo:
        expected_rest = list(rest)
    else:
        submit_rank = {rid: i for i, rid in enumerate(rest)}
        expected_rest = sorted(
            rest, key=lambda r: (-priorities.get(r, 0), submit_rank[r])
        )
    shape_ok = wave_order == [ordered_rids[0]] + expected_rest
    return first_parker, shape_ok, wave_order


# ===========================================================================
# FLEXLB_CONFIG factory + EnvSpec factories (design §4.3 env plan)
# ===========================================================================

_PREEMPT_PQ = {"allowed_victim_stages": ["PREFILL_QUEUED"]}
_PREEMPT_DECODE = {
    "allowed_victim_stages": ["DECODE_RESERVED", "DECODE_ENGINE_OWNED"],
    "engine_cancellation": {"ack_timeout_ms": 50, "completion_timeout_ms": 1000},
}
# All-three-stage superset (the live preemption config — cancel.py's
# cancel_preemption_victim runs exactly this set): DECODE_ENGINE_OWNED
# REQUIRES the engineCancellation block (FlexlbConfigValidator rejects
# the stage set otherwise); ack 50ms / completion 1000ms mirror the
# ft-line precedent.
_PREEMPT_ALL_STAGES = {
    "allowed_victim_stages": [
        "PREFILL_QUEUED",
        "DECODE_RESERVED",
        "DECODE_ENGINE_OWNED",
    ],
    "engine_cancellation": {"ack_timeout_ms": 50, "completion_timeout_ms": 1000},
}
# Metric-plane exposure for the live-eviction family: the default
# critical-only filter hides auto_tpm.* (FLEXLB_MONITOR_MODE=all is dead
# on the v2 line — see _q3_spec), so the whitelist carries the exposure.
# One bare prefix entry (WhitelistMetricsFilterConfig.matches is a
# promName.startsWith) covers the whole family the eviction cases
# assert on: auto_tpm.victim.count / auto_tpm.priority_preempt.count /
# auto_tpm.victim.kv_tokens / auto_tpm.decode.reserved.count.
_MONITOR_AUTO_TPM_ENV = {"FLEXLB_MONITOR_METRIC_WHITELIST": "flexlb_auto_tpm"}


def _prio_config(
    *,
    ordering: str = "priority",
    preemption: Optional[dict] = None,
    default_priority: Optional[int] = None,
    queue_timeout_ms: Optional[int] = None,
    max_outstanding: Optional[int] = None,
    max_inflight: Optional[int] = 1,
    max_waiting: Optional[int] = 8,
    dispatcher: str = "non_batch",
    decode_max_engine_requests: Optional[int] = None,
) -> ConfigOverride:
    """Unified priority-family override (PRIORITY + SINGLE + NON_BATCH
    base; dispatcher="batch" variant for the live-eviction family, 2026-09)
    layered on the ctx profile's base document.

    Implementation-period additions over the design's config sketch
    (all verified against the Java code):

    * ``maxInflightRequestsPerPrefillWorker=1`` (override field
      max_inflight_requests_per_worker) is what actually creates the
      master-side backlog window — RoutePrefillAdmission.reserveRoute
      leases one in-flight delivery per dispatch, and without the cap
      every request dispatches immediately (no queueing, no observable
      ordering).
    * ``maxWaitingRequestsPerPrefillWorker`` rides the native
      max_waiting_requests_per_prefill_worker override field on this
      line (the admission wave-2 passthrough, 2026-09); the queue-full
      eviction path needs the tight cap (Java default 1024).
    * ``dispatcher="batch"`` (preemption-stage coverage, 2026-09): boots
      the BATCH dispatcher so the master-owned enqueue path (WorkerBatcher
      queue + the maxWaiting cap feeding AdmissionFallback's preemption
      trigger) is reachable — the live 8400 eviction paths need it; under
      batch the per-worker inflight cap is meaningless.
    * ``decode_max_engine_requests`` replaces the former JSON-splice
      helper (_max_engine_requests_1) — the native override knob.

    Legacy-wrapper "key not emitted" semantics carry over as OMIT:
    queue_timeout_ms=None means the queueTimeoutMs key stays OUT (the
    Java default 1h — the functional profiles' 60000 must not leak in);
    max_inflight=None / max_waiting=None likewise drop their keys.
    """
    return ConfigOverride(
        ordering=ordering,
        decision="single",
        dispatcher=dispatcher,
        default_priority=default_priority,
        preemption=preemption,
        queue_timeout_ms=OMIT if queue_timeout_ms is None else queue_timeout_ms,
        max_outstanding=max_outstanding,
        max_inflight_requests_per_worker=(
            None
            if dispatcher == "batch"
            else (OMIT if max_inflight is None else max_inflight)
        ),
        max_waiting_requests_per_prefill_worker=(
            OMIT if max_waiting is None else max_waiting
        ),
        decode_max_engine_requests=decode_max_engine_requests,
    )


def _spec(
    ctx: CaseContext,
    label: str,
    *,
    n_prefill: int = 1,
    n_decode: int = 4,
    config_overrides: Optional[ConfigOverride] = None,
    raw_config: Optional[str] = None,
    master_debug_log: bool = False,
    extra_env: Optional[dict] = None,
    decode_cache_blocks: Optional[int] = None,
) -> EnvSpec:
    # extra_env carries non-config env only (e.g. the auto_tpm metric
    # whitelist); FLEXLB_CONFIG comes from config_overrides / raw_config.
    # decode_cache_blocks sizes the mock's decode KV pool
    # (totalKvTokens = blocks x blockSize — the reported capacity always
    # tracks the built pool); None keeps the harness default.
    spec_kwargs = {}
    if decode_cache_blocks is not None:
        spec_kwargs["decode_cache_blocks"] = decode_cache_blocks
    return EnvSpec(
        label=f"{label}_{ctx.profile}",
        n_prefill=n_prefill,
        n_decode=n_decode,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env=dict(extra_env) if extra_env else {},
        config_overrides=config_overrides,
        raw_config=raw_config,
        master_debug_log=master_debug_log,
        **spec_kwargs,
    )


def _q1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-Q1: ordering window env (no preemption, 1P+4D, queue cap 8)."""
    return _spec(ctx, "prio_q1", config_overrides=_prio_config())


def _q2_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-Q2: PREFILL_QUEUED preemption env (1P+4D, queue cap 8)."""
    return _spec(
        ctx,
        "atpm_q2",
        config_overrides=_prio_config(preemption=_PREEMPT_PQ, queue_timeout_ms=60_000),
    )


def _t1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-T1: queueTimeout 8s, no preemption (1P+4D)."""
    return _spec(ctx, "prio_t1", config_overrides=_prio_config(queue_timeout_ms=8_000))


def _a1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-A1: PREFILL_QUEUED preemption + queueTimeout 7s (1P+4D)."""
    return _spec(
        ctx,
        "atpm_a1",
        config_overrides=_prio_config(preemption=_PREEMPT_PQ, queue_timeout_ms=7_000),
    )


def _d1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-D1: decode-stage preemption env (2P+4D, inflight=3 so four victims
    plus one incoming dispatch concurrently).

    Decode saturation is manufactured at RUN TIME via kv_pressure on every
    decode endpoint (the decode eviction guard requires every endpoint to
    be in "needs eviction" state — the decode eviction planner returns null
    when any ordinary endpoint is available).  The design's
    decode_cache_blocks knob does not affect the mock's KV reporting (it
    only sizes the LRU block cache), hence the runtime-injection approach.
    Injection TIMING is decisive (implementation-period finding): a
    PRIORITY queue deliberately retains the strict decode KV gate in
    ordinary routing (CostBasedDecodeStrategy.applyHardFilters — "route
    failure is what enters its typed admission/preemption path"), so
    kv_pressure must be injected only AFTER the victim wave has routed
    (their decode reservations exist); injecting earlier rejects the
    victims themselves with NO_DECODE_WORKER(8403) and leaves nothing to
    evict.

    Metric exposure (behaviour-neutral for scheduling): the default
    critical-only whitelist hides auto_tpm.* (the single configurable
    filter is flexlb.monitor.metric-whitelist — see _q3_spec; the legacy
    FLEXLB_MONITOR_MODE switch is dead on this line);
    atpm_decode_reservation_priority asserts the auto_tpm.victim.count
    priority tags, so the shared family-prefix entry flexlb_auto_tpm
    (_MONITOR_AUTO_TPM_ENV — a promName.startsWith match covering the
    whole auto_tpm family) carries the exposure.
    """
    return _spec(
        ctx,
        "atpm_d1",
        n_prefill=2,
        config_overrides=_prio_config(
            preemption=_PREEMPT_DECODE, max_inflight=3, queue_timeout_ms=60_000
        ),
        extra_env=_MONITOR_AUTO_TPM_ENV,
    )


def _c1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-C1: global outstanding capacity 2 (G11b isomorphic, 2P+2D)."""
    return _spec(
        ctx,
        "atpm_c1",
        n_prefill=2,
        n_decode=2,
        config_overrides=_prio_config(
            max_outstanding=2, max_inflight=None, max_waiting=None
        ),
    )


def _n1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-N1: defaultPriority=30 (1P+4D)."""
    return _spec(ctx, "prio_n1", config_overrides=_prio_config(default_priority=30))


def _f1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-F1: FIFO control env (same shape as Q1, ordering=fifo)."""
    return _spec(ctx, "atpm_f1", config_overrides=_prio_config(ordering="fifo"))


def _o1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-O1: observability env — Q2-shaped config with a SHORT queueTimeout
    (7s, so the choreography yields timeout-attribution samples), debug log
    on, and the auto_tpm family metric whitelist.

    [EV-1-FIXED] queueTimeout 7s (was 8s, flipped at intake3
    PendingPlacementCoordinator 6ad0315f10): under the pull model the
    wave's third release slot lands at t=9s, which sat INSIDE the 8s
    deadline of the 4th submitter (70a, deadline ~8.7-9.0s) — a race
    between the coordinator pull and the expiry check.  7s puts every
    non-dispatched deadline (7.1-8.35s) strictly before the t=9 slot,
    making the client shape deterministic: ph + 30a (first parker) + 90
    dispatch and complete, the remaining seven expire 8511.

    Implementation-period corrections over the design's env sketch: the
    default critical-only whitelist does not expose auto_tpm.*, so the
    FLEXLB_MONITOR_METRIC_WHITELIST family entry is required (the legacy
    FLEXLB_MONITOR_MODE switch is dead on this line — see _q3_spec);
    pv.log writes at INFO level by default on the harness line
    (FLEXLB_PV_LOG is a load-client-line knob with no consumer here)."""
    return _spec(
        ctx,
        "atpm_o1",
        config_overrides=_prio_config(preemption=_PREEMPT_PQ, queue_timeout_ms=7_000),
        master_debug_log=True,
        extra_env=_MONITOR_AUTO_TPM_ENV,
    )


def _q3_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-Q3 (A5, Mark P1-3/PR3): metric-plane normalization env — Q1
    config plus the metric-whitelist entry exposing
    auto_tpm.request.count.  auto_tpm.request.count{priority=..}
    is counted at the schedule RPC entry for EVERY request regardless of
    outcome (FlexlbServiceImpl:723), which keeps the proto-vs-header
    channel discrimination observable even while capacity-blocked
    submitters park in the intake3 coordinator; the shared ENV-P0/Q1/F1
    specs cannot serve it because the default critical-only metrics
    filter hides auto_tpm.*.

    [2026-09-04] FLEXLB_MONITOR_MODE=all is DEAD on the v2 line — the
    single configurable exposure filter is flexlb.monitor.metric-
    whitelist (env FLEXLB_MONITOR_METRIC_WHITELIST via relaxed binding;
    WhitelistMetricsFilterConfig), whose default preset keeps only the
    six core link-latency metrics, so the old env var silently left
    auto_tpm.* hidden and prio_normalize's segment 4 scraped None
    buckets (2026-09-04 run).  The spec now pins the exact
    prometheus-form prefix flexlb_auto_tpm_request_count — the same
    entry the run_online_eval.sh collector whitelist uses — exposing
    exactly the series segment 4 sums (a bare "flexlb_" would expose
    the whole flexlb_* family; the minimal entry keeps the Q3 env's
    exposition surface tight)."""
    return _spec(
        ctx,
        "prio_q3",
        config_overrides=_prio_config(),
        extra_env={"FLEXLB_MONITOR_METRIC_WHITELIST": "flexlb_auto_tpm_request_count"},
    )


# ===========================================================================
# Observability collection: management-port Prometheus, master log, pv.log
# ===========================================================================


def _scrape_master_metrics(ops) -> list:
    """Scrape the master management port's Prometheus text exposition and
    parse it into [(name, labels dict, value)] samples.

    On this line the HTTP ladder (actuator/prometheus first, plain
    /prometheus fallback) already lives in EngineOps
    .master_prometheus_text(); the shared parse_prometheus_samples
    parser (same triple shape) is reused with an empty prefix so every
    metric family is kept for the name-substring lookups below.  A failed
    scrape degrades to an empty sample list — the callers record it as
    missing evidence rather than raising.
    """
    text = ops.master_prometheus_text()
    if not text:
        return []
    return parse_prometheus_samples(text, "")


def _metric_sum(samples: list, name_substr: str, labels_subset: dict):
    """Sum of samples whose metric name contains *name_substr* and whose
    labels are a superset of *labels_subset* (None when nothing matches).
    Name-substring matching keeps both the dotted and the Prometheus-
    underscored spellings observable (first-e2e calibration input)."""
    total = 0.0
    found = False
    for name, labels, val in samples:
        if name_substr in name and all(
            labels.get(k) == v for k, v in labels_subset.items()
        ):
            total += val
            found = True
    return total if found else None


def _metric_lines(samples: list, name_substr: str, limit: int = 12) -> str:
    """Raw sample lines for detail/diagnostics."""
    lines = [f"{name}{labels}" for name, labels, _v in samples if name_substr in name]
    return "; ".join(lines[:limit]) if lines else "<none>"


def _master_log_text(env) -> str:
    """Master log text for THIS env: the JVM stdout redirect plus the
    bytes the logback flexlbLogger file appender (~/ai-whale/logs/flexlb.log,
    shared across every master in the container) wrote since our own start
    (offset recorded by harness.start_master).  The [priority-scheduler]
    DEBUG lines land in the file appender, never in the ~11 buffered
    stdout lines (round-2 O1 finding)."""
    parts = []
    log_path = Path(env.run_dir) / "flexlb_master.log"
    try:
        parts.append(log_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        pass
    flexlb_log = Path.home() / "ai-whale" / "logs" / "flexlb.log"
    offset = getattr(env, "flexlb_log_offset", 0)
    try:
        with open(flexlb_log, "rb") as fh:
            fh.seek(offset)
            parts.append(fh.read().decode("utf-8", errors="replace"))
    except Exception:
        pass
    return "\n".join(p for p in parts if p)


def _pv_log_tail(env, rids: Optional[list] = None, max_lines: int = 400) -> str:
    """pv.log request-journal delta for THIS env (A8, Daniel P2-3): the
    file is shared across every master in the container, so read only the
    bytes written since OUR master start (offset recorded by
    harness.start_master — the same discipline as _master_log_text) and,
    when *rids* is given, keep only the JSON rows whose "requestId" field
    matches one of them (PvLogData.java serialises requestId per row) —
    eliminating sibling-instance read crosstalk."""
    path = Path.home() / "ai-whale" / "logs" / "pv.log"
    offset = getattr(env, "pv_log_offset", 0)
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            text = fh.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    lines = text.splitlines()
    if rids:
        wanted = set()
        for rid in rids:
            wanted.add(f'"requestId":{rid}')
            wanted.add(f'"requestId": {rid}')
        lines = [ln for ln in lines if any(m in ln for m in wanted)]
    return "\n".join(lines[-max_lines:])


# ===========================================================================
# Shared finally hygiene (design §5.3: perf restore + drain + inflight_clean)
# ===========================================================================


def _finally_hygiene(ops, fires: list, prefill_names: list) -> None:
    """Cross-case environment hygiene for a shared env: restore perf, drain
    every fired request to terminal (cancel the stuck tail), wait for the
    master inflight ledger to settle."""
    try:
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=100.0)
    except Exception:
        pass
    try:
        _drain(ops, [f for f in fires if f.terminal is not None or not f.ok])
    except Exception:
        pass
    try:
        _cancel_all(ops, fires)
    except Exception:
        pass
    try:
        AssertUtils.inflight_clean(f"http://127.0.0.1:{ops.master_http_port}", 30.0)
    except Exception:
        pass


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


# ===========================================================================
# Basic family — prio_* (design §2.2)
# ===========================================================================


def _lat_stats(fires: list) -> str:
    """Per-request terminal wall time (submit → stream terminal); a
    calibration-detail helper — no band consumes it (design §2.2
    prio_low_no_starvation: the non-saturated latency split is recorded,
    not asserted)."""
    total = []
    for fr in fires:
        term = fr.terminal
        if term is not None and term.terminated_s is not None:
            total.append(term.terminated_s - fr.submitted_s)
    if not total:
        return "n=0"
    return (
        f"n={len(total)} avg={sum(total) / len(total) * 1000:.0f}ms "
        f"max={max(total) * 1000:.0f}ms"
    )


def _all_ok(outcomes: list) -> bool:
    return all(ok for (_rid, ok, _code, _detail) in outcomes)


# ===========================================================================
# Preemption family — atpm_* part 1 (design §2.3)
# ===========================================================================


def _mono_to_epoch(monotonic_ts: float) -> float:
    """Convert a time.monotonic() stamp captured by this process into the
    wall-clock epoch domain (both clocks advance at the same rate — the
    offset is sampled at call time and applies retroactively to any stamp
    from the same process).  Used by the AT5 closure measurement, which
    must cross the client clock (stream terminal) with the engine's epoch
    running_ms; master, mock engines and this client run on one host, so
    the two epoch domains coincide."""
    return time.time() - (time.monotonic() - monotonic_ts)


def _outcome_map(outcomes: list) -> dict:
    """{(rid): (ok, code, detail)} from _drain results."""
    return {rid: (ok, code, detail) for (rid, ok, code, detail) in outcomes}


def _code_of(fr) -> object:
    """Unified typed terminal code of a fire: schedule-response code when
    the RPC failed, stream raw code (grpc-status-details-bin) when the
    direct stream broke, CODE_OK when completed."""
    if fr.rpc_error or fr.resp is None:
        return None
    if not fr.ok:
        return fr.code
    if fr.terminal is None:
        return None
    if fr.terminal.completed:
        return CODE_OK
    return fr.terminal.raw_error_code


# ===========================================================================
# Boundary family — atpm_* part 2 (design §2.4)
# ===========================================================================

#: NO_DECODE_WORKER — the DECODE role's selection failure (RoleType.
#: getErrorType → DefaultRouter.buildFailureResponse).  A PRIORITY queue
#: deliberately retains the strict decode KV gate in ordinary routing
#: (CostBasedDecodeStrategy.applyHardFilters), so kv_pressure-saturated
#: decode endpoints surface here as 8403 before the eviction fallback —
#: the code the infeasible same-priority decode wave actually observes.
CODE_NO_DECODE = 8403


# ===========================================================================
# atpm_config_strict_reject — strict FLEXLB_CONFIG startup rejection (AT1)
# ===========================================================================


# ===========================================================================
# atpm_decode_reservation_priority — decode-plane victim selection (AT7)
# ===========================================================================


# ===========================================================================
# atpm_observability_integrity — four signal planes on one choreography (AT8)
# ===========================================================================


# ===========================================================================
# Preemption-stage live coverage — atpm_preempt_* live family (2026-09)
#
# Design input: the preemption-stages audit (Zara,
# verdict_preemption_stages.md).  The existing atpm_preempt_prefill_queued
# ([EV-1-FIXED]) and the decode family ([EV-2]) all assert ZERO eviction —
# under the NON_BATCH pull model capacity blocking parks every submitter,
# so neither 8400 path (PREFILL_QUEUED queue replacement / DECODE_RESERVED
# shadow-reservation eviction) ever fires.  The live family boots the
# BATCH dispatcher so the master-owned enqueue path (WorkerBatcher queue
# + maxWaiting cap → AdmissionFallback → EvictionManager) is reachable,
# and pins the REAL victim terminals: exactly-8400 for both master-local
# stages, 8429 for the engine-owned tombstone settle.
# ===========================================================================

# 3-strike health demotion + restart windows (cancel.py HA-family
# precedent — copied, not imported: cancel.py is under concurrent
# modification by another session; these copies are small and frozen
# by contract).
MASTER_EVICT_S = 30.0
ENGINE_RECOVERY_WAIT_S = 3.0


def _cancel_rpc_total(ops) -> int:
    """Sum of per-engine Cancel RPC counters from /snapshot."""
    snap = ops.snapshot()
    return sum(
        int(e.get("rpc_counts", {}).get("cancel", 0)) for e in snap.get("engines", [])
    )


def _all_engine_names(ops) -> list:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", [])]


def _engine_saw(ops, rid: int) -> bool:
    """True when ANY engine has a lifecycle entry for *rid* — the
    never-delivered proof for master-local (8400) victims."""
    return any(
        str(rid) in e.get("request_lifecycle", {})
        for e in ops.snapshot().get("engines", [])
    )


def _poll_engine_finished(ops, rid: int, timeout_s: float) -> bool:
    """Wait until *rid* shows a NON-running end_state on a decode engine
    (engine-side completion — the master view may be frozen behind an
    injected status_no_respond)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for engine in ops.snapshot().get("engines", []):
            if engine.get("role") != "decode":
                continue
            lc = engine.get("request_lifecycle", {}).get(str(rid), {})
            end = lc.get("end_state")
            if end and end != "running":
                return True
        time.sleep(0.05)
    return False


def _direct_enqueue(ops, addr: str, input_pb, batch_id: int):
    """Client-side EnqueueBatch probe straight at one engine's gRPC port
    (bypasses the master — the ABSENT_FENCE probe must observe the
    ENGINE's admission decision, not the master's settled ledger)."""
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
    """True-crash + restart cycle on one engine (crash_after n=1) — the
    per-engine memory wipe (running tasks, tombstones, absent-fence
    records, RPC counters), so the restarted instance has never seen any
    pre-restart rid.  The sacrificial request's own fate is the
    empty-ack uncertain path and is deliberately not asserted."""
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


def _pq_live_spec(ctx: CaseContext) -> EnvSpec:
    """ENV for atpm_preempt_prefill_queued_live: BATCH dispatcher +
    PREFILL_QUEUED-only preemption + maxWaiting=2, so the third submitter
    (the P70 incoming) overflows the queue into AdmissionFallback's
    queue-replacement path (1P+4D)."""
    return _spec(
        ctx,
        "atpm_pq_live",
        config_overrides=_prio_config(
            dispatcher="batch",
            preemption=_PREEMPT_PQ,
            max_waiting=2,
            queue_timeout_ms=60_000,
        ),
        extra_env=_MONITOR_AUTO_TPM_ENV,
    )


def _dr_live_spec(ctx: CaseContext) -> EnvSpec:
    """ENV for atpm_preempt_decode_reserved_live: BATCH dispatcher + the
    production-baseline stage set {PREFILL_QUEUED, DECODE_RESERVED} + a
    4-block decode KV pool (4096 tokens at blockSize=1024) on a SINGLE
    decode engine, so the victim's shadow reservation — not a slot
    deficit — makes the incoming's decode placement fail."""
    return _spec(
        ctx,
        "atpm_dr_live",
        n_decode=1,
        decode_cache_blocks=4,
        config_overrides=_prio_config(
            dispatcher="batch",
            preemption={"allowed_victim_stages": ["PREFILL_QUEUED", "DECODE_RESERVED"]},
            queue_timeout_ms=60_000,
        ),
        extra_env=_MONITOR_AUTO_TPM_ENV,
    )


def _nf_spec(ctx: CaseContext) -> EnvSpec:
    """ENV for atpm_preempt_cancel_not_found: all-three-stage preemption
    (engineCancellation mandatory) + decode maxEngineRequests=1, NON_BATCH
    single-decode topology (1P+1D)."""
    return _spec(
        ctx,
        "atpm_nf",
        n_decode=1,
        config_overrides=_prio_config(
            preemption=_PREEMPT_ALL_STAGES,
            queue_timeout_ms=60_000,
            decode_max_engine_requests=1,
        ),
    )


def _ts_spec(ctx: CaseContext) -> EnvSpec:
    """ENV for atpm_preempt_cancel_tombstoned: the live preemption config
    (all three stages + engineCancellation) + decode maxEngineRequests=1
    on the BATCH dispatcher (1P+1D) — same shape as cancel.py's
    cancel_preemption_victim."""
    return _spec(
        ctx,
        "atpm_ts",
        n_decode=1,
        config_overrides=_prio_config(
            dispatcher="batch",
            preemption=_PREEMPT_ALL_STAGES,
            queue_timeout_ms=60_000,
            decode_max_engine_requests=1,
        ),
    )
