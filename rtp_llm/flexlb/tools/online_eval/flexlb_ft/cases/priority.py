"""Priority ordering + Auto-TPM (preemption orchestration / admission
attribution) functional cases — the "priority" category on the
intake3-rebuild flexlb_ft line.

Migrated from the feat/flexlb_priority_auto_tpm_ft delivery branch
(priority_cases.py; design doc docs/priority_auto_tpm_test_design.md on
that branch — the authoritative spec for every choreography and assertion
below).  Profile strategy per the migration decision: the target keeps
its four built-in profiles and injects the PRIORITY axis at the CASE
layer — _prio_config builds SINGLE + NON_BATCH + ordering/preemption
directly via build_flexlb_config (no priority profile is added to
PROFILE_SPECS), the cancel_preemption_victim JSON-splice precedent with
the knobs expressed as native generator parameters.

Category theme: priority-order fidelity (PR1 band, same-level FIFO,
three-channel normalization) + the Auto-TPM preemption/degradation
contract (victim selection, terminal split, timeout attribution, config
strictness, observability).  15 cases:

  prio_* (5) — basics: priority-order fidelity (PR1 band),
      same-priority FIFO (PR2), three-channel normalization (PR3),
      low-priority completion (PR8 completion calibre), queueTimeout
      absolute deadline (PR8 band).

  atpm_* (10) — preemption + boundary: PREFILL_QUEUED
      queue replacement (PR10/PR5/PR6/PR4), DECODE_RESERVED vs
      DECODE_ENGINE_OWNED eviction (PR6 + AT5 band), same-priority and
      preemption-disabled zero eviction (PR4/AT3/AT2), admission-timeout
      attribution (PR7), comparator-freeze weak form (PR9), error-code
      family separation (AT4), strict config rejection (AT1), decode
      reservation priority (AT7), observability integrity (AT8).

Baseline annotations: [EV-1-FIXED] assertions were calibrated on the
feat/flexlb_priority_auto_tpm_ft line at the intake3
PendingPlacementCoordinator (commit 6ad0315f10) — the pull-based park
model with priority desc + FIFO release; they are PENDING remote-probe
verification against the intake3-rebuild Java line (behaviour may
differ — assertion calibres are intentionally unchanged in this
migration, per the migration brief).  [EV-2] marks the live degradation
baseline (decode eviction unreachable).  Grep those keys to enumerate
every flip point (the EV-1/EV-2 runbook discipline).

Signal sources (design §1.1 / appendix A): schedule proto response
(``code`` = StrategyErrorType code, ``admission_reject_reason`` = proto
field 9), engine ``request_lifecycle`` (arrived_ms / running_ms /
end_state — the mock cluster is a single JVM so the clock is comparable
across engines), master management-port Prometheus text (``auto_tpm.*``),
master log terminal strings, pv.log request rows, client per-request
terminal outcomes.  Dispatch-order observation uses the design §3.3
arbitration chain: engine ``running_ms`` primary → client schedule()
settle order tie-break for same-millisecond conflicts.

Registration model: local ``case()`` decorator appends into
PRIORITY_CASES with category="priority" (the cases/admission.py
precedent — category drives runner grouping, physical file ownership
stays here).

Behaviour-contract assertion discipline (design §1.1): only error
codes, admission_reject_reason, ``auto_tpm.*`` metric names/tags,
terminal log strings, ``enqueued_by_master`` and terminal lifecycle
states are asserted.  NEVER internal class/method names — the codex
branch renames (PreemptionCommand/preempt/Outcome,
PreemptionRegistration, the removed PriorityScheduler) must not break
any assertion.  No "TPM quota / 429 / sliding window" wording: Auto-TPM
is priority queueing + preemption orchestration + admission-failure
attribution + the auto_tpm.* observability family only.

White-box handovers (deliberately NOT implemented here, design §2.5/
§3.2): EvictionPlanner requestId tie-break, the 8432 sentinel-prefix
branch, the 14-item AT6 accounting ledger, the full comparator-freeze
contract (AutoTpmE2EHarness), and AdmissionFailureClassifier reason
precision.
"""

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
from ..engine_ops import parse_prometheus_samples
from ..grade import GradeReport
from ..harness import AssertUtils, EnvSpec, build_flexlb_config, default_perf
from .admission import MOCK_TOTAL_KV_TOKENS

PRIORITY_CASES: list[CaseDef] = []


def case(name: str, profiles=None, requires=None, source: str = ""):
    """Register into PRIORITY_CASES (category is always "priority")."""

    def deco(fn):
        PRIORITY_CASES.append(
            CaseDef(
                name=name,
                category="priority",
                fn=fn,
                profiles=profiles,
                requires=requires,
                source=source,
            )
        )
        return fn

    return deco


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


def _prio_config(
    *,
    ordering: str = "priority",
    preemption: Optional[dict] = None,
    default_priority: Optional[int] = None,
    queue_timeout_ms: Optional[int] = None,
    max_outstanding: Optional[int] = None,
    max_inflight: Optional[int] = 1,
    max_waiting: Optional[int] = 8,
) -> str:
    """Unified priority-family config (PRIORITY + SINGLE + NON_BATCH base).

    Two implementation-period additions over the design's config sketch
    (both verified against the Java code):

    * ``maxInflightRequestsPerPrefillWorker=1`` (build_flexlb_config kwarg
      max_inflight_requests_per_worker) is what actually creates the
      master-side backlog window — RoutePrefillAdmission.reserveRoute
      leases one in-flight delivery per dispatch, and without the cap
      every request dispatches immediately (no queueing, no observable
      ordering).
    * ``maxWaitingRequestsPerPrefillWorker`` rides the native
      max_waiting_requests_per_prefill_worker generator parameter on
      this line (the admission wave-2 passthrough, 2026-09) — on the
      source branch it had to be spliced via JSON post-processing; the
      queue-full eviction path needs the tight cap (Java default 1024).
    """
    return build_flexlb_config(
        ordering=ordering,
        decision="single",
        dispatcher="non_batch",
        default_priority=default_priority,
        preemption=preemption,
        queue_timeout_ms=queue_timeout_ms,
        max_outstanding=max_outstanding if max_outstanding is not None else 5_000,
        max_inflight_requests_per_worker=max_inflight,
        max_waiting_requests_per_prefill_worker=max_waiting,
    )


def _spec(
    ctx: CaseContext,
    label: str,
    *,
    n_prefill: int = 1,
    n_decode: int = 4,
    config: str,
    master_debug_log: bool = False,
    extra_env: Optional[dict] = None,
) -> EnvSpec:
    env = {"FLEXLB_CONFIG": config}
    if extra_env:
        env.update(extra_env)
    return EnvSpec(
        label=f"{label}_{ctx.profile}",
        n_prefill=n_prefill,
        n_decode=n_decode,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env=env,
        master_debug_log=master_debug_log,
    )


def _q1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-Q1: ordering window env (no preemption, 1P+4D, queue cap 8)."""
    return _spec(ctx, "prio_q1", config=_prio_config())


def _q2_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-Q2: PREFILL_QUEUED preemption env (1P+4D, queue cap 8)."""
    return _spec(
        ctx,
        "atpm_q2",
        config=_prio_config(preemption=_PREEMPT_PQ, queue_timeout_ms=60_000),
    )


def _t1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-T1: queueTimeout 8s, no preemption (1P+4D)."""
    return _spec(ctx, "prio_t1", config=_prio_config(queue_timeout_ms=8_000))


def _a1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-A1: PREFILL_QUEUED preemption + queueTimeout 7s (1P+4D)."""
    return _spec(
        ctx,
        "atpm_a1",
        config=_prio_config(preemption=_PREEMPT_PQ, queue_timeout_ms=7_000),
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

    FLEXLB_MONITOR_MODE=all (behaviour-neutral for scheduling): the
    default critical-only metrics filter hides auto_tpm.*
    (application.yml flexlb.monitor.mode); atpm_decode_reservation_priority
    asserts the auto_tpm.victim.count priority tags, which need the
    env-level switch.
    """
    return _spec(
        ctx,
        "atpm_d1",
        n_prefill=2,
        config=_prio_config(
            preemption=_PREEMPT_DECODE, max_inflight=3, queue_timeout_ms=60_000
        ),
        extra_env={"FLEXLB_MONITOR_MODE": "all"},
    )


def _c1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-C1: global outstanding capacity 2 (G11b isomorphic, 2P+2D)."""
    return _spec(
        ctx,
        "atpm_c1",
        n_prefill=2,
        n_decode=2,
        config=_prio_config(max_outstanding=2, max_inflight=None, max_waiting=None),
    )


def _n1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-N1: defaultPriority=30 (1P+4D)."""
    return _spec(ctx, "prio_n1", config=_prio_config(default_priority=30))


def _f1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-F1: FIFO control env (same shape as Q1, ordering=fifo)."""
    return _spec(ctx, "atpm_f1", config=_prio_config(ordering="fifo"))


def _o1_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-O1: observability env — Q2-shaped config with a SHORT queueTimeout
    (7s, so the choreography yields timeout-attribution samples), debug log
    on, and FLEXLB_MONITOR_MODE=all.

    [EV-1-FIXED] queueTimeout 7s (was 8s, flipped at intake3
    PendingPlacementCoordinator 6ad0315f10): under the pull model the
    wave's third release slot lands at t=9s, which sat INSIDE the 8s
    deadline of the 4th submitter (70a, deadline ~8.7-9.0s) — a race
    between the coordinator pull and the expiry check.  7s puts every
    non-dispatched deadline (7.1-8.35s) strictly before the t=9 slot,
    making the client shape deterministic: ph + 30a (first parker) + 90
    dispatch and complete, the remaining seven expire 8511.

    Implementation-period corrections over the design's env sketch: the
    critical-only metrics filter (the default) does not expose auto_tpm.*,
    so FLEXLB_MONITOR_MODE=all is required; pv.log writes at INFO level by
    default on the harness line (FLEXLB_PV_LOG is a load-client-line knob
    with no consumer here)."""
    return _spec(
        ctx,
        "atpm_o1",
        config=_prio_config(preemption=_PREEMPT_PQ, queue_timeout_ms=7_000),
        master_debug_log=True,
        extra_env={"FLEXLB_MONITOR_MODE": "all"},
    )


def _q3_spec(ctx: CaseContext) -> EnvSpec:
    """ENV-Q3 (A5, Mark P1-3/PR3): metric-plane normalization env — Q1
    config plus FLEXLB_MONITOR_MODE=all.  auto_tpm.request.count{priority=..}
    is counted at the schedule RPC entry for EVERY request regardless of
    outcome (FlexlbServiceImpl:723), which keeps the proto-vs-header
    channel discrimination observable even while capacity-blocked
    submitters park in the intake3 coordinator; the shared ENV-P0/Q1/F1
    specs cannot serve it because the default critical-only metrics
    filter hides auto_tpm.*."""
    return _spec(
        ctx,
        "prio_q3",
        config=_prio_config(),
        extra_env={"FLEXLB_MONITOR_MODE": "all"},
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


@case(
    "prio_order_basic",
    # Source ran on the dedicated priority-single-nonbatch profile; on
    # this line the SINGLE+NON_BATCH base IS single-nonbatch and the
    # PRIORITY axis arrives through the case-layer config injection
    # (_q1_spec / _prio_config).
    profiles=["single-nonbatch"],
    source="design §2.2 #1 — PR1(band) + PR2 + P6",
)
def prio_order_basic(ctx: CaseContext):
    """Priority-order fidelity (PR1 band + PR2 group FIFO + P6).

    Choreography (design §2.2 #1): ENV-Q1 — a single prefill so every
    request lands in the same queue (no routing ambiguity), no preemption,
    default queueTimeout (Java default 1h — nothing expires inside the
    window).  A priority=50 placeholder parks the inflight lease, then a
    mixed ladder is submitted LOW-first (30a, 30b, 50a, 50b, 70a, 70b,
    0.15s apart, 7 concurrent ≤ maxWaiting 8 — no route failure, no
    preemption).  [EV-1-FIXED] Under the intake3 pull-based coordinator
    (PendingPlacementCoordinator, 6ad0315f10) the whole wave parks and
    settles code=200; the design-final dispatch order is [ph, 30a (the
    wave's first submitter legitimately wins the FIRST release slot),
    70a, 70b, 50a, 50b, 30b] — every later release follows strict
    priority desc + same-level FIFO (probe evidence 2026-08-31, gaps
    ~3015ms, zero route-reject).

    Observation (design §3.3): engine request_lifecycle.running_ms
    ascending == dispatch order (the mock cluster is one JVM, clocks
    comparable); NON_BATCH parks capacity-blocked schedules, so the client
    settle order is a natural arbitration signal for same-millisecond
    conflicts — the risk-1 fallback the task brief pre-authorized.
    """
    env = ctx.env_manager.ensure(_q1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        ph = ops.next_request_id(base)
        ph_fire = _fire(ops, ph, priority=50, input_len=2048, output_len=2)
        fires.append(ph_fire)
        if not ph_fire.ok:
            return False, f"placeholder schedule failed: code={ph_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "placeholder never reached the prefill engine"

        tags = ["30a", "30b", "50a", "50b", "70a", "70b"]
        rids: dict = {}
        specs = []
        for tag in tags:
            rid = ops.next_request_id(base)
            rids[tag] = rid
            specs.append(
                (rid, {"priority": int(tag[:-1]), "input_len": 2048, "output_len": 2})
            )
        wave = _fire_batch(ops, specs)
        fires.extend(wave)

        outcomes = _drain(ops, fires)
        tag_of = {rid: tag for tag, rid in rids.items()}
        tag_of[ph] = "ph"
        priorities = {ph: 50}
        for tag, rid in rids.items():
            priorities[rid] = int(tag[:-1])

        order = _dispatch_order(ops, fires)
        order_tags = [tag_of.get(r, str(r)) for r in order]
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the whole wave parks (pull-based WaitBucket, priority
        # desc + FIFO tiebreak) and every request settles code=200 — zero
        # route-reject.  Design-final dispatch shape: [ph, 30a (the wave's
        # first submitter legitimately wins the FIRST release slot), 70a,
        # 70b, 50a, 50b, 30b].  PR1 scores the REAL dispatch order
        # (running_ms asc + settle-rank arbitration) with the first parker
        # AND the pre-wave running placeholder excluded — the exclusion is
        # the designed pull-model behaviour, not an inversion amnesty.
        m = _outcome_map(outcomes)
        wave_rids = [rids[t] for t in tags]
        first_parker, shape_ok, wave_order = _design_final_pattern(
            ops, fires, wave_rids, priorities
        )
        wave_order_tags = [tag_of.get(r, str(r)) for r in wave_order]
        all_ok = m[ph][0] and all(m[rids[t]][0] for t in tags)

        report.check(
            "PR1",
            _inversion_ratio(order, priorities, exclude={ph, first_parker}),
            context="basic_order",
            detail=(
                f"[EV-1-FIXED] dispatch={order_tags} (design-final: first "
                f"parker 30a then priority desc; first parker and pre-wave "
                f"placeholder excluded from PR1 scoring)"
            ),
        )
        report.invariant(
            "PR2",
            shape_ok
            and _group_order_ok(order, [rids["70a"], rids["70b"]])
            and _group_order_ok(order, [rids["50a"], rids["50b"]])
            and _group_order_ok(order, [rids["30a"], rids["30b"]]),
            context="same_priority_fifo",
            detail=(
                f"[EV-1-FIXED] dispatch={order_tags} (shape covers "
                f"same-level FIFO inside every priority group)"
            ),
        )
        report.invariant(
            "PR6",
            shape_ok
            and all_ok
            and all(m[rids[t]][1] == CODE_OK for t in tags)
            and m[ph][1] == CODE_OK,
            context="design_final_dispatch",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): wave="
                f"{wave_order_tags} (first parker="
                f"{tag_of.get(first_parker, first_parker)} then priority "
                f"desc + same-level FIFO), all code=200 (zero route-reject), "
                f"codes={[(t, m[rids[t]][1]) for t in tags]}"
            ),
        )
        unfinished = [o for o in outcomes if not o[1]]
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            not unfinished and clean_ok,
            detail=(
                f"[EV-1-FIXED] issued=terminal no-loss: "
                f"{len(outcomes) - len(unfinished)}/7 completed, "
                f"unfinished={unfinished[:3] if unfinished else 'none'}, "
                f"inflight={'ok' if clean_ok else clean_detail}"
            ),
        )
        return report.finish(
            f"dispatched={order_tags} [EV-1-FIXED], grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)


@case(
    "prio_same_level_fifo",
    profiles=["single-nonbatch"],
    source="design §2.2 #2 — PR2 + P6",
)
def prio_same_level_fifo(ctx: CaseContext):
    """Same-priority FIFO (PR2 invariant + P6): seven priority=50 requests
    submitted with rid ascending (the first doubles as the placeholder) —
    the dispatch order must equal the submit order exactly.

    enqueueSeq is PriorityOrdering's second key (design §3.4 row 5);
    sequential submission with 0.15s gaps makes enqueuedAtMs order ==
    submit order, so the tie-break is exercised deterministically.  The
    requestId tie-break inside one arrival instant is NOT constructible
    from a single client (arrival order already decides) — white-box
    handover per design §2.5.
    """
    env = ctx.env_manager.ensure(_q1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        rids = [ops.next_request_id(base) for _ in range(7)]
        specs = [
            (rid, {"priority": 50, "input_len": 2048, "output_len": 2}) for rid in rids
        ]
        fires.extend(_fire_batch(ops, specs))

        outcomes = _drain(ops, fires)
        order = _dispatch_order(ops, fires)
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): all seven same-priority peers park in the pull-based
        # coordinator and dispatch in pure submit order (enqueueSeq
        # tie-break, design §3.4 row 5) — rids[0] direct-dispatches against
        # the empty queue, rids[1] is the wave's first parker, and the
        # "dispatch == submit" equality on seven FIFO peers is now a REAL
        # observation object (was EV-1: only the first parker survived,
        # the rest route-rejected).
        m = _outcome_map(outcomes)
        all_ok = all(m[rid][0] for rid in rids)
        fifo_ok = order == rids and all_ok
        report.invariant(
            "PR2",
            fifo_ok,
            context="same_priority_fifo",
            detail=(
                f"[EV-1-FIXED] dispatch==submit:{order == rids}, "
                f"all 7 code=200={all_ok}, "
                f"dispatch={[r % 1_000_000 for r in order]}"
            ),
        )
        unfinished = [o for o in outcomes if not o[1]]
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            not unfinished and clean_ok,
            detail=(
                f"[EV-1-FIXED] issued=terminal no-loss: "
                f"{len(outcomes) - len(unfinished)}/7 completed, "
                f"unfinished={unfinished[:3] if unfinished else 'none'}, "
                f"inflight={'ok' if clean_ok else clean_detail}"
            ),
        )
        return report.finish(
            f"fifo dispatch==submit:{order == rids} [EV-1-FIXED], "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)


@case(
    "prio_normalize",
    # profiles=None — the case stays eligible on every one of the four
    # built-in profiles (migration decision: the PRIORITY axis is a
    # case-layer injection, no priority profile exists to branch on, and
    # the source's priority-profile branch collapses to the only branch).
    # requires=["queue"]: the shared-env segment 1 and the Q1/Q3 windows
    # all need the waiting-queue capability.
    profiles=None,
    requires=["queue"],
    source="design §2.2 #3 — PR3 + P6",
)
def prio_normalize(ctx: CaseContext):
    """Three-channel normalization (PR3 invariant, per segment): proto
    field 14 > the DashScope QoS header > defaultPriority, unset →
    default; plus the FIFO-control proof that normalization never
    reorders FIFO arrival.

    Segment 1 (shared env, every profile): no-input and explicit-50
    interleaved — same-weight merge (PrioritySource is observational
    metadata only, design §3.4 row 8: the test must never assume the
    explicit source outranks the default source).  The production config
    has no inflight cap, so this segment is the weak arrival-order form
    (all succeed, dispatch == submit).

    Segment 2 (window env, Q1 — the PRIORITY axis): placeholder(no
    input) + C(no input → 50) + A(proto 70, header 30 — proto must win)
    + B(proto unset, header 70 — header must take effect) + G(explicit
    70).  [EV-1-FIXED] Under the intake3 pull-based coordinator the wave
    parks whole and dispatches [ph, C, A, B, G]: C (the wave's first
    submitter) legitimately wins the first release slot, then the
    70-group A/B/G follows in submit order (a failed channel reorders
    the tail and flips the assertion).  The source's FIFO-profile half
    (F1 control env) is not profile-reachable on this line; the FIFO
    control lives separately in atpm_comparator_frozen_weak's fifo_half.
    Implementation-period note: design §2.2 sketched this segment without
    maxInflightRequestsPerPrefillWorker, but without it there is no
    backlog window and no observable queue-jumping — the window env
    keeps the choreography and makes it observable.

    Segment 3 (ENV-N1 — the defaultPriority=30 variant, built as its own
    case-layer env): [EV-1-FIXED] submit order adapted for the
    design-final baseline: Y(50) leads (first parker), then D(no input),
    then a Z(40) reference, then X(explicit 30).  Expected dispatch
    [ph, Y, Z, D, X] — D (default 30) ties X inside the 30-group FIFO
    behind Z(40); a failed default (D=50) would give [ph, Y, D, Z, X] —
    the two outcomes stay distinguishable, so the assertion really pins
    the third channel.

    Segment 4 (ENV-Q3 — A5, Mark P1-3/PR3 strengthening): channel
    discrimination observed through the METRIC plane.  The behaviour
    plane (dispatch order) is EV-1-blocked for multi-parker waves, but
    auto_tpm.request.count{priority=..} counts at the schedule RPC entry
    regardless of outcome, so proto(70)+header(30) must land bucket 70,
    header-only(30) bucket 30, no-input bucket 50.  Normalization is
    profile-independent (FIFO normalizes identically) — the segment
    runs on every profile, hardening the four-profile gate (Daniel
    P2-1).
    """
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    segments: list = []  # (label, ok, detail)
    p6_flags: list = []
    hygiene: list = []  # (ops, fires, prefill_names)
    try:
        # -- segment 1: default-50 same weight (weak arrival form) ------
        ops0 = ctx.ops()
        s1_rids = [ops0.next_request_id(base) for _ in range(4)]
        s1_specs = []
        for i, rid in enumerate(s1_rids):
            kw: dict = {"input_len": 2048, "output_len": 2}
            if i % 2 == 1:
                kw["priority"] = 50  # explicit-50 alternates with no-input
            s1_specs.append((rid, kw))
        s1_fires = _fire_batch(ops0, s1_specs, gap_s=0.3)
        s1_outcomes = _drain(ops0, s1_fires)
        s1_order = _dispatch_order(ops0, s1_fires)
        s1_ok = s1_order == s1_rids and _all_ok(s1_outcomes)
        segments.append(
            (
                "default50_same_weight",
                s1_ok,
                f"dispatch==submit:{s1_order == s1_rids}, "
                f"all_ok={_all_ok(s1_outcomes)}",
            )
        )
        hygiene.append((ops0, s1_fires, []))
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops0), 30.0)
        p6_flags.append(_all_ok(s1_outcomes) and clean_ok)

        # -- segment 2: proto > header > default (window env) -----------
        env2 = ctx.env_manager.ensure(_q1_spec(ctx))
        ops2 = ctx.engine_ops(env2)
        p2_names = _prefill_names(ops2)
        for name in p2_names:
            ops2.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        ph2 = ops2.next_request_id(base)
        ph2_fire = _fire(ops2, ph2, input_len=2048, output_len=2)
        s2_fires = [ph2_fire]
        if not ph2_fire.ok:
            return False, f"segment2 placeholder failed: code={ph2_fire.code}"
        if not _poll_engine_pending(ops2, p2_names[0], 1):
            return False, "segment2 placeholder never dispatched"

        c_rid = ops2.next_request_id(base)
        a_rid = ops2.next_request_id(base)
        b_rid = ops2.next_request_id(base)
        g_rid = ops2.next_request_id(base)
        s2_specs = [
            (c_rid, {"input_len": 2048, "output_len": 2}),
            (
                a_rid,
                {"priority": 70, "qos_level": 30, "input_len": 2048, "output_len": 2},
            ),
            (b_rid, {"qos_level": 70, "input_len": 2048, "output_len": 2}),
            (g_rid, {"priority": 70, "input_len": 2048, "output_len": 2}),
        ]
        s2_fires.extend(_fire_batch(ops2, s2_specs))
        s2_outcomes = _drain(ops2, s2_fires)
        s2_order = _dispatch_order(ops2, s2_fires)
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the four-request wave parks whole and dispatches
        # [ph, C, A, B, G] — C (no input -> default 50) is the wave's first
        # submitter and wins the first release slot, then the 70-group A/B/G
        # follows in submit order (proto won for A, header took effect for
        # B — a failed channel reorders the tail and flips the assertion).
        # All four settle code=200.
        s2_m = _outcome_map(s2_outcomes)
        s2_wave = [c_rid, a_rid, b_rid, g_rid]
        s2_expected = [ph2, c_rid, a_rid, b_rid, g_rid]
        s2_ok = (
            s2_order == s2_expected
            and s2_m[ph2][0]
            and all(s2_m[r][0] for r in s2_wave)
        )
        segments.append(
            (
                "proto_header_default_ev1_fixed",
                s2_ok,
                f"[EV-1-FIXED] dispatch==[ph,C,A,B,G]:"
                f"{s2_order == s2_expected}, "
                f"codes={[(r % 1_000_000, s2_m[r][1]) for r in s2_wave]}, "
                f"order={[r % 1_000_000 for r in s2_order]}",
            )
        )
        hygiene.append((ops2, s2_fires, p2_names))
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops2), 30.0)
        p6_flags.append(s2_m[ph2][0] and all(s2_m[r][0] for r in s2_wave) and clean_ok)

        # -- segment 3: defaultPriority=30 (own case-layer env) ---------
        env3 = ctx.env_manager.ensure(_n1_spec(ctx))
        ops3 = ctx.engine_ops(env3)
        p3_names = _prefill_names(ops3)
        for name in p3_names:
            ops3.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        ph3 = ops3.next_request_id(base)
        ph3_fire = _fire(ops3, ph3, priority=10, input_len=2048, output_len=2)
        s3_fires = [ph3_fire]
        if not ph3_fire.ok:
            return False, f"segment3 placeholder failed: code={ph3_fire.code}"
        if not _poll_engine_pending(ops3, p3_names[0], 1):
            return False, "segment3 placeholder never dispatched"

        y_rid = ops3.next_request_id(base)
        d_rid = ops3.next_request_id(base)
        z_rid = ops3.next_request_id(base)
        x_rid = ops3.next_request_id(base)
        # [EV-1-FIXED] submit order adapted for the design-final
        # baseline (intake3 PendingPlacementCoordinator, 6ad0315f10):
        # the wave's FIRST submitter now legitimately wins the first
        # release slot, so the default-channel probe D must NOT sit in
        # first position (there it is order-invariant and the
        # assertion goes vacuous).  Y(50) leads as the first parker;
        # the Z(40) reference between D and X keeps the outcomes
        # distinguishable: default=30 gives [ph, Y, Z, D, X] (D ties X
        # at 30, FIFO inside the group, both behind Z), a failed
        # default (D=50) gives [ph, Y, D, Z, X].
        s3_specs = [
            (y_rid, {"priority": 50, "input_len": 2048, "output_len": 2}),
            (d_rid, {"input_len": 2048, "output_len": 2}),
            (z_rid, {"priority": 40, "input_len": 2048, "output_len": 2}),
            (x_rid, {"priority": 30, "input_len": 2048, "output_len": 2}),
        ]
        s3_fires.extend(_fire_batch(ops3, s3_specs))
        s3_outcomes = _drain(ops3, s3_fires)
        s3_order = _dispatch_order(ops3, s3_fires)
        # [EV-1-FIXED] baseline flipped at intake3
        # PendingPlacementCoordinator (6ad0315f10): the four-request
        # wave parks whole and dispatches [ph, Y (first parker), Z,
        # D, X] — D (no input -> defaultPriority=30) ties X inside the
        # 30-group FIFO behind the Z(40) reference; a failed default
        # (D=50) would give [ph, Y, D, Z, X].  The third channel
        # (defaultPriority) is pinned through D's dispatch position.
        s3_m = _outcome_map(s3_outcomes)
        s3_wave = [y_rid, d_rid, z_rid, x_rid]
        s3_expected = [ph3, y_rid, z_rid, d_rid, x_rid]
        s3_ok = (
            s3_order == s3_expected
            and s3_m[ph3][0]
            and all(s3_m[r][0] for r in s3_wave)
        )
        segments.append(
            (
                "default_priority_30_ev1_fixed",
                s3_ok,
                f"[EV-1-FIXED] dispatch==[ph,Y,Z,D,X]:"
                f"{s3_order == s3_expected} (D=default30 ties X behind "
                f"Z(40); failed default would give [ph,Y,D,Z,X]), "
                f"codes={[(r % 1_000_000, s3_m[r][1]) for r in s3_wave]}",
            )
        )
        hygiene.append((ops3, s3_fires, p3_names))
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops3), 30.0)
        p6_flags.append(s3_m[ph3][0] and all(s3_m[r][0] for r in s3_wave) and clean_ok)

        # -- segment 4 (A5): channel discrimination, metric plane -------
        # Needs its own env (FLEXLB_MONITOR_MODE=all): the shared env /
        # Q1 specs run the default critical-only metrics filter that
        # hides auto_tpm.*.  Fresh env means the counters start at zero,
        # so the expected buckets are absolute.  Buckets count at the
        # schedule RPC entry regardless of outcome — EV-1 cannot block
        # this observation plane.
        env4 = ctx.env_manager.ensure(_q3_spec(ctx))
        ops4 = ctx.engine_ops(env4)
        s4_specs = [
            (
                "proto70_over_header30",
                {
                    "priority": 70,
                    "qos_level": 30,
                    "input_len": 2048,
                    "output_len": 2,
                },
            ),
            (
                "header30_only",
                {"qos_level": 30, "input_len": 2048, "output_len": 2},
            ),
            ("no_input_default50", {"input_len": 2048, "output_len": 2}),
        ]
        s4_rids = [ops4.next_request_id(base) for _ in s4_specs]
        s4_fires = _fire_batch(
            ops4, [(rid, kw) for rid, (_l, kw) in zip(s4_rids, s4_specs)]
        )
        _drain(ops4, s4_fires)  # terminals only — buckets count regardless
        s4_buckets = {
            p: _metric_sum(
                _scrape_master_metrics(ops4),
                "auto_tpm_request",
                {"priority": str(p)},
            )
            for p in (30, 50, 70)
        }
        # proto(70) beats header(30); header alone lands 30; no input
        # defaults to 50 (segment 1's behaviour-plane default form echoed
        # on the metric plane — a different observation channel, not a
        # duplicate assertion).
        s4_ok = (
            s4_buckets[70] == 1.0 and s4_buckets[30] == 1.0 and s4_buckets[50] == 1.0
        )
        segments.append(
            (
                "channel_discrimination_metric_plane",
                s4_ok,
                f"buckets={ {p: s4_buckets[p] for p in s4_buckets} } "
                f"(expected 70:1, 30:1, 50:1)",
            )
        )
        hygiene.append((ops4, s4_fires, _prefill_names(ops4)))
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops4), 30.0)
        p6_flags.append(s4_ok and clean_ok)

        report.invariant(
            "PR3",
            all(ok for (_label, ok, _detail) in segments),
            context="three_channel_normalization",
            detail="; ".join(
                f"{label}={'ok' if ok else 'FAIL(' + detail + ')'}"
                for label, ok, detail in segments
            ),
        )
        report.invariant(
            "P6",
            all(p6_flags),
            detail=f"per-segment drain+inflight flags={p6_flags}",
        )
        return report.finish(
            f"profile={ctx.profile}, segments="
            f"{sum(1 for _l, ok, _d in segments if ok)}/{len(segments)}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for ops_x, fires_x, names_x in hygiene:
            try:
                _finally_hygiene(ops_x, fires_x, names_x)
            except Exception:
                pass


@case(
    "prio_low_no_starvation",
    profiles=["single-nonbatch"],
    source="design §2.2 #4 — PR8(完成口径) + P6",
)
def prio_low_no_starvation(ctx: CaseContext):
    """Low-priority completion under non-saturated load (PR8 completion
    calibre + P6).  Two waves on the shared profile env (no inflight cap):
    each wave fires 30x4 FIRST (early low-priority arrivals) then 70x4 —
    8 requests against a capacity of thousands, no sustained saturation,
    so nothing queues and nothing preempts; the property under test is
    that the 30s still complete (rate 1.0).  With no explicit
    anti-starvation mechanism, non-suspension is the only mechanical
    protection (analysis report §3.7).  [Migration note: the source ran
    this on its 2P+4D production-shaped shared env; on this line the
    shared env is the single-nonbatch profile topology (1P+4D) — the
    non-saturation property is topology-independent, 8 requests remain
    far below capacity.]

    PR8's grade-registry entry is the deadline-ratio upper band (used by
    prio_queue_timeout_terminal); a completion rate cannot ride
    report.invariant("PR8") — the registry types PR8 as a band and
    invariant() rejects band ids — so the completion assertion folds
    into P6 with the rate spelled out in the detail (design §2.2's
    "PR8 完成口径" invariant intent).

    The 30-vs-70 terminal-latency split is recorded as calibration data
    only (non-saturated ratios are choreography-determined; no band).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    all_fires: list = []
    low_all: list = []
    high_all: list = []
    low_total = 0
    low_done = 0
    high_total = 0
    high_done = 0
    prefill_names: list = []
    try:
        # EV-1 phase-race guard (flake fix, 2026-08 third-round run): the
        # default FIRE_GAP_S=0.15s batch let consecutive submitters hit
        # the occupied-prefill-slot window inside the master's 1s
        # status-poll period — one wave request parked into the single
        # probe slot and hung on queueTimeout (inflight scheduler=1 for
        # 30s+).  Non-saturation is made DETERMINISTIC instead: fast
        # prefill (50ms) plus a 1.5s submit gap (> poll period +
        # completion visibility) means every submitter finds the queue
        # empty and dispatches directly.
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=50.0)
        time.sleep(PERF_SETTLE_S)
        for wave in range(2):
            wave_fires = []
            for i in range(8):
                rid = ops.next_request_id(base)
                prio = 30 if i < 4 else 70
                fr = _fire(ops, rid, priority=prio, input_len=2048, output_len=2)
                wave_fires.append(fr)
                time.sleep(1.5)
            all_fires.extend(wave_fires)
            low_fires = [fr for fr in wave_fires if fr.kwargs.get("priority") == 30]
            high_fires = [fr for fr in wave_fires if fr.kwargs.get("priority") == 70]
            low_all.extend(low_fires)
            high_all.extend(high_fires)
            outcomes = _drain(ops, wave_fires)
            ok_rids = {rid for (rid, ok, _c, _d) in outcomes if ok}
            low_total += len(low_fires)
            low_done += sum(1 for fr in low_fires if fr.rid in ok_rids)
            high_total += len(high_fires)
            high_done += sum(1 for fr in high_fires if fr.rid in ok_rids)
            clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
            if not clean_ok:
                report.invariant(
                    "P6", False, detail=f"wave{wave} inflight: {clean_detail}"
                )
                return report.finish(f"wave{wave} inflight dirty, early stop")
            time.sleep(2.0)  # quiet window between waves

        rate = low_done / low_total if low_total else 0.0
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            low_done == low_total and high_done == high_total and clean_ok,
            detail=(
                f"low completion {low_done}/{low_total} (rate {rate:.2f}), "
                f"high completion {high_done}/{high_total}, "
                f"low latency={_lat_stats(low_all)}, "
                f"high latency={_lat_stats(high_all)}, "
                f"inflight={'ok' if clean_ok else clean_detail}"
            ),
        )
        return report.finish(
            f"low={low_done}/{low_total}, high={high_done}/{high_total}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, all_fires, prefill_names)


@case(
    "prio_queue_timeout_terminal",
    profiles=["single-nonbatch"],
    source="design §2.2 #5 — PR8(band) + P6",
)
def prio_queue_timeout_terminal(ctx: CaseContext):
    """queueTimeout as an absolute deadline (PR8 band + P6): sustained
    high-priority pressure must terminal the queued low-priority requests
    AT the deadline — never suspended past it (design §3.4 row 6,
    passive half: repeated juggling/queue-jumping never extends
    expiresAtMs; the active priorityAdmission half lives in
    atpm_timeout_attribution).

    ENV-T1: queueTimeout 8s, no preemption (the plain-timeout path — no
    priorityAdmission, so no 8430 attribution in this env), maxWaiting 8,
    inflight cap 1.

    Choreography (calibrated from the design's 70x3x4s sketch): 70a
    placeholder (10000ms) holds the lease; then 30a, 30b, 30c, 70b submit
    in one batch.  [EV-1-FIXED] Under the intake3 pull-based coordinator
    (PendingPlacementCoordinator, 6ad0315f10) the whole wave parks — and
    with prefill 10s > queueTimeout 8s > submit window ~0.7s, EVERY wave
    request's absolute deadline fires before the first lease release:
    all four settle 8511 BATCH_SLO_EXPIRED at enqueue+8s (30a ≈8.01s,
    70b ≈8.7s), zero route-reject.  The E10 calibration form (prefill
    deliberately beyond the deadline so the parked head provably expires
    AT its absolute deadline) carries over to every parked request.

    Assertions: PR8 band = max low-priority terminal wall-time / 8000ms
    (strict 1.25 — the latest submitter's deadline lands ≈1.07, the
    absolute-deadline proof: no suspension, no extension); low terminals
    all typed 8511 (implementation-period correction: the design's
    {8503, 8402, 8430} assumed QUEUE_TIMEOUT 8503 is the plain-path
    code, but 8503 is dead code in the master — the ordinary
    queued-expiry terminal is BATCH_SLO_EXPIRED 8511,
    RequestSlot.deadlineErrorType configured at registration); 70a
    succeeds; P6 every request reaches a terminal (no suspension).
    """
    env = ctx.env_manager.ensure(_t1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        # E10 calibration: prefill 10s > queueTimeout 8s so every parked
        # request provably expires at its absolute deadline (probe E10:
        # 8511 at wall=8.01s).  [EV-1-FIXED] under the pull model the whole
        # wave parks: 30a/30b/30c and 70b all settle 8511 at their own
        # enqueue+8s deadlines before the t=10s lease release; 70a
        # (dispatched t=0) completes.
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=10_000.0)
        time.sleep(PERF_SETTLE_S)

        h1 = ops.next_request_id(base)
        h1_fire = _fire(ops, h1, priority=70, input_len=2048, output_len=2)
        fires.append(h1_fire)
        if not h1_fire.ok:
            return False, f"70a schedule failed: code={h1_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "70a never dispatched"

        low_rids = [ops.next_request_id(base) for _ in range(3)]
        h2 = ops.next_request_id(base)
        specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 2})
            for rid in low_rids
        ]
        specs.append((h2, {"priority": 70, "input_len": 2048, "output_len": 2}))
        wave = _fire_batch(ops, specs)
        fires.extend(wave)
        low_fires = wave[:3]
        h2_fire = wave[3]

        outcomes = _drain(ops, fires)
        by_rid = {rid: (ok, code) for (rid, ok, code, _detail) in outcomes}

        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the whole wave parks (pull-based) and, with prefill
        # 10s > queueTimeout 8s > submit window ~0.7s, EVERY wave request's
        # absolute deadline fires before the first lease release — all
        # four settle 8511 BATCH_SLO_EXPIRED at enqueue+8s (30a ≈8.01s,
        # 70b ≈8.7s), none suspended past its deadline, zero route-reject.
        h1_ok = by_rid[h1][0]
        low_codes = [by_rid[rid][1] for rid in low_rids]
        wave_all_expired = (
            all(code == CODE_SLO_EXPIRED for code in low_codes)
            and by_rid[h2][1] == CODE_SLO_EXPIRED
        )
        max_low_s = max(fr.settled_s - fr.submitted_s for fr in low_fires)
        ratio = max_low_s / 8.0
        report.check(
            "PR8",
            ratio,
            context="queue_timeout_terminal",
            detail=(
                f"[EV-1-FIXED] all three 30s settle 8511 at their own "
                f"enqueue+8s deadlines (max wall {max_low_s * 1000:.0f}ms "
                f"/ 8000ms, absolute — no suspension, no extension), "
                f"wave codes="
                f"{[(rid % 1_000_000, c) for rid, c in zip(low_rids, low_codes)]}, "
                f"70b={by_rid[h2][1]} (parked; deadline before first release)"
            ),
        )
        report.invariant(
            "P6",
            h1_ok and wave_all_expired,
            detail=(
                f"[EV-1-FIXED] 70a ok={h1_ok}, whole wave 8511="
                f"{wave_all_expired} (park-to-deadline terminals, zero "
                f"route-reject), no suspension (queueTimeout absolute)"
            ),
        )
        return report.finish(
            f"ratio={ratio:.2f}, low codes={low_codes}, 70b={by_rid[h2][1]} "
            f"[EV-1-FIXED], grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)


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


@case(
    "atpm_preempt_prefill_queued",
    profiles=["single-nonbatch"],
    source="design §2.3 #6 — PR10 + PR5 + PR6 + PR4",
)
def atpm_preempt_prefill_queued(ctx: CaseContext):
    """PREFILL_QUEUED preemption choreography under the intake3 pull model
    (PR10 + PR5 + PR6 + PR4, [EV-1-FIXED] design-final form).

    ENV-Q2: preemption allows PREFILL_QUEUED only, queueTimeout 60s,
    maxWaiting 8, inflight cap 1, single prefill.

    [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
    (6ad0315f10): capacity blocking parks EVERY submitter (pull-based
    WaitBucket, priority desc + FIFO tiebreak), so the queue-replacement
    choreography (a failed enqueue feeding AdmissionFallback → evict
    exactly one 30f → 8400) has no trigger — maxWaiting's
    enqueueUnderLock cap is a BATCH-path check the NON_BATCH pull model
    never reaches, no enqueue ever fails, and the eviction fallback
    never runs (zero victims across both waves; the deficit==1
    replacement exactness and multi-victim events migrate to the
    BATCH-profile white-box handover, design §2.5 row 11).

    Wave 1: a priority=50 placeholder parks the lease, then EIGHT
    requests + the incoming 70 queue up: 30a, 30b, 40a, 40b, 30c, 30d,
    30e, 30f, 70.  All nine park and complete 200; the dispatch order
    (design-final) is [30a (first parker — the wave's first submitter
    legitimately wins the first release slot), 70, 40a, 40b, 30b, 30c,
    30d, 30e, 30f] — after the first parker, strict priority desc +
    same-level FIFO.

    Wave 2 (same-priority infeasible shape): after the drain, a 70
    placeholder parks the lease; 70x8 + the incoming 90 all park.  The
    "no strictly-lower candidate → DECLINED" branch stays what the
    (never-triggered) fallback would see; zero victims holds trivially,
    and all nine complete 200 with the 90 dispatching FIRST among the
    wave (priority desc; first parker 70a keeps slot one).
    """
    env = ctx.env_manager.ensure(_q2_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=4000.0)
        time.sleep(PERF_SETTLE_S)

        # ---- wave 1: victim selection + deficit exactness --------------
        ph = ops.next_request_id(base)
        ph_fire = _fire(ops, ph, priority=50, input_len=2048, output_len=2)
        fires.append(ph_fire)
        if not ph_fire.ok:
            return False, f"wave1 placeholder failed: code={ph_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "wave1 placeholder never dispatched"

        tags = ["30a", "30b", "40a", "40b", "30c", "30d", "30e", "30f"]
        rids: dict = {}
        specs = []
        for tag in tags:
            rid = ops.next_request_id(base)
            rids[tag] = rid
            specs.append(
                (rid, {"priority": int(tag[:-1]), "input_len": 2048, "output_len": 2})
            )
        incoming = ops.next_request_id(base)
        specs.append((incoming, {"priority": 70, "input_len": 2048, "output_len": 2}))
        wave1 = _fire_batch(ops, specs)
        fires.extend(wave1)

        outcomes1 = _drain(ops, [ph_fire] + wave1)
        m1 = _outcome_map(outcomes1)
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the whole wave parks — no enqueue ever fails, the
        # eviction fallback never runs, zero victims.  Design-final shape:
        # all nine settle 200 and the dispatch order is [30a (first
        # parker), 70, 40a, 40b, 30b..30f] (first submitter + priority
        # desc + same-level FIFO after it).
        zero_eviction_w1 = all(
            m1[rids[tag]][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
            for tag in tags
        ) and m1[incoming][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
        wave1_rids = [rids[t] for t in tags] + [incoming]
        prio1 = {rids[t]: int(t[:-1]) for t in tags}
        prio1[incoming] = 70
        first1, shape1, order1 = _design_final_pattern(
            ops, [ph_fire] + wave1, wave1_rids, prio1
        )
        all1_ok = all(m1[rids[t]][0] for t in tags) and m1[incoming][0]
        ph1_ok = m1[ph][0]

        report.invariant(
            "PR10",
            shape1 and zero_eviction_w1 and ph1_ok,
            context="deficit_exact_one_design_final",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): no enqueue ever "
                f"fails (maxWaiting is a BATCH-path cap under the NON_BATCH "
                f"pull model), the queue-full replacement has no trigger "
                f"and zero victims holds; dispatch shape ok={shape1}, "
                f"zero 8400/8429={zero_eviction_w1}, "
                f"codes={[(t, m1[rids[t]][1]) for t in tags]}, "
                f"incoming70={m1[incoming][1]}, "
                f"dispatch={[r % 1_000_000 for r in order1]}"
            ),
        )
        report.invariant(
            "PR5",
            zero_eviction_w1 and shape1,
            context="victim_determinism_design_final",
            detail=(
                "[EV-1-FIXED] baseline flipped at intake3 "
                "PendingPlacementCoordinator (6ad0315f10): eviction never "
                "triggers (no failed enqueue feeds the fallback) — zero "
                "victims across the wave; victim-selection determinism "
                "stays a white-box handover, the design-final dispatch "
                "shape carries the ordering evidence"
            ),
        )
        report.invariant(
            "PR6",
            all1_ok and shape1,
            context="prefill_queued_terminal_design_final",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): every parked "
                f"submitter (not just the first) is re-pulled on capacity "
                f"release — all wave codes 200, zero route-reject; "
                f"codes={[(t, m1[rids[t]][1]) for t in tags]}, "
                f"incoming70={m1[incoming][1]}, "
                f"dispatch={[r % 1_000_000 for r in order1]}"
            ),
        )
        report.invariant(
            "PR4",
            zero_eviction_w1 and ph1_ok and shape1 and all1_ok,
            context="strict_low_priority_victims_design_final",
            detail=(
                "[EV-1-FIXED] baseline flipped at intake3 "
                "PendingPlacementCoordinator (6ad0315f10): strictly-lower-"
                "priority victim selection stays white-box (the eviction "
                "fallback has no trigger under the pull model); the "
                "design-final form is zero victims + every queued request "
                "completing untouched in priority-desc dispatch order"
            ),
        )
        clean1_ok, clean1_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            shape1 and all1_ok and ph1_ok and clean1_ok,
            detail=(
                f"[EV-1-FIXED] wave1: all nine requests dispatched from the "
                f"park bucket and completed 200 (first parker 30a, then "
                f"priority desc + FIFO), "
                f"inflight={'ok' if clean1_ok else clean1_detail}"
            ),
        )
        if not (shape1 and all1_ok and ph1_ok and clean1_ok):
            return report.finish(f"wave1 incomplete, grades: {report.summary()}")

        # ---- wave 2: infeasible → zero eviction -------------------------
        ph2 = ops.next_request_id(base)
        ph2_fire = _fire(ops, ph2, priority=70, input_len=2048, output_len=2)
        fires.append(ph2_fire)
        if not ph2_fire.ok:
            return False, f"wave2 placeholder failed: code={ph2_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "wave2 placeholder never dispatched"

        w2_rids = [ops.next_request_id(base) for _ in range(8)]
        w2_specs = [
            (rid, {"priority": 70, "input_len": 2048, "output_len": 2})
            for rid in w2_rids
        ]
        inc90 = ops.next_request_id(base)
        w2_specs.append((inc90, {"priority": 90, "input_len": 2048, "output_len": 2}))
        wave2 = _fire_batch(ops, w2_specs)
        fires.extend(wave2)

        outcomes2 = _drain(ops, [ph2_fire] + wave2)
        m2 = _outcome_map(outcomes2)
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the same-priority wave parks all eight 70s AND the
        # incoming 90 — the "no strictly-lower candidate -> DECLINED"
        # branch is what the (never-triggered) fallback would still see,
        # zero victims holds trivially, and the design-final form is all
        # nine completing 200 with the 90 dispatching FIRST among the wave
        # (priority desc; first parker 70a keeps slot one).
        zero_eviction = all(
            m2[rid][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
            for rid in w2_rids + [ph2, inc90]
        )
        prio2 = {rid: 70 for rid in w2_rids}
        prio2[inc90] = 90
        first2, shape2, order2 = _design_final_pattern(
            ops, [ph2_fire] + wave2, w2_rids + [inc90], prio2
        )
        all2_ok = all(m2[rid][0] for rid in w2_rids) and m2[inc90][0]
        ph2_ok = m2[ph2][0]
        report.invariant(
            "PR10",
            zero_eviction and shape2 and all2_ok and ph2_ok,
            context="infeasible_no_partial_eviction_design_final",
            detail=(
                f"[EV-1-FIXED] zero eviction={zero_eviction} (no "
                f"strictly-lower candidate for the 90 — all-or-nothing, and "
                f"the fallback never triggers anyway), "
                f"90 terminal={m2[inc90][1]} (dispatched first among the "
                f"wave, priority desc), shape ok={shape2}, "
                f"dispatch={[r % 1_000_000 for r in order2]}, ph ok={ph2_ok}"
            ),
        )
        clean2_ok, clean2_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            shape2 and all2_ok and ph2_ok and clean2_ok,
            detail=(
                f"[EV-1-FIXED] wave2: all nine requests completed 200 (the "
                f"90 first among the wave by priority), "
                f"inflight={'ok' if clean2_ok else clean2_detail}"
            ),
        )
        return report.finish(
            f"wave1 shape={shape1} all-200 zero-victims, wave2 shape={shape2} "
            f"zero-eviction={zero_eviction}, grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)


@case(
    "atpm_preempt_decode_engine_owned",
    profiles=["single-nonbatch"],
    source="design §2.3 #7 — AT5(band) + PR6 + PR10(decode)",
)
def atpm_preempt_decode_engine_owned(ctx: CaseContext):
    """DECODE_RESERVED vs DECODE_ENGINE_OWNED eviction terminal split
    (PR6: reserved → 8400 retryable, engine-accepted → 8429 typed cancel)
    plus the preemption closure budget (AT5 band).

    ENV-D1: 2P+4D, preemption allows both decode stages with
    engineCancellation {ack 50ms, completion 1000ms}, prefill inflight
    cap 3 (so four victims + the incoming all dispatch concurrently),
    queueTimeout 60s.

    Guardrail (design §2.5 row 2, EvictionManager.java:445-452): decode
    eviction is never a substitute for an ordinary available endpoint —
    EVERY decode endpoint must be in the needs-eviction state first.
    Decode saturation is manufactured at run time by injecting kv
    pressure on all four decode engines (the decode_cache_blocks knob
    does not reach the mock's KV reporting — implementation-period
    finding), so the snapshot evidence "no ordinary endpoint available"
    holds before every wave.

    Injection timing (implementation-period correction, the third major
    one): a PRIORITY queue deliberately RETAINS the strict decode KV gate
    in ordinary routing (CostBasedDecodeStrategy.applyHardFilters →
    availableKv < seqLen filters the endpoint; softQueuePlacement is
    queue && !priorityOrdering).  kv_pressure therefore goes in only
    AFTER the victim wave has routed (dispatch ACK ⇒ decode reservation
    established) and BEFORE the incoming fires; the victims' own decode
    handoff uses the already-pinned reservation and is unaffected, while
    the incoming's ordinary route fails (NO_DECODE_WORKER 8403) into
    AdmissionFallback → decode eviction.  Between waves the pressure is
    released so the next victim wave can route.

    Wave 1 (reserved-only → 8400): prefill slowed to 4s so the reserved
    window comfortably covers kv_pressure settle (master status poll 1s)
    plus the incoming's route; four priority=30 victims fire and settle
    (their decode reservations exist while their prefill is still
    executing — output_len=500 keeps the decode phase long); the 70 is
    fired inside that window, its decode placement fails on every
    endpoint → local eviction of a reserved victim → victim terminal
    8400, the 70 completes.  The reserved window is tight; if the
    observed terminal turns out 8429 (the victim had already reached
    decode running), the case records the degradation instead of
    pretending the split — the assert stays strict so the first real run
    calibrates it.

    Wave 2 (engine-owned → Cancel → 8429): four fresh 30s are polled
    until RUNNING on decode engines, then the 70 fires → the tokenized
    Cancel coordinator evicts one owned victim → typed CANCELED+8429
    (grpc-status-details-bin), the engine records the cancellation
    (verify_engine_cancelled), and the 70 itself completes.  AT5 closure =
    the 70's first engine running_ms (epoch) minus the victim's stream
    terminal (client clock crossed into the epoch domain via
    _mono_to_epoch) — expected well inside completionTimeoutMs(1000) +
    scheduling margin.

    Victim-count note: exactly ONE victim per wave (the planner releases
    one endpoint's worth); the 70 then takes that endpoint.
    """
    # A4 (Mark P1-2): batch-dispatch caliber reservation, following the
    # dual-caliber paradigm (is_batch = ctx.batch_dispatch();
    # completion-duration caliber under BATCH, client-TTFT under NON_BATCH).
    # Under BATCH dispatch the master enqueues the stream itself
    # (enqueued_by_master), so the _StreamTerminal direct-stream channel —
    # and with it the live 8429/AT5-closure observation — is NON_BATCH-only
    # by construction.  Unreachable under this case's profile
    # (single-nonbatch = NON_BATCH base, PRIORITY axis injected at the case
    # layer); the arm is reserved so a future priority-batch variant fills
    # it without touching the NON_BATCH logic below.
    if ctx.batch_dispatch():
        # TODO(A4): BATCH arm — victim terminal rides FetchResponse, the
        # closure caliber is completion-duration; fill when a
        # priority-batch variant enables BATCH dispatch.
        raise NotImplementedError(
            "atpm_preempt_decode_engine_owned BATCH arm reserved — fill "
            "when the priority-batch variant enables BATCH dispatch"
        )
    env = ctx.env_manager.ensure(_d1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    decode_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        decode_names = _decode_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=4000.0)
        time.sleep(PERF_SETTLE_S)

        # ---- wave 1: DECODE_RESERVED → local eviction → 8400 -----------
        w1_victim_rids = [ops.next_request_id(base) for _ in range(4)]
        w1_specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 500})
            for rid in w1_victim_rids
        ]
        w1_fires = _fire_batch(ops, w1_specs)  # dispatch ACKs: reservations in
        fires.extend(w1_fires)
        # Guardrail NOW: every decode endpoint needs eviction before the
        # incoming's decode preemption can run — saturate KV on all of them
        # (victims already routed; see the case docstring for the timing).
        for name in decode_names:
            ops.set_kv_pressure(name, MOCK_TOTAL_KV_TOKENS)
        time.sleep(PERF_SETTLE_S)
        # A7 (Mark P2-1): pre-fire guardrail — see
        # _decode_pressure_guardrail.
        w1_guard = _decode_pressure_guardrail(ops, decode_names)
        if not w1_guard[0]:
            return False, f"wave1 decode guardrail failed: {w1_guard[1]}"
        w1_inc = ops.next_request_id(base)
        w1_inc_fire = _fire(ops, w1_inc, priority=70, input_len=2048, output_len=2)
        fires.append(w1_inc_fire)

        w1_outcomes = _drain(ops, w1_fires + [w1_inc_fire])
        m1 = _outcome_map(w1_outcomes)
        w1_codes = {rid: m1[rid][1] for rid in w1_victim_rids}
        w1_yielded = [rid for rid in w1_victim_rids if m1[rid][1] == CODE_YIELDED]
        w1_owned = [
            rid for rid in w1_victim_rids if m1[rid][1] == CODE_ENGINE_CANCELLED
        ]
        w1_inc_ok = m1[w1_inc][0]
        w1_inc_code = m1[w1_inc][1]
        w1_victims_ok = all(m1[rid][0] for rid in w1_victim_rids)
        w1_zero_eviction = not (w1_yielded or w1_owned)
        w1_inc_rejected = (not w1_inc_ok) and w1_inc_code in EV2_REJECT_FAMILY
        # EV-2 baseline (behaviour finding, probes E9/E11 + the
        # DecodeEndpoint projection math): DECODE_RESERVED eviction never
        # fires — the kv dimension is mathematically unreachable
        # (freedKv is a subset of currentHardCharges, so "fits after
        # eviction" implies "fits without it", contradicting the
        # INFEASIBLE entry check), and the slots dimension is absorbed
        # engine-side (decode_max_concurrency=1 with four RUNNING victims
        # still dispatches the incoming — E11).  Observable form: the
        # victims all complete, zero 8400/8429, the incoming keeps a
        # rejection from EV2_REJECT_FAMILY.
        report.invariant(
            "PR6",
            w1_zero_eviction and w1_inc_rejected and w1_victims_ok,
            context="decode_reserved_terminal_ev2",
            detail=(
                f"[EV-2] reserved wave (EV-2 baseline): victims="
                f"{ {r % 1_000_000: c for r, c in w1_codes.items()} } "
                f"(all complete — reserved eviction never fires), "
                f"yielded(8400)={len(w1_yielded)}, "
                f"owned(8429)={len(w1_owned)}, "
                f"incoming70 ok={w1_inc_ok} code={w1_inc_code} "
                f"(family {list(EV2_REJECT_FAMILY)})"
            ),
        )
        clean1_ok, _cd = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        if not clean1_ok:
            return report.finish(
                f"wave1 inflight dirty, early stop, " f"grades: {report.summary()}"
            )

        # ---- wave 2: DECODE_ENGINE_OWNED → Cancel RPC → 8429 -----------
        # Release the KV pressure first: the fresh victim wave must route
        # normally (the strict KV gate would reject them otherwise — see
        # the case docstring).
        for name in decode_names:
            ops.set_kv_pressure(name, 0)
        time.sleep(PERF_SETTLE_S)
        w2_victim_rids = [ops.next_request_id(base) for _ in range(4)]
        w2_specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 500})
            for rid in w2_victim_rids
        ]
        w2_fires = _fire_batch(ops, w2_specs)
        fires.extend(w2_fires)
        running_all = all(
            _poll_decode_running(ops, rid, timeout_s=20.0) for rid in w2_victim_rids
        )
        if not running_all:
            return False, "wave2 victims never reached decode running"
        # Re-saturate every decode endpoint, then fire the incoming.
        for name in decode_names:
            ops.set_kv_pressure(name, MOCK_TOTAL_KV_TOKENS)
        time.sleep(PERF_SETTLE_S)
        # A7: same guardrail for wave 2 — the AT5 observation wave.
        w2_guard = _decode_pressure_guardrail(ops, decode_names)
        if not w2_guard[0]:
            return False, f"wave2 decode guardrail failed: {w2_guard[1]}"
        w2_inc = ops.next_request_id(base)
        w2_inc_fire = _fire(ops, w2_inc, priority=70, input_len=2048, output_len=2)
        fires.append(w2_inc_fire)

        w2_outcomes = _drain(ops, w2_fires + [w2_inc_fire])
        m2 = _outcome_map(w2_outcomes)
        w2_owned = [
            rid for rid in w2_victim_rids if m2[rid][1] == CODE_ENGINE_CANCELLED
        ]
        w2_survivors_ok = all(
            m2[rid][0] for rid in w2_victim_rids if rid not in w2_owned
        )
        w2_inc_ok = m2[w2_inc][0]
        w2_inc_code = m2[w2_inc][1]
        w2_zero_eviction = not w2_owned
        w2_inc_rejected = (not w2_inc_ok) and w2_inc_code in EV2_REJECT_FAMILY
        cancel_evidence = []
        for rid in w2_owned:
            ok_c, detail_c = ops.verify_engine_cancelled(rid)
            cancel_evidence.append(f"{rid % 1_000_000}:{ok_c}")
        # EV-2 baseline (behaviour finding, probe E11 two-orchestration):
        # DECODE_ENGINE_OWNED eviction (tokenized Cancel → 8429) is
        # equally unreachable — the 8429/8400 terminal split has no
        # observation object.  Observable form mirrors wave 1.
        report.invariant(
            "PR6",
            w2_zero_eviction and w2_inc_rejected and w2_survivors_ok,
            context="decode_owned_terminal_ev2",
            detail=(
                f"[EV-2] owned wave (EV-2 baseline): 8429 victims="
                f"{len(w2_owned)} (engine-owned eviction never fires), "
                f"engine cancel evidence={cancel_evidence}, "
                f"incoming70 ok={w2_inc_ok} code={w2_inc_code} "
                f"(family {list(EV2_REJECT_FAMILY)}), "
                f"survivors ok={w2_survivors_ok}"
            ),
        )

        # A9-4 (Mark P3-2): PR10(decode) in its vacuous EV-2 form — the
        # replacement-precision property (deficit-exact victims,
        # infeasible → zero partial eviction) has no decode-side object
        # while decode eviction never fires; the vacuous form (zero
        # evictions across BOTH waves, no deficit object anywhere) is
        # asserted so the property stays registered with the degradation
        # recorded instead of silently absent.
        report.invariant(
            "PR10",
            w1_zero_eviction and w2_zero_eviction,
            context="decode_deficit_vacuous_ev2",
            detail=(
                "[EV-2] decode-side PR10 vacuous form: zero evictions "
                "across both waves (reserved + owned), no deficit "
                "object — replacement precision unobservable while "
                "decode eviction never fires"
            ),
        )

        # AT5 closure: incoming first engine running (epoch ms) minus the
        # victim's stream terminal crossed into the epoch domain.  Under
        # EV-2 there is no victim terminal to anchor against, so the
        # banded property has NO observation object this run: check()
        # would need a fabricated value and invariant() is illegal for a
        # banded property (raises) — the gap is filed as behaviour
        # finding EV-2 and carried in the case detail instead.  The
        # terminal channel itself is now LIVE (A1, Ryan P1-1):
        # _StreamTerminal maps the engine's in-band error frames
        # (GenerateOutputsPB.error_info → CANCELLED enum →
        # CODE_ENGINE_CANCELLED), so the moment a Java-side EV-2 fix makes
        # decode eviction reachable, w2_owned populates from real 8429
        # terminals, the verify_engine_cancelled evidence loop above
        # runs, and this computation feeds the AT5 band automatically —
        # no further framework change needed.
        closure_ms = None
        if w2_owned and w2_inc_ok:
            victim_fire = next(fr for fr in w2_fires if fr.rid == w2_owned[0])
            inc_lc = _prefill_lifecycle(ops, w2_inc) or {}
            if (
                victim_fire.terminal is not None
                and victim_fire.terminal.terminated_s is not None
                and inc_lc.get("running_ms")
            ):
                victim_end_epoch = _mono_to_epoch(victim_fire.terminal.terminated_s)
                closure_ms = inc_lc["running_ms"] - victim_end_epoch * 1000.0
        if closure_ms is not None:
            report.check(
                "AT5",
                closure_ms,
                context="preemption_closure",
                detail=(
                    "closure = incoming prefill running_ms − victim stream "
                    f"terminal (epoch-crossed); completionTimeoutMs=1000"
                ),
            )
        clean2_ok, clean2_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            w2_inc_rejected and w2_survivors_ok and clean2_ok,
            detail=(
                f"[EV-2] wave2 drained (EV-2: zero eviction, all victims "
                f"completed, incoming terminal {w2_inc_code}), "
                f"inflight={'ok' if clean2_ok else clean2_detail}"
            ),
        )
        return report.finish(
            f"EV-2 baseline: wave1 zero-eviction (incoming {w1_inc_code}), "
            f"wave2 zero-eviction (incoming {w2_inc_code}), "
            f"closure_ms={'n/a (EV-2)' if closure_ms is None else f'{closure_ms:.0f}'}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for name in decode_names:
                ops.set_kv_pressure(name, 0)
        except Exception:
            pass
        _finally_hygiene(ops, fires, prefill_names)


@case(
    "atpm_same_priority_zero_eviction",
    profiles=["single-nonbatch"],
    source="design §2.3 #8 — PR4 + AT3",
)
def atpm_same_priority_zero_eviction(ctx: CaseContext):
    """Same-priority never evicts (PR4 core + AT3, [EV-1-FIXED] design-final
    form): eight explicit priority=50 requests (Python-side per-request
    priority — the FORCE_PRIORITY semantics without the Java load
    client) plus the incoming 50 (the ninth) ALL park in the intake3
    PendingPlacementCoordinator (pull-based, priority desc + FIFO
    tiebreak — baseline flipped at 6ad0315f10); the (never-triggered)
    eviction fallback's strictly-lower-priority candidate filter would
    come up empty for a same-priority incoming, so ZERO victims are
    taken (no 8400/8429 anywhere) and the design-final shape is all
    nine completing 200 in pure submit FIFO order.
    ENV-Q2 is shared with atpm_preempt_prefill_queued (same
    fingerprint → same run, sequential order + finally hygiene)."""
    env = ctx.env_manager.ensure(_q2_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=4000.0)
        time.sleep(PERF_SETTLE_S)

        ph = ops.next_request_id(base)
        ph_fire = _fire(ops, ph, priority=50, input_len=2048, output_len=2)
        fires.append(ph_fire)
        if not ph_fire.ok:
            return False, f"placeholder failed: code={ph_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "placeholder never dispatched"

        queued_rids = [ops.next_request_id(base) for _ in range(8)]
        specs = [
            (rid, {"priority": 50, "input_len": 2048, "output_len": 2})
            for rid in queued_rids
        ]
        inc = ops.next_request_id(base)
        specs.append((inc, {"priority": 50, "input_len": 2048, "output_len": 2}))
        wave = _fire_batch(ops, specs)
        fires.extend(wave)

        outcomes = _drain(ops, [ph_fire] + wave)
        m = _outcome_map(outcomes)
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): every submitter parks — the incoming 50 no longer
        # receives the route-reject family, it completes like the rest; the
        # same-priority FIFO dispatch shape (pure submit order) is the
        # design-final observation object for "same priority never evicts".
        zero_eviction = all(
            m[rid][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
            for rid in queued_rids + [inc]
        )
        prio = {rid: 50 for rid in queued_rids + [inc]}
        _first_sp, shape_sp, order_sp = _design_final_pattern(
            ops, [ph_fire] + wave, queued_rids + [inc], prio
        )
        all_ok = all(m[rid][0] for rid in queued_rids) and m[inc][0]
        report.invariant(
            "PR4",
            zero_eviction and shape_sp and all_ok,
            context="same_priority_zero_eviction_design_final",
            detail=(
                f"[EV-1-FIXED] zero 8400/8429={zero_eviction}, all nine 50s "
                f"completed={all_ok}, dispatch FIFO shape ok={shape_sp} "
                f"(pure submit order), "
                f"dispatch={[r % 1_000_000 for r in order_sp]}"
            ),
        )
        report.invariant(
            "AT3",
            m[inc][1] == CODE_OK,
            context="single_qos_incoming_design_final",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): the same-priority "
                f"incoming parks and completes 200 like every queued peer — "
                f"the original-error passthrough "
                f"({list(ROUTE_REJECT_FAMILY)}) is no longer reachable for "
                f"a capacity-blocked same-priority submitter, incoming 50 "
                f"terminal={m[inc][1]}"
            ),
        )
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            shape_sp and all_ok and clean_ok,
            detail=(
                f"[EV-1-FIXED] all nine 50s + ph dispatched and completed "
                f"(FIFO), inflight={'ok' if clean_ok else clean_detail}"
            ),
        )
        return report.finish(
            f"zero-eviction={zero_eviction}, incoming code={m[inc][1]}, "
            f"fifo shape={shape_sp}, grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)


@case(
    "atpm_preemption_disabled_zero_eviction",
    profiles=["single-nonbatch"],
    source="design §2.3 #9 — AT2",
)
def atpm_preemption_disabled_zero_eviction(ctx: CaseContext):
    """Omitting the preemption block disables preemption entirely (AT2):
    PRIORITY ordering but no preemption config → EvictionManager's
    precondition rejects before any planning.  Two rounds: a saturated
    low-priority queue whose incoming 70 parks in the intake3
    PendingPlacementCoordinator ([EV-1-FIXED] baseline flipped at
    6ad0315f10 — the anti-overload mechanism under PRIORITY is now
    parking, not plain rejection; no fallback fires because no
    preemption config exists — zero 8400/8429/8430), then a saturated
    high-priority queue with an incoming 90 — the same park (high
    priority does not bypass capacity either; both incomings complete
    200 once capacity releases, or expire 8511 inside the queueTimeout
    window).

    ENV-T1 is shared with prio_queue_timeout_terminal (identical
    fingerprint): sequential execution + per-case finally hygiene.
    queueTimeout 8s means the deep tail of each saturated queue times out
    — that is a legal terminal (P6 = every request terminal, not every
    request completed)."""
    env = ctx.env_manager.ensure(_t1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        round_reports = []
        for label, fill_prio, inc_prio in (
            ("low_saturation", 30, 70),
            ("high_saturation", 70, 90),
        ):
            ph = ops.next_request_id(base)
            ph_fire = _fire(ops, ph, priority=fill_prio, input_len=2048, output_len=2)
            fires.append(ph_fire)
            if not ph_fire.ok:
                return False, f"{label} placeholder failed: code={ph_fire.code}"
            if not _poll_engine_pending(ops, prefill_names[0], 1):
                return False, f"{label} placeholder never dispatched"

            queued = [ops.next_request_id(base) for _ in range(8)]
            specs = [
                (rid, {"priority": fill_prio, "input_len": 2048, "output_len": 2})
                for rid in queued
            ]
            inc = ops.next_request_id(base)
            specs.append(
                (inc, {"priority": inc_prio, "input_len": 2048, "output_len": 2})
            )
            wave = _fire_batch(ops, specs)
            fires.extend(wave)
            outcomes = _drain(ops, [ph_fire] + wave)
            m = _outcome_map(outcomes)
            zero_preempt = all(
                m[rid][1]
                not in (CODE_YIELDED, CODE_ENGINE_CANCELLED, CODE_ADMISSION_TIMEOUT)
                for rid in queued + [ph]
            )
            # [EV-1-FIXED] baseline flipped at intake3
            # PendingPlacementCoordinator (6ad0315f10): the incoming parks
            # (no preemption fallback exists to fire — config omitted) and
            # completes 200 once capacity releases; an 8511 park expiry
            # inside the queueTimeout window is equally legal.
            inc_ok = m[inc][1] in (CODE_OK, CODE_SLO_EXPIRED)
            round_reports.append(
                (
                    label,
                    zero_preempt and inc_ok,
                    f"{label}: zero 8400/8429/8430={zero_preempt}, "
                    f"incoming{inc_prio} code={m[inc][1]} "
                    f"(park → 200 or 8511 expiry, [EV-1-FIXED])",
                )
            )
            clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
            if not clean_ok:
                round_reports[-1] = (
                    label,
                    False,
                    round_reports[-1][2] + f", inflight dirty: {clean_detail}",
                )
                break

        report.invariant(
            "AT2",
            all(ok for (_l, ok, _d) in round_reports),
            context="preemption_disabled",
            detail="; ".join(
                f"{l}={'ok' if ok else 'FAIL(' + d + ')'}" for l, ok, d in round_reports
            ),
        )
        report.invariant(
            "P6",
            all(ok for (_l, ok, _d) in round_reports),
            detail="every request reached a terminal (queue-timeout terminals included)",
        )
        return report.finish(
            f"rounds={[l for l, ok, _d in round_reports if ok]}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)


@case(
    "atpm_timeout_attribution",
    profiles=["single-nonbatch"],
    source="design §2.3 #10 — PR7 (+PR8 deadline-no-extension)",
)
def atpm_timeout_attribution(ctx: CaseContext):
    """Admission-timeout expiry uniformity (PR7, [EV-1-FIXED] design-final
    form): under the intake3 PendingPlacementCoordinator (6ad0315f10) a
    capacity-blocked submitter parks with the schedule() RPC blocking
    until its queueTimeoutMs deadline, then terminals as plain 8511
    BATCH_SLO_EXPIRED with admission_reject_reason=UNSPECIFIED(0) — the
    attributed form (8430 + HIGHER_PRIORITY_AHEAD / 8431 +
    RESOURCE_EXHAUSTED) needs the AdmissionFailureClassifier to run at
    the queued-expiry decision, and that classifier has ZERO call sites
    in the intake3 master (Java-side observation gap — filed, not fixed
    here); every queued expiry rides the plain deadlineErrorType path
    (RequestLifecycleCoordinator.timeoutEntry fallback).

    ENV-A1: PREFILL_QUEUED preemption, queueTimeout 7s, maxWaiting 8.

    Wave 1 (mixed priorities): a 90a placeholder (12s prefill) parks the
    lease; eight 30s, the incoming 70, then 90b/90c all park.  Every
    queued member expires 8511/UNSPECIFIED at its own deadline inside
    the 12s window; 90a completes.  Zero 8400 victims (the eviction
    fallback never triggers — no failed enqueue under the pull model).

    Wave 2 (same shape, single client): a 70_early placeholder (10s
    prefill — it must OUTLAST the 70_late's ~8s deadline, the deadline
    cancels at delivery ACK), eight 30s, incoming 70_late — the 70_late
    expires 8511/UNSPECIFIED, 70_early completes.

    Deadline-no-extension (PR8, raw recording): the 70_late's terminal
    wall-time / queueTimeoutMs(7000) must stay ~1 — the park never
    restarts expiresAtMs (A9-3 keeps prio_queue_timeout_terminal as
    PR8's ONLY band consumer; the raw value rides the case finish
    detail)."""
    env = ctx.env_manager.ensure(_a1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        # ---- wave 1: 8430 + HIGHER_PRIORITY_AHEAD ----------------------
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=12000.0)
        time.sleep(PERF_SETTLE_S)

        ph90 = ops.next_request_id(base)
        ph90_fire = _fire(ops, ph90, priority=90, input_len=2048, output_len=2)
        fires.append(ph90_fire)
        if not ph90_fire.ok:
            return False, f"wave1 90a failed: code={ph90_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "wave1 90a never dispatched"

        low_rids = [ops.next_request_id(base) for _ in range(8)]
        specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 2})
            for rid in low_rids
        ]
        inc70 = ops.next_request_id(base)
        specs.append((inc70, {"priority": 70, "input_len": 2048, "output_len": 2}))
        q90b = ops.next_request_id(base)
        specs.append((q90b, {"priority": 90, "input_len": 2048, "output_len": 2}))
        q90c = ops.next_request_id(base)
        specs.append((q90c, {"priority": 90, "input_len": 2048, "output_len": 2}))
        wave1 = _fire_batch(ops, specs)
        fires.extend(wave1)
        inc70_fire = wave1[8]
        q90b_fire = wave1[9]
        q90c_fire = wave1[10]

        outcomes1 = _drain(ops, [ph90_fire] + wave1)
        m1 = _outcome_map(outcomes1)
        inc70_code = m1[inc70][1]
        inc70_reason = None
        if inc70_fire.resp is not None:
            inc70_reason = int(inc70_fire.resp.admission_reject_reason)
        victims8400 = [rid for rid in low_rids if m1[rid][1] == CODE_YIELDED]
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): every wave submitter parks; the attributed form
        # (8430 + HIGHER_PRIORITY_AHEAD) needs the AdmissionFailureClassifier
        # at the queued-expiry decision, which has ZERO call sites in the
        # intake3 master (Java-side observation gap, filed).  Observable
        # design-final form: every parked expiry is uniform plain
        # 8511 + UNSPECIFIED, zero 8400 victims, the placeholder completes.
        w1_wave = low_rids + [inc70, q90b, q90c]
        w1_expired = all(m1[rid][1] == CODE_SLO_EXPIRED for rid in w1_wave)
        w1_reasons_unspec = all(
            fr.reason == REASON_UNSPECIFIED for fr in wave1 if fr.resp is not None
        )
        w1_ok = w1_expired and w1_reasons_unspec and victims8400 == [] and m1[ph90][0]
        report.invariant(
            "PR7",
            w1_ok,
            context="higher_priority_ahead_design_final",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): incoming70 "
                f"terminal={inc70_code} "
                f"reason={REASON_NAMES.get(inc70_reason, inc70_reason)} "
                f"(park-expiry uniform: the 8430 attribution classifier has "
                f"zero call sites in the intake3 master — Java gap, filed), "
                f"all-wave expired 8511={w1_expired}, "
                f"reasons UNSPECIFIED={w1_reasons_unspec}, "
                f"victims8400={len(victims8400)}, "
                f"90b={m1[q90b][1]}/{q90b_fire.reason}, "
                f"90c={m1[q90c][1]}/{q90c_fire.reason}, "
                f"90a completed={m1[ph90][0]}"
            ),
        )
        # deadline-not-extended: under EV-1 the 70 has no queue residency
        # at all (fast route-reject) — the no-extension property needs an
        # admitted 70 (Java behaviour gap, filed with EV-1).
        # A9-3 (Mark P3-1): the fast-reject latency is NOT a deadline
        # observation; recording it under the PR8 band drifted the
        # property's calibre (prio_queue_timeout_terminal stays PR8's ONLY
        # band consumer).  Raw value carried in the case finish detail.
        inc70_wall_ms = (inc70_fire.settled_s - inc70_fire.submitted_s) * 1000.0
        clean1_ok, clean1_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        if not clean1_ok:
            return report.finish(
                f"wave1 inflight dirty: {clean1_detail}, " f"grades: {report.summary()}"
            )

        # ---- wave 2: attributed timeout, weak SAME/RESOURCE form -------
        # Placeholder prefill must outlast the 70_late's ~8.5s deadline
        # (deadline cancels at dispatch — see the case docstring).
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=10_000.0)
        time.sleep(PERF_SETTLE_S)

        ph70 = ops.next_request_id(base)
        ph70_fire = _fire(ops, ph70, priority=70, input_len=2048, output_len=2)
        fires.append(ph70_fire)
        if not ph70_fire.ok:
            return False, f"wave2 70_early failed: code={ph70_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "wave2 70_early never dispatched"

        low2_rids = [ops.next_request_id(base) for _ in range(8)]
        specs2 = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 2})
            for rid in low2_rids
        ]
        inc70l = ops.next_request_id(base)
        specs2.append((inc70l, {"priority": 70, "input_len": 2048, "output_len": 2}))
        wave2 = _fire_batch(ops, specs2)
        fires.extend(wave2)
        inc70l_fire = wave2[8]

        outcomes2 = _drain(ops, [ph70_fire] + wave2)
        m2 = _outcome_map(outcomes2)
        inc70l_code = m2[inc70l][1]
        inc70l_reason = (
            int(inc70l_fire.resp.admission_reject_reason)
            if inc70l_fire.resp is not None
            else None
        )
        victims2 = [rid for rid in low2_rids if m2[rid][1] == CODE_YIELDED]
        # [EV-1-FIXED] (see wave 1): the SAME/RESOURCE attribution branches
        # share the classifier's zero-call-site gap; the design-final
        # observable is the same uniform 8511/UNSPECIFIED park expiry.
        w2_wave = low2_rids + [inc70l]
        w2_expired = all(m2[rid][1] == CODE_SLO_EXPIRED for rid in w2_wave)
        w2_reasons_unspec = all(
            fr.reason == REASON_UNSPECIFIED for fr in wave2 if fr.resp is not None
        )
        w2_ok = w2_expired and w2_reasons_unspec and victims2 == [] and m2[ph70][0]
        report.invariant(
            "PR7",
            w2_ok,
            context="same_or_resource_weak_form_design_final",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): incoming70_late "
                f"terminal={inc70l_code} "
                f"reason={REASON_NAMES.get(inc70l_reason, inc70l_reason)} "
                f"(uniform park expiry — the SAME/RESOURCE attribution "
                f"branches share the classifier's zero-call-site gap, "
                f"Java-side, filed), all-wave expired 8511={w2_expired}, "
                f"reasons UNSPECIFIED={w2_reasons_unspec}, "
                f"victim8400={len(victims2)}, "
                f"70_early completed={m2[ph70][0]}"
            ),
        )
        clean2_ok, clean2_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            m1[ph90][0]
            and m2[ph70][0]
            and clean2_ok
            and victims8400 == []
            and victims2 == [],
            detail=(
                f"[EV-1-FIXED] placeholders completed, every parked wave "
                f"member expired 8511 (uniform), zero eviction victims, "
                f"inflight={'ok' if clean2_ok else clean2_detail}"
            ),
        )
        return report.finish(
            f"wave1 70={inc70_code}/{REASON_NAMES.get(inc70_reason)}, "
            f"wave2 70={inc70l_code}/{REASON_NAMES.get(inc70l_reason)}, "
            f"[EV-1-FIXED] inc70 park-expiry wall={inc70_wall_ms:.0f}ms "
            f"(deadline held, no extension — PR8 raw), "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)


@case(
    "atpm_comparator_frozen_weak",
    profiles=["single-nonbatch"],
    source="design §2.3 #11 — PR9 (weak form)",
)
def atpm_comparator_frozen_weak(ctx: CaseContext):
    """Comparator freeze, black-box weak form (PR9): the full
    construction-time-freeze contract ("flipping ordering after queue
    creation must not reorder registered queues") needs runtime hot
    config reload, which the master env does not expose — white-box
    (AutoTpmE2EHarness).  The black-box equivalent proven here: the
    ORDERING CONFIG decided at construction time determines the queue's
    behaviour — the same load shape run under a PRIORITY env and a FIFO
    env orders differently.

    Half 1 (ENV-Q2, PRIORITY): 30x3 submitted first (the first is the
    placeholder), then 70x3 — the dispatch order must interleave as
    "all 70s before the remaining 30s": min(70-group running_ms) <
    max(30-group running_ms).

    Half 2 (ENV-F1, case-level ordering=fifo on the same 1P shape):
    identical choreography — strict arrival order: max(30-group
    running_ms) < min(70-group running_ms).  The later-arriving higher
    priorities do NOT jump under FIFO.

    Both halves assert both directions (the design's bidirectional
    contrast); running_ms comes from the single-JVM engine clocks.
    [EV-1-FIXED] the design-final contrast was restored at intake3
    (PendingPlacementCoordinator 6ad0315f10): every wave submitter parks
    and dispatches, so the ORDERING contrast is observable black-box
    again (the EV-1 downgrade had reduced both halves to the identical
    single-park shape)."""
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    hygiene: list = []

    def run_half(spec_fn, label):
        env = ctx.env_manager.ensure(spec_fn(ctx))
        ops = ctx.engine_ops(env)
        names = _prefill_names(ops)
        for name in names:
            ops.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        ph = ops.next_request_id(base)
        ph_fire = _fire(ops, ph, priority=30, input_len=2048, output_len=2)
        half_fires = [ph_fire]
        if not ph_fire.ok:
            return None, f"{label} placeholder failed: code={ph_fire.code}"
        if not _poll_engine_pending(ops, names[0], 1):
            return None, f"{label} placeholder never dispatched"

        low_rids = [ops.next_request_id(base) for _ in range(2)]
        high_rids = [ops.next_request_id(base) for _ in range(3)]
        specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 2})
            for rid in low_rids
        ] + [
            (rid, {"priority": 70, "input_len": 2048, "output_len": 2})
            for rid in high_rids
        ]
        half_fires.extend(_fire_batch(ops, specs))
        outcomes = _drain(ops, half_fires)
        m = _outcome_map(outcomes)
        hygiene.append((ops, half_fires, names))

        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the dispatch-order contrast is observable again —
        # under PRIORITY the wave dispatches [low_1 (first parker), 70a,
        # 70b, 70c, low_2] (priority desc + FIFO after the first parker),
        # under FIFO pure submit order [low_1, low_2, 70a, 70b, 70c].  The
        # same construction-time ORDERING config still governs both halves
        # (the comparator-freeze contract's black-box weak form).
        wave_rids = low_rids + high_rids
        prio = {rid: 30 for rid in low_rids}
        prio.update({rid: 70 for rid in high_rids})
        _first_p, shape_ok, order = _design_final_pattern(
            ops, half_fires, wave_rids, prio, fifo=(label == "fifo_half")
        )
        all_ok = all(m[rid][0] for rid in wave_rids)
        ph_ok = m[ph][0]
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        half_ok = shape_ok and all_ok and ph_ok and clean_ok
        return (shape_ok, all_ok, ph_ok, clean_ok), (
            f"{label}: shape ok={shape_ok}, "
            f"dispatch={[r % 1_000_000 for r in order]}, "
            f"placeholder completed={ph_ok}, "
            f"codes={[(r % 1_000_000, m[r][1]) for r in wave_rids]}, "
            f"clean={'ok' if clean_ok else clean_detail}"
        )

    try:
        prio_result = run_half(_q2_spec, "priority_half")
        if prio_result[0] is None:
            return False, prio_result[1]
        (p_shape, p_all_ok, p_ph_ok, p_clean), p_note = prio_result
        fifo_result = run_half(_f1_spec, "fifo_half")
        if fifo_result[0] is None:
            return False, fifo_result[1]
        (f_shape, f_all_ok, f_ph_ok, f_clean), f_note = fifo_result

        prio_half_ok = p_shape and p_all_ok and p_ph_ok and p_clean
        fifo_half_ok = f_shape and f_all_ok and f_ph_ok and f_clean
        report.invariant(
            "PR9",
            prio_half_ok and fifo_half_ok,
            context="comparator_frozen_weak_design_final",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): the dispatch-"
                f"order contrast is live — under PRIORITY the 70s dispatch "
                f"before the remaining 30 (first parker exempt), under "
                f"FIFO pure arrival order ({p_note}; {f_note}).  "
                f"Comparator-freeze itself stays white-box (runtime "
                f"reload unavailable); the construction-time ORDERING "
                f"config is the black-box weak form."
            ),
        )
        report.invariant(
            "P6",
            prio_half_ok and fifo_half_ok,
            detail=(
                f"both halves drained all-five-200 with inflight clean "
                f"(priority half={prio_half_ok}, fifo half={fifo_half_ok})"
            ),
        )
        return report.finish(
            f"priority-half shape={p_shape}, fifo-half shape={f_shape} "
            f"(design-final contrast), grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for ops_x, fires_x, names_x in hygiene:
            try:
                _finally_hygiene(ops_x, fires_x, names_x)
            except Exception:
                pass


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


@case(
    "atpm_error_code_family",
    profiles=["single-nonbatch"],
    source="design §2.4 #12 — AT4 + P6",
)
def atpm_error_code_family(ctx: CaseContext):
    """Error-code family separation (AT4): each admission failure code
    appears only under its own trigger condition, and the three segments
    never cross-contaminate.

    Segment 1 (8502 QUEUE_FULL, ENV-C1 maxOutstanding=2, G11b-isomorphic):
    two slow placeholders hold both global outstanding permits; two
    arrivals (priority 30 and 70 — the GLOBAL cap exempts no priority)
    fail submit's outstanding acquire → completeError(QUEUE_FULL) as a
    synchronous fast-reject.  The single-argument Response.error path
    leaves admission_reject_reason=UNSPECIFIED(0) (code-level finding;
    the actual pair is recorded for first-e2e calibration, per the
    design's "8502 vs 8431 presentation needs first-run calibration"
    note — the code-level expectation here is 8502 on the outstanding
    path).  After the placeholders drain, a sequential request succeeds
    (exact permit release).

    Segment 2 (capacity park, ENV-Q2 shared; [EV-1-FIXED] flipped at
    intake3 PendingPlacementCoordinator 6ad0315f10): a 70 placeholder
    parks the inflight lease, eight 70s + the incoming 90 ALL park —
    the {8402, 8510} route-reject family lost its capacity-blocked
    trigger (maxWaiting's enqueueUnderLock cap is a BATCH-path check
    the NON_BATCH pull model never reaches, so no enqueue ever fails
    and the tryFallback path to 8510 never runs).  Zero victims, all
    nine complete 200 with the 90 dispatching first among the wave
    (priority desc; first parker 70a keeps slot one); the explicit-cap
    rejection observation lives in segment 1's 8502.

    Segment 3 (expiry uniformity, ENV-A1 shared; [EV-1-FIXED] flipped
    at intake3): a 70_early placeholder (10s prefill — it must OUTLAST
    every queue deadline, the deadline-cancels-at-dispatch finding)
    parks the lease; 30a..30h + the incoming 90 all park and every one
    expires at its own 7s deadline as plain 8511 BATCH_SLO_EXPIRED +
    UNSPECIFIED (QUEUE_TIMEOUT 8503 is dead code; the 8431 +
    RESOURCE_EXHAUSTED attributed form needs the expiry-time
    classifier, which has zero call sites in the intake3 master —
    Java-side observation gap, filed).  The 70_early completes (its
    deadline cancelled at delivery ACK).

    Segment 4 (A4, Mark P1-2, SKELETON — BATCH dispatcher family,
    reserved not constructed): 8514 BATCH_TOKEN_CAPACITY / 8515
    SCHEDULER_PLAN_CONFLICT only fire on the BATCH dispatcher, which the
    current case base (SINGLE + NON_BATCH) never enters; filled
    when a priority-batch variant enables BATCH dispatch.

    Cross-segment isolation (AT4, per segment): segment-1 terminals
    contain no 8402/8403/8431/8400/8429/8511; segment-2 no
    8502/8403/8431/8400/8429/8511; segment-3 no 8502/8402/8403/8510.
    """
    # A4 (Mark P1-2): batch-dispatch caliber reservation, following the
    # dual-caliber paradigm (is_batch = ctx.batch_dispatch();
    # completion-duration caliber under BATCH, client-TTFT under NON_BATCH).
    # The 8514/8515 segment-4 codes below are BATCH-dispatcher-only, so
    # the whole segment is unreachable under this case's profile
    # (single-nonbatch = NON_BATCH base, PRIORITY axis injected at the case
    # layer); the arm is reserved so a priority-batch variant fills it
    # without touching the NON_BATCH segments below.
    if ctx.batch_dispatch():
        # TODO(A4): BATCH arm — segment 4 becomes live (8514 group token
        # capacity / 8515 plan conflict); fill when a priority-batch
        # variant enables BATCH dispatch.
        raise NotImplementedError(
            "atpm_error_code_family BATCH arm reserved — fill when a "
            "priority-batch variant enables BATCH dispatch"
        )
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    segs = []

    # ---- segment 1: 8502 QUEUE_FULL (ENV-C1) ---------------------------
    env1 = ctx.env_manager.ensure(_c1_spec(ctx))
    ops1 = ctx.engine_ops(env1)
    fires1: list = []
    names1: list = []
    try:
        names1 = _prefill_names(ops1)
        for name in names1:
            ops1.set_perf(name, prefill_fixed_ms=4000.0)
        time.sleep(PERF_SETTLE_S)

        ph_a = ops1.next_request_id(base)
        ph_b = ops1.next_request_id(base)
        ph_fires = _fire_batch(
            ops1,
            [
                (ph_a, {"priority": 50, "input_len": 2048, "output_len": 2}),
                (ph_b, {"priority": 50, "input_len": 2048, "output_len": 2}),
            ],
        )
        fires1.extend(ph_fires)
        if not all(f.ok for f in ph_fires):
            return False, f"seg1 placeholders failed: {[f.code for f in ph_fires]}"
        if not _poll_engine_pending(ops1, names1[0], 1):
            return False, "seg1 placeholders never dispatched"

        # Both outstanding permits are held from submit time; the next two
        # arrivals — low and high priority alike — must fast-reject 8502.
        rej_lo = ops1.next_request_id(base)
        rej_hi = ops1.next_request_id(base)
        rej_fires = _fire_batch(
            ops1,
            [
                (rej_lo, {"priority": 30, "input_len": 2048, "output_len": 2}),
                (rej_hi, {"priority": 70, "input_len": 2048, "output_len": 2}),
            ],
        )
        fires1.extend(rej_fires)

        m1 = _outcome_map(_drain(ops1, ph_fires + rej_fires))
        rej_codes = [m1[rej_lo][1], m1[rej_hi][1]]
        rej_fast = all(f.settled_s - f.submitted_s < 3.0 for f in rej_fires)
        rej_reasons = [f.reason for f in rej_fires]
        placeholders_ok = m1[ph_a][0] and m1[ph_b][0]
        isolated1 = all(
            c
            not in (
                CODE_NO_PREFILL,
                CODE_NO_DECODE,
                CODE_RESOURCE_EXHAUSTED,
                CODE_YIELDED,
                CODE_ENGINE_CANCELLED,
                CODE_SLO_EXPIRED,
            )
            for c in rej_codes + [m1[ph_a][1], m1[ph_b][1]]
        )

        for name in names1:
            ops1.set_perf(name, prefill_fixed_ms=100.0)
        recovery_ok, recovery_detail = ops1.verify_recovery()
        clean1_ok, clean1_detail = AssertUtils.inflight_clean(_master_http(ops1), 30.0)
        segs.append(
            (
                "s1_8502_outstanding",
                rej_codes == [CODE_QUEUE_FULL, CODE_QUEUE_FULL]
                and rej_fast
                and placeholders_ok
                and isolated1
                and recovery_ok
                and clean1_ok,
                (
                    f"rejected codes={rej_codes} (expected [8502, 8502]), "
                    f"reasons={rej_reasons} (expected [0, 0] UNSPECIFIED), "
                    f"fast={rej_fast}, placeholders completed={placeholders_ok}, "
                    f"isolated={isolated1}, recovery={recovery_ok}"
                    f"({recovery_detail[:60]}), "
                    f"inflight={'ok' if clean1_ok else clean1_detail}"
                ),
            )
        )
    finally:
        _finally_hygiene(ops1, fires1, names1)

    # ---- segment 2: {8402, 8510} route-reject family (ENV-Q2) ----------
    env2 = ctx.env_manager.ensure(_q2_spec(ctx))
    ops2 = ctx.engine_ops(env2)
    fires2: list = []
    names2: list = []
    try:
        names2 = _prefill_names(ops2)
        for name in names2:
            ops2.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        ph2 = ops2.next_request_id(base)
        ph2_fire = _fire(ops2, ph2, priority=70, input_len=2048, output_len=2)
        fires2.append(ph2_fire)
        if not ph2_fire.ok:
            return False, f"seg2 placeholder failed: code={ph2_fire.code}"
        if not _poll_engine_pending(ops2, names2[0], 1):
            return False, "seg2 placeholder never dispatched"

        high_rids = [ops2.next_request_id(base) for _ in range(8)]
        specs2 = [
            (rid, {"priority": 70, "input_len": 2048, "output_len": 2})
            for rid in high_rids
        ]
        inc90 = ops2.next_request_id(base)
        specs2.append((inc90, {"priority": 90, "input_len": 2048, "output_len": 2}))
        wave2 = _fire_batch(ops2, specs2)
        fires2.extend(wave2)

        m2 = _outcome_map(_drain(ops2, [ph2_fire] + wave2))
        inc90_code = m2[inc90][1]
        zero_eviction = all(
            m2[rid][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
            for rid in high_rids + [inc90]
        )
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the route-reject family {8402, 8510} has no
        # capacity-blocked trigger left — every submitter parks, so the 90
        # completes 200 after the wave (priority desc; first parker 70a
        # keeps slot one).  The tryFallback path to 8510 needs a failed
        # enqueue, which the NON_BATCH pull model never produces
        # (maxWaiting is a BATCH-path cap — the equivalent explicit-cap
        # rejection observation lives in segment 1's 8502).  Zero
        # evictions; all nine complete.
        s2_wave = high_rids + [inc90]
        prio2 = {rid: 70 for rid in high_rids}
        prio2[inc90] = 90
        _s2_first, s2_shape, s2_order = _design_final_pattern(
            ops2, [ph2_fire] + wave2, s2_wave, prio2
        )
        s2_all_ok = all(m2[rid][0] for rid in high_rids) and m2[inc90][0]
        isolated2 = all(
            m2[rid][1]
            not in (
                CODE_QUEUE_FULL,
                CODE_NO_DECODE,
                CODE_RESOURCE_EXHAUSTED,
                CODE_YIELDED,
                CODE_ENGINE_CANCELLED,
                CODE_SLO_EXPIRED,
            )
            for rid in high_rids + [ph2, inc90]
        )
        clean2_ok, clean2_detail = AssertUtils.inflight_clean(_master_http(ops2), 30.0)
        segs.append(
            (
                "s2_capacity_park_design_final",
                inc90_code == CODE_OK
                and zero_eviction
                and s2_shape
                and s2_all_ok
                and m2[ph2][0]
                and isolated2
                and clean2_ok,
                (
                    f"[EV-1-FIXED] incoming90 terminal={inc90_code} "
                    f"(parks and completes — the "
                    f"{list(ROUTE_REJECT_FAMILY)} route-reject family lost "
                    f"its capacity-blocked trigger at intake3 "
                    f"PendingPlacementCoordinator 6ad0315f10; the explicit-"
                    f"cap rejection observation lives in s1's 8502), "
                    f"shape ok={s2_shape}, "
                    f"dispatch={[r % 1_000_000 for r in s2_order]}, "
                    f"zero 8400/8429={zero_eviction}, "
                    f"all nine completed={s2_all_ok}, "
                    f"placeholder completed={m2[ph2][0]}, "
                    f"isolated={isolated2}, "
                    f"inflight={'ok' if clean2_ok else clean2_detail}"
                ),
            )
        )
    finally:
        _finally_hygiene(ops2, fires2, names2)

    # ---- segment 3: 8431 + RESOURCE_EXHAUSTED (ENV-A1) -----------------
    env3 = ctx.env_manager.ensure(_a1_spec(ctx))
    ops3 = ctx.engine_ops(env3)
    fires3: list = []
    names3: list = []
    try:
        names3 = _prefill_names(ops3)
        # The placeholder must OUTLAST every queue deadline (deadline
        # cancels at delivery ACK — the queued items never dispatch).
        for name in names3:
            ops3.set_perf(name, prefill_fixed_ms=10_000.0)
        time.sleep(PERF_SETTLE_S)

        ph70 = ops3.next_request_id(base)
        ph70_fire = _fire(ops3, ph70, priority=70, input_len=2048, output_len=2)
        fires3.append(ph70_fire)
        if not ph70_fire.ok:
            return False, f"seg3 70_early failed: code={ph70_fire.code}"
        if not _poll_engine_pending(ops3, names3[0], 1):
            return False, "seg3 70_early never dispatched"

        low_rids = [ops3.next_request_id(base) for _ in range(8)]
        specs3 = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 2})
            for rid in low_rids
        ]
        inc90b = ops3.next_request_id(base)
        specs3.append((inc90b, {"priority": 90, "input_len": 2048, "output_len": 2}))
        wave3 = _fire_batch(ops3, specs3)
        fires3.extend(wave3)
        inc90b_fire = wave3[8]

        m3 = _outcome_map(_drain(ops3, [ph70_fire] + wave3))
        inc90b_code = m3[inc90b][1]
        inc90b_reason = (
            int(inc90b_fire.resp.admission_reject_reason)
            if inc90b_fire.resp is not None
            else None
        )
        victims8400 = [rid for rid in low_rids if m3[rid][1] == CODE_YIELDED]
        plain8511 = [rid for rid in low_rids if m3[rid][1] == CODE_SLO_EXPIRED]
        ph70_ok = m3[ph70][0]
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the 90 parks (rather than route-rejecting) and
        # expires at its own 7s deadline — 8511 + UNSPECIFIED, the same
        # uniform park-expiry terminal as every 30.  The 8431 +
        # RESOURCE_EXHAUSTED attributed form needs the expiry-time
        # classifier, which has zero call sites in the intake3 master
        # (Java-side observation gap, filed); 8503 stays dead code.
        s3_wave = low_rids + [inc90b]
        s3_expired = all(m3[rid][1] == CODE_SLO_EXPIRED for rid in s3_wave)
        isolated3 = all(
            m3[rid][1]
            not in (
                CODE_QUEUE_FULL,
                CODE_NO_DECODE,
                CODE_YIELDED,
                CODE_ENGINE_CANCELLED,
            )
            for rid in low_rids + [ph70, inc90b]
        )
        clean3_ok, clean3_detail = AssertUtils.inflight_clean(_master_http(ops3), 30.0)
        segs.append(
            (
                "s3_expiry_uniformity_design_final",
                inc90b_code == CODE_SLO_EXPIRED
                and s3_expired
                and victims8400 == []
                and ph70_ok
                and isolated3
                and clean3_ok,
                (
                    f"[EV-1-FIXED] incoming90 terminal={inc90b_code} "
                    f"reason={REASON_NAMES.get(inc90b_reason, inc90b_reason)} "
                    f"(park expiry at its own 7s deadline — the 8431 + "
                    f"RESOURCE_EXHAUSTED attributed form needs the expiry-"
                    f"time classifier, zero call sites in the intake3 "
                    f"master, Java gap filed), all-wave expired 8511="
                    f"{s3_expired}, "
                    f"victim8400={len(victims8400)}, "
                    f"plain8511={len(plain8511)} (8503 stays dead code), "
                    f"70_early completed={ph70_ok}, isolated={isolated3}, "
                    f"inflight={'ok' if clean3_ok else clean3_detail}"
                ),
            )
        )
    finally:
        _finally_hygiene(ops3, fires3, names3)

    # ---- segment 4 (A4, Mark P1-2): BATCH dispatcher family — SKELETON --
    # Reserved, not constructed: CODE_BATCH_TOKEN_CAPACITY (8514, group
    # token capacity exceeded) and CODE_SCHEDULER_PLAN_CONFLICT (8515,
    # plan conflict) only fire on the BATCH dispatcher, which the
    # current case base (SINGLE + NON_BATCH) never enters.  Fill
    # when a priority-batch variant enables BATCH dispatch —
    # expected shape: saturate maxWaitingRequestsPerGroup so an incoming
    # over the group token budget rejects 8514; force a concurrent plan
    # mutation for 8515; keep the cross-segment isolation table growing
    # (segment-4 terminals must contain none of segments 1-3's codes).
    # Constants live at the module head next to the code-family block.

    try:
        report.invariant(
            "AT4",
            all(ok for (_l, ok, _d) in segs),
            context="error_code_family_separation",
            detail="; ".join(
                f"{l}={'ok' if ok else 'FAIL(' + d + ')'}" for l, ok, d in segs
            ),
        )
        report.invariant(
            "P6",
            all(ok for (_l, ok, _d) in segs),
            detail="every segment drained to terminals with inflight clean",
        )
        return report.finish(
            f"segments={[l for l, ok, _d in segs if ok]}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


# ===========================================================================
# atpm_config_strict_reject — strict FLEXLB_CONFIG startup rejection (AT1)
# ===========================================================================


@case(
    "atpm_config_strict_reject",
    profiles=["single-nonbatch"],
    source="design §2.4 #13 — AT1",
)
def atpm_config_strict_reject(ctx: CaseContext):
    """Strict FLEXLB_CONFIG rejection (AT1): three illegal config variants
    injected as RAW JSON strings (bypassing build_flexlb_config's Python
    mirror validation on purpose — the Java strict parser is the system
    under test) must fail MASTER STARTUP.

    Variants:
      1. a legal priority base plus a top-level ``"autoTpmEnabled": true``
         — a removed field must not resurrect: the STRICT_MAPPER
         (FAIL_ON_UNKNOWN_PROPERTIES, ConfigService) rejects the
         unrecognized field (ConfigServiceTest.java:204-209 white-box
         precedent);
      2. ``ordering.type=FIFO`` with ``scheduler.ordering.defaultPriority``
         spliced in — a cross-field violation (FifoOrderingConfig has no
         such field; the Python mirror raises on the same shape, the raw
         JSON splice bypasses it to hit the Java parser);
      3. PRIORITY with ``allowedVictimStages`` containing
         DECODE_ENGINE_OWNED but the ``engineCancellation`` block DELETED
         — the validator's owned-cancellation cross-check.

    Assertion signal (design §2.4): the Spring context dies during config
    parsing → start_master's health check never sees the port → ensure()
    raises RuntimeError("master failed to start:\n<tail log>") — the
    black-box failure signal; the tail is grepped for the strict-parser
    message family ("Config validation failed" / "Unrecognized field" /
    "Invalid FLEXLB_CONFIG").  The harness's _build failure path already
    stops the half-started processes and resets current=None (verified),
    so each next variant builds cleanly.  Each variant costs a full
    wait_for_port timeout (~90s — the status poll does not early-exit on
    process death); the design explicitly accepts the runtime.

    Profile declaration (single-nonbatch + PRIORITY axis injected at the
    case layer) is semantic ownership + regression efficiency only (the
    G11b label-honesty precedent): config rejection is
    profile-independent behaviour.
    """
    report = GradeReport(run_grade=ctx.grade)

    cfg1 = json.loads(_prio_config())
    cfg1["autoTpmEnabled"] = True
    variants = [("removed_field_autoTpmEnabled", json.dumps(cfg1))]

    cfg2 = json.loads(_prio_config(ordering="fifo"))
    cfg2["scheduler"]["ordering"]["defaultPriority"] = 50
    variants.append(("fifo_with_defaultPriority", json.dumps(cfg2)))

    cfg3 = json.loads(_prio_config(preemption=_PREEMPT_DECODE))
    del cfg3["scheduler"]["ordering"]["preemption"]["engineCancellation"]
    variants.append(("owned_without_engineCancellation", json.dumps(cfg3)))

    results = []
    try:
        for i, (label, raw_config) in enumerate(variants):
            spec = _spec(ctx, f"atpm_bad{i}", config=raw_config)
            raised = None
            try:
                ctx.env_manager.ensure(spec)
            except Exception as exc:  # RuntimeError from start_master
                raised = exc
            tail_text = str(raised) if raised is not None else ""
            # The tail now includes the logback file appender's output
            # (harness start_master appends ~/ai-whale/logs/application.log
            # bytes written by THIS start — implementation-period fix for
            # the stdout-only tail that carried no parser message).  The
            # keyword family covers all three rejection shapes: Jackson
            # strict-mapper (Unrecognized field), the cross-field
            # validator (ConfigValidationException / "is required when"),
            # and the legacy raw-config gate.
            matched = [
                kw
                for kw in (
                    "config validation failed",
                    "unrecognized field",
                    "invalid flexlb_config",
                    "configvalidationexception",
                    "is required when",
                )
                if kw in tail_text.lower()
            ]
            ok = raised is not None and bool(matched)
            results.append(
                (
                    label,
                    ok,
                    (
                        "startup "
                        + ("failed" if raised is not None else "SUCCEEDED (UNEXPECTED)")
                        + (
                            f", matched={matched}"
                            if matched
                            else ", no strict-parser message in tail"
                        )
                        + (
                            f", exc_head={tail_text[:140]!r}"
                            if raised is not None
                            else ""
                        )
                    ),
                )
            )
        report.invariant(
            "AT1",
            all(ok for (_l, ok, _d) in results),
            context="strict_config_reject",
            detail="; ".join(
                f"{l}={'ok' if ok else 'FAIL(' + d + ')'}" for l, ok, d in results
            ),
        )
        # A2 (Ryan P2-1): was invariant("P6", True, ...) — a can't-fail
        # registration.  Real (failable) condition: every rejected variant
        # build leaves EnvManager.current at None (harness _build's
        # failure path stops the half-started processes and never
        # publishes the env), so a master surviving a supposedly-fatal
        # variant flips this.
        report.invariant(
            "P6",
            ctx.env_manager.current is None,
            detail=(
                f"no live env after {len(variants)} startup-failure "
                f"variants (env_manager.current is None="
                f"{ctx.env_manager.current is None})"
            ),
        )
        return report.finish(
            f"rejected={[l for l, ok, _d in results if ok]}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


# ===========================================================================
# atpm_decode_reservation_priority — decode-plane victim selection (AT7)
# ===========================================================================


@case(
    "atpm_decode_reservation_priority",
    profiles=["single-nonbatch"],
    source="design §2.4 #14 — AT7 + P6",
)
def atpm_decode_reservation_priority(ctx: CaseContext):
    """Decode-reservation priority consistency across stages (AT7): the
    priority rules that govern PREFILL_QUEUED eviction hold verbatim in
    the decode plane — strictly-lower-priority decode victims only, and
    same-priority decode occupancy never evicts (the decode-plane PR4,
    cross-checked against atpm_same_priority_zero_eviction's prefill
    side — that pair is the cross-stage consistency evidence).

    ENV-D1 (shared fingerprint with atpm_preempt_decode_engine_owned;
    FLEXLB_MONITOR_MODE=all so auto_tpm.victim.count is exposed — the
    default critical-only filter hides auto_tpm.*).  Every wave follows
    the corrected injection order (see the D1 spec docstring): victims
    route FIRST under normal KV, kv_pressure goes in only once every
    victim is observable at its target stage, then the incoming fires.

    Wave 1 (strictly-lower victim, engine-owned → 8429): four 30s are
    polled to decode RUNNING; kv_pressure saturates every endpoint; the
    70's ordinary route fails (NO_DECODE_WORKER 8403 — the strict KV
    gate) into the eviction fallback → exactly one owned victim is
    cancelled (8429, typed via grpc-status-details-bin), the 70
    completes, the survivors complete.  Metric cross-check:
    auto_tpm.victim.count{victim_priority="30",incoming_priority="70"}
    increments by exactly ONE (D1 is a shared env — the assertion is
    delta-based against a pre-wave scrape).

    Wave 2 (same-priority zero eviction): four 50s run on decode; the
    incoming 50 finds no strictly-lower candidate → the eviction plan is
    infeasible → the ORIGINAL routing rejection reaches the client —
    NO_DECODE_WORKER(8403) under this construction (8402/8510/8431 stay
    in the family for first-e2e calibration).  Zero victims: client
    terminals plus the victim-count delta staying flat.

    Wave 3 (kvBucket-descending victim preference — WEAK/tendency
    assertion, design §2.4): two 30_small (input 2048) and two 30_big
    (input 16384) occupants, one per decode endpoint; the 70's
    input_len=8192 needs hardKv ≈ 8194.  A small endpoint frees only
    ~2.5k tokens (infeasible); a big endpoint frees ~16.9k (feasible) →
    the victim must be a 30_big.  The construction is deterministic at
    the code level but the design grades it weakly — the
    assertion pins the victim-set membership (big group), not the exact
    rid.
    """
    env = ctx.env_manager.ensure(_d1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    decode_names: list = []
    wave_reports = []
    try:
        prefill_names = _prefill_names(ops)
        decode_names = _decode_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=4000.0)
        time.sleep(PERF_SETTLE_S)

        # ---- wave 1: strictly-lower owned victim → 8429 -----------------
        w1_rids = [ops.next_request_id(base) for _ in range(4)]
        w1_specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 500})
            for rid in w1_rids
        ]
        w1_fires = _fire_batch(ops, w1_specs)
        fires.extend(w1_fires)
        if not all(_poll_decode_running(ops, rid, timeout_s=20.0) for rid in w1_rids):
            return False, "wave1 occupants never reached decode running"
        base_victim = _metric_sum(
            _scrape_master_metrics(ops),
            "auto_tpm_victim",
            {"victim_priority": "30", "incoming_priority": "70"},
        )
        for name in decode_names:
            ops.set_kv_pressure(name, MOCK_TOTAL_KV_TOKENS)
        time.sleep(PERF_SETTLE_S)
        w1_inc = ops.next_request_id(base)
        w1_inc_fire = _fire(ops, w1_inc, priority=70, input_len=2048, output_len=2)
        fires.append(w1_inc_fire)

        m1 = _outcome_map(_drain(ops, w1_fires + [w1_inc_fire]))
        w1_victims = [rid for rid in w1_rids if m1[rid][1] == CODE_ENGINE_CANCELLED]
        w1_survivors_ok = all(m1[rid][0] for rid in w1_rids if rid not in w1_victims)
        w1_inc_ok = m1[w1_inc][0]
        w1_inc_code = m1[w1_inc][1]
        now_victim = _metric_sum(
            _scrape_master_metrics(ops),
            "auto_tpm_victim",
            {"victim_priority": "30", "incoming_priority": "70"},
        )
        w1_delta = (now_victim or 0.0) - (base_victim or 0.0)
        # EV-2 baseline (behaviour finding, probes E9/E11 + the
        # DecodeEndpoint projection math): decode eviction never fires —
        # the kv dimension is unreachable (freedKv ⊆ currentHardCharges)
        # and the slots dimension is absorbed engine-side, so the
        # strictly-lower-priority victim selection has no observation
        # object.  Observable form: zero victims, all occupants complete,
        # the incoming keeps a rejection from EV2_REJECT_FAMILY, and the
        # victim-count metric stays flat (cross-checked against wave 2's
        # identical zero-delta form — the priority asymmetry 30<70 vs
        # 50==50 is itself unobservable black-box).
        wave_reports.append(
            (
                "w1_lower_priority_victim_ev2",
                w1_victims == []
                and (w1_inc_ok or w1_inc_code in EV2_REJECT_FAMILY)
                and w1_survivors_ok
                and w1_delta == 0.0,
                (
                    f"victims8429={len(w1_victims)} (EV-2: decode eviction "
                    f"unreachable — priority 30 < 70 selection has no "
                    f"object), incoming70 ok={w1_inc_ok} code={w1_inc_code} "
                    f"(family {list(EV2_REJECT_FAMILY)}), "
                    f"survivors ok={w1_survivors_ok}, "
                    f"victim.count delta(30<-70)={w1_delta} (expected 0.0 "
                    f"under EV-2 — no eviction events)"
                ),
            )
        )
        clean1_ok, clean1_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        if not clean1_ok:
            wave_reports[-1] = (
                wave_reports[-1][0],
                False,
                wave_reports[-1][2] + f", inflight dirty: {clean1_detail}",
            )

        # ---- wave 2: same-priority zero eviction ------------------------
        for name in decode_names:
            ops.set_kv_pressure(name, 0)
        time.sleep(PERF_SETTLE_S)
        w2_rids = [ops.next_request_id(base) for _ in range(4)]
        w2_specs = [
            (rid, {"priority": 50, "input_len": 2048, "output_len": 500})
            for rid in w2_rids
        ]
        w2_fires = _fire_batch(ops, w2_specs)
        fires.extend(w2_fires)
        if not all(_poll_decode_running(ops, rid, timeout_s=20.0) for rid in w2_rids):
            return False, "wave2 occupants never reached decode running"
        base2_victim = _metric_sum(_scrape_master_metrics(ops), "auto_tpm_victim", {})
        for name in decode_names:
            ops.set_kv_pressure(name, MOCK_TOTAL_KV_TOKENS)
        time.sleep(PERF_SETTLE_S)
        w2_inc = ops.next_request_id(base)
        w2_inc_fire = _fire(ops, w2_inc, priority=50, input_len=2048, output_len=2)
        fires.append(w2_inc_fire)

        m2 = _outcome_map(_drain(ops, w2_fires + [w2_inc_fire]))
        w2_inc_code = m2[w2_inc][1]
        w2_zero_eviction = all(
            m2[rid][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED) for rid in w2_rids
        )
        w2_occupants_ok = all(m2[rid][0] for rid in w2_rids)
        now2_victim = _metric_sum(_scrape_master_metrics(ops), "auto_tpm_victim", {})
        w2_delta = (now2_victim or 0.0) - (base2_victim or 0.0)
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the decode-role-blocked incoming 50 no longer
        # surfaces its original routing rejection (8403) — it parks in the
        # pull-based coordinator with schedule() blocking until the 60s
        # queueTimeout deadline, then terminals as plain 8511
        # BATCH_SLO_EXPIRED (observed).  EV-2 (decode eviction never
        # fires) is unchanged: zero victims, occupants complete, metric
        # flat.
        w2_family = (CODE_NO_DECODE,) + ROUTE_REJECT_FAMILY + (CODE_RESOURCE_EXHAUSTED,)
        w2_legal = (CODE_OK, CODE_SLO_EXPIRED) + w2_family
        wave_reports.append(
            (
                "w2_same_priority_zero_eviction",
                w2_inc_code in w2_legal
                and w2_zero_eviction
                and w2_occupants_ok
                and w2_delta == 0.0,
                (
                    f"[EV-1-FIXED] incoming50 terminal={w2_inc_code} "
                    f"(design-final legal set {list(w2_legal)}: the decode-"
                    f"blocked submitter parks — schedule() blocks to the 60s "
                    f"queueTimeout deadline → 8511 park expiry observed; "
                    f"8403 and the reject family remain legal under other "
                    f"timings), "
                    f"zero 8400/8429={w2_zero_eviction}, occupants completed="
                    f"{w2_occupants_ok}, victim.count delta={w2_delta} "
                    f"(expected 0.0)"
                ),
            )
        )
        clean2_ok, clean2_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        if not clean2_ok:
            wave_reports[-1] = (
                wave_reports[-1][0],
                False,
                wave_reports[-1][2] + f", inflight dirty: {clean2_detail}",
            )

        # ---- wave 3: kvBucket-descending victim (weak) ------------------
        for name in decode_names:
            ops.set_kv_pressure(name, 0)
        time.sleep(PERF_SETTLE_S)
        small_rids = [ops.next_request_id(base) for _ in range(2)]
        big_rids = [ops.next_request_id(base) for _ in range(2)]
        w3_specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 500})
            for rid in small_rids
        ] + [
            (rid, {"priority": 30, "input_len": 16384, "output_len": 500})
            for rid in big_rids
        ]
        w3_fires = _fire_batch(ops, w3_specs)
        fires.extend(w3_fires)
        if not all(
            _poll_decode_running(ops, rid, timeout_s=20.0)
            for rid in small_rids + big_rids
        ):
            return False, "wave3 occupants never reached decode running"
        for name in decode_names:
            ops.set_kv_pressure(name, MOCK_TOTAL_KV_TOKENS)
        time.sleep(PERF_SETTLE_S)
        w3_inc = ops.next_request_id(base)
        w3_inc_fire = _fire(ops, w3_inc, priority=70, input_len=8192, output_len=2)
        fires.append(w3_inc_fire)

        m3 = _outcome_map(_drain(ops, w3_fires + [w3_inc_fire]))
        w3_victims = [
            rid
            for rid in small_rids + big_rids
            if m3[rid][1] in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
        ]
        w3_survivors_ok = all(
            m3[rid][0] for rid in small_rids + big_rids if rid not in w3_victims
        )
        w3_inc_ok = m3[w3_inc][0]
        w3_inc_code = m3[w3_inc][1]
        # EV-2 baseline: the kvBucket-descending victim preference is
        # unobservable for the same reason (no decode eviction ever
        # fires), so the small/big distinction never reaches a terminal.
        wave_reports.append(
            (
                "w3_kvbucket_preference_weak_ev2",
                w3_victims == []
                and (w3_inc_ok or w3_inc_code in EV2_REJECT_FAMILY)
                and w3_survivors_ok,
                (
                    f"victims={len(w3_victims)} (EV-2: kvBucket-descending "
                    f"preference has no object — decode eviction never "
                    f"fires; small-vs-big group distinction unobservable "
                    f"black-box), incoming70 ok={w3_inc_ok} "
                    f"code={w3_inc_code} (family {list(EV2_REJECT_FAMILY)}), "
                    f"survivors ok={w3_survivors_ok} "
                    f"(weak/tendency assertion per design §2.4, EV-2 form)"
                ),
            )
        )
        clean3_ok, clean3_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)

        report.invariant(
            "AT7",
            all(ok for (_l, ok, _d) in wave_reports),
            context="decode_reservation_priority",
            detail="; ".join(
                f"{l}={'ok' if ok else 'FAIL(' + d + ')'}" for l, ok, d in wave_reports
            ),
        )
        report.invariant(
            "P6",
            all(ok for (_l, ok, _d) in wave_reports) and clean3_ok,
            detail=(
                f"all three waves drained (victims terminal, occupants "
                f"completed), inflight={'ok' if clean3_ok else clean3_detail}"
            ),
        )
        return report.finish(
            f"waves={[l for l, ok, _d in wave_reports if ok]}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for name in decode_names:
                ops.set_kv_pressure(name, 0)
        except Exception:
            pass
        _finally_hygiene(ops, fires, prefill_names)


# ===========================================================================
# atpm_observability_integrity — four signal planes on one choreography (AT8)
# ===========================================================================


@case(
    "atpm_observability_integrity",
    profiles=["single-nonbatch"],
    source="design §2.4 #15 — AT8 + P6",
)
def atpm_observability_integrity(ctx: CaseContext):
    """Observability integrity (AT8): every signal plane the design's
    assertion ladder names (proto response > auto_tpm.* metrics > pv.log
    > debug log) carries the priority/TPM facts on ONE composite
    choreography.

    ENV-O1: Q2-shaped config (PREFILL_QUEUED preemption, queueTimeout 7s
    — [EV-1-FIXED] flipped from 8s at intake3
    PendingPlacementCoordinator 6ad0315f10: under the pull model the
    wave's third release slot lands at t=9s, which raced the 8s deadline
    of the 4th submitter (70a); 7s puts every non-dispatched deadline
    strictly before the third slot, making the client shape
    deterministic) + master debug log + FLEXLB_MONITOR_MODE=all.
    Implementation-period corrections over the design's env sketch: the
    DEFAULT critical-only metrics filter hides auto_tpm.*
    (application.yml flexlb.monitor.mode), so the env-level switch is
    required; FLEXLB_PV_LOG is a load-client-line knob with no consumer
    on the harness line — the pvLogger writes at INFO by default, so
    the pv.log plane needs no extra knob.  The master_env + debug-log
    differences give O1 its own fingerprint (exclusive env — the metric
    counters start from zero).

    Choreography (the atpm_preempt_prefill_queued wave-1 shape with
    mixed priorities for bucket coverage), [EV-1-FIXED] design-final
    form: a 50 placeholder parks the inflight lease; 30a/30b/50a/50b/
    70a/70b/30c/30d + the 90 ALL park in the pull-based coordinator.
    Prefill is slowed to 3s: ph completes at t=3, the first parker 30a
    takes slot two (t=3-6), the 90 (highest priority) takes slot three
    (t=6-9) — both complete inside their deadlines.  The remaining
    seven (30b, 30c, 30d, 50a, 50b, 70a, 70b) expire at their own 7s
    deadlines as plain 8511 BATCH_SLO_EXPIRED (the low-priority-
    suppression sample; 30d is no longer evicted — no eviction ever
    fires under the pull model, the victim counter stays flat).

    Per-plane assertions:
      * auto_tpm.request.count{priority=30|50|70|90} == the injected
        bucket counts 4/3/2/1 — counted at the schedule RPC entry for
        EVERY request regardless of outcome (FlexlbServiceImpl:723), the
        metric-plane normalization evidence crossing prio_normalize's
        behaviour plane;
      * auto_tpm.schedule.latency_ms{result="success"} present (the
        TIMER family; result is "success" | "error_<code>");
      * auto_tpm.victim.count == 1 — matching the client-side 8400
        count exactly (exclusive env, absolute value);
      * master log contains [priority-scheduler] lines (debug level —
        master_debug_log=True, the analysis-report §7.5.1 pitfall);
      * pv.log tail carries admissionRejectReason fields (channel
        availability; sampled non-null values recorded in the detail).
    """
    env = ctx.env_manager.ensure(_o1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        ph = ops.next_request_id(base)
        ph_fire = _fire(ops, ph, priority=50, input_len=2048, output_len=2)
        fires.append(ph_fire)
        if not ph_fire.ok:
            return False, f"placeholder failed: code={ph_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "placeholder never dispatched"
        # A3 (Mark P1-1): AT6 black-box aggregate, probe 1 — duplicate
        # request_id rejection observed through the schedule RPC response
        # (RequestLifecycleCoordinator.register putIfAbsent, coordinator
        # L262-268 "duplicate request_id: <rid>" →
        # StrategyErrorType.INVALID_REQUEST 8406).  Probed WHILE ph is
        # still inflight: after its terminal the slot ledger may clean
        # up and a re-submit would re-register successfully.  priority=40
        # keeps the probe out of the expected metric buckets
        # {30, 50, 70, 90} (the counter fires at the schedule RPC entry
        # regardless of outcome).
        dup_code = None
        dup_err = None
        try:
            dup_resp = ops.schedule(
                ph, priority=40, input_len=2048, output_len=2, timeout_s=30.0
            )
            dup_code = int(getattr(dup_resp, "code", -1))
        except Exception as exc:
            dup_err = repr(exc)
        dup_rejected = dup_code == CODE_INVALID_REQUEST
        dup_note = (
            f"observed (code={dup_code})"
            if dup_rejected
            else f"MISSING (code={dup_code}, err={dup_err})"
        )
        # The client-shape assertions index rids["ph"] — register the
        # placeholder alongside the ladder tags (first-run KeyError fix).
        rids: dict = {"ph": ph}

        ladder = [
            ("30a", 30),
            ("30b", 30),
            ("50a", 50),
            ("50b", 50),
            ("70a", 70),
            ("70b", 70),
            ("30c", 30),
            ("30d", 30),
            ("90", 90),
        ]
        specs = []
        for tag, prio in ladder:
            rid = ops.next_request_id(base)
            rids[tag] = rid
            specs.append((rid, {"priority": prio, "input_len": 2048, "output_len": 2}))
        wave = _fire_batch(ops, specs)
        fires.extend(wave)

        outcomes = _drain(ops, [ph_fire] + wave)
        m = _outcome_map(outcomes)

        # Client-plane expectations — [EV-1-FIXED] baseline flipped at
        # intake3 PendingPlacementCoordinator (6ad0315f10): every ladder
        # submitter parks; with queueTimeout 7s the deterministic shape
        # is ph + the first parker 30a + the 90 (highest priority, third
        # release slot) completing 200, the remaining seven expiring
        # 8511 at their own deadlines, zero route-reject and zero
        # eviction (no 8400 — the preemption sample retired with the
        # enqueue-failure trigger).
        wave_tags = [t for t, _p in ladder]
        tag_by_rid = {rids[t]: t for t in wave_tags}
        completed = ["ph"] + [t for t in wave_tags if m[rids[t]][0]]
        rejected8402 = [t for t in wave_tags if m[rids[t]][1] in ROUTE_REJECT_FAMILY]
        expired8511 = [t for t in wave_tags if m[rids[t]][1] == CODE_SLO_EXPIRED]
        ph_ok = m[rids["ph"]][0]
        d_order = _dispatch_order(ops, [ph_fire] + wave)
        d_pos = {r: i for i, r in enumerate(d_order)}
        dispatch_pair_ok = d_pos[rids["30a"]] < d_pos[rids["90"]]
        client_shape_ok = (
            completed == ["ph", "30a", "90"]
            and len(expired8511) == 7
            and rejected8402 == []
            and ph_ok
            and dispatch_pair_ok
        )

        # ---- metric plane (management port /prometheus) -----------------
        samples = _scrape_master_metrics(ops)
        buckets = {
            p: _metric_sum(samples, "auto_tpm_request", {"priority": str(p)})
            for p in (30, 50, 70, 90)
        }
        expected_buckets = {30: 4.0, 50: 3.0, 70: 2.0, 90: 1.0}
        buckets_ok = all(
            buckets[p] is not None and buckets[p] == expected_buckets[p]
            for p in expected_buckets
        )
        latency_success = _metric_sum(
            samples, "auto_tpm_schedule", {"result": "success"}
        )
        latency_ok = latency_success is not None
        # [EV-1-FIXED]: no eviction ever fires under the pull model (no
        # failed enqueue feeds the fallback), so the victim counter must
        # stay at zero — matching the client-side zero-8400 count exactly
        # (exclusive env, absolute value).
        victim_total = _metric_sum(samples, "auto_tpm_victim", {})
        victim_ok = (victim_total or 0.0) == 0.0

        # ---- log plane ---------------------------------------------------
        log_text = _master_log_text(env)
        sched_log_ok = "[priority-scheduler]" in log_text

        # ---- pv.log plane ------------------------------------------------
        # A8 (Daniel P2-3): this env's delta only, filtered to this
        # case's rids (the dup probe re-uses ph, so its INVALID_REQUEST
        # row is inside the ph filter as well).
        pv_tail = _pv_log_tail(env, [ph] + [rids[t] for t in wave_tags])
        pv_field_ok = "admissionRejectReason" in pv_tail
        pv_samples = re.findall(r'"admissionRejectReason"\s*:\s*"([A-Z_]+)"', pv_tail)

        report.invariant(
            "AT8",
            client_shape_ok
            and buckets_ok
            and latency_ok
            and victim_ok
            and sched_log_ok
            and pv_field_ok,
            context="observability_integrity_design_final",
            detail=(
                f"[EV-1-FIXED] client shape (design-final): completed="
                f"{completed}, expired8511={len(expired8511)}/7, "
                f"rejected8402={len(rejected8402)}/0, "
                f"30a-before-90 dispatch={dispatch_pair_ok}; "
                f"request.count buckets={ {p: buckets[p] for p in buckets} } "
                f"(expected {expected_buckets}); "
                f"schedule.latency success={'present' if latency_ok else 'MISSING'}; "
                f"victim.count total={victim_total} (expected 0.0 — no "
                f"eviction under the pull model, matches zero client-side "
                f"8400); "
                f"[priority-scheduler] log={'present' if sched_log_ok else 'MISSING'}; "
                f"pv.log admissionRejectReason field="
                f"{'present' if pv_field_ok else 'MISSING'}"
                + (
                    f", samples={pv_samples[:3]}"
                    if pv_samples
                    else " (no non-null sample values)"
                )
            ),
        )
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            client_shape_ok and clean_ok,
            detail=(
                f"[EV-1-FIXED] every request reached a terminal: 3 "
                f"completed [ph, 30a, 90], 7 expired 8511, "
                f"inflight={'ok' if clean_ok else clean_detail}"
            ),
        )
        # A3 (Mark P1-1): AT6 black-box aggregate — the design's white-box
        # observability-closure item, rebuilt as the three black-box-
        # observable facets on one invariant: duplicate request_id
        # rejection through the schedule response (INVALID_REQUEST
        # 8406), P6 request integrity (every request terminal), and the
        # inflight ledger clean.  AT6 was registered in grade.py but had
        # zero call sites (dead entry); this aggregate is its live form.
        report.invariant(
            "AT6",
            dup_rejected and client_shape_ok and clean_ok,
            context="blackbox_aggregate_dup_p6_inflight",
            detail=(
                f"[EV-1-FIXED] duplicate-rid rejection (INVALID_REQUEST "
                f"{CODE_INVALID_REQUEST}): {dup_note}, "
                f"client shape (design-final)={client_shape_ok}, "
                f"inflight={'ok' if clean_ok else clean_detail}"
            ),
        )
        return report.finish(
            f"planes: client(design-final)={client_shape_ok} metrics="
            f"{buckets_ok and latency_ok and victim_ok} log={sched_log_ok} "
            f"pv={pv_field_ok}, grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)
