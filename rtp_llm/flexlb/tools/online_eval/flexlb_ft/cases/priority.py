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

Category theme: priority-order fidelity, terminal completeness, and the
observable Auto-TPM boundary contracts.  Cases distinguish global waiting
from endpoint-local preemption: PR4/PR5/PR6/PR10 and AT5 are recorded only
when a real victim terminal and replacement evidence exist.  Unsupported
black-box victim selection and per-decision log coverage remain explicit
gaps backed separately by Java contract tests.

Baseline annotations: [EV-1-FIXED] assertions were calibrated on the
feat/flexlb_priority_auto_tpm_ft line at the intake3
PendingPlacementCoordinator (commit 6ad0315f10) — the pull-based park
model with priority desc + FIFO release; they are PENDING remote-probe
verification against the intake3-rebuild Java line (behaviour may
differ — assertion calibres are intentionally unchanged in this
migration, per the migration brief).  [EV-2] marks the historical
decode-pressure black-box baseline, not the Java planner contract.  Grep
those keys to enumerate every flip point (the EV-1/EV-2 runbook discipline).

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
from ..engine_ops import (
    _fence_residue_stable,
    engine_inflight_clean,
    inject_type,
    parse_prometheus_samples,
)
from ..grade import GradeReport
from ..harness import AssertUtils, EnvSpec, build_flexlb_config, default_perf, wait_for
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

# Historical terminal family from the E9/E11 decode-pressure probes.  New
# assertions no longer accept this family interchangeably: each wave pins one
# exact observed outcome, while real victim contracts remain evidence-gated.
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
# downgrades; grep [EV-1-FIXED] for every flip point.  EV-2 names only the
# historical black-box pressure construction, not the Java planner contract.
EXPECTED_BASELINES = {
    "EV-1": (
        "FIXED (flipped at intake3 PendingPlacementCoordinator, commit "
        "6ad0315f10): capacity blocking parks every submitter (pull-based, "
        "priority desc + FIFO); all-200 design-final dispatch shape — "
        "zero route-reject (was: single park slot, later submitters "
        "route-reject {8402, 8510})"
    ),
    "EV-2": (
        "historical decode-pressure probes produced zero 8400/8429 victims; "
        "current cases pin exact outcomes and do not grade zero victims"
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
    if not decode_names:
        return False, "no decode engines"
    ok = True
    try:
        snap = ops.snapshot_by_name()
    except Exception as exc:
        return False, f"snapshot failed: {exc!r}"
    for name in decode_names:
        entry = snap.get(name)
        if not isinstance(entry, dict):
            evidence.append(f"{name}:missing")
            ok = False
            continue
        raw_available = entry.get("available_kv_tokens")
        try:
            avail = int(raw_available)
        except (TypeError, ValueError):
            evidence.append(f"{name}:invalid({raw_available!r})")
            ok = False
            continue
        evidence.append(f"{name}:{avail}")
        if avail != 0:
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
    running_ms asc + settle-rank arbitration).  ``exclude`` removes work
    that was already running before the measured wait wave.  It never
    exempts a member of the globally ordered wave.  0.0 is ideal."""
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
# slot baseline was superseded by the pull-based coordinator.  The current
# coordinator publishes every capacity-blocked request into one ordered
# WaitBucket before a release is selected, so the complete waiting wave is
# ordered by priority desc + sequence asc.  Earlier calibration treated the
# first submitter as an already-selected parker; current end-to-end evidence
# shows that exemption no longer exists.
def _design_final_pattern(
    ops, fires: list, ordered_rids: list, priorities: dict, fifo: bool = False
) -> tuple:
    """Classify one wave under the current pull-coordinator ordering.

    Returns ``(first_submitted_rid, shape_ok, wave_dispatch_order)``.  FIFO
    preserves submission order; PRIORITY orders the whole waiting wave by
    priority descending with submission order as its stable tiebreaker.
    The first return value is retained for call-site compatibility only.
    """
    order = _dispatch_order(ops, fires)
    wave_set = set(ordered_rids)
    wave_order = [r for r in order if r in wave_set]
    first_submitted = ordered_rids[0] if ordered_rids else None
    if len(wave_order) != len(ordered_rids):
        return first_submitted, False, wave_order
    if fifo:
        expected = list(ordered_rids)
    else:
        submit_rank = {rid: i for i, rid in enumerate(ordered_rids)}
        expected = sorted(
            ordered_rids,
            key=lambda r: (-priorities.get(r, 0), submit_rank[r]),
        )
    return first_submitted, wave_order == expected, wave_order


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
# The staged BATCH constructions must prove that the lower-priority request
# reached the endpoint-local WorkerBatcher before the incoming is submitted.
# Expose the queue-depth gauge as well as the preemption counters; this changes
# observability only, not scheduler behavior.
_MONITOR_PREEMPT_LIVE_ENV = {
    "FLEXLB_MONITOR_METRIC_WHITELIST": "flexlb_auto_tpm,flexlb_app_routing_queue_length"
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
    dispatcher: str = "non_batch",
    max_inflight_batches: int = 4,
) -> str:
    """Unified priority-family config (PRIORITY + SINGLE + NON_BATCH base;
    dispatcher="batch" variant for the live-eviction family, 2026-09).

    Implementation-period additions over the design's config sketch
    (all verified against the Java code):

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
    * ``dispatcher="batch"`` exposes the endpoint-local WorkerBatcher.  A
      functional preemption probe must still prove that a lower-priority
      request entered that queue before submitting the higher-priority
      request; concurrent submission alone can leave both in the global
      queue and only test PR1 ordering.  Under BATCH the per-worker inflight
      cap is meaningless and ``max_inflight_batches`` is the caliber.
    """
    return build_flexlb_config(
        ordering=ordering,
        decision="single",
        dispatcher=dispatcher,
        default_priority=default_priority,
        preemption=preemption,
        queue_timeout_ms=queue_timeout_ms,
        max_outstanding=max_outstanding if max_outstanding is not None else 5_000,
        max_inflight_batches=max_inflight_batches,
        max_inflight_requests_per_worker=(
            None if dispatcher == "batch" else max_inflight
        ),
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
    decode_cache_blocks: Optional[int] = None,
) -> EnvSpec:
    env = {"FLEXLB_CONFIG": config}
    if extra_env:
        env.update(extra_env)
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
        master_env=env,
        master_debug_log=master_debug_log,
        **spec_kwargs,
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
        config=_prio_config(
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
    on, and the auto_tpm family metric whitelist.

    [EV-1-FIXED] queueTimeout 7s (was 8s, flipped at intake3
    PendingPlacementCoordinator 6ad0315f10): under the pull model the
    wave's third release slot lands at t=9s, which sat INSIDE the 8s
    deadline of the 4th submitter (70a, deadline ~8.7-9.0s) — a race
    between the coordinator pull and the expiry check.  7s puts every
    non-dispatched deadline (7.1-8.35s) strictly before the t=9 slot,
    making the client shape deterministic: ph + 90 + 70a dispatch and
    complete, while the remaining seven expire 8511.

    Implementation-period corrections over the design's env sketch: the
    default critical-only whitelist does not expose auto_tpm.*, so the
    FLEXLB_MONITOR_METRIC_WHITELIST family entry is required (the legacy
    FLEXLB_MONITOR_MODE switch is dead on this line — see _q3_spec);
    pv.log writes at INFO level by default on the harness line
    (FLEXLB_PV_LOG is a load-client-line knob with no consumer here)."""
    return _spec(
        ctx,
        "atpm_o1",
        config=_prio_config(preemption=_PREEMPT_PQ, queue_timeout_ms=7_000),
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
        config=_prio_config(),
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


def _wait_batch_queue_priority(
    ops, priority: int, min_depth: int = 1, timeout_s: float = 6.0
) -> tuple:
    """Wait for endpoint-local WorkerBatcher depth at one priority.

    This is the synchronization barrier that distinguishes a real endpoint
    replacement attempt from two requests racing inside the global queue.
    Returns ``(seen, last_depth, raw_metric_lines)``.
    """
    deadline = time.monotonic() + timeout_s
    last_depth = None
    last_samples = []
    while time.monotonic() < deadline:
        last_samples = _scrape_master_metrics(ops)
        last_depth = _metric_sum(
            last_samples,
            "routing_queue_length",
            {
                "type": "batchQueue",
                "role": "PREFILL",
                "priority": str(priority),
            },
        )
        if last_depth is not None and last_depth >= min_depth:
            return (
                True,
                last_depth,
                _metric_lines(last_samples, "routing_queue_length"),
            )
        time.sleep(0.2)
    return (
        False,
        last_depth,
        _metric_lines(last_samples, "routing_queue_length"),
    )


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
    flexlb_log = Path(
        getattr(
            env,
            "flexlb_log_path",
            Path.home() / "ai-whale" / "logs" / "flexlb.log",
        )
    )
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
    path = Path(
        getattr(env, "pv_log_path", Path.home() / "ai-whale" / "logs" / "pv.log")
    )
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
            wanted.add(f'"requestId":"{rid}"')
            wanted.add(f'"requestId": "{rid}"')
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
    settles code=200; the current dispatch order is [ph, 70a, 70b, 50a,
    50b, 30a, 30b].  The entire waiting wave follows priority desc plus
    same-level FIFO; only ph was already running before it formed.

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
        # The pull coordinator orders the complete waiting wave by priority
        # descending and submission sequence ascending.  The placeholder was
        # already running before this wave, so it alone is outside PR1.
        m = _outcome_map(outcomes)
        wave_rids = [rids[t] for t in tags]
        _first_submitted, shape_ok, wave_order = _design_final_pattern(
            ops, fires, wave_rids, priorities
        )
        wave_order_tags = [tag_of.get(r, str(r)) for r in wave_order]
        all_ok = m[ph][0] and all(m[rids[t]][0] for t in tags)

        report.check(
            "PR1",
            _inversion_ratio(order, priorities, exclude={ph}),
            context="basic_order",
            detail=(
                f"[EV-1-FIXED] dispatch={order_tags} (the complete waiting "
                f"wave is priority-desc; only the pre-wave running "
                f"placeholder is excluded from PR1 scoring)"
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
        unfinished = [o for o in outcomes if not o[1]]
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            shape_ok and all_ok and not unfinished and clean_ok,
            detail=(
                f"[EV-1-FIXED] issued=terminal no-loss: "
                f"{len(outcomes) - len(unfinished)}/7 completed, "
                f"waiting wave={wave_order_tags}, "
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
        # the empty queue; the remaining six wait and retain submission order.
        # The "dispatch == submit" equality on all seven FIFO peers is now a
        # real observation object (the former route-reject baseline is gone).
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
    parks whole behind the placeholder.  [2026-09 recalibration] the
    post-codex admission release is pure priority-desc: observed
    dispatch [ph, A, B, G, C].  The assertion is now STRUCTURAL, not
    exact-sequence (the
    exact form lagged twice already — intake3, then codex admission):
    the 70-group A/B/G must precede C (either channel failing demotes
    that member to <=50 and behind C), the group must keep submit
    order A→B→G (FIFO survives park→release), and all settle code=200.
    The source's FIFO-profile half
    (F1 control env) is not profile-reachable on this line; the FIFO
    control lives separately in atpm_comparator_frozen_weak's fifo_half.
    Implementation-period note: design §2.2 sketched this segment without
    maxInflightRequestsPerPrefillWorker, but without it there is no
    backlog window and no observable queue-jumping — the window env
    keeps the choreography and makes it observable.

    Segment 3 (ENV-N1 — the defaultPriority=30 variant, built as its own
    case-layer env): Y(50) leads by priority, then D(no input),
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
        # [2026-09 recalibration, post-codex admission] the wave parks
        # whole behind the placeholder and releases pure priority-desc:
        # observed [ph, A, B, G, C].  Structural form (sep-anchored like
        # _two_cluster_split,
        # robust to release-order drift that does not cross a priority
        # group): (i) the placeholder and every wave member dispatched;
        # (ii) the placeholder is first (it held the slot before the
        # wave parked); (iii) the 70-group A/B/G strictly precedes C —
        # proto must win for A (a header-loser A lands 30, behind C's
        # 50) and the header must take effect for B (a default-loser B
        # ties C at 50 and FIFO puts submitted-later B behind
        # first-submitter C); (iv) the 70-group keeps submit order
        # A→B→G (FIFO survives park→release); (v) all settle code=200.
        s2_m = _outcome_map(s2_outcomes)
        s2_wave = [c_rid, a_rid, b_rid, g_rid]
        _s2_miss = len(s2_order) + 99  # sentinel: never dispatched
        s2_pos = {
            rid: (s2_order.index(rid) if rid in s2_order else _s2_miss)
            for rid in (ph2, c_rid, a_rid, b_rid, g_rid)
        }
        s2_all = all(p < _s2_miss for p in s2_pos.values())
        s2_ph_first = s2_pos[ph2] == 0
        s2_group_before_c = all(
            s2_pos[r] < s2_pos[c_rid] for r in (a_rid, b_rid, g_rid)
        )
        s2_fifo_in_group = s2_pos[a_rid] < s2_pos[b_rid] < s2_pos[g_rid]
        s2_ok = (
            s2_all
            and s2_ph_first
            and s2_group_before_c
            and s2_fifo_in_group
            and s2_m[ph2][0]
            and all(s2_m[r][0] for r in s2_wave)
        )
        segments.append(
            (
                "proto_header_default_ev1_fixed",
                s2_ok,
                f"[2026-09 recal] dispatch="
                f"{[r % 1_000_000 for r in s2_order]} "
                f"(structural: ph first, 70-group A/B/G before C, "
                f"FIFO in group, all code=200), "
                f"codes={[(r % 1_000_000, s2_m[r][1]) for r in s2_wave]}",
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
        # The complete waiting wave is priority ordered.  Y(50) leads;
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
        # wave parks whole and dispatches [ph, Y, Z,
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
        # Needs its own env (ENV-Q3): the default critical-only metrics
        # whitelist (six link-latency presets) hides auto_tpm.*, and the
        # v2 filter has NO mode switch — the whitelist entry
        # flexlb_auto_tpm_request_count is what exposes the series (see
        # the _q3_spec docstring).  Fresh env means the counters start
        # at zero, so the expected buckets are absolute.  Buckets count
        # at the schedule RPC entry regardless of outcome — EV-1 cannot
        # block this observation plane.
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
        s4_samples = _scrape_master_metrics(ops4)
        s4_buckets = {
            p: _metric_sum(
                s4_samples,
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
                f"(expected 70:1, 30:1, 50:1), "
                f"auto_tpm_lines="
                f"{_metric_lines(s4_samples, 'auto_tpm_request')}",
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


def _queue_deadline_terminal(fr, timeout_ms: int) -> tuple[bool, float]:
    """Recognize a real parked-request deadline, not merely error code 8511.

    The schedule future must remain open for the configured absolute queue
    timeout.  The narrow clock window allows timer/poller scheduling jitter
    while preventing an immediate or unrelated 8511 from being accepted as
    the current pull-coordinator contract.
    """
    elapsed_ms = max(0.0, (fr.settled_s - fr.submitted_s) * 1000.0)
    timing_ok = timeout_ms - 2_000 <= elapsed_ms <= timeout_ms + 5_000
    return (not fr.ok and fr.code == CODE_SLO_EXPIRED and timing_ok), elapsed_ms


@case(
    "atpm_preempt_prefill_queued",
    profiles=["single-nonbatch"],
    source="design §2.3 #6 — current global wait ordering (PR1 + PR2 + P6)",
)
def atpm_preempt_prefill_queued(ctx: CaseContext):
    """Priority ordering under a PREFILL_QUEUED-enabled NON_BATCH config.

    ENV-Q2: preemption allows PREFILL_QUEUED only, queueTimeout 60s,
    maxWaiting 8, inflight cap 1, single prefill.

    NON_BATCH capacity blocking leaves every member in the global ordered
    wait set.  Consequently these waves observe PR1/PR2 only: they do not
    create an endpoint-local victim and must not be scored as PR4/PR5/PR6/
    PR10 preemption coverage.  The BATCH boundary case records the same
    global-vs-endpoint distinction; exact replacement remains in the Java
    eviction contract tests until a functional hook can hold endpoint work.

    Wave 1: a priority=50 placeholder parks the lease, then EIGHT
    requests + the incoming 70 queue up: 30a, 30b, 40a, 40b, 30c, 30d,
    30e, 30f, 70.  All nine park and complete 200; the dispatch order
    is [70, 40a, 40b, 30a, 30b, 30c, 30d, 30e, 30f]: the complete
    waiting wave is ordered by priority desc + same-level FIFO.

    Wave 2: after the drain, a 70 placeholder parks the lease; 70x8 plus
    a 90 all wait globally.  All nine complete 200, with the 90 first and
    the 70 group retaining FIFO order.
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

        # ---- wave 1: mixed-priority global wait ordering ----------------
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
        # eviction fallback never runs, zero victims.  All nine settle 200;
        # the complete waiting wave dispatches priority desc + same-level FIFO.
        zero_eviction_w1 = all(
            m1[rids[tag]][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
            for tag in tags
        ) and m1[incoming][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
        wave1_rids = [rids[t] for t in tags] + [incoming]
        prio1 = {rids[t]: int(t[:-1]) for t in tags}
        prio1[incoming] = 70
        _first1, shape1, order1 = _design_final_pattern(
            ops, [ph_fire] + wave1, wave1_rids, prio1
        )
        all1_ok = all(m1[rids[t]][0] for t in tags) and m1[incoming][0]
        ph1_ok = m1[ph][0]

        report.check(
            "PR1",
            _inversion_ratio(order1, prio1),
            context="prefill_enabled_nonbatch_wave1",
            detail=(
                f"global wait dispatch={[r % 1_000_000 for r in order1]}, "
                f"shape_ok={shape1}; no endpoint-local victim was created"
            ),
        )
        report.invariant(
            "PR2",
            _group_order_ok(order1, [rids["40a"], rids["40b"]])
            and _group_order_ok(order1, [rids[t] for t in tags if t.startswith("30")]),
            context="prefill_enabled_nonbatch_wave1_fifo",
            detail=(
                f"same-priority groups retain submit order; dispatch="
                f"{[r % 1_000_000 for r in order1]}"
            ),
        )
        clean1_ok, clean1_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            shape1 and all1_ok and ph1_ok and zero_eviction_w1 and clean1_ok,
            detail=(
                f"[EV-1-FIXED] wave1: all nine requests dispatched from the "
                f"park bucket and completed 200 (whole-wave priority desc "
                f"+ FIFO), "
                f"inflight={'ok' if clean1_ok else clean1_detail}"
            ),
        )
        if not (shape1 and all1_ok and ph1_ok and clean1_ok):
            return report.finish(f"wave1 incomplete, grades: {report.summary()}")

        # ---- wave 2: high-priority global waiter + equal-priority FIFO ---
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
        # incoming 90.  All remain in the global wait set, so this proves
        # ordering only; zero victims is evidence that no endpoint-local
        # preemption observation exists in this NON_BATCH construction.
        zero_eviction = all(
            m2[rid][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
            for rid in w2_rids + [ph2, inc90]
        )
        prio2 = {rid: 70 for rid in w2_rids}
        prio2[inc90] = 90
        _first2, shape2, order2 = _design_final_pattern(
            ops, [ph2_fire] + wave2, w2_rids + [inc90], prio2
        )
        all2_ok = all(m2[rid][0] for rid in w2_rids) and m2[inc90][0]
        ph2_ok = m2[ph2][0]
        report.check(
            "PR1",
            _inversion_ratio(order2, prio2),
            context="prefill_enabled_nonbatch_wave2",
            detail=(
                f"global wait dispatch={[r % 1_000_000 for r in order2]}, "
                f"90 terminal={m2[inc90][1]}, shape_ok={shape2}"
            ),
        )
        report.invariant(
            "PR2",
            _group_order_ok(order2, w2_rids),
            context="prefill_enabled_nonbatch_wave2_fifo",
            detail=(
                f"all priority-70 waiters retain submit order; dispatch="
                f"{[r % 1_000_000 for r in order2]}"
            ),
        )
        clean2_ok, clean2_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            shape2 and all2_ok and ph2_ok and zero_eviction and clean2_ok,
            detail=(
                f"[EV-1-FIXED] wave2: all nine requests completed 200 (the "
                f"90 first among the wave by priority), "
                f"inflight={'ok' if clean2_ok else clean2_detail}"
            ),
        )
        return report.finish(
            f"NON_BATCH global wait: wave1 shape={shape1}, wave2 shape={shape2}, "
            f"endpoint victims=0, grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)


@case(
    "atpm_preempt_decode_engine_owned",
    profiles=["single-nonbatch"],
    source="design §2.3 #7 — guarded decode-pressure terminal contract",
)
def atpm_preempt_decode_engine_owned(ctx: CaseContext):
    """Attempt both decode victim stages behind an explicit KV guardrail.

    Four lower-priority requests are first admitted under normal KV, then
    every Decode endpoint is driven to zero available KV before P70 is
    submitted.  Wave 1 targets the reserved window; wave 2 first proves the
    lower-priority requests are Decode RUNNING and targets engine ownership.

    The current black-box contract is narrower than PR6/PR10: no 8400/8429
    victim is observed.  Wave 1 terminates the incoming exactly as 8431.
    Without a real victim, wave 2 must park and terminate with exact 8511 at
    the configured 60s queue deadline; an immediate 8431, another error code,
    success, or an early 8511 fails.  This describes global-queue settlement
    only and is not scored as preemption.
    PR6 and AT5 are emitted only if real victim terminals and a successful
    replacement make those properties measurable; zero victims never count
    as preemption coverage.
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
        w1_survivors_ok = all(
            m1[rid][0]
            for rid in w1_victim_rids
            if rid not in w1_yielded and rid not in w1_owned
        )
        w1_zero_eviction = not (w1_yielded or w1_owned)
        w1_inc_rejected = (not w1_inc_ok) and w1_inc_code == CODE_RESOURCE_EXHAUSTED
        # This guarded black-box construction currently produces no victim:
        # all occupants complete and the incoming terminates exactly 8431.
        # That is a pressure-terminal observation only; the Java planner and
        # manager tests carry the endpoint-victim contract separately.
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
        w2_deadline_expired, w2_wait_ms = _queue_deadline_terminal(
            w2_inc_fire, 60_000
        )
        w2_immediate_rejected = (
            not w2_inc_ok and w2_inc_code == CODE_RESOURCE_EXHAUSTED
        )
        w2_terminal_ok = w2_deadline_expired
        w2_terminal_mode = (
            "resource_exhausted_unexpected"
            if w2_immediate_rejected
            else "queue_deadline"
            if w2_deadline_expired
            else "unexpected"
        )
        cancel_evidence = []
        for rid in w2_owned:
            ok_c, detail_c = ops.verify_engine_cancelled(rid)
            cancel_evidence.append(f"{rid % 1_000_000}:{ok_c}")
        # EV-2 baseline (behaviour finding, probe E11 two-orchestration):
        # DECODE_ENGINE_OWNED eviction (tokenized Cancel → 8429) is
        # equally unreachable — the 8429/8400 terminal split has no
        # observation object.  Observable form mirrors wave 1.
        real_victim_observed = bool(w1_yielded or w1_owned or w2_owned)
        split_ok = False
        if real_victim_observed:
            split_ok = (
                len(w1_yielded) == 1
                and not w1_owned
                and w1_inc_ok
                and w1_survivors_ok
                and len(w2_owned) == 1
                and w2_inc_ok
                and w2_survivors_ok
                and all(ok.endswith(":True") for ok in cancel_evidence)
            )
            report.invariant(
                "PR6",
                split_ok,
                context="decode_terminal_split_real_victims",
                detail=(
                    f"reserved yielded={len(w1_yielded)}, reserved-owned="
                    f"{len(w1_owned)}, engine-owned={len(w2_owned)}, "
                    f"incoming success={w1_inc_ok}/{w2_inc_ok}, "
                    f"cancel evidence={cancel_evidence}"
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
        zero_victim_baseline_ok = (
            w1_zero_eviction
            and w1_inc_rejected
            and w1_victims_ok
            and w2_zero_eviction
            and w2_terminal_ok
            and w2_survivors_ok
        )
        observed_path_ok = (
            split_ok if real_victim_observed else zero_victim_baseline_ok
        )
        report.invariant(
            "P6",
            w1_guard[0]
            and w2_guard[0]
            and observed_path_ok
            and clean2_ok,
            detail=(
                f"wave1 guard=[{w1_guard[1]}], occupants="
                f"{ {r % 1_000_000: c for r, c in w1_codes.items()} }, "
                f"victims 8400/8429={len(w1_yielded)}/{len(w1_owned)}, "
                f"incoming={w1_inc_code} (expected 8431); wave2 guard="
                f"[{w2_guard[1]}], victims8429={len(w2_owned)}, "
                f"incoming={w2_inc_code}, mode={w2_terminal_mode}, "
                f"settlement={w2_wait_ms:.0f}ms, exact-deadline="
                f"{w2_deadline_expired}, observed_path="
                f"{'real_victim' if real_victim_observed else 'zero_victim'}, "
                f"inflight={'ok' if clean2_ok else clean2_detail}"
            ),
        )
        return report.finish(
            f"guarded decode pressure: wave1 incoming={w1_inc_code}, "
            f"wave2 incoming={w2_inc_code}/{w2_wait_ms:.0f}ms, victims="
            f"{len(w1_yielded) + len(w1_owned) + len(w2_owned)}, "
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
    source="design §2.3 #8 — single-QoS global wait (AT3 + P6)",
)
def atpm_same_priority_zero_eviction(ctx: CaseContext):
    """Single-QoS global-wait behavior (AT3, not endpoint-local PR4).

    Eight priority-50 requests plus one more priority-50 request wait in
    the pull coordinator.  All complete in FIFO order with zero victim
    terminals.  Because the fallback is never reached, this case does not
    claim the stronger PR4 endpoint-victim filter contract.

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
            "AT3",
            zero_eviction and shape_sp and all_ok and m[inc][1] == CODE_OK,
            context="single_qos_incoming_design_final",
            detail=(
                f"all nine priority-50 waiters complete in FIFO order; "
                f"incoming={m[inc][1]}, zero 8400/8429={zero_eviction}, "
                f"dispatch={[r % 1_000_000 for r in order_sp]}"
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
        # under PRIORITY the wave dispatches [70a, 70b, 70c, low_1, low_2]
        # (whole-wave priority desc + FIFO),
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
                f"before both 30s, under "
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
    (whole-wave priority desc + same-level FIFO); the explicit-cap
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
        # dispatches first in the whole wave.  The tryFallback path to 8510
        # needs a failed
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
# atpm_decode_reservation_priority — guarded decode-pressure deadline contract
# ===========================================================================


@case(
    "atpm_decode_reservation_priority",
    profiles=["single-nonbatch"],
    source="design §2.4 #14 — guarded decode-pressure deadline (P6)",
)
def atpm_decode_reservation_priority(ctx: CaseContext):
    """Three guarded Decode-pressure waves with one exact terminal shape.

    Existing occupants route under normal KV.  After they reach Decode
    RUNNING, every Decode endpoint is saturated and verified by
    ``_decode_pressure_guardrail`` before the incoming request is fired.
    With the current global queue, each incoming remains parked and must
    terminate exactly at the configured 60s deadline as code 8511.  A
    success, immediate rejection, early stale-TTL 8511, 8400, or 8429 is a
    failure; occupants must complete and victim metrics must stay flat.

    This is deliberately P6/deadline coverage, not AT7.  The three waves
    differ in priority and KV shape, but none currently yields an endpoint
    victim from which cross-stage victim-selection parity could be inferred.
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
        w1_guard = _decode_pressure_guardrail(ops, decode_names)
        if not w1_guard[0]:
            return False, f"wave1 decode guardrail failed: {w1_guard[1]}"
        w1_inc = ops.next_request_id(base)
        w1_inc_fire = _fire(ops, w1_inc, priority=70, input_len=2048, output_len=2)
        fires.append(w1_inc_fire)

        m1 = _outcome_map(_drain(ops, w1_fires + [w1_inc_fire]))
        w1_victims = [rid for rid in w1_rids if m1[rid][1] == CODE_ENGINE_CANCELLED]
        w1_survivors_ok = all(m1[rid][0] for rid in w1_rids if rid not in w1_victims)
        w1_inc_code = m1[w1_inc][1]
        now_victim = _metric_sum(
            _scrape_master_metrics(ops),
            "auto_tpm_victim",
            {"victim_priority": "30", "incoming_priority": "70"},
        )
        w1_delta = (now_victim or 0.0) - (base_victim or 0.0)
        w1_deadline, w1_wait_ms = _queue_deadline_terminal(w1_inc_fire, 60_000)
        wave_reports.append(
            (
                "w1_lower_priority_exact_deadline",
                w1_victims == []
                and w1_deadline
                and w1_survivors_ok
                and w1_delta == 0.0,
                (
                    f"guard=[{w1_guard[1]}], victims8429="
                    f"{len(w1_victims)}, incoming70={w1_inc_code}, "
                    f"waited={w1_wait_ms:.0f}ms, exact-deadline="
                    f"{w1_deadline}, "
                    f"survivors ok={w1_survivors_ok}, "
                    f"victim.count delta(30<-70)={w1_delta}"
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
        w2_guard = _decode_pressure_guardrail(ops, decode_names)
        if not w2_guard[0]:
            return False, f"wave2 decode guardrail failed: {w2_guard[1]}"
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
        w2_deadline, w2_wait_ms = _queue_deadline_terminal(w2_inc_fire, 60_000)
        wave_reports.append(
            (
                "w2_same_priority_exact_deadline",
                w2_deadline
                and w2_zero_eviction
                and w2_occupants_ok
                and w2_delta == 0.0,
                (
                    f"guard=[{w2_guard[1]}], incoming50={w2_inc_code}, "
                    f"waited={w2_wait_ms:.0f}ms, exact-deadline="
                    f"{w2_deadline}, "
                    f"zero 8400/8429={w2_zero_eviction}, occupants completed="
                    f"{w2_occupants_ok}, victim.count delta={w2_delta}"
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
        w3_guard = _decode_pressure_guardrail(ops, decode_names)
        if not w3_guard[0]:
            return False, f"wave3 decode guardrail failed: {w3_guard[1]}"
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
        w3_inc_code = m3[w3_inc][1]
        w3_deadline_expired, w3_wait_ms = _queue_deadline_terminal(
            w3_inc_fire, 60_000
        )
        wave_reports.append(
            (
                "w3_mixed_kv_exact_deadline",
                w3_victims == []
                and w3_deadline_expired
                and w3_survivors_ok,
                (
                    f"guard=[{w3_guard[1]}], victims="
                    f"{len(w3_victims)}, incoming70={w3_inc_code}, "
                    f"waited={w3_wait_ms:.0f}ms, exact-deadline="
                    f"{w3_deadline_expired}, survivors ok={w3_survivors_ok}"
                ),
            )
        )
        clean3_ok, clean3_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)

        report.invariant(
            "P6",
            all(ok for (_l, ok, _d) in wave_reports) and clean3_ok,
            detail=(
                "; ".join(f"{label}: {detail}" for label, _ok, detail in wave_reports)
                + f"; inflight={'ok' if clean3_ok else clean3_detail}"
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
# atpm_observability_integrity — supported metric/PV accounting planes
# ===========================================================================


@case(
    "atpm_observability_integrity",
    profiles=["single-nonbatch"],
    source="design §2.4 #15 — AT6 + P6 (AT8 decision-log gap)",
)
def atpm_observability_integrity(ctx: CaseContext):
    """Observable priority ordering plus metric/PV accounting integrity.

    ENV-O1: Q2-shaped config (PREFILL_QUEUED preemption, queueTimeout 7s
    — [EV-1-FIXED] flipped from 8s at intake3
    PendingPlacementCoordinator 6ad0315f10: under the pull model the
    wave's third release slot lands at t=9s, which raced the 8s deadline
    of the 4th submitter (70a); 7s puts every non-dispatched deadline
    strictly before the third slot, making the client shape
    deterministic) + master debug log + the auto_tpm family whitelist.
    Implementation-period corrections over the design's env sketch: the
    DEFAULT critical-only whitelist hides auto_tpm.* (the legacy
    flexlb.monitor.mode switch is dead on this line), so the
    FLEXLB_MONITOR_METRIC_WHITELIST entry is required; FLEXLB_PV_LOG is
    a load-client-line knob with no consumer
    on the harness line — the pvLogger writes at INFO by default, so
    the pv.log plane needs no extra knob.  The master_env + debug-log
    differences give O1 its own fingerprint (exclusive env — the metric
    counters start from zero).

    Choreography (the atpm_preempt_prefill_queued wave-1 shape with
    mixed priorities for bucket coverage), [EV-1-FIXED] design-final
    form: a 50 placeholder parks the inflight lease; 30a/30b/50a/50b/
    70a/70b/30c/30d + the 90 ALL park in the pull-based coordinator.
    Prefill is slowed to 3s: ph completes at t=3, then the globally highest
    priority 90 takes slot two (t=3-6), followed by the earliest 70 in slot
    three (t=6-9); both complete inside their deadlines.  The remaining
    seven (30a, 30b, 30c, 30d, 50a, 50b, and the other 70) expire at their 7s
    deadlines as plain 8511 BATCH_SLO_EXPIRED (the low-priority-
    suppression sample; 30d is no longer evicted — no eviction ever
    fires under the pull model, the victim counter stays flat).

    Supported-plane assertions:
      * auto_tpm.request.count{priority=30|50|70|90} == the injected
        bucket counts 4/3/2/1 — counted at the schedule RPC entry for
        EVERY request regardless of outcome (FlexlbServiceImpl:723), the
        metric-plane normalization evidence crossing prio_normalize's
        behaviour plane;
      * auto_tpm.schedule.latency_ms{result="success"} present (the
        TIMER family; result is "success" | "error_<code>");
      * auto_tpm.victim.count == 0, matching zero client-side victim
        terminals in this global-wait choreography;
      * pv.log tail carries admissionRejectReason fields (channel
        availability; sampled non-null values recorded in the detail).

    The master currently emits only startup configuration text for this
    zero-victim flow, not a per-decision scheduler/preemption event.  Startup
    text is retained as environment sanity only and is not accepted as AT8
    decision observability; AT8 therefore remains intentionally ungraded.
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
        # is ph + 90 + the earliest 70 completing 200.  The complete wait
        # set is globally priority ordered.  The remaining seven expire as
        # 8511, with zero
        # route rejection and zero eviction.
        wave_tags = [t for t, _p in ladder]
        completed = ["ph"] + [t for t in wave_tags if m[rids[t]][0]]
        rejected8402 = [t for t in wave_tags if m[rids[t]][1] in ROUTE_REJECT_FAMILY]
        expired8511 = [t for t in wave_tags if m[rids[t]][1] == CODE_SLO_EXPIRED]
        ph_ok = m[rids["ph"]][0]
        d_order = _dispatch_order(ops, [ph_fire] + wave)
        d_pos = {r: i for i, r in enumerate(d_order)}
        dispatch_pair_ok = (
            rids["90"] in d_pos
            and rids["70a"] in d_pos
            and d_pos[rids["90"]] < d_pos[rids["70a"]]
        )
        client_shape_ok = (
            set(completed) == {"ph", "70a", "90"}
            and len(completed) == 3
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

        # Startup configuration is only an environment sanity check.  It is
        # not a scheduler decision or preemption event and cannot satisfy AT8.
        log_text = _master_log_text(env)
        config_sanity = (
            "ordering=PRIORITY" in log_text and "dispatcher=NON_BATCH" in log_text
        )

        # ---- pv.log plane ------------------------------------------------
        # A8 (Daniel P2-3): this env's delta only, filtered to this
        # case's rids (the dup probe re-uses ph, so its INVALID_REQUEST
        # row is inside the ph filter as well).
        pv_tail = _pv_log_tail(env, [ph] + [rids[t] for t in wave_tags])
        pv_field_ok = "admissionRejectReason" in pv_tail
        pv_samples = re.findall(r'"admissionRejectReason"\s*:\s*"([A-Z_]+)"', pv_tail)

        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            client_shape_ok and clean_ok,
            detail=(
                f"[EV-1-FIXED] every request reached a terminal: 3 "
                f"completed [ph, 70a, 90], 7 expired 8511, "
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
            dup_rejected
            and client_shape_ok
            and buckets_ok
            and latency_ok
            and victim_ok
            and pv_field_ok
            and clean_ok,
            context="supported_metric_pv_accounting",
            detail=(
                f"duplicate-rid {CODE_INVALID_REQUEST}: {dup_note}; "
                f"client completed={completed}, expired8511="
                f"{len(expired8511)}/7, route-rejected={rejected8402}; "
                f"request.count={ {p: buckets[p] for p in buckets} } "
                f"(expected {expected_buckets}), schedule.latency="
                f"{'present' if latency_ok else 'MISSING'}, victim.count="
                f"{victim_total} (expected 0), pv field="
                f"{'present' if pv_field_ok else 'MISSING'}"
                + (f", pv samples={pv_samples[:3]}" if pv_samples else "")
                + f"; startup-config sanity={config_sanity} (not decision evidence); "
                f"inflight={'ok' if clean_ok else clean_detail}; AT8 ungraded: "
                f"no per-decision scheduler/preemption event emitted"
            ),
        )
        return report.finish(
            f"planes: client(design-final)={client_shape_ok} metrics="
            f"{buckets_ok and latency_ok and victim_ok} pv={pv_field_ok}, "
            f"startup-config={config_sanity} (not AT8), grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)


# ===========================================================================
# Priority boundary and cancellation coverage — atpm_preempt_* family
#
# The BATCH boundary cases explicitly probe whether an older P30 reaches the
# endpoint queue before P70 arrives.  Under the current single-credit path it
# remains global, so those cases grade PR1/accounting and record zero victim
# evidence.  Cancel NOT_FOUND and TOMBSTONED cover their exact reconciliation
# branches but are not relabeled as PR6/PR10 preemption terminals.
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
    sacrificial = ops.next_request_id()
    try:
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
    return dropped, restored, sacrificial


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


def _max_engine_requests_1(config: str) -> str:
    """JSON post-processing: decode availability maxEngineRequests=1 (the
    knob is not a build_flexlb_config generator — cancel.py's
    cancel_preemption_victim established the splice)."""
    parsed = json.loads(config)
    parsed["router"]["roles"]["decode"]["availability"]["maxEngineRequests"] = 1
    return json.dumps(parsed, separators=(",", ":"))


def _pq_live_spec(ctx: CaseContext) -> EnvSpec:
    """Single-credit BATCH env for staged endpoint queue replacement."""
    return _spec(
        ctx,
        "atpm_pq_live",
        config=_prio_config(
            dispatcher="batch",
            preemption=_PREEMPT_PQ,
            max_waiting=2,
            queue_timeout_ms=60_000,
            max_inflight_batches=1,
        ),
        extra_env=_MONITOR_PREEMPT_LIVE_ENV,
    )


def _dr_live_spec(ctx: CaseContext) -> EnvSpec:
    """Single-credit BATCH env with one 4-block Decode pool."""
    return _spec(
        ctx,
        "atpm_dr_live",
        n_decode=1,
        decode_cache_blocks=4,
        config=_prio_config(
            dispatcher="batch",
            preemption={"allowed_victim_stages": ["PREFILL_QUEUED", "DECODE_RESERVED"]},
            queue_timeout_ms=60_000,
            max_inflight_batches=1,
        ),
        extra_env=_MONITOR_PREEMPT_LIVE_ENV,
    )


def _nf_spec(ctx: CaseContext) -> EnvSpec:
    """ENV for atpm_preempt_cancel_not_found: all-three-stage preemption
    (engineCancellation mandatory) + decode maxEngineRequests=1, NON_BATCH
    single-decode topology (1P+1D)."""
    return _spec(
        ctx,
        "atpm_nf",
        n_decode=1,
        config=_max_engine_requests_1(
            _prio_config(preemption=_PREEMPT_ALL_STAGES, queue_timeout_ms=60_000)
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
        config=_max_engine_requests_1(
            _prio_config(
                dispatcher="batch",
                preemption=_PREEMPT_ALL_STAGES,
                queue_timeout_ms=60_000,
            )
        ),
    )


@case(
    "atpm_preempt_prefill_queued_live",
    profiles=["single-batch"],
    source="preemption-stages audit (2026-09) — BATCH global-wait boundary",
)
def atpm_preempt_prefill_queued_live(ctx: CaseContext):
    """BATCH global-wait ordering at the PREFILL_QUEUED boundary.

    A slow P50 holds the sole delivery credit.  P30 is submitted first and
    given six seconds to reach the endpoint queue, then P70 is submitted.
    Under the current single-credit coordinator P30 remains global: the
    endpoint priority-30 queue gauge never appears, P70 dispatches before the
    older P30, both complete, and victim metrics remain zero.  This is PR1
    plus accounting evidence, not endpoint-local PR4/PR6/PR10 coverage.
    """
    env = ctx.env_manager.ensure(_pq_live_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    handles: dict = {}
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=8000.0)
        time.sleep(PERF_SETTLE_S)

        ph = ops.next_request_id(base)
        ph_resp = ops.schedule(
            ph, priority=50, input_len=2048, output_len=2, timeout_s=90.0
        )
        if ph_resp.code != CODE_OK or not ph_resp.success:
            return False, f"placeholder schedule failed: {ph_resp.error_message}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "placeholder never dispatched"

        victim = ops.next_request_id(base)
        inc = ops.next_request_id(base)
        with ThreadPoolExecutor(max_workers=2) as pool:
            victim_future = pool.submit(
                ops.schedule,
                victim,
                priority=30,
                input_len=2048,
                output_len=2,
                timeout_s=90.0,
            )
            queue_seen, queue_depth, queue_lines = _wait_batch_queue_priority(
                ops, 30, min_depth=1, timeout_s=6.0
            )
            inc_future = pool.submit(
                ops.schedule,
                inc,
                priority=70,
                input_len=2048,
                output_len=2,
                timeout_s=90.0,
            )
            victim_resp = victim_future.result(timeout=95.0)
            inc_resp = inc_future.result(timeout=95.0)

        queue_channel_ok = queue_lines != "<none>"
        responses = {ph: ph_resp, victim: victim_resp, inc: inc_resp}
        all_scheduled = all(
            resp.code == CODE_OK and resp.success for resp in responses.values()
        )

        completed = {}
        for rid, resp in responses.items():
            if resp.code != CODE_OK or not resp.success:
                continue
            handle = ops.start_stream(resp, rid)
            handles[rid] = handle
            ended = handle.wait_end(45.0)
            completed[rid] = ended and handle.snap.completed
        all_completed = all(completed.get(r) for r in responses)
        all_seen = all(_engine_saw(ops, r) for r in responses)
        lifecycle = {rid: _prefill_lifecycle(ops, rid) or {} for rid in responses}
        wait_order = sorted(
            (victim, inc),
            key=lambda rid: lifecycle[rid].get("running_ms", float("inf")),
        )
        priorities = {victim: 30, inc: 70}

        samples = _scrape_master_metrics(ops)
        pq_victim = _metric_sum(
            samples, "auto_tpm_victim_count", {"stage": "prefill_queued"}
        )
        pq_victim_30_70 = _metric_sum(
            samples,
            "auto_tpm_victim_count",
            {
                "stage": "prefill_queued",
                "victim_priority": "30",
                "incoming_priority": "70",
            },
        )
        pq_preempt = _metric_sum(
            samples,
            "auto_tpm_priority_preempt_count",
            {"stage": "prefill_queued"},
        )

        report.check(
            "PR1",
            _inversion_ratio(wait_order, priorities),
            context="batch_prefill_global_wait",
            detail=(
                f"P30 submitted at least 6s before P70; dispatch="
                f"{[r % 1_000_000 for r in wait_order]} (expected P70 then "
                f"P30), endpoint P30 queue observed={queue_seen}, depth="
                f"{queue_depth}, metrics={queue_lines}"
            ),
        )
        report.invariant(
            "AT6",
            queue_channel_ok
            and not queue_seen
            and all_scheduled
            and all_completed
            and all_seen
            and (pq_victim or 0.0) == 0.0
            and (pq_preempt or 0.0) == 0.0,
            context="batch_prefill_global_wait_accounting",
            detail=(
                f"schedule codes="
                f"{ {r % 1_000_000: resp.code for r, resp in responses.items()} }, "
                f"completed={ {r % 1_000_000: completed.get(r) for r in responses} }, "
                f"all engine-seen={all_seen}; queue metric channel="
                f"{queue_channel_ok}; victim.count="
                f"{pq_victim} (expected absent/0), 30<-70 sample="
                f"{pq_victim_30_70}, priority_preempt.count="
                f"{pq_preempt} (expected absent/0), victim samples="
                f"{_metric_lines(samples, 'auto_tpm_victim_count')}"
            ),
        )
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()
        report.invariant(
            "P6",
            queue_channel_ok
            and not queue_seen
            and all_scheduled
            and all_completed
            and all_seen
            and clean_ok
            and engine_clean
            and recovery_ok,
            detail=(
                f"all three requests completed normally, endpoint victims=0, "
                f"inflight="
                f"{'ok' if clean_ok else clean_detail}, "
                f"engine={'ok' if engine_clean else engine_detail}, "
                f"recovery={recovery_msg}"
            ),
        )
        return report.finish(
            f"BATCH global wait: queue_seen={queue_seen}, dispatch="
            f"{[r % 1_000_000 for r in wait_order]}, victim.count={pq_victim}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for handle in handles.values():
            handle.cancel()
        _finally_hygiene(ops, [], prefill_names)


@case(
    "atpm_preempt_decode_reserved_live",
    profiles=["single-batch"],
    source="preemption-stages audit (2026-09) — BATCH Decode global-wait boundary",
)
def atpm_preempt_decode_reserved_live(ctx: CaseContext):
    """BATCH global-wait ordering at the Decode-reservation boundary.

    P90 (512 tokens) holds the single delivery credit.  P30 (512 tokens)
    is submitted six seconds before P70.  It still remains in the global
    queue, as proved by the missing endpoint priority-30 queue gauge.  P70
    uses input_len=2048, comfortably below the 4096-token pool's default 90%
    guard after publication, dispatches before P30, and all three requests
    complete.  Zero stage-victim metrics mean this is PR1/accounting coverage,
    not DECODE_RESERVED PR4/PR6/PR10 coverage.
    """
    env = ctx.env_manager.ensure(_dr_live_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    handles: dict = {}
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=8000.0)
        time.sleep(PERF_SETTLE_S)

        ph = ops.next_request_id(base)
        ph_resp = ops.schedule(
            ph, priority=90, input_len=512, output_len=2, timeout_s=90.0
        )
        if ph_resp.code != CODE_OK or not ph_resp.success:
            return False, f"placeholder schedule failed: {ph_resp.error_message}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "placeholder never dispatched"

        victim = ops.next_request_id(base)
        inc = ops.next_request_id(base)
        with ThreadPoolExecutor(max_workers=2) as pool:
            v_future = pool.submit(
                ops.schedule,
                victim,
                priority=30,
                input_len=512,
                output_len=2,
                timeout_s=90.0,
            )
            queue_seen, queue_depth, queue_lines = _wait_batch_queue_priority(
                ops, 30, min_depth=1, timeout_s=6.0
            )
            inc_future = pool.submit(
                ops.schedule,
                inc,
                priority=70,
                input_len=2048,
                output_len=2,
                timeout_s=90.0,
            )
            v_resp = v_future.result(timeout=95.0)
            inc_resp = inc_future.result(timeout=95.0)

        queue_channel_ok = queue_lines != "<none>"
        responses = {ph: ph_resp, victim: v_resp, inc: inc_resp}
        all_scheduled = all(
            resp.code == CODE_OK and resp.success for resp in responses.values()
        )

        completed = {}
        for rid, resp in responses.items():
            if resp.code != CODE_OK or not resp.success:
                continue
            handle = ops.start_stream(resp, rid)
            handles[rid] = handle
            ended = handle.wait_end(45.0)
            completed[rid] = ended and handle.snap.completed
        all_completed = all(completed.get(r) for r in responses)
        all_seen = all(_engine_saw(ops, r) for r in responses)
        lifecycle = {rid: _prefill_lifecycle(ops, rid) or {} for rid in responses}
        wait_order = sorted(
            (victim, inc),
            key=lambda rid: lifecycle[rid].get("running_ms", float("inf")),
        )
        priorities = {victim: 30, inc: 70}

        samples = _scrape_master_metrics(ops)
        dr_victim = _metric_sum(
            samples, "auto_tpm_victim_count", {"stage": "decode_reserved"}
        )
        dr_kv = _metric_sum(
            samples, "auto_tpm_victim_kv_tokens", {"stage": "decode_reserved"}
        )
        pq_victim = _metric_sum(
            samples, "auto_tpm_victim_count", {"stage": "prefill_queued"}
        )

        report.check(
            "PR1",
            _inversion_ratio(wait_order, priorities),
            context="batch_decode_global_wait",
            detail=(
                f"P30 submitted at least 6s before P70; dispatch="
                f"{[r % 1_000_000 for r in wait_order]} (expected P70 then "
                f"P30), endpoint P30 queue observed={queue_seen}, depth="
                f"{queue_depth}, metrics={queue_lines}"
            ),
        )
        report.invariant(
            "AT6",
            queue_channel_ok
            and not queue_seen
            and all_scheduled
            and all_completed
            and all_seen
            and (dr_victim or 0.0) == 0.0
            and (dr_kv or 0.0) == 0.0
            and (pq_victim or 0.0) == 0.0,
            context="batch_decode_global_wait_accounting",
            detail=(
                f"schedule codes="
                f"{ {r % 1_000_000: resp.code for r, resp in responses.items()} }, "
                f"completed={ {r % 1_000_000: completed.get(r) for r in responses} }, "
                f"all engine-seen={all_seen}; queue metric channel="
                f"{queue_channel_ok}; decode_reserved victim.count="
                f"{dr_victim}, victim.kv_tokens={dr_kv}, prefill_queued "
                f"victims={pq_victim} (all expected absent/0), samples="
                f"{_metric_lines(samples, 'auto_tpm_victim')}"
            ),
        )
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()
        report.invariant(
            "P6",
            queue_channel_ok
            and not queue_seen
            and all_scheduled
            and all_completed
            and all_seen
            and clean_ok
            and engine_clean
            and recovery_ok,
            detail=(
                f"all three requests completed normally, endpoint victims=0, "
                f"inflight="
                f"{'ok' if clean_ok else clean_detail}, "
                f"engine={'ok' if engine_clean else engine_detail}, "
                f"recovery={recovery_msg}"
            ),
        )
        return report.finish(
            f"BATCH Decode global wait: queue_seen={queue_seen}, dispatch="
            f"{[r % 1_000_000 for r in wait_order]}, victim.count={dr_victim}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for handle in handles.values():
            handle.cancel()
        _finally_hygiene(ops, [], prefill_names)


@case(
    "atpm_preempt_cancel_not_found",
    profiles=["single-nonbatch"],
    source="preemption-stages audit (2026-09) — Cancel NOT_FOUND (AT6 + P6)",
)
def atpm_preempt_cancel_not_found(ctx: CaseContext):
    """Preemption-chain Cancel NOT_FOUND branch: the victim has FINISHED
    by the time the preemption Cancel reaches its original prefill, and
    the master's consumer side closes per DecodePreemptionCoordinator's
    NOT_FOUND semantics (cleanSingleNotFound → abort → the incoming
    settles 8431, never a hang and never a false success).

    Construction (the engine cancelRequest branch ORDER is decisive):
    a RUNNING victim's Cancel is answered by the downstreamDecodeOwners
    branch (P→D conduction → ACCEPTED) BEFORE the finished-check, so the
    victim must have cleared BOTH ownership directions (decode
    completion runs clearUpstreamOwnership) AND hold a non-running
    prefill-side lifecycle (the prefill marks its lifecycle entry
    finished when the prefill phase ends) — that routes the Cancel to
    alreadyFinished → NOT_FOUND.

    Choreography: victim P30 (input=512, output=200 ≈ 1.5s of decode)
    fires and reaches RUNNING on the single decode engine; the decode
    engine then stops answering the master's status polls
    (status_no_respond) — the master's view freezes on victim RUNNING +
    slot 1/1 (decode maxEngineRequests=1) while the engine-side victim
    runs to completion; the P70 incoming then fires: its decode
    placement BLOCKS on the frozen view → DECODE_ENGINE_OWNED eviction
    → the tokenized Cancel reaches the original prefill with the victim
    already finished → NOT_FOUND → the incoming settles 8431.

    The frozen-view window is bounded by the master's 3-strike health
    demotion (3 consecutive 1s status RPC timeouts ≈ 3s): the victim's
    remaining decode (~0.7s at injection) keeps the whole choreography
    inside it — a slow finish would let the master demote the decode
    engine first and the incoming would surface 8403 instead (recorded
    as a construction miss, not a contract relaxation).

    Contract:
      * the incoming settles exactly 8431 RESOURCE_EXHAUSTED (the
        coordinator's NOT_FOUND semantics); the victim completes
        NORMALLY (full output, never engine-cancelled);
      * the preemption Cancel really went out (Cancel RPC delta >= 1);
      * after the injection clears, the master ledger drains (the
        victim's stale RUNNING settles from the resumed decode status)
        and recovery works.
    """
    env = ctx.env_manager.ensure(_nf_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    decode_name = None
    injected = False
    try:
        decode_name = _decode_names(ops)[0]
        victim = ops.next_request_id(base)
        victim_fire = _fire(ops, victim, priority=30, input_len=512, output_len=200)
        fires.append(victim_fire)
        if not victim_fire.ok:
            return False, f"victim schedule failed: code={victim_fire.code}"
        if not _poll_decode_running(ops, victim):
            return False, "victim never reached decode running"
        time.sleep(0.6)

        baseline_cancel = _cancel_rpc_total(ops)
        inject_type(ops, decode_name, "status_no_respond")
        injected = True
        # Engine-side completion — the MASTER view stays frozen on
        # RUNNING (the stale slot the incoming will block on).
        if not _poll_engine_finished(ops, victim, 3.0):
            return False, "victim never finished engine-side (3s window)"

        inc = ops.next_request_id(base)
        inc_fire = _fire(ops, inc, priority=70, input_len=512, output_len=2)
        fires.append(inc_fire)

        inc_rejected = not inc_fire.ok and inc_fire.code == CODE_RESOURCE_EXHAUSTED
        victim_completed = bool(
            victim_fire.terminal is not None
            and victim_fire.terminal.wait(STREAM_WAIT_S)
            and victim_fire.terminal.completed
        )
        victim_cancelled, victim_cancel_detail = ops.verify_engine_cancelled(victim)
        cancel_delta = _cancel_rpc_total(ops) - baseline_cancel
        settlement_ms = max(
            0.0, (inc_fire.settled_s - inc_fire.submitted_s) * 1000.0
        )

        # Clear the injection: the master resumes consuming decode
        # status, the victim's stale RUNNING settles, the ledger drains.
        inject_type(ops, decode_name, "status_no_respond", enabled=False)
        injected = False
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 20.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        report.invariant(
            "AT6",
            inc_rejected
            and victim_completed
            and not victim_cancelled
            and cancel_delta >= 1,
            context="cancel_not_found_accounting",
            detail=(
                f"incoming terminal={inc_fire.code} (expected exactly "
                f"{CODE_RESOURCE_EXHAUSTED} — cleanSingleNotFound abort), "
                f"victim completed={victim_completed} (normal "
                f"completion), victim engine-cancelled={victim_cancelled} "
                f"[{victim_cancel_detail}] (expect False — the Cancel "
                f"arrived AFTER the finish, nothing to cancel), Cancel RPC "
                f"delta={cancel_delta}, schedule settlement="
                f"{settlement_ms:.0f}ms (diagnostic only, not AT5)"
            ),
        )
        report.invariant(
            "P6",
            inc_rejected
            and victim_completed
            and not victim_cancelled
            and cancel_delta >= 1
            and clean_ok
            and engine_clean
            and recovery_ok,
            detail=(
                f"after injection cleared: "
                f"inflight={'ok' if clean_ok else clean_detail}, "
                f"engine={'ok' if engine_clean else engine_detail}, "
                f"recovery={recovery_msg}"
            ),
        )
        return report.finish(
            f"cancel-not-found: incoming={inc_fire.code}, victim completed"
            f"={victim_completed}, cancel_delta={cancel_delta}, "
            f"settlement_ms={settlement_ms:.0f} (not AT5), "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if injected and decode_name is not None:
            try:
                inject_type(ops, decode_name, "status_no_respond", enabled=False)
            except Exception:
                pass
        _finally_hygiene(ops, fires, [])


@case(
    "atpm_preempt_cancel_tombstoned",
    profiles=["single-batch"],
    requires=["enqueue_batch"],
    source="preemption-stages audit (2026-09) — TOMBSTONED fence (AT6 + P6)",
)
def atpm_preempt_cancel_tombstoned(ctx: CaseContext):
    """Observable TOMBSTONED + ABSENT_FENCE contract after Prefill restart.

    ``crash_after`` kills the Prefill while the sacrificial BATCH delivery is
    uncertain.  The coordinator reconciles that exact request against the
    fresh, empty Prefill; its Cancel is therefore TOMBSTONED and installs an
    ABSENT_FENCE.  The same rid must be recorded as TOMBSTONED in this env's
    PV journal and rejected by a direct late Enqueue with typed 8429.

    A former version tried to keep a separate Decode victim alive across the
    crash and then force a P70 admission preemption.  Stream-break cleanup
    legitimately settled that occupancy first, so no capacity victim existed
    and the assertion was testing a retired choreography rather than product
    behaviour.
    """
    env = ctx.env_manager.ensure(_ts_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    try:
        dropped, restored, sacrificial = _crash_and_restart(ops, "prefill-0")
        if not (dropped and restored):
            return False, (
                f"crash/restart failed: dropped={dropped}, " f"restored={restored}"
            )

        # start_engine resets per-instance counters.  Any Cancel now visible
        # landed on the fresh generation and belongs to uncertain-delivery
        # reconciliation for the sacrificial rid.
        cancel_reached = wait_for(
            lambda: _cancel_rpc_total(ops) >= 1, 15.0, 0.5
        )
        cancel_total = _cancel_rpc_total(ops)
        pv_tail = _pv_log_tail(env, [sacrificial])
        tombstoned_journal = "engine reported TOMBSTONED" in pv_tail

        fence_ok, fence_detail = False, "no probe"
        try:
            prefill_addr = ops.snapshot_by_name()["prefill-0"]["grpc_addr"]
            probe = ops.build_generate_input(sacrificial, output_len=2)
            ack = _direct_enqueue(
                ops, prefill_addr, probe, sacrificial * 10 + 1
            )
            fence_ok, fence_detail = _fence_rejected_8429(ack, sacrificial)
        except Exception as exc:
            fence_detail = repr(exc)

        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 30.0
        )
        residue_ok, residue_detail = _fence_residue_stable(ops, 1)
        recovery_ok, recovery_msg = ops.verify_recovery()

        report.invariant(
            "AT6",
            dropped
            and restored
            and cancel_reached
            and tombstoned_journal
            and fence_ok,
            context="uncertain_delivery_tombstone_accounting",
            detail=(
                f"crash/restart={dropped}/{restored}, sacrificial="
                f"{sacrificial}, fresh-instance cancel_total={cancel_total}, "
                f"PV engine-reported-TOMBSTONED={tombstoned_journal}, "
                f"ABSENT_FENCE direct-enqueue rejection 8429={fence_ok} "
                f"({fence_detail}), fresh-instance Cancel reached="
                f"{cancel_reached}"
            ),
        )
        report.invariant(
            "P6",
            dropped
            and restored
            and cancel_reached
            and tombstoned_journal
            and fence_ok
            and engine_clean
            and residue_ok
            and recovery_ok,
            detail=(
                f"engine={'ok' if engine_clean else engine_detail}, "
                f"master residue stable={residue_ok} "
                f"({residue_detail}), recovery={recovery_msg}"
            ),
        )
        return report.finish(
            f"tombstoned uncertain delivery: sacrificial={sacrificial}, "
            f"cancel_total={cancel_total}, fence8429={fence_ok}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _restore_engines(ops)
