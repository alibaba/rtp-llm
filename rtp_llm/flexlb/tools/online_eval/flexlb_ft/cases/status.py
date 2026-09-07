"""Status-category cases: the engine→master status-report contract.

The engine→master status channel is the authoritative terminal source for
the master's inflight ledgers: WorkerStatus facts (ACTIVE / TERMINAL) both
settle request slots and refresh their stale-inflight activity clock
(RequestSlot.observeWorkerStatus → reduceStaleSlot expires a slot only when
now - lastWorkerStatusAtMs > staleInflightTimeoutMs).  This category
injects faults into exactly that channel (mock /inject "status_*" /
"enqueue_ack_*" types) and pins the CORRECT master contract, NOT the
current behaviour: assertions state what a correct master MUST do.  Former
finding probes become ordinary regression cases once their product fixes
land; this file currently has no expected-fail registrations.

Injection interface (mock side, parallel implementation; field names per
the agreed spec — do not invent alternatives):

    status_suppress_finished(bool) / status_suppress_running(bool)
    status_suppress_rids([rid...]) / status_no_respond(bool)
    status_fake_task({rid required, batchId, phase: RUNNING | KV_ALLOCATED
                      | RECEIVED | finished(+errorCode)})
    status_duplicate_finished(bool) / status_cursor_regress(int n)
    status_version_regress(bool) / status_zombie_running(bool)
    enqueue_ack_partial_fail(int k) / enqueue_ack_error_code(int code)
    enqueue_ack_drop(bool)

Shared environment (_status_spec): 2P+2D, legacy fault axes pinned via
FLEXLB_CONFIG (PRIORITY + FIXED_WINDOW + BATCH), staleInflightTimeoutMs=30s
(TTL observations cap at TTL+margin) and scheduler.queueTimeoutMs=10s —
zombie keep-alive scenarios (a suppressed-finished request keeps appearing
RUNNING, which refreshes lastWorkerStatusAtMs and disarms the stale TTL)
need a short deadline bottom line.

Master-side cleanup observability — two channels (task #103 step 2):
  * Counter channel: the master prometheus counters behind
    app.flexlb.inflight.ttl.expired.qps (role=SCHEDULER per-request slot
    sweep / role=PREFILL|DECODE per-endpoint orphan sweep), read via
    ops.master_ttl_eviction_counts() with a before/after DELTA (>= bound,
    never equality — process-cumulative counters, uncontrolled merges).
    The counter only advances on the PASSIVE stale-inflight sweep, so a
    >= bound is assertable ONLY in a case that deliberately constructs
    a stale inflight with no other exit — status_inflight_ttl_cleanup
    (scheduler role).  Cases whose ledgers drain through retire/settle
    completion paths assert ledger zero + their behaviour contract and
    only require the channel to stay REACHABLE (UNREACHABLE =
    environment failure), reporting the counter delta as observation:
    status_prefill_suppress_all, status_status_no_respond,
    status_version_regress (a retired engine's endpoint row disappears
    from the inflight view, so its ledger cleanup does not ride the TTL
    counter channel).
  * Log-anchor channel (informational only): the drained ledgers are the
    hard assertions for every other case.  Both anchors are logged via
    org.flexlb.util.Logger → logback "flexlbLogger" → the FLEXLB file
    appender (additivity=false), i.e. ~/ai-whale/logs/flexlb.log — NOT
    the JVM stdout redirect (a structural 0 there):
    event=scheduler_inflight_ttl_eviction   (ExpirationTimer.maintain)
    event=endpoint_inflight_ttl_eviction    (EndpointRegistry)

CAVEAT (event channel): the TTL maintenance sweep runs at a 60s cadence
(SchedulerRuntime.maintainExpiration, fixedRate=60s), so eviction COUNTERS
lag the eviction event — after-side reads must poll (wait_for) inside
TTL_EVENT_WINDOW_S, never sample once.  An unreachable prometheus endpoint
(ops.master_ttl_eviction_counts() is None) is an environment failure the
cases fail on, never a pass reason.

Case index (P0 = release-blocking contract, P1 = robustness, P2 = extended
contract probe):

    P0 status_ack_partial_fail          k-of-batch ack failure isolates
                                       + prompt ledger release
    P0 status_ack_partial_fail_permanent
                                       permanent 8431 fails fast, no retry
    P0 status_ack_partial_fail_transient_retry
                                       transient code 13 retries to queue SLO
    P0 status_batch_async_partial_fail  in-batch execution-phase partial
                                       failure: typed terminal + mixed-batch
                                       lease closure
    P0 status_ack_multi_error           per-request error-code passthrough
    P0 status_ack_empty_no_crash        empty ack → uncertain fence, bounded + clearable
    P0 status_prefill_suppress_all      full status silence → TTL eviction
    P0 status_prefill_suppress_finished running keep-alive → queueTimeout is the only exit
    P0 status_status_no_respond         status RPC silence → generation retirement
    P0 status_unknown_rid_finished      unknown-rid terminal ignored
    P0 status_version_regress           stale version → generation retirement
    P1 status_decode_suppress_finished       decode-side terminal suppression
    P1 status_decode_before_prefill          D terminal settles + event-driven P cleanup
    P1 status_decode_running_before_prefill  D ACTIVE-only: no premature P cleanup
    P1 status_decode_waiting_before_prefill  D RECEIVED-only: no premature P cleanup
    P1 status_unknown_rid_running       one-shot ghost running entry
    P1 status_unknown_batchid           mismatched batchId must not settle a real rid
    P1 status_special_ids               sentinel/boundary ids: rid -1/0, batch 0/-1
    P1 status_unbatched_single_request  batch-less facts, omitted/0 × RUNNING/finished
    P1 status_foreign_batchid           out-of-range batchId, real traffic unaffected
    P1 status_duplicate_finished        duplicate terminal replay idempotent
    P1 status_cursor_regress            completion cursor rewind idempotent
    P1 status_finished_then_running     terminal must not be resurrected
    P1 status_zombie_completed_running  zombie running vs tombstone
    P2 status_zombie_fake_running       persistent ghost cleanup regression

Migrated in from the legacy fault families (task #85 category reorg):

    status_inflight_ttl_cleanup         stuck inflight → TTL cleanup (S1 port)
    status_fetch_error                  batch FetchResponse fault surfacing
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Optional

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
    EnvSpec,
    _accepted,
    _fault_spec,
    build_flexlb_config,
    default_perf,
    http_get_status,
)
from ..harness import master_decode_requests_sum as _harness_decode_requests_sum
from ..harness import master_prefill_batches_sum as _harness_prefill_batches_sum
from ..harness import master_prefill_requests_sum as _harness_prefill_requests_sum
from ..harness import ttl_spec, wait_for

STATUS_CASES: list[CaseDef] = []

STREAM_TIMEOUT_S = 15.0
# > staleInflightTimeoutMs (30s): lets a TTL-eviction terminal reach the
# client stream instead of the client's own stream deadline firing first.
LONG_STREAM_TIMEOUT_S = 45.0
STALE_INFLIGHT_TTL_S = 30.0
TTL_MARGIN_S = 30.0
QUEUE_TIMEOUT_S = 10.0
# Longer than the dispatcher's maximum 500ms ACK retry backoff.  Permanent
# rejection cases hold the counter observation through this window so a late
# retry cannot slip past the assertion.
ACK_NO_RETRY_OBSERVATION_S = 1.25
# Event-driven prefill-cleanup window (decode-before-prefill contract):
# once the decode terminal has settled the request, the prefill ledger
# entry for the same member must be released by the settle path itself,
# not parked until the 30s stale TTL + 60s sweep — 10s covers several
# status-poll rounds (statusRpcMs=1s) with margin.
EVENT_DRIVEN_CLEANUP_S = 10.0
# 3-strike health demotion + eviction window (fault-family MASTER_EVICT_S
# precedent).
MASTER_EVICT_S = 30.0
# TTL-eviction EVENT window (task #103 step 2): the drain window
# (TTL_DRAIN_TIMEOUT_S = 95s, ledger-side) plus event margin — the eviction
# counters are reported by the 60s maintenance sweep
# (SchedulerRuntime.maintainExpiration) and only then become visible in the
# prometheus exposition, so the after-side read must poll for up to this
# long SEPARATELY from the drain wait.  Dov's ruling: the observation
# window must stay >= 100s (worst-phase eviction lands at TTL + a full
# sweep).
TTL_EVENT_WINDOW_S = 105.0
# Fake/ghost rid offset: far above every rid this process will hand out
# (next_request_ids stay within base + small offsets) so the master has
# never seen these ids.
GHOST_RID_OFFSET = 900_000


def case(
    name: str,
    profiles=None,
    requires=None,
    source: str = "",
    expected_fail: bool = False,
):
    """Register into STATUS_CASES (category is always "status").

    ``expected_fail=True`` declares a declared-finding probe (task #101):
    failing confirms the finding, passing resolves it — neither counts
    toward failed_count / the suite verdict / the exit code."""

    def deco(fn):
        STATUS_CASES.append(
            CaseDef(
                name=name,
                category="status",
                fn=fn,
                profiles=profiles,
                requires=requires,
                source=source,
                expected_fail=expected_fail,
            )
        )
        return fn

    return deco


# ===========================================================================
# Shared environment
# ===========================================================================


def _status_spec(ctx: CaseContext) -> EnvSpec:
    """Family env: 2P+2D, legacy fault axes, TTL=30s, queueTimeout=10s.

    queueTimeoutMs=10s is the zombie keep-alive bottom line: a request
    whose terminal is suppressed but which keeps appearing RUNNING on the
    status channel refreshes lastWorkerStatusAtMs forever, disarming the
    stale-inflight TTL — the queue/deadline path is then the ONLY legal
    exit, and it must fire quickly.
    """
    return EnvSpec(
        label=f"status_fault_{ctx.profile}",
        n_prefill=2,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={
            "FLEXLB_CONFIG": build_flexlb_config(
                ordering="priority",
                decision="fixed_window",
                dispatcher="batch",
                queue_timeout_ms=int(QUEUE_TIMEOUT_S * 1000),
                stale_inflight_ms=int(STALE_INFLIGHT_TTL_S * 1000),
            )
        },
    )


# ===========================================================================
# Shared helpers
# ===========================================================================


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


def _master_ok(ops) -> bool:
    """Master liveness probe (GET inflight_status == 200)."""
    return (
        http_get_status(f"{_master_http(ops)}/rtp_llm/inflight_status", timeout=5)
        == 200
    )


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"]


def _decode_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "decode"]


def _timeout_typed(err) -> bool:
    """Deadline/timeout-class terminal (admission_slo_queue_deadline
    keyword set plus the run_one_request "stream did not complete"
    client-timeout form)."""
    text = str(err or "").lower()
    return any(
        kw in text
        for kw in (
            "deadline",
            "timeout",
            "timed out",
            "not complete",
            "expire",
            "8511",
        )
    )


def _run_requests(
    ops,
    base: int,
    n: int,
    output_len: int = 2,
    concurrency: int = 8,
    stream_timeout_s: float = STREAM_TIMEOUT_S,
    typed_stream_error: bool = False,
) -> list:
    """Fire *n* requests (bounded concurrency); returns the per-request
    error list (None = success).  Error types are stashed on the function
    as ``last_error_types`` for failure diagnostics.
    ``typed_stream_error`` forwards to run_one_request: in-band error
    frames surface their typed code/message instead of the generic
    "stream did not complete" (execution-phase failure family)."""
    rids = [ops.next_request_id(base) for _ in range(n)]

    def run(rid: int):
        _, err = ops.run_one_request(
            rid,
            output_len=output_len,
            stream_timeout_s=stream_timeout_s,
            typed_stream_error=typed_stream_error,
        )
        return err

    with ThreadPoolExecutor(max_workers=min(n, concurrency)) as pool:
        errs = list(pool.map(run, rids))
    _run_requests.last_error_types = sorted(
        {str(e)[:70] for e in errs if e is not None}
    )
    return errs


def _recovery_rate(ops, base: int, n: int = 20) -> tuple:
    """AssertUtils.recovery_rate's >=95% contract on the direct gRPC path
    (a fresh n-request batch; driving the JavaLoadClient subprocess for
    this is overkill — the semantic is identical)."""
    errs = _run_requests(ops, base, n, concurrency=8)
    ok = sum(1 for e in errs if e is None)
    rate = ok / n if n else 0.0
    return rate >= 0.95, f"recovery {ok}/{n} ({rate:.1%})"


def _prefill_batches_sum(ops) -> int:
    """Sum of master-side prefill inflight_batches across every endpoint
    (shared caliber — see harness.master_prefill_batches_sum; kept as a
    local alias so existing status.py call sites read unchanged)."""
    return _harness_prefill_batches_sum(ops)


def _decode_requests_sum(ops) -> int:
    """Sum of master-side decode inflight_requests across every endpoint
    (shared caliber — see harness.master_decode_requests_sum)."""
    return _harness_decode_requests_sum(ops)


def _prefill_requests_sum(ops) -> int:
    """Sum of master-side prefill inflight_requests (locally-owned member
    accounting) across every endpoint — the member-count caliber behind
    the batch bookkeeping (shared caliber, harness.master_prefill_requests_sum;
    inflight_batches alone cannot expose whether a failed ack member still
    occupies the batch ledger)."""
    return _harness_prefill_requests_sum(ops)


def _enqueue_rpc_count(ops, names: list) -> int:
    """Total EnqueueBatch RPC count across the named engines (mock snapshot
    rpc_counts.enqueue_batch) — the retry-observation caliber: a master-side
    retry re-dispatches, which shows up as NEW EnqueueBatch RPCs."""
    snap = ops.snapshot_by_name()
    total = 0
    for n in names:
        counts = snap.get(n, {}).get("rpc_counts", {}) or {}
        total += int(counts.get("enqueue_batch", 0) or 0)
    return total


def _inflight_fingerprint(ops):
    """Comparable summary of /rtp_llm/inflight_status: scheduler count +
    per-endpoint (ip_port, inflight_batches, inflight_requests).  Two
    equal fingerprints mean "no ledger mutation" (the ghost-task /
    replay-idempotency assertions)."""
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


def _log_count(env, anchor: str) -> int:
    """Occurrences of *anchor* in the master's flexlbLogger file appender
    (~/ai-whale/logs/flexlb.log, shared across every master in the
    container) since OUR master started — env.flexlb_log_offset is
    recorded by harness.start_master.  The TTL-eviction anchors are
    logged via org.flexlb.util.Logger → logback "flexlbLogger" → the
    FLEXLB appender with additivity=false, so they NEVER reach the JVM
    stdout redirect (mp.log_file) the old reader watched — a structural
    0 there.  Cases take a before/after delta because the shared env
    keeps one master log across cases."""
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
            if offset > 0:
                fh.seek(offset)
            return fh.read().decode("utf-8", errors="replace").count(anchor)
    except Exception:
        return 0


def _ttl_anchor_deltas(env, before: tuple) -> tuple:
    """(scheduler_evictions, endpoint_evictions) delta since *before*."""
    return (
        _log_count(env, "event=scheduler_inflight_ttl_eviction") - before[0],
        _log_count(env, "event=endpoint_inflight_ttl_eviction") - before[1],
    )


def _ttl_eviction_delta(ops, before: dict, role: str) -> Optional[int]:
    """Prometheus TTL-eviction counter delta for *role* since *before*.

    Role keys: "scheduler" (per-request slot sweep), "prefill" / "decode"
    (per-endpoint orphan sweeps).  None means the master prometheus
    endpoint is unreachable — NOT zero evictions; the sparse-counter
    "never happened" state is a role-level None that reads as a 0 baseline
    (see engine_ops.master_ttl_eviction_counts).
    """
    after = ops.master_ttl_eviction_counts()
    if after is None:
        return None
    return int((after.get(role) or 0) - (before.get(role) or 0))


def _ttl_eviction_events(
    ops,
    before: dict,
    role: str,
    min_delta: int,
    window_s: float = TTL_EVENT_WINDOW_S,
) -> tuple:
    """wait_for the *role* TTL-eviction counter to advance by >= min_delta.

    Event channel (task #103 step 2): the master reports evictions via
    app.flexlb.inflight.ttl.expired.qps (prometheus
    flexlb_app_flexlb_inflight_ttl_expired_qps_total) at the 60s
    maintenance-sweep granularity, so the after side POLLS instead of
    sampling once.  Deliberately a >= bound, never equality: the counter
    is process-cumulative on a shared env and its merges are
    uncontrolled (residue from earlier cases can land inside this
    window), so only the lower bound carries assertion semantics.

    A persistently unreachable endpoint FAILS: a missing observability
    channel is an environment problem, not a pass reason.  A transient
    miss just keeps polling inside the window.
    """
    final_delta: Optional[int] = None

    def _delta_reached() -> bool:
        nonlocal final_delta
        delta = _ttl_eviction_delta(ops, before, role)
        if delta is None:
            return False  # transiently unreachable: keep polling
        final_delta = delta
        return delta >= min_delta

    reached = wait_for(_delta_reached, window_s, 2.0)
    if final_delta is None:
        return False, (
            f"{role}_ttl_eviction=UNREACHABLE — master prometheus endpoint "
            f"never answered within {window_s:.0f}s (observability channel "
            f"missing: environment failure, not a pass)"
        )
    return reached, f"{role}_ttl_eviction_delta={final_delta} (need>={min_delta})"


def _ttl_counter_observe(ops, before: dict, role: str) -> tuple:
    """Observational TTL-eviction counter delta for *role* — NO >= bound.

    The counter channel only advances on the PASSIVE stale-inflight sweep
    (ExpirationTimer / EndpointRegistry orphan expiry); a ledger drained
    through retire or settle completion paths clears cleanly WITHOUT
    touching it — correct semantics, not a lost eviction.  So a case that
    does not deliberately construct a stale inflight (that is
    status_inflight_ttl_cleanup's job) asserts only channel REACHABILITY
    here: a master whose prometheus endpoint never answers is an
    environment failure.  Returns (channel_ok, detail)."""
    after = ops.master_ttl_eviction_counts()
    if after is None:
        return False, (
            f"{role}_ttl_eviction=UNREACHABLE — master prometheus endpoint "
            f"never answered (observability channel missing: environment "
            f"failure, not a pass)"
        )
    delta = int((after.get(role) or 0) - (before.get(role) or 0))
    return True, f"{role}_ttl_eviction_delta={delta} (observational)"


def _fire_and_forget(ops, base: int, n: int, output_len: int = 10) -> tuple:
    """Schedule *n* requests WITHOUT consuming their streams — the master
    has enqueued the batches and the ledgers hold live entries (the
    status_inflight_ttl_cleanup precedent).  Returns (rids, error)."""
    rids: list[int] = []
    for _ in range(n):
        rid = ops.next_request_id(base)
        try:
            resp = ops.schedule(rid, output_len=output_len)
        except Exception as exc:
            return rids, f"schedule rpc failed for rid={rid}: {exc!r}"
        if resp.code != 200 or not resp.success:
            return rids, f"schedule failed for rid={rid}: {resp.error_message}"
        rids.append(rid)
    return rids, None


def _wait_scheduler_zero(ops, timeout_s: float = TTL_DRAIN_TIMEOUT_S):
    # TTL-aware default: the settle rides the 30s stale TTL PLUS the
    # ExpirationTimer's 60s sweep period (worst-phase ~90s).  The legacy
    # TTL+margin=60s default lost that race whenever the TTL expiry landed
    # in the sweeper's second half: the case itself false-FAILed on the
    # drain and the surviving residue poisoned the next case on this
    # shared env (integration-round cascade, task #87).
    return wait_for(lambda: ops.master_scheduler_inflight() == 0, timeout_s, 2.0)


def _stale_inflight_clean(ops, timeout_s: float = TTL_DRAIN_TIMEOUT_S) -> tuple:
    """Master inflight drain with the TTL-aware window (30s TTL + 60s
    ExpirationTimer sweep + margin — the worst-case settle path)."""
    return AssertUtils.inflight_clean(_master_http(ops), timeout_s)


# ===========================================================================
# Migrated from the legacy fault families: stuck-inflight TTL cleanup (S1)
# ===========================================================================


@case(
    "status_inflight_ttl_cleanup",
    profiles=["batch-window"],
    source="flexlb_behavior_test.sh S1 (stuck inflight TTL cleanup)",
)
def inflight_ttl_cleanup(ctx: CaseContext):
    """S1 port, suppressed-alive form: the TTL sweep is the ONLY exit.

    Construction (task #107 fix): the legacy kill-engine form never
    reached the TTL sweep — the 3-strike engine-death verdict fires the
    fence / failure-terminal cleanup path first (Tara's forensics:
    ttl_eviction_delta=0, zero log anchors), so the
    staleInflightTimeoutMs machinery itself went untested.  The engines
    now stay ALIVE (no stop, no fence, no death verdict): the batch's
    rids are pre-generated and the suppression is armed BEFORE the
    requests are sent, on EVERY engine of BOTH roles (prefills AND
    decodes suppress all facts for those rids via status_suppress_rids
    — no RUNNING, no finished; without the D-side arm a decode
    finished fact would settle the slots and bypass the TTL exit
    again).  The requests really reach the engines (fire-and-forget
    Schedule + the prefills' accepted counters asserted >= the wave
    size up front — fail loudly otherwise) and the slow prefills (10s)
    stretch the in-flight window, but the ledger never sees ANY worker
    fact for them: no terminal, no engine-down signal — the only exit
    left is the scheduler slot's staleInflightTimeoutMs (30s) sweep,
    reported by the 60s maintainExpiration cycle.

    Expected (contract): the suppressed wave is accepted by the engines
    (precondition asserted BEFORE the observation — a suppression that
    never reached the engines is a construction failure, not a pass);
    while the suppression holds the scheduler inflight STAYS > 0 (no
    early cleanup: the engines are alive and silence alone must not
    settle anything); the ledger then drains to zero within the
    TTL-aware window (30s TTL + 60s sweep + margin); the
    scheduler-role TTL-eviction counter advances by >= the suppressed
    population (105s event window); the fleet still serves after the
    release (recovery).

    Profile semantics (v2, task #55): ttl_spec pins the legacy fault
    axes (PRIORITY + FIXED_WINDOW + BATCH) via FLEXLB_CONFIG — the
    declaration stays batch-window (label honesty + regression
    efficiency).  fault_env_config sets no queueTimeoutMs, so the Java
    default (1h) cannot expire these requests before the TTL — the
    deadline path is not an exit here.
    """
    env = ctx.env_manager.ensure(ttl_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    pnames = _prefill_names(ops)
    dnames = _decode_names(ops)
    if not pnames or not dnames:
        return False, (f"engines missing (prefill={len(pnames)}, decode={len(dnames)})")
    all_names = pnames + dnames
    # Pre-generate the wave's rids so the suppression can be armed
    # BEFORE the requests are sent (rids are client-assigned) — no
    # RUNNING fact may leak into the ledger first.
    rids = [ops.next_request_id(base) for _ in range(6)]
    try:
        # Slow both prefills (10s) so the suppressed requests stay
        # genuinely in-flight at the engines: the in-flight window is
        # real, only its reporting is silenced.
        ops.set_perf("prefill-0", prefill_fixed_ms=10000.0)
        ops.set_perf("prefill-1", prefill_fixed_ms=10000.0)

        # Event-channel baseline (task #103 step 2), BEFORE any eviction
        # this case can cause.  Unreachable = environment failure, not a
        # pass reason.
        ttl_before = ops.master_ttl_eviction_counts()
        if ttl_before is None:
            return False, (
                "master prometheus unreachable before injection — "
                "TTL-eviction observability missing (environment failure)"
            )

        # Arm the suppression on every engine of both roles, then fire:
        # the requests travel for real, but neither role ever reports a
        # fact for them.
        inject_type_all(ops, all_names, "status_suppress_rids", rids=rids)

        # Fire-and-forget: schedule without consuming the response
        # streams — the observation target is the LEDGER.
        for rid in rids:
            resp = ops.schedule(rid, output_len=10)
            if resp.code != 200 or not resp.success:
                return False, f"schedule failed for rid={rid}: {resp.error_message}"

        # PRECONDITION (fail loudly): the suppressed wave must have
        # actually reached the engines — accepted >= 6 on the prefills —
        # otherwise the scenario degenerates into "nothing was ever
        # in-flight" and the TTL sweep would pass vacuously.
        accepted_ok = wait_for(
            lambda: _accepted(ops, "prefill-0") + _accepted(ops, "prefill-1") >= 6,
            15.0,
            0.5,
        )
        accepted_total = _accepted(ops, "prefill-0") + _accepted(ops, "prefill-1")
        if not accepted_ok:
            return False, (
                f"precondition failed: suppressed wave never reached the "
                f"engines (accepted={accepted_total} < 6 after 15s) — the "
                f"TTL scenario is vacuous, not proven"
            )
        inflight_before = ops.master_scheduler_inflight()

        # Observation window INSIDE the suppression: the engines are
        # alive and (unreported) RUNNING — worker silence alone must
        # settle nothing.  12s sits deep inside the 30s stale TTL and
        # past the 10s prefill horizon.
        time.sleep(12.0)
        inflight_held = ops.master_scheduler_inflight()

        # The ONLY exit: the TTL sweep drains the ledger (30s stale TTL
        # + 60s ExpirationTimer cycle + margin — the same worst-case
        # settle window as TTL_DRAIN_TIMEOUT_S).
        cleanup_ok = wait_for(
            lambda: ops.master_scheduler_inflight() == 0, TTL_DRAIN_TIMEOUT_S, 2.0
        )
        inflight_final = ops.master_scheduler_inflight()

        # task #103 step 2 — event-channel assertion, SEPARATE from the
        # drain above: the scheduler-level TTL-eviction counter must
        # have advanced by at least the suppressed population (105s
        # window — the 60s maintenance sweep reports the eviction with
        # lag).
        ttl_events_ok, ttl_events_detail = _ttl_eviction_events(
            ops, ttl_before, "scheduler", len(rids)
        )

        # The engines were never stopped: release the suppression and
        # prove the fleet still serves.
        clear_type_all(ops, all_names, "status_suppress_rids")
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            inflight_held > 0
            and cleanup_ok
            and inflight_final == 0
            and ttl_events_ok
            and recovery_ok
        )
        return passed, (
            f"accepted={accepted_total}/6 (precondition), "
            f"inflight_before={inflight_before}, "
            f"inflight_held_after_12s={inflight_held} (no early cleanup), "
            f"ttl_cleanup_within_{TTL_DRAIN_TIMEOUT_S:.0f}s={cleanup_ok}, "
            f"inflight_final={inflight_final}, "
            f"scheduler_ttl_evictions[{ttl_events_detail}], "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            clear_type_all(ops, all_names, "status_suppress_rids")
            ops.set_perf("prefill-0", prefill_fixed_ms=100.0)
            ops.set_perf("prefill-1", prefill_fixed_ms=100.0)
        except Exception:
            pass


# ===========================================================================
# P0 — enqueue-ack fault shapes
# ===========================================================================


def _ack_partial_spec(ctx: CaseContext) -> EnvSpec:
    """One slow prefill and a wide collection window for exact 4-member ACKs."""
    perf = default_perf()
    perf["prefill"] = {"fixed_ms": 3000.0, "scale": 1.0}
    return EnvSpec(
        label=f"status_ack_partial_{ctx.profile}",
        n_prefill=1,
        n_decode=2,
        perf=perf,
        master_profile=ctx.profile,
        master_env={
            "FLEXLB_CONFIG": build_flexlb_config(
                ordering="priority",
                decision="fixed_window",
                dispatcher="batch",
                max_requests=4,
                max_collection_wait_ms=300,
                max_predicted_execution_ms=5000,
                queue_timeout_ms=int(QUEUE_TIMEOUT_S * 1000),
                stale_inflight_ms=int(STALE_INFLIGHT_TTL_S * 1000),
            )
        },
    )


def _ack_partial_wave(ops, base: int, names: list[str], code: int) -> dict:
    """Submit four Schedule RPCs together and retain every member outcome."""
    rids = [ops.next_request_id(base) for _ in range(4)]
    ready = Barrier(len(rids) + 1)
    rpc_before = _enqueue_rpc_count(ops, names)
    accepted_before = _accepted(ops, names[0])

    def schedule_member(rid: int) -> dict:
        ready.wait(timeout=5.0)
        started = time.monotonic()
        try:
            response = ops.schedule(rid, output_len=2)
        except Exception as exc:
            return {
                "rid": rid,
                "ok": False,
                "response": None,
                "response_code": None,
                "error": repr(exc),
                "elapsed_s": time.monotonic() - started,
                "started_s": started,
            }
        ok = response.code == 200 and response.success
        return {
            "rid": rid,
            "ok": ok,
            "response": response if ok else None,
            "response_code": int(response.code),
            "error": None if ok else str(response.error_message),
            "elapsed_s": time.monotonic() - started,
            "started_s": started,
        }

    inject_type_all(ops, names, "enqueue_ack_partial_fail", k=1)
    inject_type_all(ops, names, "enqueue_ack_error_code", code=code)
    try:
        with ThreadPoolExecutor(max_workers=len(rids)) as pool:
            futures = [pool.submit(schedule_member, rid) for rid in rids]
            ready.wait(timeout=5.0)
            results = [future.result(timeout=35.0) for future in futures]
    finally:
        clear_type_all(ops, names, "enqueue_ack_partial_fail")
        clear_type_all(ops, names, "enqueue_ack_error_code")

    # Hold permanent rejections beyond the implementation's maximum retry
    # backoff.  A wrong delayed retry must be visible in rpc_delta.
    time.sleep(ACK_NO_RETRY_OBSERVATION_S if code != 13 else 0.5)
    starts = [item["started_s"] for item in results]
    return {
        "rids": rids,
        "results": results,
        "successes": [item for item in results if item["ok"]],
        "failures": [item for item in results if not item["ok"]],
        "rpc_delta": _enqueue_rpc_count(ops, names) - rpc_before,
        "accepted_delta": _accepted(ops, names[0]) - accepted_before,
        "submit_spread_ms": (max(starts) - min(starts)) * 1000.0,
    }


def _start_ack_survivor_streams(ops, successes: list[dict]) -> tuple[list, list]:
    handles = []
    errors = []
    for item in successes:
        try:
            handles.append(
                (item["rid"], ops.start_stream(item["response"], item["rid"]))
            )
        except Exception as exc:
            errors.append((item["rid"], repr(exc)))
    return handles, errors


def _finish_ack_survivor_streams(handles: list, timeout_s: float = 15.0) -> list:
    outcomes = []
    for rid, handle in handles:
        ended = handle.wait_end(timeout_s)
        completed = ended and handle.snap.completed and not handle.snap.error
        outcomes.append(
            (
                rid,
                completed,
                None if completed else handle.snap.error or "stream did not complete",
            )
        )
    return outcomes


def _ack_failure_details(failures: list[dict]) -> list[str]:
    return [
        f"rid={item['rid']} response_code={item['response_code']} "
        f"elapsed={item['elapsed_s']:.3f}s error={item['error']}"
        for item in failures
    ]


@case(
    "status_ack_partial_fail",
    profiles=["batch-window"],
    source="P0 status fault family: enqueue_ack_partial_fail(k=1) on a 4-request batch",
)
def status_ack_partial_fail(ctx: CaseContext):
    """One exact 4-member ACK loses one member without harming survivors.

    This ordinary P0 case owns the construction and isolation contract:
    four concurrent Schedule calls must become one EnqueueBatch RPC, exactly
    one permanent 8431 member must fail, the other three streams must
    complete, and the failed member must leave the prompt ledger while those
    survivors are still executing. The separate transient probe owns retry
    policy independently so it cannot hide a construction or isolation
    regression.
    """
    ops = ctx.engine_ops(ctx.env_manager.ensure(_ack_partial_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if len(names) != 1:
        return False, f"construction requires exactly one prefill, found {names}"

    try:
        wave = _ack_partial_wave(ops, base, names, code=8431)
        failures = wave["failures"]
        code_ok = len(failures) == 1 and "error_code=8431" in str(
            failures[0]["error"]
        )
        constructed = (
            len(wave["results"]) == 4
            and len(wave["successes"]) == 3
            and len(failures) == 1
            and wave["rpc_delta"] == 1
            and wave["accepted_delta"] == 3
            and code_ok
        )

        handles, stream_start_errors = _start_ack_survivor_streams(
            ops, wave["successes"]
        )
        ledger_samples = []
        release_latency_ms = None
        release_started = time.monotonic()
        deadline = release_started + 1.5
        while time.monotonic() < deadline:
            value = _prefill_requests_sum(ops)
            ledger_samples.append(value)
            if value == 3:
                release_latency_ms = (time.monotonic() - release_started) * 1000.0
                break
            time.sleep(0.05)
        prompt_released = release_latency_ms is not None
        survivor_outcomes = _finish_ack_survivor_streams(handles)
        survivor_isolated = (
            not stream_start_errors
            and len(survivor_outcomes) == 3
            and all(completed for _, completed, _ in survivor_outcomes)
        )
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        master_ok = _master_ok(ops)

        passed = (
            constructed
            and prompt_released
            and survivor_isolated
            and inflight_ok
            and master_ok
        )
        return passed, (
            f"constructed_1batch_4members={constructed} "
            f"(rpc_delta={wave['rpc_delta']}, accepted_delta="
            f"{wave['accepted_delta']}, successes={len(wave['successes'])}, "
            f"failures={_ack_failure_details(failures)}, "
            f"submit_spread_ms={wave['submit_spread_ms']:.1f}), "
            f"prompt_member_released={prompt_released} "
            f"(latency_ms={release_latency_ms}, samples={ledger_samples[:8]}), "
            f"survivor_isolation={survivor_isolated} "
            f"(start_errors={stream_start_errors}, outcomes={survivor_outcomes}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "enqueue_ack_partial_fail")
        clear_type_all(ops, names, "enqueue_ack_error_code")


@case(
    "status_ack_partial_fail_transient_retry",
    profiles=["batch-window"],
    source=(
        "P0 status fault family: transient code 13 must retry as a "
        "fresh EnqueueBatch dispatch"
    ),
)
def status_ack_partial_fail_transient_retry(ctx: CaseContext):
    """A transient partial ACK must retry instead of leaking raw code 13.

    The fault remains armed on the sole prefill, so a correct implementation
    retries to the 10-second queue SLO and returns a timeout-shaped terminal.
    The ordinary status_ack_partial_fail case separately guards construction
    and survivor isolation; this case owns the retry-to-deadline contract.
    """
    ops = ctx.engine_ops(ctx.env_manager.ensure(_ack_partial_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if len(names) != 1:
        return False, f"construction requires exactly one prefill, found {names}"

    try:
        wave = _ack_partial_wave(ops, base, names, code=13)
        handles, stream_start_errors = _start_ack_survivor_streams(
            ops, wave["successes"]
        )
        survivor_outcomes = _finish_ack_survivor_streams(handles)
        failures = wave["failures"]
        construction_guard = (
            len(wave["results"]) == 4
            and len(wave["successes"]) == 3
            and len(failures) == 1
            and wave["accepted_delta"] == 3
        )
        survivors_ok = (
            not stream_start_errors
            and len(survivor_outcomes) == len(wave["successes"])
            and all(completed for _, completed, _ in survivor_outcomes)
        )
        retry_dispatched = wave["rpc_delta"] >= 2
        raw_code_13 = [
            item
            for item in failures
            if "error_code=13" in str(item["error"])
            or "enqueue_ack_partial_fail" in str(item["error"])
        ]
        deadline_terminal = (
            len(failures) == 1
            and failures[0]["response_code"] == 8511
            and QUEUE_TIMEOUT_S - 2.0
            <= failures[0]["elapsed_s"]
            <= QUEUE_TIMEOUT_S + 5.0
            and _timeout_typed(failures[0]["error"])
        )
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        master_ok = _master_ok(ops)

        passed = (
            construction_guard
            and survivors_ok
            and retry_dispatched
            and not raw_code_13
            and deadline_terminal
            and inflight_ok
            and master_ok
        )
        return passed, (
            f"construction_guard={construction_guard} "
            f"(accepted_delta={wave['accepted_delta']}, "
            f"successes={len(wave['successes'])}, failures={len(failures)}), "
            f"retry_dispatched={retry_dispatched} "
            f"(enqueue_rpc_delta={wave['rpc_delta']}), "
            f"raw_code_13_leaked={bool(raw_code_13)}, "
            f"terminal_exact_8511_at_deadline={deadline_terminal}, "
            f"failure_details={_ack_failure_details(failures)}, "
            f"survivors_ok={survivors_ok} (outcomes={survivor_outcomes}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "enqueue_ack_partial_fail")
        clear_type_all(ops, names, "enqueue_ack_error_code")


@case(
    "status_ack_partial_fail_permanent",
    profiles=["batch-window"],
    source=(
        "P0 status fault family: permanent 8431 partial ACK terminates one "
        "member without retrying or harming survivors"
    ),
)
def status_ack_partial_fail_permanent(ctx: CaseContext):
    """A permanent 8431 partial ACK fails one member fast and never retries."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_ack_partial_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if len(names) != 1:
        return False, f"construction requires exactly one prefill, found {names}"

    try:
        wave = _ack_partial_wave(ops, base, names, code=8431)
        failures = wave["failures"]
        permanent_typed_fast = (
            len(failures) == 1
            and "error_code=8431" in str(failures[0]["error"])
            and failures[0]["elapsed_s"] <= 3.0
        )
        constructed = (
            len(wave["results"]) == 4
            and len(wave["successes"]) == 3
            and len(failures) == 1
            and wave["accepted_delta"] == 3
        )
        no_retry = wave["rpc_delta"] == 1
        handles, stream_start_errors = _start_ack_survivor_streams(
            ops, wave["successes"]
        )
        survivor_outcomes = _finish_ack_survivor_streams(handles)
        survivors_ok = (
            not stream_start_errors
            and len(survivor_outcomes) == 3
            and all(completed for _, completed, _ in survivor_outcomes)
        )
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        master_ok = _master_ok(ops)

        passed = (
            constructed
            and permanent_typed_fast
            and no_retry
            and survivors_ok
            and inflight_ok
            and master_ok
        )
        return passed, (
            f"constructed_1batch_4members={constructed} "
            f"(accepted_delta={wave['accepted_delta']}, "
            f"successes={len(wave['successes'])}), "
            f"permanent_8431_fast={permanent_typed_fast} "
            f"(failures={_ack_failure_details(failures)}), "
            f"no_retry={no_retry} (enqueue_rpc_delta={wave['rpc_delta']}), "
            f"survivors_ok={survivors_ok} (outcomes={survivor_outcomes}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "enqueue_ack_partial_fail")
        clear_type_all(ops, names, "enqueue_ack_error_code")


@case(
    "status_batch_async_partial_fail",
    profiles=["batch-window"],  # _status_spec pins the legacy fault axes
    source="P0 status fault family: prefill_async_partial_fail(k=1, code=8500) "
    "on 4 concurrent requests (execution-phase in-batch failure)",
)
def status_batch_async_partial_fail(ctx: CaseContext):
    """Scenario: 4 concurrent requests (output_len=2, concurrency=4) land on
    prefills injected with prefill_async_partial_fail k=1 code=8500: the
    FIRST non-cancelled member of every executing prefill batch fails AT
    the batch completion callback (execution phase), while the EnqueueBatch
    ack stays fully successful for all members — the execution-phase
    counterpart of status_ack_partial_fail.

    Behaviour (mock, production-aligned): the batch executes normally, then
    the injected member surfaces a TYPED terminal
    (task_info.error_code=8500, "injected prefill_async_partial_fail") on
    the same finished_task_list channel the survivors' terminals ride; the
    failed member starts no decode, leaves no output frame and no KV lease;
    the survivors hand off to decode and finish normally.

    Expectation (contract), five layers:
    1. Batch isolation: the survivors of each affected batch complete
       NORMALLY (>= 2 of 4 requests succeed).
    2. Typed failure terminal: every failed request's client-visible error
       carries the engine error code 8500 (master path:
       WorkerTerminalObservation -> "worker error code 8500" ->
       WORKER_EXECUTION_FAILED); 1 <= failed <= 2 — 4 requests form 1-2
       batches, each losing exactly k=1 member (the ack_partial_fail
       interval calibre).
    3. Ledger integrity: the master inflight ledger drains (inflight_clean)
       — a failure terminal missing or mismatching the dispatched batchId
       falls into the reconcile mismatch branch and leaks the slot, so this
       layer also covers batchId correctness. The failed member's slot
       settles immediately through the typed FAILED terminal (scheduler +
       prefill ledger drain fast), while its decode-side reservation is a
       soft hold reclaimed by the master's stale-inflight eviction
       (stale_inflight_ms TTL plus the eviction sweep's fixed-rate
       interval), so the drain window uses the relaxed TTL calibre
       (STALE_INFLIGHT_TTL_S + TTL_MARGIN_S + 60.0), not a fast path.
    4. No resurrection / duplicate settle: the master stays HTTP-200 and
       the drained ledger fingerprint stays stable across a late-terminal
       observation window.
    5. Batch lease closure: mixed-terminal batches release their dispatch
       capacity (BatchCompletionProjection counts members outcome-blind,
       the LAST member closes the lease) — a fresh healthy round of 4
       requests recovers >= 95%.

    Optional second form (prefill_fixed_ms=3000): stretching prefill
    execution past the batching window forces queued-serial batches — the
    same typed-terminal interval must hold when batch boundaries are
    unambiguous.

    Production alignment notes (ground truth survey):
    * Splitting is master-transparent: production EnqueueBatch has no
      split (each request enters FIFOScheduler as its own stream; partial
      admit / residual groups only reorganize execution) and the status
      report is a per-request terminal stream — batch_id travels unchanged
      from the stream group_id into TaskInfo. The mock models no split:
      whether the 4 requests form 1 or 2 batches, the shape above is the
      legal outcome.
    * In-batch async failure semantics: after admission an execution-phase
      failure goes stream->reportError -> the dequeue fills
      task_info.error_code / error_message -> finished_task_list,
      independent of the same batch's surviving members (no batch-level
      failure propagation); each member settles on its own.
      stream->reportError is a TWO-channel terminal — the client-visible
      half terminates the request's own output stream with an error frame
      (the mock engine delivers RpcErrorPB carrying the injected code),
      the master half rides finished_task_list into the reconcile; layer 2
      asserts the former (client error string carries 8500), layer 3 the
      latter (ledger drains through the typed FAILED terminal).
    * Master settlement: a P-side terminal fact with errorCode != 0 maps
      to Kind.FAILED -> WorkerStatusFact.terminal -> the slot's failure
      terminal (errorCode == 0 non-cancel prefill returns null — the
      success terminal is left for decode, which is exactly what the
      survivors exercise in parallel); no re-dispatch, no whole-batch
      abandon.

    Grade: P0."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"

    try:
        # ── Arm 1 (default perf): the batching window forms 1-2 batches.
        # typed_stream_error: the injected member's output stream terminates
        # with an in-band error frame (production stream->reportError's
        # client-visible half) — the failed request's error carries the
        # engine code 8500 instead of the generic stream-timeout form.
        inject_type_all(ops, names, "prefill_async_partial_fail", k=1, code=8500)
        try:
            errs = _run_requests(
                ops,
                base,
                4,
                output_len=2,
                concurrency=4,
                typed_stream_error=True,
            )
        finally:
            clear_type_all(ops, names, "prefill_async_partial_fail")
        failed = [e for e in errs if e is not None]
        ok = len(errs) - len(failed)
        typed_terminal = (
            1 <= len(failed) <= 2 and ok >= 2 and all("8500" in str(e) for e in failed)
        )
        error_shapes = sorted({str(e)[:80] for e in failed})[:3]

        # ── Layers 3+4: ledger integrity + no resurrection.
        # Relaxed window: the failed member's decode-side reservation is a
        # soft hold reclaimed by stale-inflight eviction (TTL + the sweep's
        # fixed-rate interval), so use the relaxed TTL calibre like the
        # other relaxed precedents in this file.
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), STALE_INFLIGHT_TTL_S + TTL_MARGIN_S + 60.0
        )
        fingerprint = _inflight_fingerprint(ops)
        time.sleep(3.0)  # late-terminal / duplicate-report window
        stable = _inflight_fingerprint(ops) == fingerprint
        master_ok = _master_ok(ops)

        # ── Layer 5: batch lease closure via a healthy round.
        recovery_ok, recovery_msg = _recovery_rate(ops, base, 4)

        # ── Arm 2 (prefill_fixed_ms=3000): queued-serial batches.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)
        inject_type_all(ops, names, "prefill_async_partial_fail", k=1, code=8500)
        try:
            errs2 = _run_requests(
                ops,
                base,
                4,
                output_len=2,
                concurrency=4,
                typed_stream_error=True,
            )
        finally:
            clear_type_all(ops, names, "prefill_async_partial_fail")
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        failed2 = [e for e in errs2 if e is not None]
        ok2 = len(errs2) - len(failed2)
        typed_terminal_2 = (
            1 <= len(failed2) <= 2
            and ok2 >= 2
            and all("8500" in str(e) for e in failed2)
        )
        inflight_ok2, inflight_detail2 = AssertUtils.inflight_clean(
            _master_http(ops), STALE_INFLIGHT_TTL_S + TTL_MARGIN_S + 60.0
        )

        passed = (
            typed_terminal
            and inflight_ok
            and stable
            and master_ok
            and recovery_ok
            and typed_terminal_2
            and inflight_ok2
        )
        return passed, (
            f"arm1_typed_terminal={typed_terminal} "
            f"(failed={len(failed)}, ok={ok}, shapes={error_shapes}), "
            f"arm1_inflight_clean={inflight_ok}({inflight_detail}), "
            f"no_resurrection={stable}, master_200={master_ok}, "
            f"recovery={recovery_msg}, "
            f"arm2_serial_typed_terminal={typed_terminal_2} "
            f"(failed={len(failed2)}, ok={ok2}), "
            f"arm2_inflight_clean={inflight_ok2}({inflight_detail2})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "prefill_async_partial_fail")
        for n in names:
            try:
                ops.set_perf(n, prefill_fixed_ms=100.0)
            except Exception:
                pass


@case(
    "status_ack_multi_error",
    profiles=["batch-window"],
    source="P0 status fault family: enqueue_ack_error_code with two distinct codes",
)
def status_ack_multi_error(ctx: CaseContext):
    """Scenario: two enqueue batches fail with DIFFERENT injected error
    codes (enqueue_ack_error_code 8431 then 8510).

    Behaviour: each batch's ack carries its own injected code for every
    member of that batch.

    Expectation (contract): the error code is passed through PER REQUEST —
    every failed request of batch A surfaces 8431 and every one of batch B
    surfaces 8510 in its client-visible error; the master never crashes
    (HTTP 200 throughout); the failed batches do not resurrect (no retry:
    scheduler inflight settles to zero and stays there).

    Grade: P0."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        results = []
        for code in (8431, 8510):
            inject_type_all(ops, names, "enqueue_ack_error_code", code=code)
            try:
                errs = _run_requests(ops, base, 2, concurrency=2)
            finally:
                clear_type_all(ops, names, "enqueue_ack_error_code")
            results.append((code, errs))

        # Per-request passthrough: every request fails and carries ITS
        # batch's injected code in the client-visible error text.
        passthrough = all(
            errs and all(str(code) in str(e) for e in errs) for code, errs in results
        )
        err_samples = {
            code: [str(e)[:70] for e in errs if e][:2] for code, errs in results
        }

        # No resurrection: the failed batches must not re-enter the ledger.
        settle_ok = _wait_scheduler_zero(ops, 15.0)
        time.sleep(3.0)
        stable = ops.master_scheduler_inflight() == 0
        no_resurrect = settle_ok and stable

        master_ok = _master_ok(ops)
        passed = passthrough and no_resurrect and master_ok
        return passed, (
            f"code_passthrough={passthrough} (samples={err_samples}), "
            f"no_resurrect={no_resurrect} "
            f"(settle={settle_ok}, stable_after_3s={stable}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "enqueue_ack_error_code")


@case(
    "status_ack_empty_no_crash",
    profiles=["batch-window"],
    source="P0 status fault family: enqueue_ack_drop — empty ack (dispatch-uncertain)",
)
def status_ack_empty_no_crash(ctx: CaseContext):
    """Scenario: the prefill drops the whole enqueue ack
    (enqueue_ack_drop) — the master sees an EMPTY ack for a batch it
    dispatched.

    Behaviour: the master classifies the batch dispatch-uncertain and
    installs a BATCH_ACK_UNCERTAIN engine fence.

    Expectation (contract): the fence residue stays BOUNDED and
    non-growing (reuse _fence_residue_stable), AND the quarantined entries
    are ultimately clearable — the scheduler inflight must drain to zero
    within TTL+margin.  The master itself must stay up (HTTP 200)
    throughout.  This was previously a declared finding; the request-slot
    tombstone/TTL cleanup now makes the uncertain fence clearable, so the
    contract is enforced as an ordinary regression case.

    Grade: P0."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        inject_type_all(ops, names, "enqueue_ack_drop")
        try:
            errs = _run_requests(ops, base, 4, concurrency=4)
        finally:
            clear_type_all(ops, names, "enqueue_ack_drop")

        # Dispatch-uncertain: the requests themselves end (their fate is
        # reported, not asserted); the ledger contract is below.
        failed = sum(1 for e in errs if e is not None)
        # 4 requests -> at most 4 fence entries (one slot per request).
        residue_ok, residue_detail = _fence_residue_stable(ops, 4)
        # Contract: no permanently-resident entries — TTL+margin drain.
        drained = _wait_scheduler_zero(ops)
        final = ops.master_scheduler_inflight()
        master_ok = _master_ok(ops)

        passed = residue_ok and drained and final == 0 and master_ok
        return passed, (
            f"request_fate: {4 - failed}/4 ok, "
            f"err_types={getattr(_run_requests, 'last_error_types', [])[:2]}, "
            f"fence_residue={residue_ok}({residue_detail}), "
            f"ttl_drained={drained} (final={final}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "enqueue_ack_drop")


# ===========================================================================
# P0 — status-channel suppression (3 cases)
# ===========================================================================


@case(
    "status_prefill_suppress_all",
    profiles=["batch-window"],
    source="P0 status fault family: status_suppress_running+finished on every prefill",
)
def status_prefill_suppress_all(ctx: CaseContext):
    """Scenario: every prefill suppresses BOTH the running and the finished
    facts (status_suppress_running + status_suppress_finished) — the status
    channel goes fully silent for those tasks while the requests are live.

    Behaviour: with no ACTIVE fact the slot's lastWorkerStatusAtMs freezes,
    so the stale-inflight TTL (30s) is the ONLY ledger exit; the requests
    themselves terminate (success if the data plane stays up, or a
    timeout-class terminal otherwise — both are contract-acceptable; a
    non-timeout internal error or an infinite hang is not).

    Expectation (contract): master stays HTTP 200; every request ends with
    a legal terminal (ok or timeout-typed); scheduler_inflight AND the
    prefill inflight_batches both drain to zero within TTL(30s)+margin;
    TTL eviction anchors and the prefill endpoint TTL-eviction counter
    are observational — the counter only advances on the PASSIVE
    stale-inflight sweep, while these ledgers clear through the
    retire/settle completion paths (queueTimeout deadline / data-plane
    terminal), which never touch it; the counter channel must stay
    REACHABLE (UNREACHABLE = environment failure); after the injection
    is cleared a fresh batch recovers (verify_recovery).

    Grade: P0."""
    env = ctx.env_manager.ensure(_status_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        anchors_before = (
            _log_count(env, "event=scheduler_inflight_ttl_eviction"),
            _log_count(env, "event=endpoint_inflight_ttl_eviction"),
        )
        # Event-channel baseline (task #103 step 2), before any eviction
        # this case can cause.  Unreachable = environment failure.
        ttl_before = ops.master_ttl_eviction_counts()
        if ttl_before is None:
            return False, (
                "master prometheus unreachable before injection — "
                "TTL-eviction observability missing (environment failure)"
            )
        inject_type_all(ops, names, "status_suppress_running")
        inject_type_all(ops, names, "status_suppress_finished")
        try:
            errs = _run_requests(
                ops, base, 4, concurrency=4, stream_timeout_s=LONG_STREAM_TIMEOUT_S
            )
            # Suppress stays ON so the TTL is the only cleanup path.
            sched_zero = _wait_scheduler_zero(ops)
            batches_zero = wait_for(
                lambda: _prefill_batches_sum(ops) == 0,
                TTL_DRAIN_TIMEOUT_S,
                2.0,
            )
            anchors_after = _ttl_anchor_deltas(env, anchors_before)
            # TTL counter OBSERVATION, not an assertion: the counter only
            # advances on the passive stale-inflight sweep, and these
            # ledgers clear through the retire/settle completion paths
            # (queueTimeout deadline / data-plane terminal), which never
            # touch it — the hard assertion is the drain itself
            # (sched_zero / batches_zero / final == 0).  The channel must
            # stay reachable: UNREACHABLE is an environment failure.
            ttl_channel_ok, ttl_channel_detail = _ttl_counter_observe(
                ops, ttl_before, "prefill"
            )
            # Scheduler-side delta rides the same 60s sweep — also
            # observational here; the hard scheduler assertion lives in
            # status_inflight_ttl_cleanup.
            sched_channel_ok, sched_channel_detail = _ttl_counter_observe(
                ops, ttl_before, "scheduler"
            )
        finally:
            clear_type_all(ops, names, "status_suppress_running")
            clear_type_all(ops, names, "status_suppress_finished")

        ok = sum(1 for e in errs if e is None)
        legal_terminal = all(e is None or _timeout_typed(e) for e in errs)
        bad_errs = [
            str(e)[:70] for e in errs if e is not None and not _timeout_typed(e)
        ]
        final_sched = ops.master_scheduler_inflight()
        final_batches = _prefill_batches_sum(ops)
        master_ok = _master_ok(ops)
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            legal_terminal
            and sched_zero
            and batches_zero
            and final_sched == 0
            and final_batches == 0
            and ttl_channel_ok
            and sched_channel_ok
            and master_ok
            and recovery_ok
        )
        return passed, (
            f"request_terminals: ok={ok}/4, "
            f"illegal_errors={bad_errs[:2]}, "
            f"scheduler_zero={sched_zero} (final={final_sched}), "
            f"prefill_batches_zero={batches_zero} (final={final_batches}), "
            f"ttl_anchors(sched,endp)={anchors_after}, "
            f"prefill_ttl_counter[{ttl_channel_detail}], "
            f"observability: {sched_channel_detail}, "
            f"master_200={master_ok}, recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_suppress_running")
        clear_type_all(ops, names, "status_suppress_finished")


@case(
    "status_prefill_suppress_finished",
    profiles=["batch-window"],
    source="P0 status fault family: status_suppress_finished on every prefill",
)
def status_prefill_suppress_finished(ctx: CaseContext):
    """Scenario: every prefill suppresses only the finished facts
    (status_suppress_finished) — the requests keep appearing RUNNING.

    Behaviour: the persistent RUNNING fact refreshes the slot's
    lastWorkerStatusAtMs on every poll, DISARMING the stale-inflight TTL.
    That keep-alive is allowed; consequently the queue/deadline path is
    the ONLY legal exit for the stuck ledger.

    Expectation (contract): master stays HTTP 200; every request ends
    within queueTimeout(10s)+margin with a legal terminal (success if the
    data plane stays up, or a timeout-class terminal via the deadline
    bottom line); AFTER the injection is cleared the whole ledger drains
    (inflight_clean within a TTL+margin window — the keep-alive stops, the
    frozen activity clock finally expires).

    Grade: P0."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        inject_type_all(ops, names, "status_suppress_finished")
        try:
            # queueTimeout(10s) + margin: the deadline bottom line must
            # fire well inside this window when the data plane also hangs.
            errs = _run_requests(
                ops, base, 4, concurrency=4, stream_timeout_s=QUEUE_TIMEOUT_S + 10.0
            )
        finally:
            clear_type_all(ops, names, "status_suppress_finished")

        ok = sum(1 for e in errs if e is None)
        legal_terminal = all(e is None or _timeout_typed(e) for e in errs)
        bad_errs = [
            str(e)[:70] for e in errs if e is not None and not _timeout_typed(e)
        ]
        # Clear -> keep-alive stops -> frozen clock expires -> drain.
        inflight_ok, inflight_detail = _stale_inflight_clean(ops)
        master_ok = _master_ok(ops)

        passed = legal_terminal and inflight_ok and master_ok
        return passed, (
            f"request_terminals: ok={ok}/4 within "
            f"{QUEUE_TIMEOUT_S + 10.0:.0f}s, illegal_errors={bad_errs[:2]}, "
            f"inflight_clean_after_clear={inflight_ok}({inflight_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_suppress_finished")


@case(
    "status_status_no_respond",
    profiles=["batch-window"],
    source="P0 status fault family: status_no_respond — engine stops answering the status RPC",
)
def status_status_no_respond(ctx: CaseContext):
    """Scenario: prefills stop answering the WorkerStatus poll entirely
    (status_no_respond) while batches are live in their ledgers.

    Behaviour: the health poller accumulates strikes (3 consecutive
    failures) and demotes/retires the whole engine generation; the live
    slots freeze (no ACTIVE fact), so the stale TTL reclaims them.

    Expectation (contract): the alive count DROPS within the 3-strike
    window (generation retirement); master stays HTTP 200; the scheduler
    inflight drains to zero within TTL+margin; the TTL-eviction counter
    is observational here — the generation retirement itself clears the
    ledger through the retire path, which does not ride the TTL counter
    channel (the counter only advances on the passive stale-inflight
    sweep), but the channel must stay REACHABLE (UNREACHABLE =
    environment failure); after the injection is cleared the topology
    fully recovers (alive back to 2P) and a fresh request succeeds.

    Grade: P0."""
    env = ctx.env_manager.ensure(_status_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if len(names) < 2:
        return False, "need >=2 prefill engines"
    try:
        # Slow prefills widen the in-flight window so the injection lands
        # before the engines report their terminals.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)
        rids, sched_err = _fire_and_forget(ops, base, 4)
        if sched_err:
            return False, f"could not stage live inflight: {sched_err}"

        anchors_before = (
            _log_count(env, "event=scheduler_inflight_ttl_eviction"),
            _log_count(env, "event=endpoint_inflight_ttl_eviction"),
        )
        # Event-channel baseline (task #103 step 2).  Unreachable =
        # environment failure.
        ttl_before = ops.master_ttl_eviction_counts()
        if ttl_before is None:
            return False, (
                "master prometheus unreachable before injection — "
                "TTL-eviction observability missing (environment failure)"
            )
        inject_type_all(ops, names, "status_no_respond")
        try:
            alive_dropped = wait_for(
                lambda: ops.master_alive_count("PREFILL") <= len(names) - 1,
                MASTER_EVICT_S,
                0.5,
            )
            all_retired = wait_for(
                lambda: ops.master_alive_count("PREFILL") == 0,
                MASTER_EVICT_S,
                0.5,
            )
            drained = _wait_scheduler_zero(ops)
            anchors_after = _ttl_anchor_deltas(env, anchors_before)
            # TTL counter OBSERVATION, not an assertion: the generation
            # retirement clears the ledger through the retire path, which
            # does not advance the counter (it only moves on the passive
            # stale-inflight sweep) — the hard assertion is the drain
            # itself (drained / final_sched == 0).  The channel must stay
            # reachable: UNREACHABLE is an environment failure.
            ttl_channel_ok, ttl_channel_detail = _ttl_counter_observe(
                ops, ttl_before, "scheduler"
            )
        finally:
            clear_type_all(ops, names, "status_no_respond")

        final_sched = ops.master_scheduler_inflight()
        master_ok = _master_ok(ops)
        alive_back = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= len(names),
            MASTER_EVICT_S,
            0.5,
        )
        time.sleep(2.0)  # channel reconnect settle (crash_after precedent)
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            alive_dropped
            and all_retired
            and drained
            and final_sched == 0
            and ttl_channel_ok
            and master_ok
            and alive_back
            and recovery_ok
        )
        return passed, (
            f"generation_retired={all_retired} "
            f"(alive={ops.master_alive_count('PREFILL')}), "
            f"scheduler_zero={drained} (final={final_sched}), "
            f"ttl_anchors(sched,endp)={anchors_after}, "
            f"scheduler_ttl_counter[{ttl_channel_detail}], "
            f"master_200={master_ok}, topology_recovered={alive_back}, "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_no_respond")
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass


# ===========================================================================
# P0 — ghost tasks & generation regress (2 cases)
# ===========================================================================


@case(
    "status_unknown_rid_finished",
    profiles=["batch-window"],
    source="P0 status fault family: status_fake_task(finished, unknown rid), one-shot",
)
def status_unknown_rid_finished(ctx: CaseContext):
    """Scenario: an engine reports a TERMINAL (finished, errorCode 8500)
    for a request id the master has never seen (status_fake_task, one-shot
    on the first prefill).

    Behaviour: the master's slot lookup for the ghost rid finds nothing.

    Expectation (contract): the master IGNORES the unknown-rid terminal —
    the inflight ledgers are bit-identical before vs after (fingerprint
    comparison: scheduler count + every endpoint row), no new terminal is
    produced, and the master stays HTTP 200.

    Grade: P0."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    ghost_rid = base + GHOST_RID_OFFSET
    try:
        # Clean baseline so "no mutation" is observable against zero.
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        before = _inflight_fingerprint(ops)

        inject_type(
            ops,
            names[0],
            "status_fake_task",
            rid=ghost_rid,
            phase="finished",
            errorCode=8500,
        )
        try:
            time.sleep(3.0)  # several status poll rounds
        finally:
            clear_type_all(ops, names, "status_fake_task")

        after = _inflight_fingerprint(ops)
        unchanged = before is not None and after is not None and before == after
        master_ok = _master_ok(ops)

        passed = clean0 and unchanged and master_ok
        return passed, (
            f"baseline_clean={clean0}({clean0_detail}), "
            f"ledger_unchanged={unchanged} "
            f"(before={before}, after={after}), "
            f"ghost_rid={ghost_rid}, master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")


@case(
    "status_version_regress",
    profiles=["batch-window"],
    source="P0 status fault family: status_version_regress — stale status version",
)
def status_version_regress(ctx: CaseContext):
    """Scenario: prefills keep answering the status RPC but with a
    REGRESSED version (status_version_regress) while batches are live.

    Behaviour: the master rejects the stale-version reports as invalid;
    sustained invalid reports accumulate into the health 3-strike, so the
    whole engine generation retires; the live slots freeze and the stale
    TTL reclaims them.

    Expectation (contract): the alive count DROPS (generation retirement);
    master stays HTTP 200; the scheduler inflight drains to zero within
    TTL+margin; the TTL-eviction counter is observational here — same
    rationale as status_status_no_respond (the retire path clears the
    ledger without advancing the passive-sweep counter), but the channel
    must stay REACHABLE (UNREACHABLE = environment failure).

    Grade: P0."""
    env = ctx.env_manager.ensure(_status_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)
        rids, sched_err = _fire_and_forget(ops, base, 4)
        if sched_err:
            return False, f"could not stage live inflight: {sched_err}"

        anchors_before = (
            _log_count(env, "event=scheduler_inflight_ttl_eviction"),
            _log_count(env, "event=endpoint_inflight_ttl_eviction"),
        )
        # Event-channel baseline (task #103 step 2).  Unreachable =
        # environment failure.
        ttl_before = ops.master_ttl_eviction_counts()
        if ttl_before is None:
            return False, (
                "master prometheus unreachable before injection — "
                "TTL-eviction observability missing (environment failure)"
            )
        inject_type_all(ops, names, "status_version_regress")
        try:
            alive_dropped = wait_for(
                lambda: ops.master_alive_count("PREFILL") <= len(names) - 1,
                MASTER_EVICT_S,
                0.5,
            )
            drained = _wait_scheduler_zero(ops)
            anchors_after = _ttl_anchor_deltas(env, anchors_before)
            # TTL counter OBSERVATION, not an assertion — same rationale
            # as status_status_no_respond: the retire path clears the
            # ledger without advancing the passive-sweep counter; the
            # channel must stay reachable (UNREACHABLE = environment
            # failure).
            ttl_channel_ok, ttl_channel_detail = _ttl_counter_observe(
                ops, ttl_before, "scheduler"
            )
        finally:
            clear_type_all(ops, names, "status_version_regress")

        final_sched = ops.master_scheduler_inflight()
        master_ok = _master_ok(ops)
        # Post-clear topology recovery (observational — the spec asserts
        # retirement + drain; recovery is the shared-env hygiene proof).
        alive_back = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= len(names),
            MASTER_EVICT_S,
            0.5,
        )

        passed = (
            alive_dropped
            and drained
            and final_sched == 0
            and master_ok
            and ttl_channel_ok
        )
        return passed, (
            f"generation_retired={alive_dropped} "
            f"(alive={ops.master_alive_count('PREFILL')}), "
            f"scheduler_zero={drained} (final={final_sched}), "
            f"ttl_anchors(sched,endp)={anchors_after}, "
            f"scheduler_ttl_counter[{ttl_channel_detail}], "
            f"master_200={master_ok}, topology_recovered={alive_back}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_version_regress")
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass


# ===========================================================================
# P1 — decode-side suppression & cross-role settle (4 cases)
# ===========================================================================


@case(
    "status_decode_suppress_finished",
    profiles=["batch-window"],
    source="P1 status fault family: status_suppress_finished on every decode engine",
)
def status_decode_suppress_finished(ctx: CaseContext):
    """Scenario: every DECODE engine suppresses its finished facts
    (status_suppress_finished) while prefills report normally.

    Behaviour: the prefill stage settles normally (prefill inflight_batches
    drain at the usual pace); the decode ledger keeps its entries alive via
    the still-reported ACTIVE facts (TTL disarmed).

    Expectation (contract): every request ends within deadline/TTL
    (success or a timeout-class terminal); the prefill batches drain fast
    (normal settle); AFTER the injection is cleared the decode
    inflight_requests drain to zero within a RELAXED TTL+margin window
    (the fence-exemption margin — decode-side cleanup may lag the strict
    scheduler TTL); master stays HTTP 200.

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    dnames = _decode_names(ops)
    if not dnames:
        return False, "no decode engines found"
    try:
        inject_type_all(ops, dnames, "status_suppress_finished")
        try:
            errs = _run_requests(
                ops, base, 4, concurrency=4, stream_timeout_s=LONG_STREAM_TIMEOUT_S
            )
            # Prefill stage settles normally — fast drain while the
            # injection is still ON.
            p_batches_zero = wait_for(lambda: _prefill_batches_sum(ops) == 0, 20.0, 1.0)
        finally:
            clear_type_all(ops, dnames, "status_suppress_finished")

        ok = sum(1 for e in errs if e is None)
        legal_terminal = all(e is None or _timeout_typed(e) for e in errs)
        bad_errs = [
            str(e)[:70] for e in errs if e is not None and not _timeout_typed(e)
        ]
        # Relaxed window: TTL + margin + the fence-exemption margin.
        d_requests_zero = wait_for(
            lambda: _decode_requests_sum(ops) == 0,
            STALE_INFLIGHT_TTL_S + TTL_MARGIN_S + 60.0,
            2.0,
        )
        final_d = _decode_requests_sum(ops)
        master_ok = _master_ok(ops)

        passed = (
            legal_terminal
            and p_batches_zero
            and d_requests_zero
            and final_d == 0
            and master_ok
        )
        return passed, (
            f"request_terminals: ok={ok}/4, illegal_errors={bad_errs[:2]}, "
            f"prefill_settled={p_batches_zero}, "
            f"decode_drained={d_requests_zero} (final={final_d}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, dnames, "status_suppress_finished")


@case(
    "status_decode_before_prefill",
    profiles=["batch-window"],
    source="P1 status fault family: status_suppress_rids(full batch) on prefills, decodes normal",
)
def status_decode_before_prefill(ctx: CaseContext):
    """Scenario (finished arm of the decode-before-prefill matrix): the
    prefills suppress ALL facts for the whole batch's rids
    (status_suppress_rids on pre-generated ids) while the decodes report
    normally — the decode side delivers the terminal.

    Behaviour: the prefill ledger never hears about these rids; the decode
    finished fact settles the request slots.

    Expectation (contract — "a D terminal drives the P cleanup, no TTL
    waiting"): under P/D separation a decode-side finished fact settles
    the request — all 4 requests reach a SUCCESSFUL terminal — AND the
    prefill inflight_batches for those members are released by the
    settle path itself within a short event-driven window (<= 10s,
    several status-poll rounds — the master already knows the request is
    finished, so no TTL wait is justified); master stays HTTP 200 (no
    crash from the cross-role settle).

    Regression contract: counterpart cleanup applies to both
    ROUTE_DECISION and BATCH_ENQUEUE delivery claims.  It uses the exact
    ScheduledRequest identity, so a stale decode terminal cannot release a
    replacement generation that reused the same request id.  The TTL fallback
    observation below remains diagnostic only; it is not accepted as a pass.

    Grade: P1."""
    env = ctx.env_manager.ensure(_status_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    # Pre-generate the batch's rids so the suppression can be armed
    # BEFORE the requests are sent (rids are client-assigned).
    rids = [ops.next_request_id(base) for _ in range(4)]
    try:
        anchors_before = (
            _log_count(env, "event=scheduler_inflight_ttl_eviction"),
            _log_count(env, "event=endpoint_inflight_ttl_eviction"),
        )
        inject_type_all(ops, names, "status_suppress_rids", rids=rids)
        try:

            def run(rid: int):
                _, err = ops.run_one_request(
                    rid, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
                )
                return err

            with ThreadPoolExecutor(max_workers=4) as pool:
                errs = list(pool.map(run, rids))
            # Event-driven window: every stream above has completed, so
            # the decode terminal has ALREADY settled the requests — the
            # frozen prefill entries must now be released by the settle
            # path itself, not parked until the stale TTL.
            p_batches_fast = wait_for(
                lambda: _prefill_batches_sum(ops) == 0,
                EVENT_DRIVEN_CLEANUP_S,
                1.0,
            )
            # TTL fallback observation (NOT in `passed`): the entries must
            # at least expire — a permanent hang is a separate bug.
            p_batches_ttl = p_batches_fast or wait_for(
                lambda: _prefill_batches_sum(ops) == 0,
                TTL_DRAIN_TIMEOUT_S,
                2.0,
            )
            anchors_after = _ttl_anchor_deltas(env, anchors_before)
        finally:
            clear_type_all(ops, names, "status_suppress_rids")

        ok = sum(1 for e in errs if e is None)
        err_kinds = sorted({str(e)[:70] for e in errs if e})[:3]
        final_p = _prefill_batches_sum(ops)
        master_ok = _master_ok(ops)
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            ok == 4 and p_batches_fast and final_p == 0 and master_ok and recovery_ok
        )
        return passed, (
            f"requests_succeeded_via_decode_terminal={ok}/4 "
            f"(err_kinds={err_kinds}), "
            f"prefill_event_drained={p_batches_fast} "
            f"(<= {EVENT_DRIVEN_CLEANUP_S:.0f}s), "
            f"ttl_fallback_drained={p_batches_ttl} (final={final_p}), "
            f"ttl_anchors(sched,endp)={anchors_after}, "
            f"master_200={master_ok}, recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_suppress_rids")


@case(
    "status_decode_running_before_prefill",
    profiles=["batch-window"],
    source=(
        "decode-before-prefill matrix, RUNNING arm: decode ACTIVE-only "
        "(status_suppress_finished on decodes) + full prefill suppression "
        "(status_suppress_rids)"
    ),
)
def status_decode_running_before_prefill(ctx: CaseContext):
    """Scenario (RUNNING arm of the decode-before-prefill matrix): the
    prefills suppress ALL facts for the batch's rids (status_suppress_rids)
    AND every decode engine suppresses its finished facts
    (status_suppress_finished) — the decode side reports the requests as
    RUNNING (real ACTIVE facts while executing, then silence once the
    mock finishes) but never a terminal.

    Behaviour: with the D-side terminal absent, the master's live facts
    for these rids are the short-lived decode ACTIVE reports and then
    nothing; the prefill ledger holds the frozen batch entries.

    Expectation (contract — "D intermediate states drive NOTHING"): while
    no D terminal exists the P entries must NOT be cleaned early — an
    ACTIVE-fact-driven or silence-driven early cleanup would corrupt the
    bookkeeping (cleanup is terminal-driven only; the finished arm pins
    the positive direction, this arm pins the negative) — and the
    requests must stay unsettled (scheduler inflight keeps them); after
    the injections are cleared the frozen slots converge to zero within
    the TTL-aware window (no permanent hang); the master stays HTTP 200.

    Grade: P1."""
    env = ctx.env_manager.ensure(_status_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    pnames = _prefill_names(ops)
    dnames = _decode_names(ops)
    if not pnames or not dnames:
        return False, (f"engines missing (prefill={len(pnames)}, decode={len(dnames)})")
    rids = [ops.next_request_id(base) for _ in range(4)]
    try:
        inject_type_all(ops, pnames, "status_suppress_rids", rids=rids)
        inject_type_all(ops, dnames, "status_suppress_finished")
        try:
            # Schedule without consuming streams: the observation target is
            # the LEDGER, not the client outcome (the terminal is absent by
            # construction, so the streams would only hang).
            for rid in rids:
                resp = ops.schedule(rid, output_len=2)
                if resp.code != 200 or not resp.success:
                    return False, (
                        f"schedule failed for rid={rid}: {resp.error_message}"
                    )
            dispatched = wait_for(lambda: _prefill_batches_sum(ops) > 0, 15.0, 0.5)
            # Observation window: the decode execution finishes
            # (sub-second) and its ACTIVE facts disappear — the P entries
            # must survive BOTH the ACTIVE period and the silence after.
            time.sleep(10.0)
            p_batches_held = _prefill_batches_sum(ops)
            p_held = p_batches_held > 0
            sched_held = ops.master_scheduler_inflight() > 0
        finally:
            clear_type_all(ops, pnames, "status_suppress_rids")
            clear_type_all(ops, dnames, "status_suppress_finished")

        # After release: the suppressed decode terminal is permanently lost
        # (the completion-cursor head-trim ran under the injection), so the
        # frozen slots must expire via the stale TTL within the TTL-aware
        # window — convergence, not a hang.
        drained, drained_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )
        master_ok = _master_ok(ops)

        passed = dispatched and p_held and sched_held and drained and master_ok
        return passed, (
            f"dispatched={dispatched}, "
            f"p_entries_held_without_d_terminal={p_held} "
            f"(batches={p_batches_held}), "
            f"requests_unsettled={sched_held}, "
            f"post_release_drained={drained}({drained_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, pnames, "status_suppress_rids")
        clear_type_all(ops, dnames, "status_suppress_finished")


@case(
    "status_decode_waiting_before_prefill",
    profiles=["batch-window"],
    source=(
        "decode-before-prefill matrix, waiting arm: decode reports an "
        "early-phase RECEIVED fact only (status_suppress_rids + "
        "status_fake_task(RECEIVED) on decodes) while prefills suppress "
        "the whole batch (status_suppress_rids)"
    ),
)
def status_decode_waiting_before_prefill(ctx: CaseContext):
    """Scenario (waiting arm of the decode-before-prefill matrix): the
    prefills suppress ALL facts for the batch's rids AND every decode
    engine suppresses the real facts for those rids
    (status_suppress_rids on both roles), while a synthetic EARLY-PHASE
    fact per rid (status_fake_task, phase=RECEIVED) is appended on every
    decode — the decode side reports the requests as
    received-but-never-executing, forever, with no terminal.

    Behaviour: the only D-side fact the master ever sees for these rids
    is a RECEIVED intermediate (the queued/waiting shape); the prefill
    ledger holds the frozen batch entries.

    Expectation (contract — "D intermediate states drive NOTHING"): the
    RECEIVED fact must neither settle the requests (no early client
    terminal) nor release the P entries early (the prefill ledger stays
    populated while the synthetic fact streams — the finished arm pins
    the terminal-driven direction, this arm pins the
    intermediate-driven prohibition); after the injections are cleared
    the synthetic fact stops and the frozen slots converge to zero
    within the TTL-aware window; the master stays HTTP 200.

    Grade: P1."""
    env = ctx.env_manager.ensure(_status_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    pnames = _prefill_names(ops)
    dnames = _decode_names(ops)
    if not pnames or not dnames:
        return False, (f"engines missing (prefill={len(pnames)}, decode={len(dnames)})")
    rids = [ops.next_request_id(base) for _ in range(4)]
    try:
        inject_type_all(ops, pnames, "status_suppress_rids", rids=rids)
        inject_type_all(ops, dnames, "status_suppress_rids", rids=rids)
        # Synthetic RECEIVED per rid on every decode (fake facts are
        # appended AFTER the rid-suppression filter — mock
        # applyStatusReportFaults ordering — so the real facts are dropped
        # while the synthetic early-phase fact streams on every poll).
        for dname in dnames:
            for rid in rids:
                inject_type(ops, dname, "status_fake_task", rid=rid, phase="RECEIVED")
        try:
            for rid in rids:
                resp = ops.schedule(rid, output_len=2)
                if resp.code != 200 or not resp.success:
                    return False, (
                        f"schedule failed for rid={rid}: {resp.error_message}"
                    )
            dispatched = wait_for(lambda: _prefill_batches_sum(ops) > 0, 15.0, 0.5)
            # Observation window: the RECEIVED facts stream on every poll —
            # the P entries must survive them (no intermediate cleanup) and
            # the requests must stay unsettled (no intermediate settle).
            time.sleep(10.0)
            p_batches_held = _prefill_batches_sum(ops)
            p_held = p_batches_held > 0
            sched_held = ops.master_scheduler_inflight() > 0
        finally:
            clear_type_all(ops, pnames, "status_suppress_rids")
            clear_type_all(ops, dnames, "status_suppress_rids")
            clear_type_all(ops, dnames, "status_fake_task")

        # After release: the real decode terminal is permanently lost (the
        # completion-cursor head-trim ran under the suppression), so the
        # frozen slots must expire via the stale TTL within the TTL-aware
        # window — convergence, not a hang.
        drained, drained_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )
        master_ok = _master_ok(ops)

        passed = dispatched and p_held and sched_held and drained and master_ok
        return passed, (
            f"dispatched={dispatched}, "
            f"p_entries_held_under_received={p_held} "
            f"(batches={p_batches_held}), "
            f"requests_unsettled={sched_held}, "
            f"post_release_drained={drained}({drained_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, pnames, "status_suppress_rids")
        clear_type_all(ops, dnames, "status_suppress_rids")
        clear_type_all(ops, dnames, "status_fake_task")


# ===========================================================================
# P1 — ghost / mismatched task reports (2 cases)
# ===========================================================================


@case(
    "status_unknown_rid_running",
    profiles=["batch-window"],
    source="P1 status fault family: status_fake_task(running, unknown rid), one-shot",
)
def status_unknown_rid_running(ctx: CaseContext):
    """Scenario: an engine reports a RUNNING fact for a request id the
    master has never seen (status_fake_task, one-shot on the first
    prefill).

    Behaviour: correct masters ignore the ghost ACTIVE (no slot exists);
    an implementation that registers it creates a resident ghost entry —
    which the stale TTL must still reclaim once the one-shot report stops.

    Expectation (contract): the master stays HTTP 200 and the scheduler
    inflight is ZERO within TTL+margin after the one-shot injection (both
    implementations converge to zero — the contract forbids a permanent
    ghost resident either way).

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    ghost_rid = base + GHOST_RID_OFFSET + 1
    try:
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        before = _inflight_fingerprint(ops)

        inject_type(ops, names[0], "status_fake_task", rid=ghost_rid, phase="RUNNING")
        try:
            time.sleep(3.0)  # the one-shot report has landed
        finally:
            clear_type_all(ops, names, "status_fake_task")

        # Did the ghost register? (observational — either answer must
        # still converge to zero below).
        peak = _inflight_fingerprint(ops)
        registered = peak is not None and peak != before
        drained = _wait_scheduler_zero(ops)
        final = ops.master_scheduler_inflight()
        master_ok = _master_ok(ops)

        passed = clean0 and drained and final == 0 and master_ok
        return passed, (
            f"baseline_clean={clean0}({clean0_detail}), "
            f"ghost_registered={registered} "
            f"(before={before}, after_injection={peak}), "
            f"scheduler_zero_within_ttl={drained} (final={final}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")


@case(
    "status_unknown_batchid",
    profiles=["batch-window"],
    source="P1 status fault family: status_fake_task(real rid + fake batchId, finished), concurrent with live traffic",
)
def status_unknown_batchid(ctx: CaseContext):
    """Scenario: while a real request is in flight, the engine also reports
    a TERMINAL (finished, errorCode 8500) for that SAME rid but under a
    batchId the master never issued (status_fake_task), concurrent with a
    normal control request.

    Behaviour: the fake fact carries the real rid but a mismatched batchId
    — a settle keyed only on rid would kill the live request.

    Expectation (contract): the real member's settlement is UNAFFECTED —
    the targeted request completes successfully, the concurrent control
    request completes successfully, and the master stays HTTP 200.  (An
    implementation that settles on rid alone fails here — that failure is
    the finding.)

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    addr_map = ops.addr_to_name()
    fake_batch_id = 987_654_321
    # Widen the in-flight window so the injection lands mid-execution.
    for n in names:
        ops.set_perf(n, prefill_fixed_ms=3000.0)
    try:
        # Control request first: the env is healthy before the probe.
        control_rid = ops.next_request_id(base)
        _, control_err = ops.run_one_request(
            control_rid, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
        )
        if control_err:
            return False, f"control request failed: {control_err}"

        target_rid = ops.next_request_id(base)
        response = ops.schedule(target_rid, output_len=2)
        if response.code != 200 or not response.success:
            return False, (f"target schedule failed: {response.error_message}")
        landing = addr_map.get(f"{ops.role_addr(response, 'PREFILL')}", names[0])
        # The fake terminal races the real execution (3s prefill window).
        inject_type(
            ops,
            landing,
            "status_fake_task",
            rid=target_rid,
            batchId=fake_batch_id,
            phase="finished",
            errorCode=8500,
        )
        try:
            input_pb = (
                None
                if response.enqueued_by_master
                else ops.build_generate_input(target_rid, output_len=2)
            )
            handle = ops.start_stream(response, target_rid, input_pb=input_pb)
            handle.wait_end(STREAM_TIMEOUT_S)
            if handle.snap.error:
                target_err = str(handle.snap.error)
            elif not handle.snap.completed:
                target_err = "stream did not complete"
            else:
                target_err = None
        finally:
            clear_type_all(ops, names, "status_fake_task")

        # Window-insufficient instability fix: the settle here rides the
        # same stale-TTL + ExpirationTimer physical drain (worst ~90s) as
        # every other residue contract — 30s let a normal slow drain read
        # as a FAIL.  Aligned to the TTL_DRAIN_TIMEOUT_S standard; the
        # all-zero assertion itself is unchanged (a true leak still
        # times out and fails).
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )
        master_ok = _master_ok(ops)

        passed = (
            control_err is None and target_err is None and inflight_ok and master_ok
        )
        return passed, (
            f"control_ok={control_err is None}, "
            f"target_settled_normally={target_err is None}"
            f"{'' if target_err is None else ' err=' + target_err[:80]}, "
            f"landing={landing}, fake_batchId={fake_batch_id}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass


# ===========================================================================
# P1 — unknown-id defense matrix (id-space robustness, 3 cases)
# ===========================================================================


@case(
    "status_special_ids",
    profiles=["batch-window"],
    source=(
        "unknown-id defense matrix: sentinel/boundary ids "
        "(rid -1 / rid 0 / batch_id 0 / batch_id -1) on the fake-task channel"
    ),
)
def status_special_ids(ctx: CaseContext):
    """Scenario: the engine reports fake status facts whose id coordinates
    sit on the boundary of the id space — rid=-1 (negative ghost), and a
    REAL in-flight rid paired with batch_id=0 (the unbatched sentinel) /
    batch_id=-1 (negative), injected one variant at a time
    (status_fake_task, snake_case fields per the mock contract).

    Behaviour: the negative ids exercise lookup paths that an id-as-index
    or id-as-hash implementation might mishandle (negative hash,
    index underflow, sentinel-as-wildcard); the batch-id variants attack
    the settle key of a LIVE request — a settle that treats batch_id=0 as
    "no batch context, match on rid alone" would kill the live member.

    Expectation (contract): the ghost variant leaves the ledgers
    bit-identical (fingerprint comparison); the real-rid variants do NOT
    settle the live request (it completes successfully); the master stays
    HTTP 200 throughout.  The rid=0 variant CANNOT be built: the mock
    injection channel uses 0 as its missing-field sentinel and rejects
    rid=0 with 400 "requires 'rid'" — the case verifies and records that
    infrastructure limit instead of asserting master behaviour for it.

    NOTE (semantic ambiguity, batch_id=0): batch_id=0 may be a protocol
    sentinel meaning "no batch context" rather than a literal batch id.
    The contract still holds: a fact without batch context must be
    IGNORED, never treated as a wildcard that settles any live member.
    (Relation to status_unknown_batchid: that case's camelCase batchId
    kwarg is silently ignored by the mock's snake_case reader, so its
    effective form has also been batch_id=0 — this case makes that
    sentinel form explicit and adds the negative batch_id=-1 arm.)

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    addr_map = ops.addr_to_name()
    # Widen the in-flight window so the batch-id variants land mid-execution
    # (the status_fake_task racing pattern of status_unknown_batchid).
    for n in names:
        ops.set_perf(n, prefill_fixed_ms=3000.0)
    try:
        # Clean baseline so "no mutation" is observable against zero.
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)

        # ── Variant 1: rid=-1 (negative ghost id), terminal form. ──
        before = _inflight_fingerprint(ops)
        inject_type(
            ops,
            names[0],
            "status_fake_task",
            rid=-1,
            phase="finished",
            error_code=8500,
        )
        try:
            time.sleep(3.0)  # several status poll rounds
        finally:
            clear_type_all(ops, names, "status_fake_task")
        after_neg = _inflight_fingerprint(ops)
        rid_neg_ignored = (
            before is not None and after_neg is not None and before == after_neg
        )

        # ── Variant 2: rid=0 — the injection channel itself is expected to
        # refuse (missing-field sentinel), so the master-side defence for
        # rid=0 is NOT observable through this channel: recorded, not
        # asserted.  An accepted rid=0 would be a channel-behaviour change.
        rid_zero_note = "channel accepted rid=0 (unexpected — sentinel changed?)"
        try:
            inject_type(ops, names[0], "status_fake_task", rid=0, phase="finished")
            clear_type_all(ops, names, "status_fake_task")
        except RuntimeError as exc:
            rid_zero_note = f"channel refused rid=0 (expected): {str(exc)[:70]}"

        # ── Variants 3+4: real rid + sentinel/negative batch_id, racing a
        # live execution — the settle key must reject both forms. ──
        variant_results = []
        for label, batch_id in (("batch_id=0", 0), ("batch_id=-1", -1)):
            target_rid = ops.next_request_id(base)
            response = ops.schedule(target_rid, output_len=2)
            if response.code != 200 or not response.success:
                return False, (f"{label} schedule failed: {response.error_message}")
            landing = addr_map.get(f"{ops.role_addr(response, 'PREFILL')}", names[0])
            inject_type(
                ops,
                landing,
                "status_fake_task",
                rid=target_rid,
                batch_id=batch_id,
                phase="finished",
                error_code=8500,
            )
            try:
                input_pb = (
                    None
                    if response.enqueued_by_master
                    else ops.build_generate_input(target_rid, output_len=2)
                )
                handle = ops.start_stream(response, target_rid, input_pb=input_pb)
                # Deliver the malformed fact while the 3s Prefill is live, then
                # remove it before the real completion. Keeping a duplicate rid
                # through completion overwrites the real terminal in the
                # response-local id map and tests cursor loss instead of the
                # special batch-id lookup contract.
                time.sleep(1.0)
                clear_type_all(ops, names, "status_fake_task")
                handle.wait_end(STREAM_TIMEOUT_S)
                if handle.snap.error:
                    target_err = str(handle.snap.error)
                elif not handle.snap.completed:
                    target_err = "stream did not complete"
                else:
                    target_err = None
            finally:
                clear_type_all(ops, names, "status_fake_task")
            variant_results.append((label, batch_id, target_err))

        real_rid_unaffected = all(err is None for _, _, err in variant_results)

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        master_ok = _master_ok(ops)

        passed = (
            clean0
            and rid_neg_ignored
            and real_rid_unaffected
            and inflight_ok
            and master_ok
        )
        variant_bits = ", ".join(
            f"{label}(err={'none' if err is None else err[:50]})"
            for label, _, err in variant_results
        )
        return passed, (
            f"baseline_clean={clean0}({clean0_detail}), "
            f"rid_neg_ignored={rid_neg_ignored} "
            f"(before={before}, after={after_neg}), "
            f"rid=0: {rid_zero_note}, "
            f"real_rid_unaffected={real_rid_unaffected} "
            f"[{variant_bits}], "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass


@case(
    "status_unbatched_single_request",
    profiles=["batch-window"],
    source=(
        "unknown-id defense matrix: unbatched single-request facts "
        "(batch_id omitted vs 0, RUNNING vs finished)"
    ),
)
def status_unbatched_single_request(ctx: CaseContext):
    """Scenario: the engine reports isolated single-request facts with NO
    batch context — batch_id omitted vs batch_id=0 explicitly, each in
    both a RUNNING and a finished phase (status_fake_task, four
    combinations injected one at a time, each on a fresh never-dispatched
    rid far above the live id space).

    Behaviour: in BATCH dispatch every legitimate status fact carries the
    batch context the master issued; an unbatched fact is malformed input
    from the master's bookkeeping perspective (its rid is unknown too).

    Expectation (contract): EVERY combination is a no-op for the ledgers —
    the fingerprint is bit-identical before vs after each injection window
    (no ghost registration on RUNNING, no phantom settle on finished); the
    master stays HTTP 200 and the ledger is clean at the end.

    Distinction from status_unknown_rid_finished / _running (same channel,
    default batch_id): those pin the single-phase default form; this case
    makes the unbatched semantics EXPLICIT (batch_id=0 declared, omitted
    field declared) and matrixes both phases under it — if the mock's
    default batch_id ever changes, the omitted arms drift away from the
    explicit-0 arms and this case surfaces it.

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)

        combos = []
        for i, (label, params) in enumerate(
            (
                ("omitted/RUNNING", {"phase": "RUNNING"}),
                ("omitted/finished", {"phase": "finished", "error_code": 8500}),
                ("batch0/RUNNING", {"batch_id": 0, "phase": "RUNNING"}),
                (
                    "batch0/finished",
                    {"batch_id": 0, "phase": "finished", "error_code": 8500},
                ),
            )
        ):
            rid = base + GHOST_RID_OFFSET + 210 + i
            before = _inflight_fingerprint(ops)
            inject_type(ops, names[0], "status_fake_task", rid=rid, **params)
            try:
                time.sleep(3.0)  # several status poll rounds
            finally:
                clear_type_all(ops, names, "status_fake_task")
            after = _inflight_fingerprint(ops)
            combos.append(
                (
                    label,
                    before is not None and after is not None and before == after,
                )
            )

        all_noop = all(ok for _, ok in combos)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 20.0
        )
        master_ok = _master_ok(ops)

        passed = clean0 and all_noop and inflight_ok and master_ok
        combo_bits = ", ".join(
            f"{label}={'noop' if ok else 'MUTATED'}" for label, ok in combos
        )
        return passed, (
            f"baseline_clean={clean0}({clean0_detail}), "
            f"all_noop={all_noop} [{combo_bits}], "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")


@case(
    "status_foreign_batchid",
    profiles=["batch-window"],
    source=(
        "unknown-id defense matrix: out-of-range batchId "
        "(a foreign dispatch space) with concurrent real traffic"
    ),
)
def status_foreign_batchid(ctx: CaseContext):
    """Scenario: the engine reports a finished fact for a ghost rid under
    a batchId far outside this master's dispatch space (10_000_000 — ids a
    SECOND master sharing the same engines would issue), and the injection
    stays armed while real traffic flows concurrently.

    Behaviour: a shared-engine misconfiguration makes the engine's status
    stream interleave facts the local master never dispatched — both the
    rid and the batchId are unknown in the local dispatch space.

    Expectation (contract): the foreign fact is ignored wholesale — the
    ledger fingerprint is unchanged across the ghost window; the
    concurrent REAL requests all succeed (none is settled by the foreign
    batchId — no cross-space aliasing); the ledger drains and the master
    stays HTTP 200.

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    ghost_rid = base + GHOST_RID_OFFSET + 220
    foreign_batch_id = 10_000_000
    try:
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        before = _inflight_fingerprint(ops)

        # Arm the foreign fact on every prefill (a shared-engine
        # misconfiguration would surface on any of them) and let it ride
        # through several status polls BEFORE the real traffic starts.
        inject_type_all(
            ops,
            names,
            "status_fake_task",
            rid=ghost_rid,
            batch_id=foreign_batch_id,
            phase="finished",
            error_code=8500,
        )
        try:
            time.sleep(3.0)
            ghost_after = _inflight_fingerprint(ops)
            # Real traffic concurrent with the still-armed foreign fact:
            # no real member may be settled by the out-of-range batchId.
            errs = _run_requests(ops, base, 4, concurrency=4)
        finally:
            clear_type_all(ops, names, "status_fake_task")

        ghost_ignored = (
            before is not None and ghost_after is not None and before == ghost_after
        )
        failed = [e for e in errs if e is not None]
        real_unaffected = not failed
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        master_ok = _master_ok(ops)

        passed = (
            clean0 and ghost_ignored and real_unaffected and inflight_ok and master_ok
        )
        return passed, (
            f"baseline_clean={clean0}({clean0_detail}), "
            f"foreign_ignored={ghost_ignored} "
            f"(before={before}, after_ghost={ghost_after}), "
            f"ghost_rid={ghost_rid}, foreign_batchId={foreign_batch_id}, "
            f"real_unaffected={real_unaffected} (failed={len(failed)}/4, "
            f"kinds={sorted({str(e)[:60] for e in failed})[:2]}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")


# ===========================================================================
# P1 — terminal replay & rewind idempotency (3 cases)
# ===========================================================================


@case(
    "status_duplicate_finished",
    profiles=["batch-window"],
    source="P1 status fault family: status_duplicate_finished — same terminal reported twice",
)
def status_duplicate_finished(ctx: CaseContext):
    """Scenario: the engine replays its finished reports
    (status_duplicate_finished) while a batch of requests completes.

    Behaviour: the master receives the SAME terminal fact more than once
    per request.

    Expectation (contract): replay is idempotent — no double settle, no
    ledger resurrection: after the batch completes the inflight
    fingerprint stays BIT-IDENTICAL across the replay window and the
    ledger stays clean; master stays HTTP 200.  (Mock-side completed /
    terminal counters are recorded as observables.)

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        inject_type_all(ops, names, "status_duplicate_finished")
        try:
            errs = _run_requests(ops, base, 4, concurrency=4)
            baseline_clean, baseline_detail = AssertUtils.inflight_clean(
                _master_http(ops), 30.0
            )
            # Fingerprint pair taken INSIDE the replay window (the injection
            # is still armed) — a clear-then-compare pair would only observe
            # the post-injection calm and never the replay itself.
            before = _inflight_fingerprint(ops)
            time.sleep(5.0)  # replay window: terminals re-delivered
            after = _inflight_fingerprint(ops)
        finally:
            clear_type_all(ops, names, "status_duplicate_finished")

        ok = sum(1 for e in errs if e is None)
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        stable = before is not None and after is not None and before == after

        # Mock-side observables (field names defensive — recorded, not
        # asserted; the master-side fingerprint IS the idempotency proof).
        snap = ops.snapshot_by_name()
        mock_obs = {
            n: {
                f: snap.get(n, {}).get(f)
                for f in ("completed", "terminal", "finished")
                if f in snap.get(n, {})
            }
            for n in names
        }
        master_ok = _master_ok(ops)

        passed = ok == 4 and baseline_clean and clean_ok and stable and master_ok
        return passed, (
            f"requests_ok={ok}/4, "
            f"baseline_clean={baseline_clean}({baseline_detail}), "
            f"inflight_clean={clean_ok}({clean_detail}), "
            f"replay_ledger_stable={stable} (fp_before={before}, "
            f"fp_after={after}), mock_counters={mock_obs}, "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_duplicate_finished")


@case(
    "status_cursor_regress",
    profiles=["batch-window"],
    source="P1 status fault family: status_cursor_regress(3) — completion cursor rewinds",
)
def status_cursor_regress(ctx: CaseContext):
    """Scenario: after three requests complete, the engine's completion
    cursor REGRESSES by 3 (status_cursor_regress) — the three old
    completion facts are re-delivered.

    Behaviour: the master sees stale terminals for requests it has already
    settled.

    Expectation (contract): old-completion re-delivery is idempotent — no
    duplicate terminals, no ledger resurrection: the inflight fingerprint
    stays bit-identical across the rewind window, the ledger stays clean,
    master stays HTTP 200, and a fresh request still succeeds.

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        # Build completion history first (the cursor rewinds over these).
        errs0 = _run_requests(ops, base, 3, concurrency=3)
        ok0 = sum(1 for e in errs0 if e is None)
        if ok0 < 3:
            return False, (
                f"history batch failed ({ok0}/3): "
                f"{getattr(_run_requests, 'last_error_types', [])[:2]}"
            )
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)

        inject_type_all(ops, names, "status_cursor_regress", n=3)
        try:
            before = _inflight_fingerprint(ops)
            time.sleep(5.0)  # rewind window: old completions re-delivered
            after = _inflight_fingerprint(ops)
        finally:
            clear_type_all(ops, names, "status_cursor_regress")

        stable = before is not None and after is not None and before == after
        still_clean, still_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        master_ok = _master_ok(ops)
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = clean_ok and stable and still_clean and master_ok and recovery_ok
        return passed, (
            f"history_ok=3/3, baseline_clean={clean_ok}({clean_detail}), "
            f"rewind_ledger_stable={stable} (fp_before={before}, "
            f"fp_after={after}), "
            f"still_clean={still_clean}({still_detail}), "
            f"master_200={master_ok}, recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_cursor_regress")


@case(
    "status_finished_then_running",
    profiles=["batch-window"],
    source="P1 status fault family: fake_task sequence — finished replay then persistent RUNNING for a settled rid",
)
def status_finished_then_running(ctx: CaseContext):
    """Scenario: after a request completes, the engine FIRST replays its
    finished fact (fake_task phase=finished), THEN keeps reporting it
    RUNNING (fake_task phase=RUNNING, persistent).

    Behaviour: a terminal replay followed by an out-of-order ACTIVE for an
    already-settled request.

    Expectation (contract): no resurrection, no rollback — the terminal is
    final: the inflight ledger stays clean across both injections (the
    settled slot must not re-enter inflight) and the master stays HTTP 200.

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        rid = ops.next_request_id(base)
        _, err0 = ops.run_one_request(
            rid, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
        )
        if err0:
            return False, f"baseline request failed: {err0}"
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        fp_baseline = _inflight_fingerprint(ops)

        # Phase 1 — replay the terminal (idempotency).
        inject_type(ops, names[0], "status_fake_task", rid=rid, phase="finished")
        try:
            time.sleep(2.0)
        finally:
            clear_type_all(ops, names, "status_fake_task")

        # Phase 2 — persistent RUNNING for the settled rid (resurrection
        # attempt).  Hold it open across several poll rounds.  The window
        # fingerprint is a hard assertion, not an observation: a transient
        # resurrection that only disappears after the injection is cleared
        # would escape a post-clear-only check.
        inject_type(ops, names[0], "status_fake_task", rid=rid, phase="RUNNING")
        try:
            time.sleep(5.0)
            fp_during = _inflight_fingerprint(ops)
        finally:
            clear_type_all(ops, names, "status_fake_task")
        no_resurrect_during = fp_during is not None and fp_during == fp_baseline

        clean_final, clean_final_detail = AssertUtils.inflight_clean(
            _master_http(ops), 20.0
        )
        master_ok = _master_ok(ops)

        passed = clean_ok and no_resurrect_during and clean_final and master_ok
        return passed, (
            f"terminal_final={clean_final} "
            f"(baseline_clean={clean_ok}, no_resurrect_during_window="
            f"{no_resurrect_during}, after_resurrection_attempt="
            f"{clean_final_detail}, fingerprint_during_running={fp_during}, "
            f"fingerprint_baseline={fp_baseline}), "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")


# ===========================================================================
# P1 — zombie running vs tombstone (1 case)
# ===========================================================================


@case(
    "status_zombie_completed_running",
    profiles=["batch-window"],
    source="P1 status fault family: status_zombie_running — completed tasks re-reported RUNNING",
)
def status_zombie_completed_running(ctx: CaseContext):
    """Scenario: the DECODE engines keep re-reporting already-completed
    tasks as RUNNING (status_zombie_running) while a batch of requests
    completes.

    Behaviour: every terminal the master settles is followed by zombie
    ACTIVE facts for the same reservations — the tombstone path must absorb
    them without re-confirming.

    Expectation (contract): master stays HTTP 200; NO new confirmed entries
    leak on the decode side (decode inflight_requests stays drained); the
    whole ledger stays clean (inflight_clean).  The unknown/zombie counter
    behaviour is RECORDED as an observational item (defensive field reads),
    not asserted.

    Grade: P1."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    dnames = _decode_names(ops)
    if not dnames:
        return False, "no decode engines found"
    try:
        inject_type_all(ops, dnames, "status_zombie_running")
        try:
            errs = _run_requests(ops, base, 4, concurrency=4)
            # Hold the zombie window open across several poll rounds.
            time.sleep(10.0)
        finally:
            clear_type_all(ops, dnames, "status_zombie_running")

        ok = sum(1 for e in errs if e is None)
        clean_ok, clean_detail = _stale_inflight_clean(ops)
        d_requests = _decode_requests_sum(ops)
        master_ok = _master_ok(ops)

        # Observational: unknown/zombie counter fields (defensive reads).
        snap = ops.snapshot_by_name()
        zombie_obs = {
            n: {
                f: snap.get(n, {}).get(f)
                for f in (
                    "unknown_tasks",
                    "unknown_count",
                    "zombie_reports",
                    "tombstone_hits",
                    "confirmed",
                )
                if f in snap.get(n, {})
            }
            for n in dnames
        }

        passed = ok == 4 and clean_ok and d_requests == 0 and master_ok
        return passed, (
            f"requests_ok={ok}/4, "
            f"inflight_clean={clean_ok}({clean_detail}), "
            f"decode_confirmed_leak={d_requests} (need 0), "
            f"zombie_counters_observed={zombie_obs}, "
            f"master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, dnames, "status_zombie_running")


# ===========================================================================
# P2 — persistent ghost cleanup (1 case)
# ===========================================================================


@case(
    "status_zombie_fake_running",
    profiles=["batch-window"],
    source="P2 status fault family: persistent fake RUNNING for N ghost rids, >= 2x TTL",
)
def status_zombie_fake_running(ctx: CaseContext):
    """Scenario: the engine PERSISTENTLY reports RUNNING facts for several
    request ids the master has never seen (status_fake_task, ghost rids,
    held for >= 2x the stale TTL).

    Behaviour: every status poll re-delivers the ghost ACTIVE facts while
    the injection remains armed.

    Expectation (contract — this is the probe): the master must NOT retain
    inflight entries that cannot be cleared.  Concretely: after the
    injection is cleared (the ghost reports stop), the scheduler inflight
    MUST drain to zero within TTL(30s)+margin.  The request-slot
    tombstone/TTL cleanup fixes the former permanent-residency finding, so
    this is now an ordinary regression contract.

    Grade: P2 (contract-level finding probe)."""
    env = ctx.env_manager.ensure(_status_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    n_ghosts = 3
    ghost_rids = [base + GHOST_RID_OFFSET + 100 + i for i in range(n_ghosts)]
    try:
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        sched_before = ops.master_scheduler_inflight()

        # Arm the persistent ghost RUNNING reports (one inject per rid; a
        # MERGE-semantics server accumulates them, a replace-semantics
        # server keeps the last — the probe only needs >= 1 resident).
        for rid in ghost_rids:
            inject_type(ops, names[0], "status_fake_task", rid=rid, phase="RUNNING")
        try:
            # Observation window: >= 2x TTL with the reports flowing.
            deadline = time.monotonic() + 2 * STALE_INFLIGHT_TTL_S
            samples = []
            while time.monotonic() < deadline:
                samples.append(ops.master_scheduler_inflight())
                time.sleep(5.0)
            resident = ops.master_scheduler_inflight()
            peak = max(samples) if samples else -1
            bounded = resident <= sched_before + n_ghosts
            master_ok_during = _master_ok(ops)
        finally:
            for rid in ghost_rids:
                clear_type_all(ops, names, "status_fake_task")

        # Contract: once the reports stop, nothing may stay resident.
        drained = _wait_scheduler_zero(ops)
        final = ops.master_scheduler_inflight()
        master_ok = _master_ok(ops)

        passed = (
            clean0
            and bounded
            and master_ok_during
            and drained
            and final == 0
            and master_ok
        )
        return passed, (
            f"baseline_clean={clean0}({clean0_detail}), "
            f"resident_after_2xTTL={resident} (peak={peak}, "
            f"bounded={bounded} <= {sched_before + n_ghosts}), "
            f"drained_after_clear={drained} (final={final}), "
            f"master_200=(during={master_ok_during}, after={master_ok}), "
            f"ghost_rids={n_ghosts}; persistent reports may refresh activity, "
            f"but entries must drain after the injection clears"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_fake_task")


# ===========================================================================
# Migrated from the legacy fault families: batch FetchResponse fault
# ===========================================================================


@case(
    "status_fetch_error",
    profiles=["batch-window"],
    source="gap G6/G7: /inject type=fetch_error (cross-process, batch FetchResponse path)",
)
def inject_fetch_error(ctx: CaseContext):
    """fetch_error makes the batch-mode FetchResponse stream fail after
    emitting one unfinished output.  The client must observe the error;
    the engine-side inflight drains immediately; the master-side ledger
    entry is cleaned by the 30s stale-inflight TTL (verified contract);
    a fresh request succeeds once the injection is cleared.

    Profile semantics (v2, task #55): the fault is checked only at the
    engine's fetchResponse entry, which exists only under the BATCH
    dispatcher — and _fault_spec pins the legacy fault axes
    (PRIORITY + FIXED_WINDOW + BATCH) via FLEXLB_CONFIG, so re-running
    under another --profile would execute the identical configuration.
    The declaration stays batch-window (regression efficiency + label
    honesty); a NON_BATCH master-path generate_error variant is
    dedicated-phase material.
    """
    ops = ctx.engine_ops(ctx.env_manager.ensure(_fault_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    rid = ops.next_request_id(base)
    try:
        inject_type_all(ops, names, "fetch_error")
        try:
            response = ops.schedule(rid)
            if response.code != 200 or not response.success:
                surfaced, detail = True, (f"schedule failed: {response.error_message}")
            else:
                handle = ops.start_stream(response, rid, input_pb=None)
                handle.wait_end(10.0)
                if handle.snap.error:
                    surfaced, detail = True, f"stream error: {handle.snap.error}"
                elif not handle.snap.completed:
                    surfaced, detail = True, "stream did not complete"
                else:
                    surfaced, detail = False, "request completed despite fetch_error"
                # NOTE: no explicit master cancel here.  The stream already
                # terminated with the engine's error, and a cancel would set
                # cancellationReason, which the TTL cleaner SKIPS (it waits
                # for an authoritative engine terminal through the cancel
                # fence instead) — verified on the Java side
                # (PriorityScheduler.cleanupInflight).
        finally:
            clear_type_all(ops, names, "fetch_error")

        rid2 = ops.next_request_id(base)
        _, err2 = ops.run_one_request(rid2, stream_timeout_s=STREAM_TIMEOUT_S)
        inflight_ok, inflight_detail = _stale_inflight_clean(ops)
        engine_clean, engine_detail = engine_inflight_clean(ops, names)
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            surfaced and err2 is None and inflight_ok and engine_clean and recovery_ok
        )
        return passed, (
            f"error_surfaced={surfaced} ({detail}), "
            f"recovered={err2 is None}, "
            f"master_inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_inflight_clean={engine_clean}({engine_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "fetch_error")
