"""Status-category cases: the engine→master status-report contract.

The engine→master status channel is the authoritative terminal source for
the master's inflight ledgers: WorkerStatus facts (ACTIVE / TERMINAL) both
settle request slots and refresh their stale-inflight activity clock
(RequestSlot.observeWorkerStatus → reduceStaleSlot expires a slot only when
now - lastWorkerStatusAtMs > staleInflightTimeoutMs).  This category
injects faults into exactly that channel (mock /inject "status_*" /
"enqueue_ack_*" types) and pins the CORRECT master contract, NOT the
current behaviour: assertions state what a correct master MUST do, and
cases the current implementation cannot satisfy are expected to FAIL —
that failure is the finding (status_zombie_fake_running is the declared
P2 probe; the fence-TTL drain of status_ack_empty_no_crash is a second
structural candidate per the verified quarantine semantics).

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

Master-side cleanup observability anchors (log grep, informational — the
hard assertions are the drained ledgers themselves):
    event=scheduler_inflight_ttl_eviction   (ExpirationTimer.maintain)
    event=endpoint_inflight_ttl_eviction    (EndpointRegistry)

Case index (P0 = release-blocking contract, P1 = robustness, P2 = declared
contract-level finding probe):

    P0 status_ack_partial_fail          k-of-batch ack failure isolates
    P0 status_ack_multi_error           per-request error-code passthrough
    P0 status_ack_empty_no_crash        empty ack → uncertain fence, bounded + clearable
    P0 status_prefill_suppress_all      full status silence → TTL eviction
    P0 status_prefill_suppress_finished running keep-alive → queueTimeout is the only exit
    P0 status_status_no_respond         status RPC silence → generation retirement
    P0 status_unknown_rid_finished      unknown-rid terminal ignored
    P0 status_version_regress           stale version → generation retirement
    P1 status_decode_suppress_finished  decode-side terminal suppression
    P1 status_decode_before_prefill     D-side terminal settles the request
    P1 status_unknown_rid_running       one-shot ghost running entry
    P1 status_unknown_batchid           mismatched batchId must not settle a real rid
    P1 status_duplicate_finished        duplicate terminal replay idempotent
    P1 status_cursor_regress            completion cursor rewind idempotent
    P1 status_finished_then_running     terminal must not be resurrected
    P1 status_zombie_completed_running  zombie running vs tombstone
    P2 status_zombie_fake_running       permanent-resident inflight probe (expected finding)

Migrated in from the legacy fault families (task #85 category reorg):

    status_inflight_ttl_cleanup         stuck inflight → TTL cleanup (S1 port)
    status_fetch_error                  batch FetchResponse fault surfacing
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
    ttl_spec,
    wait_for,
)

STATUS_CASES: list[CaseDef] = []

STREAM_TIMEOUT_S = 15.0
# > staleInflightTimeoutMs (30s): lets a TTL-eviction terminal reach the
# client stream instead of the client's own stream deadline firing first.
LONG_STREAM_TIMEOUT_S = 45.0
STALE_INFLIGHT_TTL_S = 30.0
TTL_MARGIN_S = 30.0
QUEUE_TIMEOUT_S = 10.0
# 3-strike health demotion + eviction window (fault-family MASTER_EVICT_S
# precedent).
MASTER_EVICT_S = 30.0
# Fake/ghost rid offset: far above every rid this process will hand out
# (next_request_ids stay within base + small offsets) so the master has
# never seen these ids.
GHOST_RID_OFFSET = 900_000


def case(name: str, profiles=None, requires=None, source: str = ""):
    """Register into STATUS_CASES (category is always "status")."""

    def deco(fn):
        STATUS_CASES.append(
            CaseDef(
                name=name,
                category="status",
                fn=fn,
                profiles=profiles,
                requires=requires,
                source=source,
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
            "exhaust",
            "8400",
            "8511",
            "8431",
        )
    )


def _run_requests(
    ops,
    base: int,
    n: int,
    output_len: int = 2,
    concurrency: int = 8,
    stream_timeout_s: float = STREAM_TIMEOUT_S,
) -> list:
    """Fire *n* requests (bounded concurrency); returns the per-request
    error list (None = success).  Error types are stashed on the function
    as ``last_error_types`` for failure diagnostics."""
    rids = [ops.next_request_id(base) for _ in range(n)]

    def run(rid: int):
        _, err = ops.run_one_request(
            rid, output_len=output_len, stream_timeout_s=stream_timeout_s
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
    (-1 when the inflight endpoint is unreachable)."""
    data = ops.master_inflight()
    if data is None:
        return -1
    total = 0
    for ep in data.get("prefill_endpoints", []) or []:
        batches = ep.get("inflight_batches", 0)
        total += len(batches) if isinstance(batches, list) else int(batches)
    return total


def _decode_requests_sum(ops) -> int:
    """Sum of master-side decode inflight_requests across every endpoint."""
    data = ops.master_inflight()
    if data is None:
        return -1
    return sum(
        int(ep.get("inflight_requests", 0) or 0)
        for ep in data.get("decode_endpoints", []) or []
    )


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
    """Occurrences of *anchor* in the master's current log file (0 when the
    master is down or the log is unreadable).  Cases take a before/after
    delta because the shared env keeps one master log across cases."""
    mp = getattr(env, "master", None)
    if mp is None:
        return 0
    try:
        log_file = Path(mp.log_file)
        if not log_file.is_file():
            return 0
        return log_file.read_text(encoding="utf-8", errors="replace").count(anchor)
    except Exception:
        return 0


def _ttl_anchor_deltas(env, before: tuple) -> tuple:
    """(scheduler_evictions, endpoint_evictions) delta since *before*."""
    return (
        _log_count(env, "event=scheduler_inflight_ttl_eviction") - before[0],
        _log_count(env, "event=endpoint_inflight_ttl_eviction") - before[1],
    )


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
    """S1 port: slow prefill → inflight stuck → /stop_engine → TTL cleans.

    Profile semantics (v2, task #55): the stuck-inflight state is built
    from fire-and-forget requests whose master-side batch bookkeeping
    never settles, and ttl_spec pins the legacy fault axes (PRIORITY +
    FIXED_WINDOW + BATCH) via FLEXLB_CONFIG — the declaration stays
    batch-window (label honesty + regression efficiency).  A NON_BATCH
    variant would need its own stuck-direct-ledger construction
    (dedicated-phase material).
    """
    env = ctx.env_manager.ensure(ttl_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    try:
        # Slow both prefills (10s) so scheduled requests stay inflight.
        ops.set_perf("prefill-0", prefill_fixed_ms=10000.0)
        ops.set_perf("prefill-1", prefill_fixed_ms=10000.0)

        # Fire-and-forget: schedule without consuming the response stream —
        # the master has already enqueued these batches into the engines.
        rids = [ops.next_request_id(base) for _ in range(6)]
        for rid in rids:
            resp = ops.schedule(rid, output_len=10)
            if resp.code != 200 or not resp.success:
                return False, f"schedule failed for rid={rid}: {resp.error_message}"

        enqueued = wait_for(lambda: _accepted(ops, "prefill-0") > 0, 15.0, 0.5)
        inflight_before = ops.master_scheduler_inflight()

        # Cut the engine mid-flight; its batches will never complete.
        ops.stop_engine("prefill-0")
        time.sleep(5.0)  # let the gRPC failures propagate
        # Per-endpoint view: the evicted engine's row disappears from
        # prefill_endpoints, so observe the stuck batches via the global
        # scheduler inflight (survives eviction until TTL cleanup).
        inflight_after_kill = ops.master_scheduler_inflight()

        # TTL (30s) + sync margin: poll to zero within 90s.
        cleanup_ok = wait_for(lambda: ops.master_scheduler_inflight() == 0, 90.0, 2.0)
        inflight_final = ops.master_scheduler_inflight()

        # The surviving prefill keeps serving normally.
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            enqueued
            and inflight_after_kill > 0
            and cleanup_ok
            and inflight_final == 0
            and recovery_ok
        )
        return passed, (
            f"enqueued={enqueued}, inflight_before_stop={inflight_before}, "
            f"stuck_after_kill={inflight_after_kill}, "
            f"cleanup_within_90s={cleanup_ok}, inflight_final={inflight_final}, "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            snap = ops.snapshot_by_name()
            if snap.get("prefill-0", {}).get("stopped"):
                ops.start_engine("prefill-0")
            ops.set_perf("prefill-0", prefill_fixed_ms=100.0)
            ops.set_perf("prefill-1", prefill_fixed_ms=100.0)
        except Exception:
            pass


# ===========================================================================
# P0 — enqueue-ack fault shapes (3 cases)
# ===========================================================================


@case(
    "status_ack_partial_fail",
    profiles=["batch-window"],  # _status_spec pins the legacy fault axes
    source="P0 status fault family: enqueue_ack_partial_fail(k=1) on a 4-request batch",
)
def status_ack_partial_fail(ctx: CaseContext):
    """Scenario: a 4-request enqueue batch lands on prefills whose ack marks
    k=1 members failed (enqueue_ack_partial_fail).

    Behaviour: the mock answers EnqueueBatch with a partial failure — the
    k members carry a terminal error, the rest are acknowledged.

    Expectation (contract): the k members receive a TERMINAL error carrying
    the injected code while the remaining members STILL SUCCEED — a partial
    failure must not poison the whole batch; the master ledger drains
    (inflight_clean) and a fresh 20-request batch recovers >= 95%.

    Grade: P0."""
    ops = ctx.engine_ops(ctx.env_manager.ensure(_status_spec(ctx)))
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        inject_type_all(ops, names, "enqueue_ack_partial_fail", k=1)
        try:
            errs = _run_requests(ops, base, 4, concurrency=4)
        finally:
            clear_type_all(ops, names, "enqueue_ack_partial_fail")

        failed = [e for e in errs if e is not None]
        ok = len(errs) - len(failed)
        # k=1 per landing batch; 4 near-simultaneous requests form 1-2
        # batches across the 2 prefills, so 1-2 terminal errors are the
        # contract-shaped outcome (never 0, never the whole batch).
        partial = 1 <= len(failed) <= 2 and ok >= 2
        err_kinds = sorted({str(e)[:70] for e in failed})[:3]

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = _recovery_rate(ops, base)
        master_ok = _master_ok(ops)

        passed = partial and inflight_ok and recovery_ok and master_ok
        return passed, (
            f"partial_isolated={partial} (failed={len(failed)}, ok={ok}, "
            f"err_kinds={err_kinds}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"{recovery_msg}, master_200={master_ok}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "enqueue_ack_partial_fail")


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
    within TTL+margin.  NOTE: the verified current behaviour parks
    uncertain-fence entries in quarantine forever (cleanupInflight skips
    engineFence entries from the stale TTL), so the drain assertion is a
    declared contract-level candidate to FAIL — that failure is the
    finding.  The master itself must stay up (HTTP 200) regardless.

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
    TTL eviction anchors advance (observational); after the injection is
    cleared a fresh batch recovers (verify_recovery).

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
            and master_ok
            and recovery_ok
        )
        return passed, (
            f"request_terminals: ok={ok}/4, "
            f"illegal_errors={bad_errs[:2]}, "
            f"scheduler_zero={sched_zero} (final={final_sched}), "
            f"prefill_batches_zero={batches_zero} (final={final_batches}), "
            f"ttl_anchors(sched,endp)={anchors_after}, "
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
    inflight drains to zero within TTL+margin; after the injection is
    cleared the topology fully recovers (alive back to 2P) and a fresh
    request succeeds.

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
            and master_ok
            and alive_back
            and recovery_ok
        )
        return passed, (
            f"generation_retired={all_retired} "
            f"(alive={ops.master_alive_count('PREFILL')}), "
            f"scheduler_zero={drained} (final={final_sched}), "
            f"ttl_anchors(sched,endp)={anchors_after}, "
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
    TTL+margin.

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
        inject_type_all(ops, names, "status_version_regress")
        try:
            alive_dropped = wait_for(
                lambda: ops.master_alive_count("PREFILL") <= len(names) - 1,
                MASTER_EVICT_S,
                0.5,
            )
            drained = _wait_scheduler_zero(ops)
            anchors_after = _ttl_anchor_deltas(env, anchors_before)
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

        passed = alive_dropped and drained and final_sched == 0 and master_ok
        return passed, (
            f"generation_retired={alive_dropped} "
            f"(alive={ops.master_alive_count('PREFILL')}), "
            f"scheduler_zero={drained} (final={final_sched}), "
            f"ttl_anchors(sched,endp)={anchors_after}, "
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
# P1 — decode-side suppression & cross-role settle (2 cases)
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
    """Scenario: the prefills suppress ALL facts for the whole batch's rids
    (status_suppress_rids on pre-generated ids) while the decodes report
    normally.

    Behaviour: the prefill ledger never hears about these rids; the decode
    side delivers the terminal.

    Expectation (contract): under P/D separation a decode-side finished
    fact settles the request — all 4 requests reach a SUCCESSFUL terminal;
    the prefill inflight_batches drain to zero within TTL+margin (the
    frozen prefill slots finally expire); master stays HTTP 200 (no
    crash from the cross-role settle).

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
            # Suppression stays ON: the prefill ledger's only exit is the
            # stale TTL on the frozen slots.
            p_batches_zero = wait_for(
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
            ok == 4 and p_batches_zero and final_p == 0 and master_ok and recovery_ok
        )
        return passed, (
            f"requests_succeeded_via_decode_terminal={ok}/4 "
            f"(err_kinds={err_kinds}), "
            f"prefill_batches_ttl_drained={p_batches_zero} (final={final_p}), "
            f"ttl_anchors(sched,endp)={anchors_after}, "
            f"master_200={master_ok}, recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_suppress_rids")


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

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
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
        finally:
            clear_type_all(ops, names, "status_duplicate_finished")

        ok = sum(1 for e in errs if e is None)
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        before = _inflight_fingerprint(ops)
        # Replay window: hold several poll rounds open.
        time.sleep(5.0)
        after = _inflight_fingerprint(ops)
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

        passed = ok == 4 and clean_ok and stable and master_ok
        return passed, (
            f"requests_ok={ok}/4, "
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

        # Phase 1 — replay the terminal (idempotency).
        inject_type(ops, names[0], "status_fake_task", rid=rid, phase="finished")
        try:
            time.sleep(2.0)
        finally:
            clear_type_all(ops, names, "status_fake_task")

        # Phase 2 — persistent RUNNING for the settled rid (resurrection
        # attempt).  Hold it open across several poll rounds.
        inject_type(ops, names[0], "status_fake_task", rid=rid, phase="RUNNING")
        try:
            time.sleep(5.0)
            fp_during = _inflight_fingerprint(ops)
        finally:
            clear_type_all(ops, names, "status_fake_task")

        clean_final, clean_final_detail = AssertUtils.inflight_clean(
            _master_http(ops), 20.0
        )
        master_ok = _master_ok(ops)

        passed = clean_ok and clean_final and master_ok
        return passed, (
            f"terminal_final={clean_final} "
            f"(baseline_clean={clean_ok}, after_resurrection_attempt="
            f"{clean_final_detail}, fingerprint_during_running={fp_during}), "
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
# P2 — declared contract-level finding probe (1 case)
# ===========================================================================


@case(
    "status_zombie_fake_running",
    profiles=["batch-window"],
    source="P2 status fault family (DECLARED FINDING PROBE): persistent fake RUNNING for N ghost rids, >= 2x TTL",
)
def status_zombie_fake_running(ctx: CaseContext):
    """Scenario: the engine PERSISTENTLY reports RUNNING facts for several
    request ids the master has never seen (status_fake_task, ghost rids,
    held for >= 2x the stale TTL).

    Behaviour: every status poll re-delivers the ghost ACTIVE facts.  On
    the current implementation each report refreshes the entry's activity
    clock (lastWorkerStatusAtMs), so the stale TTL can NEVER fire — the
    expected failure mode is permanently-resident inflight entries
    (ConfirmedTask-style) that survive the whole observation window.

    Expectation (contract — this is the probe): the master must NOT retain
    inflight entries that cannot be cleared.  Concretely: after the
    injection is cleared (the ghost reports stop), the scheduler inflight
    MUST drain to zero within TTL(30s)+margin.  EXPECTED TO FAIL on the
    current implementation — the failure IS the finding (record the
    resident count and the non-draining ledger as evidence).

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
            f"ghost_rids={n_ghosts} — expected finding: persistent "
            f"activity-clock refresh keeps ghost entries resident"
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
