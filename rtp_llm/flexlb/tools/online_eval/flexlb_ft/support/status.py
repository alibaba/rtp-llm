"""Shared status scenario components. Resource and timing semantics live here."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
    OMIT,
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    ConfigOverride,
    EnvSpec,
    _accepted,
    default_perf,
    fault_env_perf,
    http_get_status,
)
from ..harness import master_decode_requests_sum as _harness_decode_requests_sum
from ..harness import master_prefill_batches_sum as _harness_prefill_batches_sum
from ..harness import master_prefill_requests_sum as _harness_prefill_requests_sum
from ..harness import wait_for

STREAM_TIMEOUT_S = 15.0
# > staleInflightTimeoutMs (30s): lets a TTL-eviction terminal reach the
# client stream instead of the client's own stream deadline firing first.
LONG_STREAM_TIMEOUT_S = 45.0
STALE_INFLIGHT_TTL_S = 30.0
TTL_MARGIN_S = 30.0
QUEUE_TIMEOUT_S = 10.0
# Event-driven prefill-cleanup window (decode-before-prefill contract):
# once the decode terminal has settled the request, the prefill ledger
# entry for the same member must be released by the settle path itself,
# not parked until the 30s stale TTL + 60s sweep — 10s covers several
# status-poll rounds (statusRpcMs=1s) with margin.
EVENT_DRIVEN_CLEANUP_S = 10.0
# 3-strike health demotion + eviction window (fault-family MASTER_EVICT_S
# precedent).
MASTER_EVICT_S = 30.0
# TTL-eviction EVENT window (event channel): the drain window
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


# ===========================================================================
# Shared environment
# ===========================================================================


def _status_spec(ctx: CaseContext) -> EnvSpec:
    """Family env: 2P+2D, PRIORITY ordering over the profile's own
    decision/dispatcher axes (profile-aware since the tier2 spec
    unpick), TTL=30s, queueTimeout=10s.

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
        config_overrides=ConfigOverride(
            ordering="priority",
            queue_timeout_ms=int(QUEUE_TIMEOUT_S * 1000),
            stale_inflight_ms=int(STALE_INFLIGHT_TTL_S * 1000),
        ),
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
    flexlb_log = Path.home() / "ai-whale" / "logs" / "flexlb.log"
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

    Event channel: the master reports evictions via
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
    # shared env (integration-round cascade).
    return wait_for(lambda: ops.master_scheduler_inflight() == 0, timeout_s, 2.0)


def _stale_inflight_clean(ops, timeout_s: float = TTL_DRAIN_TIMEOUT_S) -> tuple:
    """Master inflight drain with the TTL-aware window (30s TTL + 60s
    ExpirationTimer sweep + margin — the worst-case settle path)."""
    return AssertUtils.inflight_clean(_master_http(ops), timeout_s)


# ===========================================================================
# Migrated from the legacy fault families: stuck-inflight TTL cleanup (S1)
# ===========================================================================


# ===========================================================================
# P0 — enqueue-ack fault shapes (3 cases)
# ===========================================================================


# ===========================================================================
# P0 — status-channel suppression (3 cases)
# ===========================================================================


# ===========================================================================
# P0 — ghost tasks & generation regress (2 cases)
# ===========================================================================


# ===========================================================================
# P1 — decode-side suppression & cross-role settle (4 cases)
# ===========================================================================


# ===========================================================================
# P1 — ghost / mismatched task reports (2 cases)
# ===========================================================================


# ===========================================================================
# P1 — unknown-id defense matrix (id-space robustness, 3 cases)
# ===========================================================================


# ===========================================================================
# P1 — terminal replay & rewind idempotency (3 cases)
# ===========================================================================


# ===========================================================================
# P1 — zombie running vs tombstone (1 case)
# ===========================================================================


# ===========================================================================
# P2 — declared contract-level finding probe (1 case)
# ===========================================================================


# ===========================================================================
# Migrated from the legacy fault families: batch FetchResponse fault
# ===========================================================================
