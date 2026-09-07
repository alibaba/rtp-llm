from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import TTL_DRAIN_TIMEOUT_S, wait_for
from ...registry import case
from ...support.status import (
    EVENT_DRIVEN_CLEANUP_S,
    STREAM_TIMEOUT_S,
    _log_count,
    _master_ok,
    _prefill_batches_sum,
    _prefill_names,
    _status_spec,
    _ttl_anchor_deltas,
)


@case(
    "status_decode_before_prefill",
    category="status",
    profiles=["batch-window"],
    source="P1 status fault family: status_suppress_rids(full batch) on prefills, decodes normal",
    expected_fail=True,  # MIXED form (see docstring) — whole-case probe
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

    PREDICTED FINDING (cleanup linkage missing): in the current BATCH
    dispatch the decode terminal's counterpart cleanup only runs on the
    ROUTE_DECISION path (RequestRegistry.workerStatusCounterpartCleanup
    → exactPrefillCounterpartCleanup); a batch-delivered decode terminal
    does NOT release the prefill accounting, whose only exit is
    PrefillState.evictExpiredBatches (30s stale TTL + 60s sweep).  The
    event-driven assertion is EXPECTED TO FAIL and that failure is the
    finding; the TTL fallback observation below then documents that the
    entries do eventually expire (a permanent hang would be a worse,
    separate bug).

    Expected-fail marking (MIXED form): the case mixes
    should-pass dimensions (all-4-successful terminals via the decode
    settle, eventual drain, master health, recovery) with the
    predicted-fail <= 10s event-driven cleanup dimension
    (p_batches_fast), and the expected_fail granularity is whole-case —
    so the whole case is marked expected_fail: its expected failure
    classifies as finding-confirmed (the cleanup-linkage finding
    stands), its unexpected pass as finding-resolved (the counterpart
    cleanup landed on the batch path).  CAVEAT: a regression in the
    should-pass dimensions ALSO shows up as finding-confirmed — read the
    detail flags (requests_succeeded_via_decode_terminal /
    ttl_fallback_drained / master_200) to tell a regression apart from
    the declared finding.

    Grade: P1 (+ P2 cleanup-linkage probe)."""
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
