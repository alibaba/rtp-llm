from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, engine_inflight_clean, inject_type_all
from ...harness import ConfigOverride, EnvSpec, default_perf
from ...registry import case
from ...support.status import STREAM_TIMEOUT_S, _prefill_names, _stale_inflight_clean


@case(
    "status_fetch_error",
    category="status",
    profiles=["batch-window"],
    source="gap G6/G7: /inject type=fetch_error (cross-process, batch FetchResponse path)",
)
def inject_fetch_error(ctx: CaseContext):
    """fetch_error makes the batch-mode FetchResponse stream fail after
    emitting one unfinished output.  The client must observe the error;
    the engine-side inflight drains immediately; the master-side ledger
    entry is cleaned by the 30s stale-inflight TTL (verified contract);
    a fresh request succeeds once the injection is cleared.

    Profile semantics (v2): the fault is checked only at the
    engine's fetchResponse entry, which exists only under the BATCH
    dispatcher — and the env below pins the legacy fault axes
    (PRIORITY + FIXED_WINDOW + BATCH; formerly harness._fault_spec)
    via the config override layer, so re-running
    under another --profile would execute the identical configuration.
    The declaration stays batch-window (regression efficiency + label
    honesty); a NON_BATCH master-path generate_error variant is
    dedicated-phase material.
    """
    ops = ctx.engine_ops(
        ctx.env_manager.ensure(
            EnvSpec(
                label=f"inject_fault_{ctx.profile}",
                n_prefill=2,
                n_decode=2,
                perf=default_perf(),
                master_profile=ctx.profile,
                config_overrides=ConfigOverride(
                    ordering="priority",
                    decision="fixed_window",
                    dispatcher="batch",
                    queue_timeout_ms=60_000,
                    stale_inflight_ms=30_000,
                ),
            )
        )
    )
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
