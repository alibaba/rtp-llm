from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type
from ...harness import TTL_DRAIN_TIMEOUT_S, AssertUtils
from ...registry import case
from ...support.status import (
    STREAM_TIMEOUT_S,
    _master_http,
    _master_ok,
    _prefill_names,
    _status_spec,
)


@case(
    "status_unknown_batchid",
    category="status",
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
