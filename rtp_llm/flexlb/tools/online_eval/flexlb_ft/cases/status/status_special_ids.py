from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type
from ...harness import AssertUtils
from ...registry import case
from ...support.status import (
    STREAM_TIMEOUT_S,
    _inflight_fingerprint,
    _master_http,
    _master_ok,
    _prefill_names,
    _status_spec,
)


@case(
    "status_special_ids",
    category="status",
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
