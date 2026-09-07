from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import TTL_DRAIN_TIMEOUT_S, AssertUtils
from ...registry import case
from ...support.cancel import _engine_cancel_receipt_within, _master_http


@case("cancel_phase_timing", category="cancel", source="cancel_smoke.py T6")
def cancel_phase_timing(ctx: CaseContext):
    ops = ctx.ops()
    base = rid_base(ctx, "cancel")
    try:
        # A: cancel in prefill phase (before any output)
        rid_a = ops.next_request_id(base)
        resp_a = ops.schedule(rid_a)
        if resp_a.code != 200 or not resp_a.success:
            return False, f"schedule A failed: {resp_a.error_message}"
        input_pb_a = (
            None if resp_a.enqueued_by_master else ops.build_generate_input(rid_a)
        )
        handle_a = ops.start_stream(resp_a, rid_a, input_pb=input_pb_a)
        time.sleep(0.1)
        a_in_prefill = not handle_a.snap.first_received
        a_cancel_at = time.monotonic()
        ops.cancel(rid_a, resp_a)
        a_ended = handle_a.wait_end(5.0)
        # 时效契约：A 的 engine 收证须在 A cancel 发出后 5s 内（BATCH）。
        if resp_a.enqueued_by_master:
            engine_cancelled_a, cancel_detail_a = _engine_cancel_receipt_within(
                ops, rid_a, timeout_s=5.0, since=a_cancel_at
            )
        else:
            engine_cancelled_a, cancel_detail_a = ops.verify_engine_cancelled(rid_a)

        # B: cancel in decode phase (after first output)
        rid_b = ops.next_request_id(base)
        resp_b = ops.schedule(rid_b)
        if resp_b.code != 200 or not resp_b.success:
            return False, f"schedule B failed: {resp_b.error_message}"
        input_pb_b = (
            None if resp_b.enqueued_by_master else ops.build_generate_input(rid_b)
        )
        handle_b = ops.start_stream(resp_b, rid_b, input_pb=input_pb_b)
        b_got_first = handle_b.wait_first_output()
        if not b_got_first:
            handle_b.cancel()
            return False, "B never received first output (decode phase)"
        b_cancel_at = time.monotonic()
        ops.cancel(rid_b, resp_b)
        b_ended = handle_b.wait_end(5.0)
        # 时效契约：B 的 engine 收证须在 B cancel 发出后 5s 内（BATCH）。
        if resp_b.enqueued_by_master:
            engine_cancelled_b, cancel_detail_b = _engine_cancel_receipt_within(
                ops, rid_b, timeout_s=5.0, since=b_cancel_at
            )
        else:
            engine_cancelled_b, cancel_detail_b = ops.verify_engine_cancelled(rid_b)

        recovery_ok, recovery_msg = ops.verify_recovery()
        method_a = "enqueue_batch" if resp_a.enqueued_by_master else "generate_stream"
        method_b = "enqueue_batch" if resp_b.enqueued_by_master else "generate_stream"
        engine_recv_a, _ = ops.verify_engine_received(rid_a, method_a)
        engine_recv_b, _ = ops.verify_engine_received(rid_b, method_b)
        if resp_a.enqueued_by_master or resp_b.enqueued_by_master:
            # Same window-insufficient fix as the three unstable cases: the
            # cancelled A/B tasks linger on the master ledger until the
            # stale-inflight drain (~90s physical) completes, so the default
            # 10s inflight-clean window aborts early. Aligned to the
            # TTL_DRAIN_TIMEOUT_S standard; assertion semantics unchanged.
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), TTL_DRAIN_TIMEOUT_S
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"

        # 修复（eval batch A）：engine 收证（5s 时效契约）与 master 账本
        # 排空升格进 passed——BATCH 分支按各自交付模式断言。
        passed = a_ended and b_ended and a_in_prefill and recovery_ok
        if resp_a.enqueued_by_master:
            passed = passed and engine_cancelled_a
        if resp_b.enqueued_by_master:
            passed = passed and engine_cancelled_b
        if resp_a.enqueued_by_master or resp_b.enqueued_by_master:
            passed = passed and inflight_ok
        return passed, (
            f"A_prefill_phase={a_in_prefill}, A_terminated={a_ended}, "
            f"A_outputs={len(handle_a.snap.outputs)}, "
            f"B_decode_phase={b_got_first}, B_terminated={b_ended}, "
            f"B_outputs={len(handle_b.snap.outputs)}, "
            f"engine_recv_A={engine_recv_a}, "
            f"engine_cancel_A={engine_cancelled_a}({cancel_detail_a}), "
            f"engine_recv_B={engine_recv_b}, "
            f"engine_cancel_B={engine_cancelled_b}({cancel_detail_b}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
