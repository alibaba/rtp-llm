from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import AssertUtils
from ...registry import case
from ...support.cancel import (
    _cancel_rpc_total,
    _engine_cancel_receipt_within,
    _master_http,
)


@case("cancel_idempotent", category="cancel", source="cancel_smoke.py T2")
def cancel_idempotent(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            handle.cancel()
            return False, "no output received before cancel window"
        # 修复（eval batch A）：以 engine 侧 Cancel RPC 计数
        # （_cancel_rpc_total / snapshot rpc_counts.cancel）为收证口径——
        # 二次 cancel 不重复触发 engine 取消正是本用例语义核心。
        baseline_cancel = _cancel_rpc_total(ops)
        first_cancel_at = time.monotonic()
        ops.cancel(rid, response)
        # 时效契约：第一次 cancel 的 engine 收证须在 5s 内（BATCH）。
        if response.enqueued_by_master:
            engine_cancelled, cancel_detail = _engine_cancel_receipt_within(
                ops, rid, timeout_s=5.0, since=first_cancel_at
            )
        else:
            engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        first_cancel_delta = _cancel_rpc_total(ops) - baseline_cancel
        second_cancel_ok, second_cancel_err = True, ""
        try:
            ops.cancel(rid, response)
        except Exception as exc:
            second_cancel_ok, second_cancel_err = False, repr(exc)
        ended = handle.wait_end(5.0)
        recovery_ok, recovery_msg = ops.verify_recovery()
        method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
        engine_recv, recv_detail = ops.verify_engine_received(rid, method)
        # recovery 之后读取：二次 cancel 若错误地又触发 engine 取消，异步
        # 转发已在 recovery 窗口内落地，此处计数已覆盖。
        second_cancel_delta = (
            _cancel_rpc_total(ops) - baseline_cancel - first_cancel_delta
        )
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        # 修复（eval batch A）：BATCH 分支升格 engine 收证（5s 时效）+
        # 收证计数恰为 1（第二次 cancel 命中终态快照，不得再触发 engine
        # 取消）；NON_BATCH 二次 cancel 直发 worker（幂等 NOT_FOUND），
        # 计数断言不适用，保持现状。
        if response.enqueued_by_master:
            passed = (
                second_cancel_ok
                and ended
                and recovery_ok
                and engine_cancelled
                and first_cancel_delta >= 1
                and second_cancel_delta == 0
            )
        else:
            passed = second_cancel_ok and ended and recovery_ok
        return passed, (
            f"second_cancel_ok={second_cancel_ok} {second_cancel_err}, "
            f"stream_terminated={ended}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"cancel_rpc_delta=first:{first_cancel_delta}/"
            f"second:{second_cancel_delta}"
            f"[{'hard' if response.enqueued_by_master else 'observational'}], "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
