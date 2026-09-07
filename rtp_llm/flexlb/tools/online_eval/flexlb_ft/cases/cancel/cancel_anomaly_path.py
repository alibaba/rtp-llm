from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import TTL_DRAIN_TIMEOUT_S, AssertUtils
from ...registry import case
from ...support.cancel import _master_http


@case(
    "cancel_anomaly_path",
    category="cancel",
    source="anomaly_smoke.py E1",
)
def cancel_anomaly_path(ctx: CaseContext):
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
        cancel_at = time.monotonic()
        ops.cancel(rid, response)
        ended = handle.wait_end(5.0)
        cancel_latency = time.monotonic() - cancel_at
        recovery_ok, recovery_msg = ops.verify_recovery()
        if response.enqueued_by_master:
            # Window-insufficient instability fix: the post-cancel ledger
            # settle can ride the stale-TTL + ExpirationTimer drain (worst
            # ~90s) instead of the immediate explicit-cancel release —
            # the 10s window let a normal slow drain read as a FAIL.
            # Aligned to the TTL_DRAIN_TIMEOUT_S standard; the all-zero
            # assertion itself is unchanged.
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), TTL_DRAIN_TIMEOUT_S
            )
        else:
            # NON_BATCH: a client Cancel on a delivered request cannot
            # safely release the master ledger entry (fence probe NOT_FOUND
            # is not a safe-release fact), so immediate-zero is not a
            # contract here — see kv_decode_capacity_park's watermark
            # rationale.
            inflight_ok, inflight_detail = True, "N/A (NON_BATCH residue contract)"
        # P0.5 升格（refactor wave）：BATCH 投递下 master 账本排空进
        # verdict（对齐 _anomaly_error_case 的口径——client 终态、账本
        # 清零、恢复三合一）；NON_BATCH 分支 inflight_ok 恒 True，语义
        # 不变。
        passed = ended and recovery_ok and inflight_ok
        return passed, (
            f"cancel_latency={cancel_latency:.3f}s, stream_terminated={ended}, "
            f"outputs={len(handle.snap.outputs)}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
