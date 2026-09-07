from __future__ import annotations

import grpc

from ...context import CaseContext
from ...harness import AssertUtils
from ...registry import case
from ...support.cancel import _inflight_fingerprint, _master_http


@case("cancel_unknown_rid", category="cancel", source="cancel_smoke.py T5")
def cancel_unknown_rid(ctx: CaseContext):
    """Cancel for a rid the master has never seen: typed NOT_FOUND, zero
    ledger mutation.

    Rewritten in the 2026-09 eval batch A: the legacy port only asserted
    "the Cancel RPC did not raise" — nearly vacuous (any well-formed
    gRPC error also satisfies it).  The real contract has two layers:

      * response semantics: the master answers the unknown-rid Cancel
        with a typed NOT_FOUND — either an OK response carrying
        found=false (RequestRegistry has no slot for the rid) or a gRPC
        NOT_FOUND status; any other outcome (a found=true hallucination,
        INTERNAL/UNAVAILABLE/...) fails;
      * ledger invariant: the master inflight fingerprint is
        bit-identical before vs after the cancel (scheduler count +
        every endpoint row — same construction as the status family's
        _inflight_fingerprint, copied locally to avoid cross-category
        import churn).
    """
    ops = ctx.ops()
    try:
        fake_rid = 99999
        # 清零基线（status_unknown_rid_finished 先例）：共享 env 的前序
        # 残渣排空后，"逐位不变"才可观察。
        clean0, clean0_detail = AssertUtils.inflight_clean(_master_http(ops), 20.0)
        before = _inflight_fingerprint(ops)
        semantics_ok, semantics_detail = False, "no attempt"
        try:
            stub = ops.schedule_pb2_grpc.FlexlbServiceStub(
                ops._channel(ops.master_target())
            )
            ack = stub.Cancel(
                ops.schedule_pb2.FlexlbCancelRequestPB(
                    request_id=fake_rid,
                    reason=ops.schedule_pb2.CANCEL_REASON_CLIENT_CANCELLED,
                ),
                timeout=10.0,
            )
            if not ack.found:
                semantics_ok = True
                semantics_detail = "typed NOT_FOUND (found=false)"
            else:
                semantics_detail = (
                    f"found=true for unknown rid={fake_rid} "
                    f"(lifecycle={ack.lifecycle})"
                )
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.NOT_FOUND:
                semantics_ok = True
                semantics_detail = "gRPC NOT_FOUND status"
            else:
                semantics_detail = f"gRPC {exc.code()}: {exc.details()}"
        except Exception as exc:
            semantics_detail = repr(exc)

        after = _inflight_fingerprint(ops)
        # 修复（eval batch A）：两层真断言——响应语义（NOT_FOUND 或幂等
        # found=false，其它 gRPC 状态皆 fail）+ 账本指纹逐位不变。
        ledger_unchanged = before is not None and after is not None and before == after
        passed = clean0 and semantics_ok and ledger_unchanged
        return passed, (
            f"cancel(rid={fake_rid}): semantics={semantics_ok}"
            f"({semantics_detail}), "
            f"baseline_clean={clean0}({clean0_detail}), "
            f"ledger_unchanged={ledger_unchanged} "
            f"(before={before}, after={after})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
