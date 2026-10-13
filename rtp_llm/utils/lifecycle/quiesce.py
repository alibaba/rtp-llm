"""Frontend phase barriers; execution-round coordination belongs to backends."""

import asyncio

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2


def _check_coverage(results, addresses, phase):
    if len(results) != len(addresses) or {r.get("address") for r in results} != set(
        addresses
    ):
        return [
            {
                "error": f"{phase}: control-rank coverage is incomplete",
                "grpc_status": "FAILED_PRECONDITION",
            }
        ]
    return [result for result in results if "error" in result]


async def prepare_sleep_quiesce(
    request, addresses, statuses, call, broadcast, rpc_timeout_s
):
    """Drain all peers, freeze, then ask backends to establish a safe boundary.

    The caller owns the instance lease and must roll back this token on ANY
    pre-commit failure/cancellation. No round IDs cross the frontend boundary.
    Freeze ACKs must be collected before backends enter CPU group coordination.
    Both drain stages receive the full request budget; the RPC timeout separately
    includes transport headroom and must not extend the backend drain budget.
    """
    drain_timeout_ms = request.timeout_ms
    failures = _check_coverage(statuses, addresses, "drain preflight")
    if failures:
        return failures
    prepared = []
    for status in statuses:
        if (
            status.get("state") != "RUNNING"
            or int(status.get("quiesce_protocol", 0)) != 2
            or not status.get("worker_incarnation")
        ):
            return [
                {
                    "address": status["address"],
                    "error": "all ranks must be RUNNING and support backend CPU quiesce protocol 2",
                    "grpc_status": "FAILED_PRECONDITION",
                }
            ]
        drain = pb2.SleepRequestPB()
        drain.CopyFrom(request)
        drain.prepare_only = True
        drain.drain_only = True
        drain.expected_incarnation = status["worker_incarnation"]
        drain.expected_sleep_epoch = int(status["sleep_epoch"])
        prepared.append((status["address"], drain))
    results = await asyncio.gather(
        *(
            call(address, "SleepServing", drain, rpc_timeout_s)
            for address, drain in prepared
        )
    )
    failures = _check_coverage(results, addresses, "drain")
    if failures:
        return failures

    frozen = await broadcast(
        "QuiesceSleep",
        pb2.SleepQuiesceRequestPB(
            token=request.quiesce_token,
            freeze_only=True,
            protocol=2,
            timeout_ms=drain_timeout_ms,
        ),
        rpc_timeout_s,
    )
    failures = _check_coverage(frozen, addresses, "freeze")
    if failures:
        return failures
    results = await broadcast(
        "QuiesceSleep",
        pb2.SleepQuiesceRequestPB(
            token=request.quiesce_token, protocol=2, timeout_ms=60000
        ),
        max(75.0, rpc_timeout_s),
    )
    return _check_coverage(results, addresses, "quiesce") or results
