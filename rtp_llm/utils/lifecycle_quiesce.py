"""On-demand, host-only coordination of a common executor stopping round."""

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


async def prepare_sleep_rounds(
    request, addresses, statuses, call, broadcast, timeout_s
):
    """Drain all peers, freeze their admitted rounds, then catch up to max(round).

    The caller owns the instance lease and must roll back this token on ANY
    pre-commit failure/cancellation. Never run a collective from a serving thread.
    Freeze ACKs must be collected before ANY target is sent, including target 0.
    """
    failures = _check_coverage(statuses, addresses, "drain preflight")
    if failures:
        return failures
    prepared = []
    for status in statuses:
        if (
            status.get("state") != "RUNNING"
            or int(status.get("quiesce_protocol", 0)) != 1
            or not status.get("worker_incarnation")
        ):
            return [
                {
                    "address": status["address"],
                    "error": "all ranks must be RUNNING and support sleep round-fence protocol 1",
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
            call(address, "SleepServing", drain, timeout_s)
            for address, drain in prepared
        )
    )
    failures = _check_coverage(results, addresses, "drain")
    if failures:
        return failures

    frozen = await broadcast(
        "QuiesceSleep",
        pb2.SleepQuiesceRequestPB(token=request.quiesce_token, freeze_only=True),
        30.0,
    )
    failures = _check_coverage(frozen, addresses, "freeze")
    if failures:
        return failures
    try:
        rounds = [int(result["frozen_round"]) for result in frozen]
        if any(round_ < 0 or round_ >= 1 << 63 for round_ in rounds):
            raise ValueError("round out of range")
    except (KeyError, TypeError, ValueError) as error:
        return [
            {
                "error": f"invalid freeze acknowledgement: {error}",
                "grpc_status": "FAILED_PRECONDITION",
            }
        ]
    results = await broadcast(
        "QuiesceSleep",
        pb2.SleepQuiesceRequestPB(
            token=request.quiesce_token, target_round=max(rounds), timeout_ms=60000
        ),
        max(75.0, timeout_s),
    )
    return _check_coverage(results, addresses, "quiesce") or results
