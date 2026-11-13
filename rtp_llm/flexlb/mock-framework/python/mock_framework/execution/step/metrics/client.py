import json
import math

from ... import ExecutionError, metrics
from ...metrics import LatencyArgs, MetricArgs, query_latency
from ..step import StepContext
from ..step_registry import register_step


@register_step("client/query/success_rate", effect="read", unit="ratio")
async def query_success_rate(ctx: StepContext, args: MetricArgs) -> float | list[float]:
    """Query the share of successful terminal requests in the selected window."""
    millis = metrics.counter_window_ms(ctx, args)
    labels, targets = metrics.select_metric_targets(ctx, args, "client")
    name = "mock_client_requests_finished_total"
    success = f'{name}{{{labels},outcome="success"}}'
    total = f"{name}{{{labels}}}"
    expression = (
        f"sum(increase({success}[{millis}ms])) / sum(increase({total}[{millis}ms]))"
    )
    value = await metrics.query_metric_window(
        ctx,
        args,
        expression,
        [
            (name, labels + ",outcome=" + json.dumps(outcome), targets)
            for outcome in ("success", "error", "timeout", "cancelled")
        ],
    )
    if any(
        not math.isfinite(v) or not 0 <= v <= 1
        for v in (value if isinstance(value, list) else [value])
    ):
        raise ExecutionError(
            "Client success rate requires terminal observations in the selected interval"
        )
    return value


@register_step("client/query/qps", effect="read", unit="requests/s")
async def query_qps(ctx: StepContext, args: MetricArgs) -> float | list[float]:
    """Rate of Client calls started, not successful completions or Master acceptance."""
    return await metrics.query_counter(
        ctx, args, "client", "mock_client_requests_issued_total"
    )


@register_step("client/query/ttft", effect="read", unit="s")
async def query_ttft(ctx: StepContext, args: LatencyArgs) -> float | list[float]:
    return await query_latency(ctx, args, "mock_client_ttft_seconds")


@register_step("client/query/tpot", effect="read", unit="s")
async def query_tpot(ctx: StepContext, args: LatencyArgs) -> float | list[float]:
    return await query_latency(ctx, args, "mock_client_tpot_seconds")
