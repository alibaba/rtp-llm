from ..metrics import LatencyArgs, latency
from .step import StepContext
from .step_registry import register_step


@register_step("client/query/ttft", effect="read", unit="s")
async def client_query_ttft(ctx: StepContext, args: LatencyArgs) -> float | list[float]:
    return await latency(ctx, args, "mock_client_ttft_seconds")
