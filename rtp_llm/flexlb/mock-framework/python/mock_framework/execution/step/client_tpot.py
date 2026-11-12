from ..metrics import LatencyArgs, latency
from .step import StepContext
from .step_registry import register_step


@register_step("client/tpot", effect="read", unit="s")
async def client_tpot(ctx: StepContext, args: LatencyArgs) -> float | list[float]:
    return await latency(ctx, args, "mock_client_tpot_seconds")
