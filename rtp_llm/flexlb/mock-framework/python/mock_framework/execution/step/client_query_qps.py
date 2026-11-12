from .. import metrics
from ..metrics import MetricArgs
from .step import StepContext
from .step_registry import register_step


@register_step("client/query/qps", effect="read", unit="requests/s")
async def client_query_qps(ctx: StepContext, args: MetricArgs) -> float | list[float]:
    """Rate of Client calls started, not successful completions or Master acceptance."""
    return await metrics.query_counter(
        ctx, args, "client", "mock_client_requests_issued_total"
    )
