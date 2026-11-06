from .step_registry import register_step
from .step import StepContext
from . import _metrics
from ._metrics import MetricArgs


@register_step('client/qps', effect='read', unit='requests/s')
async def client_qps(ctx: StepContext, args: MetricArgs) -> float | list[float]:
    """Rate of Client calls started, not successful completions or Master acceptance."""
    return await _metrics.metric(ctx, args, 'client', 'mock_client_requests_issued_total')
