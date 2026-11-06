from .registry import register_step
from .step import Step, StepContext
from . import _metrics
from ._metrics import MetricArgs


@register_step('client/qps', effect='read', unit='requests/s')
class ClientQps(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: MetricArgs) -> float | list[float]:
        """Rate of Client calls started, not successful completions or Master acceptance."""
        return await _metrics.metric(ctx, args, 'client', 'mock_client_requests_issued_total')
