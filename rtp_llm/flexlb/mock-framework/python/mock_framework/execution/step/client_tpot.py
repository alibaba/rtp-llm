from dataclasses import dataclass
from typing import Literal
from .registry import register_step
from .step import Step, StepContext
from . import _metrics


@dataclass(frozen=True)
class LatencyArgs(_metrics.MetricArgs):
    statistic: Literal['mean', 'p95', 'p99'] = 'mean'


@register_step('client/tpot', effect='read', unit='s')
class ClientTpot(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: LatencyArgs) -> float | list[float]:
        return await _metrics.latency(ctx, args, 'mock_client_tpot_seconds')
