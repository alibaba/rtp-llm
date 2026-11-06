from dataclasses import dataclass
from typing import Literal
from .step_registry import register_step
from .step import StepContext
from . import _metrics


@dataclass(frozen=True)
class LatencyArgs(_metrics.MetricArgs):
    statistic: Literal['mean', 'p95', 'p99'] = 'mean'


@register_step('client/ttft', effect='read', unit='s')
async def client_ttft(ctx: StepContext, args: LatencyArgs) -> float | list[float]:
    return await _metrics.latency(ctx, args, 'mock_client_ttft_seconds')
