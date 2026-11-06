from dataclasses import dataclass
import json
import math
from typing import Annotated, Literal
from .step_registry import register_step
from .step import StepContext
from .. import ExecutionError
from ._targets import Targets
from . import _metrics


@dataclass(frozen=True, kw_only=True)
class SuccessRateArgs(Targets):
    range: Annotated[list[str], 'range']
    mode: Literal['interval', 'series'] = 'interval'


@register_step('client/success_rate', effect='read', unit='ratio')
async def client_success_rate(ctx: StepContext, args: SuccessRateArgs) -> float | list[float]:
    """Success share of terminal observations in an explicit [start, end] window; no request-count floor."""
    at, millis, _ = _metrics.evaluation_window(ctx, args)
    labels, targets = _metrics.scope(ctx, args, 'client')
    ctx.records.emit(event='metric_window', query_time=ctx.query_time, start=at - millis / 1000, end=at)
    name = 'mock_client_requests_finished_total'
    for outcome in ('success', 'error', 'timeout', 'cancelled'):
        await _metrics.counter_history(ctx, name, labels + ',outcome=' + json.dumps(outcome), targets, at, millis)
    success = f'{name}{{{labels},outcome="success"}}'
    total = f'{name}{{{labels}}}'
    expression = f'sum(increase({success}[{millis}ms])) / sum(increase({total}[{millis}ms]))'
    value = await _metrics.evaluate(ctx, args, expression,
        [(name, labels + ',outcome=' + json.dumps(outcome), targets)
         for outcome in ('success', 'error', 'timeout', 'cancelled')])
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (value if isinstance(value, list) else [value])):
        raise ExecutionError('Client success rate requires terminal observations in the selected interval')
    return value
