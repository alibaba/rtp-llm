from dataclasses import dataclass
import json
from typing import Annotated
from .registry import register_step
from .step import Step, StepContext
from .. import ExecutionError
from ._master import Targets
from . import _metrics


@dataclass(frozen=True, kw_only=True)
class SuccessRateArgs(Targets):
    range: Annotated[list[str], 'range']


@register_step('client/success_rate', effect='read', unit='ratio')
class ClientSuccessRate(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: SuccessRateArgs) -> float:
        """Success share of terminal observations in an explicit [start, end] window; no request-count floor."""
        at, millis, _ = _metrics.window(ctx, args)
        labels, targets = _metrics.scope(ctx, args, 'client')
        ctx.records.emit(event='metric_window', query_time=ctx.query_time, start=at - millis / 1000, end=at)
        name = 'mock_client_requests_finished_total'
        for outcome in ('success', 'error', 'timeout', 'cancelled'):
            await _metrics.counter_history(ctx, name, labels + ',outcome=' + json.dumps(outcome), targets, at, millis)
        success = f'{name}{{{labels},outcome="success"}}'
        total = f'{name}{{{labels}}}'
        expression = f'sum(increase({success}[{millis}ms])) / sum(increase({total}[{millis}ms]))'
        values = await _metrics.instant(ctx, expression, at)
        if len(values) != 1 or not 0 <= values[0].value <= 1:
            raise ExecutionError('Client success rate requires terminal observations in the selected interval')
        return values[0].value
