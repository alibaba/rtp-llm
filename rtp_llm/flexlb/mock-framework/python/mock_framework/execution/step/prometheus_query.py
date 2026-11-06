from dataclasses import dataclass
from .registry import register_step
from .step import Step, StepContext
from . import _metrics
from ._metrics import QuerySample


@dataclass(frozen=True)
class Promql:
    promql: str

    def __post_init__(self):
        if not self.promql.strip():
            raise ValueError('promql must not be blank')


@dataclass(frozen=True)
class QueryResult:
    samples: list[QuerySample]
    query_time: str


@register_step('prometheus/query', effect='read')
class PrometheusQuery(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: Promql) -> QueryResult:
        """Execute raw PromQL at the fixed step time without rewriting it."""
        return QueryResult(await _metrics.instant(ctx, args.promql, _metrics.epoch(ctx.query_time)), ctx.query_time)
