from dataclasses import dataclass

from . import _metrics
from ._metrics import QuerySample
from .step import StepContext
from .step_registry import register_step


@dataclass(frozen=True)
class Promql:
    promql: str

    def __post_init__(self):
        if not self.promql.strip():
            raise ValueError("promql must not be blank")


@dataclass(frozen=True)
class QueryResult:
    samples: list[QuerySample]
    query_time: str


@register_step("prometheus/query", effect="read")
async def prometheus_query(ctx: StepContext, args: Promql) -> QueryResult:
    """Execute raw PromQL at the fixed step time without rewriting it."""
    return QueryResult(
        await _metrics.instant(ctx, args.promql, _metrics.epoch(ctx.query_time)),
        ctx.query_time,
    )
