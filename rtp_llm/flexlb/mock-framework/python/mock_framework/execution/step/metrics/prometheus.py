import math
from dataclasses import dataclass

from ... import ExecutionError, metrics
from ...roles.prometheus import QuerySample
from ..step import StepContext
from ..step_registry import register_step


@dataclass(frozen=True)
class PrometheusQueryArgs:
    promql: str

    def __post_init__(self):
        if not self.promql.strip():
            raise ValueError("promql must not be blank")


@dataclass(frozen=True)
class QueryResult:
    samples: list[QuerySample]
    query_time: str


@register_step("prometheus/query", effect="read")
async def query_promql(ctx: StepContext, args: PrometheusQueryArgs) -> QueryResult:
    """Execute raw PromQL at the fixed step time without rewriting it."""
    return QueryResult(
        await metrics.query_instant(ctx, args.promql, metrics.epoch(ctx.query_time)),
        ctx.query_time,
    )


@register_step("prometheus/query_scalar", effect="read")
async def query_promql_scalar(ctx: StepContext, args: PrometheusQueryArgs) -> float:
    """Execute PromQL that must yield exactly one finite sample so expect gates can threshold it."""
    values = await metrics.query_instant(
        ctx, args.promql, metrics.epoch(ctx.query_time)
    )
    if len(values) != 1 or not math.isfinite(values[0].value):
        raise ExecutionError(
            f"Scalar query must return exactly one finite sample, got {len(values)}"
        )
    return values[0].value
