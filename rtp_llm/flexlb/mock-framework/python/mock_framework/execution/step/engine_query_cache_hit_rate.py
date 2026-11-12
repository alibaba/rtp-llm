from .. import metrics
from ..metrics import MetricArgs
from .step import StepContext
from .step_registry import register_step


@register_step("engine/query/cache_hit_rate", effect="read", unit="ratio")
async def engine_query_cache_hit_rate(
    ctx: StepContext, args: MetricArgs
) -> float | list[float]:
    return await metrics.query_counter(
        ctx,
        args,
        "prefill",
        "rtp_llm_cache_hit_tokens_total",
        "rtp_llm_cache_lookup_tokens_total",
    )
