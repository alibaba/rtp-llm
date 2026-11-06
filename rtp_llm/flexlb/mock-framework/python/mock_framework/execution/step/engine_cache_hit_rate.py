from .step_registry import register_step
from .step import StepContext
from . import _metrics
from ._metrics import MetricArgs


@register_step('engine/cache/hit_rate', effect='read', unit='ratio')
async def engine_cache_hit_rate(ctx: StepContext, args: MetricArgs) -> float | list[float]:
    return await _metrics.metric(ctx, args, 'prefill', 'rtp_llm_cache_hit_tokens_total', 'rtp_llm_cache_lookup_tokens_total')
