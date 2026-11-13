from .. import metrics
from ..metrics import MetricArgs
from .step import StepContext
from .step_registry import register_step


@register_step("engine/query/prefill_tps_with_cache", effect="read", unit="tokens/s")
async def engine_query_prefill_tps_with_cache(
    ctx: StepContext, args: MetricArgs
) -> float | list[float]:
    return await metrics.query_gauge(
        ctx, args, "prefill", "rtp_llm_context_tps_with_cache"
    )
