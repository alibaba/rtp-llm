from ... import metrics
from ...metrics import MetricArgs
from ..step import StepContext
from ..step_registry import register_step


@register_step("engine/query/cache_hit_rate", effect="read", unit="ratio")
async def query_cache_hit_rate(
    ctx: StepContext, args: MetricArgs
) -> float | list[float]:
    return await metrics.query_counter(
        ctx,
        args,
        "prefill",
        "rtp_llm_cache_hit_tokens_total",
        "rtp_llm_cache_lookup_tokens_total",
    )


@register_step("engine/query/prefill_tps", effect="read", unit="tokens/s")
async def query_prefill_tps(ctx: StepContext, args: MetricArgs) -> float | list[float]:
    return await metrics.query_gauge(ctx, args, "prefill", "rtp_llm_context_tps")


@register_step("engine/query/prefill_tps_with_cache", effect="read", unit="tokens/s")
async def query_prefill_tps_with_cache(
    ctx: StepContext, args: MetricArgs
) -> float | list[float]:
    return await metrics.query_gauge(
        ctx, args, "prefill", "rtp_llm_context_tps_with_cache"
    )


@register_step("engine/query/decode_tps", effect="read", unit="tokens/s")
async def query_decode_tps(ctx: StepContext, args: MetricArgs) -> float | list[float]:
    return await metrics.query_counter(
        ctx, args, "decode", "rtp_llm_forward_tokens_total", kind="output"
    )
