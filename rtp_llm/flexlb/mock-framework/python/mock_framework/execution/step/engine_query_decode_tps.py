from .. import metrics
from ..metrics import MetricArgs
from .step import StepContext
from .step_registry import register_step


@register_step("engine/query/decode_tps", effect="read", unit="tokens/s")
async def engine_query_decode_tps(
    ctx: StepContext, args: MetricArgs
) -> float | list[float]:
    return await metrics.metric(
        ctx, args, "decode", "rtp_llm_forward_tokens_total", kind="output"
    )
