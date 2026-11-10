from . import _metrics
from ._metrics import MetricArgs
from .step import StepContext
from .step_registry import register_step


@register_step("engine/prefill/tps", effect="read", unit="tokens/s")
async def engine_prefill_tps(ctx: StepContext, args: MetricArgs) -> float | list[float]:
    return await _metrics.gauge(ctx, args, "prefill", "rtp_llm_context_tps")
