from .registry import register_step
from .step import Step, StepContext
from . import _metrics
from ._metrics import MetricArgs


@register_step('engine/prefill/tps', effect='read', unit='tokens/s')
class EnginePrefillTps(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: MetricArgs) -> float:
        return await _metrics.metric(ctx, args, 'prefill', 'rtp_llm_forward_tokens_total', 'rtp_llm_context_forward_seconds_total', kind='compute')
