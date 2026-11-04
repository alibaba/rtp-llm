from .registry import register_step
from .step import Step, StepContext
from . import _metrics
from ._metrics import MetricArgs


@register_step('engine/decode/tps', effect='read', unit='tokens/s')
class EngineDecodeTps(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: MetricArgs) -> float:
        return await _metrics.metric(ctx, args, 'decode', 'rtp_llm_forward_tokens_total', kind='output')
