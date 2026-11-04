from dataclasses import dataclass
from typing import Annotated, Literal
from .registry import Bounds, register_step
from .step import Step, StepContext


@dataclass(frozen=True)
class ScaleOut:
    role: Literal['prefill', 'decode'] = 'prefill'
    count: Annotated[int, Bounds(1, 1024)] = 1


@register_step('engine/scale_out', effect='write')
class EngineScaleOut(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: ScaleOut) -> list[str]:
        """Create count instances from the role's configuration; return their service addresses."""
        return await ctx.runtime.engines.create(args.role, args.count)
