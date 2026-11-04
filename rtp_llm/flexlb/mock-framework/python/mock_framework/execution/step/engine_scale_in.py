from dataclasses import dataclass
from typing import Annotated, Literal
from .registry import Bounds, register_step
from .step import Step, StepContext
from ._master import Targets


@dataclass(frozen=True)
class ScaleIn(Targets):
    role: Literal['prefill', 'decode'] = 'prefill'
    count: Annotated[int, Bounds(1, 1024)] | None = None


@register_step('engine/scale_in', effect='write')
class EngineScaleIn(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: ScaleIn) -> list[str]:
        """Stop all selected instances by default; count limits the fixed selection."""
        return await ctx.runtime.engines.remove(args.role, args.targets, args.count)
