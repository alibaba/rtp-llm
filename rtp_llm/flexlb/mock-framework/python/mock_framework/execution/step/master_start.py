from dataclasses import dataclass
from typing import Annotated
from .registry import Bounds, register_step
from .step import Step, StepContext


@dataclass(frozen=True)
class Start:
    count: Annotated[int, Bounds(1, 1024)] = 1


@register_step('master/start', effect='write')
class MasterStart(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: Start) -> list[str]:
        """Create new Master JVMs in this run's election group; never reuse stopped identities."""
        return await ctx.runtime.masters.start(args.count)
