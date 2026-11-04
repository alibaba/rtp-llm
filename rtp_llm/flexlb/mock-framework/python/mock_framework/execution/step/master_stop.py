from dataclasses import dataclass
from typing import Literal
from .registry import register_step
from .step import Step, StepContext
from ._master import Targets


@dataclass(frozen=True)
class Stop(Targets):
    mode: Literal['terminate', 'kill'] = 'terminate'


@register_step('master/stop', effect='write')
class MasterStop(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: Stop) -> list[str]:
        """Stop selected Master JVMs, or all active Masters when targets is omitted."""
        return await ctx.runtime.masters.stop(args.targets, args.mode)
