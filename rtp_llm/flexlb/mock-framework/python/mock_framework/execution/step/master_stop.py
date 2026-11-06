from dataclasses import dataclass
from typing import Literal
from .step_registry import register_step
from .step import StepContext
from ._targets import Targets


@dataclass(frozen=True)
class Stop(Targets):
    mode: Literal['terminate', 'kill'] = 'terminate'


@register_step('master/stop', effect='write')
async def master_stop(ctx: StepContext, args: Stop) -> list[str]:
    """Stop selected Master JVMs, or all active Masters when targets is omitted."""
    return await ctx.runtime.masters.stop(args.targets, args.mode)
