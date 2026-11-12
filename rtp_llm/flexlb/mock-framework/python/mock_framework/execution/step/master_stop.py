from dataclasses import dataclass
from typing import Literal

from .step import StepContext
from .step_registry import Targets, register_step


@dataclass(frozen=True)
class MasterStopArgs(Targets):
    mode: Literal["terminate", "kill"] = "terminate"


@register_step("master/stop", effect="write")
async def master_stop(ctx: StepContext, args: MasterStopArgs) -> list[str]:
    """Stop selected Master JVMs, or all active Masters when targets is omitted."""
    return await ctx.runtime.masters.stop(args.targets, args.mode)
