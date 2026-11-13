from dataclasses import dataclass
from typing import Literal

from .step import StepContext
from .step_registry import RequiredTargets, register_step


@dataclass(frozen=True)
class MasterStopArgs(RequiredTargets):
    mode: Literal["terminate", "kill"] = "terminate"


@register_step("master/stop", effect="write")
async def master_stop(ctx: StepContext, args: MasterStopArgs) -> list[str]:
    """Stop the selected active Master JVMs and return their addresses."""
    return await ctx.runtime.masters.stop(args.targets, args.mode)
