from dataclasses import dataclass
from typing import Annotated

from .step import StepContext
from .step_registry import Bounds, register_step


@dataclass(frozen=True)
class Start:
    count: Annotated[int, Bounds(1, 1024)] = 1


@register_step("master/start", effect="write")
async def master_start(ctx: StepContext, args: Start) -> list[str]:
    """Create new Master JVMs in this run's election group; never reuse stopped identities."""
    return await ctx.runtime.masters.start(args.count)
