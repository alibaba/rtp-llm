from dataclasses import dataclass
from typing import Annotated

from ...config.validation import Bounds
from .step import StepContext
from .step_registry import register_step


@dataclass(frozen=True)
class MasterStartArgs:
    count: Annotated[int, Bounds(1, 1024)] = 1


@register_step("master/start", effect="write")
async def master_start(ctx: StepContext, args: MasterStartArgs) -> list[str]:
    """Create new Master JVMs in this run's election group; never reuse stopped identities."""
    return await ctx.runtime.masters.start(args.count)
