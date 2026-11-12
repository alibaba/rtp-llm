from .step import StepContext
from .step_registry import Targets, register_step


@register_step("master/restart", effect="write")
async def master_restart(ctx: StepContext, args: Targets) -> list[str]:
    """Restore stopped Masters at their original service addresses with fresh process artifacts."""
    return await ctx.runtime.masters.restart(args.targets)
