from .registry import register_step
from .step import Step, StepContext
from ._targets import Targets


@register_step('master/restart', effect='write')
class MasterRestart(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: Targets) -> list[str]:
        """Restore stopped Masters at their original service addresses with fresh process artifacts."""
        return await ctx.runtime.masters.restart(args.targets)
