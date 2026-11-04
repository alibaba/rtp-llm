from dataclasses import dataclass

from .. import ExecutionError
from .registry import register_step
from .step import Step, StepContext


@dataclass(frozen=True)
class StopArgs:
    target: str


@register_step('step/stop', effect='write')
class Stop(Step):
    allow_background = False

    @staticmethod
    async def execute(ctx: StepContext, args: StopArgs) -> bool:
        """Cancel a background Step in this phase and wait for its task to exit."""
        if ctx.stop_step is None:
            raise ExecutionError('step/stop requires a running Phase')
        return await ctx.stop_step(args.target)
