from dataclasses import dataclass
from ..roles.client import ClientResult
from .registry import register_step
from .step import Step, StepContext


@dataclass(frozen=True)
class Empty:
    pass


@register_step('client/stop', effect='write')
class ClientStop(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: Empty) -> dict[str, ClientResult]:
        """Stop all Clients and collect bounded cleanup; natural completion returns its saved result."""
        return await ctx.runtime.clients.stop_playback(ctx.query_time, ctx.timeout)
