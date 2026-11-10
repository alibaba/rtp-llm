from dataclasses import dataclass

from ..roles.client import ClientResult
from .step import StepContext
from .step_registry import register_step


@dataclass(frozen=True)
class Empty:
    pass


@register_step("client/stop", effect="write")
async def client_stop(ctx: StepContext, args: Empty) -> dict[str, ClientResult]:
    """Stop all Clients and collect bounded cleanup; natural completion returns its saved result."""
    return await ctx.runtime.clients.stop_playback(ctx.timeout)
