import asyncio
from dataclasses import dataclass

from ..roles.client import ClientResult
from .step import StepContext
from .step_registry import register_step


@dataclass(frozen=True)
class ClientStopPlaybackArgs:
    pass


@register_step("client/stop/playback", effect="write")
async def client_stop_playback(
    ctx: StepContext, args: ClientStopPlaybackArgs
) -> dict[str, ClientResult]:
    """Stop playback, collect results and give Prometheus time to scrape."""
    clients = ctx.runtime.clients
    result = await clients.stop_playback(ctx.timeout)
    if result and clients.monitoring:
        await asyncio.sleep(10)
    return result
