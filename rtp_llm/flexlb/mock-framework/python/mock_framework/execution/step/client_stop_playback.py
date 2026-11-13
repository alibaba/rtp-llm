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
    """Stop playback and wait for Prometheus to scrape the final Client counters."""
    clients = ctx.runtime.clients
    result = await clients.stop_playback(ctx.timeout)
    if result and clients.monitoring:
        client_ids = [
            node["id"] for node in ctx.runtime.topology.list_active_endpoints("client")
        ]
        await clients.monitoring.wait_for_final_scrape(client_ids)
    return result
