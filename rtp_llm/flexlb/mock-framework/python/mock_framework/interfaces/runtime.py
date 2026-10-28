import asyncio
from dataclasses import dataclass
from typing import Annotated
from ..config.registry import case_interface
from ..config.values import duration


@dataclass(frozen=True)
class WaitArgs:
    duration: Annotated[str, 'duration']


@case_interface('runtime/wait', effect='read')
async def wait(ctx, args: WaitArgs) -> None:
    """Wait within the current step deadline while process failure remains monitored."""
    await asyncio.sleep(float(duration(args.duration)))
