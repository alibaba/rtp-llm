"""Wait for a Case measurement interval to elapse."""

import asyncio
from dataclasses import dataclass
from typing import Annotated

from ...config.validation import duration
from .step import StepContext
from .step_registry import register_step


@dataclass(frozen=True)
class WaitArgs:
    duration: Annotated[str, "duration"]


@register_step("case/wait", effect="read")
async def case_wait(ctx: StepContext, args: WaitArgs) -> None:
    """Wait before running the next step."""
    ctx.records.emit(event="wait_started", duration=args.duration)
    await asyncio.sleep(float(duration(args.duration, positive=True)))
