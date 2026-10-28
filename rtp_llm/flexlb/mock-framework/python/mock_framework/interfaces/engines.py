"""Select logical instances once; the host only creates or closes individual objects."""
from dataclasses import dataclass
from typing import Annotated, Literal
from ..config.registry import Bounds, case_interface
from .master import Targets


@dataclass(frozen=True)
class ScaleOut:
    role: Literal['prefill', 'decode'] = 'prefill'
    count: Annotated[int, Bounds(1, 1024)] = 1


@dataclass(frozen=True)
class ScaleIn(Targets):
    role: Literal['prefill', 'decode'] = 'prefill'
    count: Annotated[int, Bounds(1, 1024)] | None = None


@case_interface('engine/scale_out', effect='write')
async def scale_out(ctx, args: ScaleOut) -> list[str]:
    """Create count instances from the role's configuration; return their service addresses."""
    return await ctx.processes.engines.create(args.role, args.count)


@case_interface('engine/scale_in', effect='write')
async def scale_in(ctx, args: ScaleIn) -> list[str]:
    """Stop all selected instances by default; count limits the fixed selection."""
    return await ctx.processes.engines.remove(args.role, args.targets, args.count)
