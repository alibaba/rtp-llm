from dataclasses import dataclass
from .step_registry import register_step
from .step import StepContext
from ._targets import Targets


@dataclass(frozen=True)
class MasterSelection(Targets):
    is_leader: bool | None = None


@register_step('master/get', effect='read')
async def master_get(ctx: StepContext, args: MasterSelection) -> list[str]:
    """List active Masters, optionally filtering their actual leadership claims."""
    return await ctx.runtime.masters.get(args.targets, args.is_leader,
                                           query_time=ctx.query_time, remaining=lambda: ctx.timeout)
