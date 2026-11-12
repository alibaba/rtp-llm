from dataclasses import dataclass

from .step import StepContext
from .step_registry import Targets, register_step


@dataclass(frozen=True)
class MasterGetArgs(Targets):
    is_leader: bool | None = None


@register_step("master/get", effect="read")
async def master_get(ctx: StepContext, args: MasterGetArgs) -> list[str]:
    """List active Masters, optionally filtering their actual leadership claims."""
    return await ctx.runtime.masters.list_addresses(
        args.targets, args.is_leader, remaining=lambda: ctx.timeout
    )
