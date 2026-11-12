from dataclasses import dataclass

from .step import StepContext
from .step_registry import Targets, register_step


@dataclass(frozen=True)
class MasterListAddressesArgs(Targets):
    is_leader: bool | None = None


@register_step("master/list/addresses", effect="read")
async def master_list_addresses(
    ctx: StepContext, args: MasterListAddressesArgs
) -> list[str]:
    """List active Masters, optionally filtering their actual leadership claims."""
    return await ctx.runtime.masters.list_addresses(
        args.targets, args.is_leader, remaining=lambda: ctx.timeout
    )
