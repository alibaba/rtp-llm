from dataclasses import dataclass
from typing import Literal

from .step import StepContext
from .step_registry import Targets, register_step


@dataclass(frozen=True)
class MasterListEnginesArgs(Targets):
    role: Literal["prefill", "decode", "vit"] = "prefill"


@register_step("master/list/engines", effect="read")
async def master_list_engines(
    ctx: StepContext, args: MasterListEnginesArgs
) -> dict[str, list[str]]:
    """Keep each Master's discovered Engine addresses separate."""
    return await ctx.runtime.masters.list_engine_addresses(
        args.targets, args.role, remaining=lambda: ctx.timeout
    )
