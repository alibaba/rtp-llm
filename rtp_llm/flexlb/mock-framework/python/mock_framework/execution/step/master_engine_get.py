from dataclasses import dataclass
from typing import Literal

from ._targets import Targets
from .step import StepContext
from .step_registry import register_step


@dataclass(frozen=True)
class MasterEngineGetArgs(Targets):
    role: Literal["prefill", "decode", "vit"] = "prefill"


@register_step("master/engine/get", effect="read")
async def master_engine_get(
    ctx: StepContext, args: MasterEngineGetArgs
) -> dict[str, list[str]]:
    """Keep each Master's discovered Engine addresses separate."""
    return await ctx.runtime.masters.list_engine_addresses(
        args.targets, args.role, remaining=lambda: ctx.timeout
    )
