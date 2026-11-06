from dataclasses import dataclass
from typing import Literal
from .step_registry import register_step
from .step import StepContext
from ._targets import Targets


@dataclass(frozen=True)
class EngineSelection(Targets):
    role: Literal['prefill', 'decode', 'vit'] = 'prefill'


@register_step('master/engine/get', effect='read')
async def master_engine_get(ctx: StepContext, args: EngineSelection) -> dict[str, list[str]]:
    """Keep each Master's discovered Engine addresses separate."""
    return await ctx.runtime.masters.get_engines(args.targets, args.role,
                                           query_time=ctx.query_time, remaining=lambda: ctx.timeout)
