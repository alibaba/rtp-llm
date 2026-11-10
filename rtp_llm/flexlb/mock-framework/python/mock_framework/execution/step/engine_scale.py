from dataclasses import dataclass
from typing import Annotated, Literal

from ._targets import Targets
from .step import StepContext
from .step_registry import Bounds, register_step


@dataclass(frozen=True, kw_only=True)
class Scale(Targets):
    direction: Literal["in", "out"]
    role: Literal["prefill", "decode"] = "prefill"
    count: Annotated[int, Bounds(1, 1024)] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.direction == "out" and self.targets is not None:
            raise ValueError("targets may only be supplied when scaling in")


@register_step("engine/scale", effect="write")
async def engine_scale(ctx: StepContext, args: Scale) -> list[str]:
    """Scale the selected Engine role in or out and return the affected service addresses."""
    engines = ctx.runtime.engines
    if args.direction == "in":
        return await engines.scale_in(args.role, args.targets, args.count)
    return await engines.scale_out(args.role, 1 if args.count is None else args.count)
