from dataclasses import dataclass
from typing import Annotated, Literal

from ...config.validation import Bounds
from ..roles.engine import MAX_ENGINE_START_COUNT
from .step import StepContext
from .step_registry import Targets, register_step


@dataclass(frozen=True, kw_only=True)
class EngineScaleArgs(Targets):
    direction: Literal["in", "out"]
    role: Literal["prefill", "decode"] = "prefill"
    count: Annotated[int, Bounds(minimum=1)] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.direction == "in" and self.targets is None and self.count is None:
            raise ValueError("scale-in requires targets or count")
        if self.direction == "out":
            if self.targets is not None:
                raise ValueError("targets may only be supplied when scaling in")
            if self.count is not None and self.count > MAX_ENGINE_START_COUNT:
                raise ValueError(
                    f"scale-out count must be at most {MAX_ENGINE_START_COUNT}"
                )


@register_step("engine/scale", effect="write")
async def engine_scale(ctx: StepContext, args: EngineScaleArgs) -> list[str]:
    """Scale the selected Engine role in or out and return the affected service addresses."""
    engines = ctx.runtime.engines
    if args.direction == "in":
        return await engines.scale_in(args.role, args.targets, args.count)
    return await engines.scale_out(args.role, 1 if args.count is None else args.count)
