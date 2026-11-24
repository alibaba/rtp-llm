"""Case steps for Master discovery and lifecycle."""

from dataclasses import dataclass
from typing import Annotated, Literal

from ...config.validation import Bounds
from .. import ExecutionError
from .step import StepContext
from .step_registry import RequiredTargets, Targets, register_step


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


@dataclass(frozen=True)
class MasterEngineCountArgs(Targets):
    role: Literal["prefill", "decode", "vit"] = "prefill"


@register_step("master/query/engine_count", effect="read", unit="engines")
async def master_engine_count(ctx: StepContext, args: MasterEngineCountArgs) -> float:
    """Catalog size of one Engine role; all Masters must agree on it."""
    observed = await ctx.runtime.masters.list_engine_addresses(
        args.targets, args.role, remaining=lambda: ctx.timeout
    )
    counts = {len(addresses) for addresses in observed.values()}
    if len(counts) != 1:
        raise ExecutionError(
            f"Masters disagree on the {args.role} catalog size: {observed}"
        )
    return float(counts.pop())


@dataclass(frozen=True)
class MasterStartArgs:
    count: Annotated[int, Bounds(1, 1024)] = 1


@register_step("master/start", effect="write")
async def master_start(ctx: StepContext, args: MasterStartArgs) -> list[str]:
    """Create new Master JVMs in this run's election group; never reuse stopped identities."""
    return await ctx.runtime.masters.start(args.count)


@dataclass(frozen=True)
class MasterStopArgs(RequiredTargets):
    mode: Literal["terminate", "kill"] = "terminate"


@register_step("master/stop", effect="write")
async def master_stop(ctx: StepContext, args: MasterStopArgs) -> list[str]:
    """Stop the selected active Master JVMs and return their addresses."""
    return await ctx.runtime.masters.stop(args.targets, args.mode)


@register_step("master/restart", effect="write")
async def master_restart(ctx: StepContext, args: RequiredTargets) -> list[str]:
    """Restore stopped Masters at their original service addresses with fresh process artifacts."""
    return await ctx.runtime.masters.restart(args.targets)
