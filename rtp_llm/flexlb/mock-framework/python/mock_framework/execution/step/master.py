"""Case steps for Master discovery and lifecycle."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from ...config.validation import Bounds
from .. import ExecutionError
from .frontend import latest_terminal_outcomes
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


@dataclass(frozen=True)
class MasterPvRequestsArgs:
    schedule_origin: str | None = None
    success: bool | None = None
    code: int | None = None
    decode_requeue_count_gte: Annotated[int, Bounds(minimum=0)] | None = None
    require_frontend_outcome: (
        Literal["success", "error", "timeout", "cancelled"] | None
    ) = None

    def __post_init__(self):
        if self.schedule_origin is not None and not self.schedule_origin.strip():
            raise ValueError("schedule_origin must not be blank when supplied")


def _master_pv_logs(components: Path) -> list[Path]:
    logs: list[Path] = []
    for master_dir in sorted(components.glob("master-*")):
        if not master_dir.is_dir():
            continue
        candidates = list(master_dir.glob("logs/pv.log*")) + list(
            master_dir.glob("restart-*/logs/pv.log*")
        )
        logs.extend(path for path in sorted(candidates) if path.is_file())
    return logs


def _parse_pv_record(line: str) -> dict | None:
    marker = line.find(" - {")
    payload = line[marker + 3 :] if marker != -1 else line.strip()
    if not payload.startswith("{"):
        return None
    try:
        record = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def _matches_pv_record(
    record: dict,
    args: MasterPvRequestsArgs,
    frontend_outcomes: dict[str, set[str]] | None,
) -> bool:
    if (
        args.schedule_origin is not None
        and record.get("scheduleOrigin") != args.schedule_origin
    ):
        return False
    if args.success is not None and record.get("success") is not args.success:
        return False
    if args.code is not None and record.get("code") != args.code:
        return False
    if args.decode_requeue_count_gte is not None:
        requeue_count = record.get("decodeRequeueCount", 0)
        if (
            not isinstance(requeue_count, int)
            or requeue_count < args.decode_requeue_count_gte
        ):
            return False
    if args.require_frontend_outcome is not None and frontend_outcomes is not None:
        request_id = record.get("requestId")
        if request_id is None:
            return False
        if frontend_outcomes.get(str(request_id)) != {args.require_frontend_outcome}:
            return False
    return True


@register_step("master/query/pv_requests", effect="read", unit="requests")
async def master_pv_requests(ctx: StepContext, args: MasterPvRequestsArgs) -> float:
    """Count per-request Master PV log records matching the filters across all Master instances."""
    components = ctx.records.directory / "components"
    logs = _master_pv_logs(components)
    if not logs:
        raise ExecutionError("No recorded Master pv.log files to inspect")
    frontend_outcomes = (
        latest_terminal_outcomes(components)
        if args.require_frontend_outcome is not None
        else None
    )
    matched = 0
    sample_ids: list[str] = []
    for path in logs:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = _parse_pv_record(line)
                if record is None:
                    continue
                if not _matches_pv_record(record, args, frontend_outcomes):
                    continue
                matched += 1
                request_id = record.get("requestId")
                if request_id is not None and len(sample_ids) < 20:
                    sample_ids.append(str(request_id))
    ctx.records.emit(
        event="master_pv_requests_count",
        schedule_origin=args.schedule_origin,
        success=args.success,
        code=args.code,
        decode_requeue_count_gte=args.decode_requeue_count_gte,
        require_frontend_outcome=args.require_frontend_outcome,
        matched=matched,
        sample_request_ids=sample_ids,
    )
    return float(matched)
