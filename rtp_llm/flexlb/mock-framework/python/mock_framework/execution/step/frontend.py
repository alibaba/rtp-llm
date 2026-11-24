"""Case steps for Frontend playback."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

from ...config.validation import Bounds, duration
from .. import ExecutionError
from ..roles.frontend import FrontendResult
from .step import StepContext
from .step_registry import register_step


@dataclass(frozen=True)
class Warmup:
    duration: Annotated[str, "duration"]
    rate_scale: float

    def __post_init__(self):
        millis = duration(self.duration, positive=True) * 1000
        if millis != int(millis) or millis > 86_400_000:
            raise ValueError("Warmup duration must be whole milliseconds up to 24h")
        if not math.isfinite(self.rate_scale) or not 0 < self.rate_scale < 1:
            raise ValueError("Warmup rate_scale must be between 0 and 1, exclusive")


@dataclass(frozen=True)
class FrontendStartPlaybackArgs:
    requests_file: str
    total_concurrency: Annotated[int, Bounds(minimum=1, maximum=2147483647)]
    qps: float | None = None
    replay_speed: float | None = None
    warmup: Warmup | None = None
    masters: list[str] | None = None
    queue_capacity: Annotated[int, Bounds(minimum=1, maximum=100000)] = 16
    request_timeout: Annotated[str, "duration"] = "30s"
    drain_timeout: Annotated[str, "duration"] = "30s"
    reuse_cache: bool = False
    fallback_max_output_tokens: (
        Annotated[int, Bounds(minimum=1, maximum=2147483647)] | None
    ) = None
    duplicate_every: Annotated[int, Bounds(minimum=2)] | None = None

    def __post_init__(self):
        if self.masters is not None and (
            not self.masters
            or len(set(self.masters)) != len(self.masters)
            or any(not v.strip() for v in self.masters)
        ):
            raise ValueError("masters must be nonempty and unique when supplied")
        rates = [v for v in (self.qps, self.replay_speed) if v is not None]
        if len(rates) != 1 or not math.isfinite(rates[0]) or rates[0] <= 0:
            raise ValueError("Supply exactly one positive finite qps or replay_speed")
        if "://" in self.requests_file or Path(self.requests_file).suffix != ".mckidx":
            raise ValueError("requests_file must be a local .mckidx input path")
        for value in (self.request_timeout, self.drain_timeout):
            millis = duration(value, positive=True) * 1000
            if millis != int(millis) or not 1 <= millis <= 3600000:
                raise ValueError(
                    "Frontend timeouts must be whole milliseconds between 1ms and 1h"
                )


@register_step("frontend/start/playback", effect="write")
async def frontend_start_playback(
    ctx: StepContext, args: FrontendStartPlaybackArgs
) -> bool:
    """Start the same finite plan on all Frontends; rate and concurrency are global budgets."""
    body = {key: value for key, value in asdict(args).items() if value is not None}
    case = ctx.runtime.case
    candidate = Path(args.requests_file)
    if candidate.parts and candidate.parts[0] == "data":
        path = (case.shared_root / candidate).resolve()
        if not path.is_relative_to((case.shared_root / "data").resolve()):
            raise ExecutionError(
                f"Frontend requests_file escapes shared data: {candidate}"
            )
    else:
        path = (case.directory / candidate).resolve()
    if not path.is_file():
        raise ExecutionError(f"Frontend requests_file is not a regular file: {path}")
    body["requests_file"] = str(path)
    body["request_timeout_ms"] = int(duration(body.pop("request_timeout")) * 1000)
    body["drain_timeout_ms"] = int(duration(body.pop("drain_timeout")) * 1000)
    if args.warmup is not None:
        body["warmup"] = {
            "duration_ms": int(duration(args.warmup.duration) * 1000),
            "rate_scale": args.warmup.rate_scale,
        }
    return await ctx.runtime.frontends.start_playback(body, ctx.timeout)


@dataclass(frozen=True)
class FrontendStopPlaybackArgs:
    pass


@register_step("frontend/stop/playback", effect="write")
async def frontend_stop_playback(
    ctx: StepContext, args: FrontendStopPlaybackArgs
) -> dict[str, FrontendResult]:
    """Stop playback and wait for Prometheus to scrape the final Frontend counters."""
    frontends = ctx.runtime.frontends
    result = await frontends.stop_playback(ctx.timeout)
    if result and frontends.monitoring:
        frontend_ids = [
            node["id"]
            for node in ctx.runtime.topology.list_active_endpoints("frontend")
        ]
        await frontends.monitoring.wait_for_final_scrape(frontend_ids)
    return result


@dataclass(frozen=True)
class PlaybackRequestsArgs:
    kind: str = "TERMINAL"
    field: str | None = None
    equals: str | None = None

    def __post_init__(self):
        if not self.kind.strip():
            raise ValueError("kind must not be blank")
        if self.field is not None and not self.field.strip():
            raise ValueError("field must not be blank when supplied")
        if self.equals is not None and self.field is None:
            raise ValueError("equals requires field")


def _latest_journal(frontend_dir: Path) -> Path | None:
    playback = frontend_dir / "playback"
    if not playback.is_dir():
        return None

    def round_number(entry: Path) -> tuple[int, str]:
        return (int(entry.name) if entry.name.isdigit() else -1, entry.name)

    journals = [
        entry / "requests.jsonl"
        for entry in sorted(playback.iterdir(), key=round_number)
        if entry.is_dir() and (entry / "requests.jsonl").is_file()
    ]
    return journals[-1] if journals else None


def count_journal_events(journal: Path, args: PlaybackRequestsArgs) -> int:
    """Count evidence events; header and seal lines never carry a request kind."""
    matched = 0
    with journal.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("kind") != args.kind:
                continue
            if args.field is not None:
                value = event.get(args.field)
                if value is None or (args.equals is not None and value != args.equals):
                    continue
            matched += 1
    return matched


@register_step("frontend/query/playback_requests", effect="read", unit="events")
async def frontend_playback_requests(
    ctx: StepContext, args: PlaybackRequestsArgs
) -> float:
    """Count per-request evidence events across every Frontend's latest playback round."""
    components = ctx.records.directory / "components"
    total = 0
    found = False
    for frontend_dir in sorted(components.glob("frontend-*")):
        journal = _latest_journal(frontend_dir)
        if journal is None:
            continue
        found = True
        total += count_journal_events(journal, args)
    if not found:
        raise ExecutionError("No recorded Frontend playback to inspect")
    ctx.records.emit(
        event="playback_requests_count",
        kind=args.kind,
        field=args.field,
        equals=args.equals,
        matched=total,
    )
    return float(total)
