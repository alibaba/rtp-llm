import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

from ...config.validation import Bounds, duration
from .. import ExecutionError
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
class ClientStartPlaybackArgs:
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
                    "Client timeouts must be whole milliseconds between 1ms and 1h"
                )


@register_step("client/start/playback", effect="write")
async def client_start_playback(
    ctx: StepContext, args: ClientStartPlaybackArgs
) -> bool:
    """Start the same finite plan on all Clients; rate and concurrency are global budgets."""
    body = {key: value for key, value in asdict(args).items() if value is not None}
    case = ctx.runtime.case
    candidate = Path(args.requests_file)
    if candidate.parts and candidate.parts[0] == "data":
        path = (case.shared_root / candidate).resolve()
        if not path.is_relative_to((case.shared_root / "data").resolve()):
            raise ExecutionError(
                f"Client requests_file escapes shared data: {candidate}"
            )
    else:
        path = (case.directory / candidate).resolve()
    if not path.is_file():
        raise ExecutionError(f"Client requests_file is not a regular file: {path}")
    body["requests_file"] = str(path)
    body["request_timeout_ms"] = int(duration(body.pop("request_timeout")) * 1000)
    body["drain_timeout_ms"] = int(duration(body.pop("drain_timeout")) * 1000)
    if args.warmup is not None:
        body["warmup"] = {
            "duration_ms": int(duration(args.warmup.duration) * 1000),
            "rate_scale": args.warmup.rate_scale,
        }
    return await ctx.runtime.clients.start_playback(body, ctx.timeout)
