import json
import math
from dataclasses import dataclass
from typing import Annotated, Literal

from .. import ExecutionError, metrics
from ..roles.client import ClientResult
from ._targets import Targets
from .step import StepContext
from .step_registry import register_step


@dataclass(frozen=True, kw_only=True)
class SuccessRateArgs(Targets):
    range: Annotated[list[str], "range"] | None = None
    mode: Literal["interval", "series"] = "interval"
    playback_result: dict[str, ClientResult] | None = None

    def __post_init__(self):
        super().__post_init__()
        if (self.range is None) == (self.playback_result is None):
            raise ValueError("Supply exactly one range or playback_result")
        if self.playback_result is not None and (
            self.targets is not None or self.mode != "interval"
        ):
            raise ValueError(
                "playback_result is the complete playback and does not support targets or series"
            )


@register_step("client/success_rate", effect="read", unit="ratio")
async def client_success_rate(
    ctx: StepContext, args: SuccessRateArgs
) -> float | list[float]:
    """Success share from a Prometheus window or one completed client/stop result."""
    if args.playback_result is not None:
        clients = args.playback_result.values()
        if any(client.unsent for client in clients):
            raise ExecutionError("Playback ended with unsent requests")
        finished = sum(client.finished for client in clients)
        if not finished:
            raise ExecutionError("Playback has no terminal observations")
        return sum(client.outcomes["success"] for client in clients) / finished

    millis = metrics.counter_window_ms(ctx, args)
    labels, targets = metrics.scope(ctx, args, "client")
    name = "mock_client_requests_finished_total"
    success = f'{name}{{{labels},outcome="success"}}'
    total = f"{name}{{{labels}}}"
    expression = (
        f"sum(increase({success}[{millis}ms])) / sum(increase({total}[{millis}ms]))"
    )
    value = await metrics.evaluate(
        ctx,
        args,
        expression,
        [
            (name, labels + ",outcome=" + json.dumps(outcome), targets)
            for outcome in ("success", "error", "timeout", "cancelled")
        ],
    )
    if any(
        not math.isfinite(v) or not 0 <= v <= 1
        for v in (value if isinstance(value, list) else [value])
    ):
        raise ExecutionError(
            "Client success rate requires terminal observations in the selected interval"
        )
    return value
