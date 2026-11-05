from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Annotated
from .registry import Bounds, register_step
from ...config.validation import duration
from .step import Step, StepContext
from .. import ExecutionError


@dataclass(frozen=True)
class Start:
    requests_file: str
    concurrency: Annotated[int, Bounds(minimum=1, maximum=2147483647)]
    qps: float | None = None
    replay_speed: float | None = None
    masters: list[str] | None = None
    queue_capacity: Annotated[int, Bounds(minimum=1, maximum=100000)] = 16
    request_timeout: Annotated[str, 'duration'] = '30s'
    drain_timeout: Annotated[str, 'duration'] = '30s'
    reuse_cache: bool = False

    def __post_init__(self):
        if self.masters is not None and (not self.masters or len(set(self.masters)) != len(self.masters) or any(not v.strip() for v in self.masters)):
            raise ValueError('masters must be nonempty and unique when supplied')
        rates = [v for v in (self.qps, self.replay_speed) if v is not None]
        if len(rates) != 1 or not math.isfinite(rates[0]) or rates[0] <= 0:
            raise ValueError('Supply exactly one positive finite qps or replay_speed')
        if not self.requests_file.strip() or '://' in self.requests_file:
            raise ValueError('requests_file must be a local JSONL or Protobuf+Zstd input path')
        for value in (self.request_timeout, self.drain_timeout):
            millis = duration(value, positive=True) * 1000
            if millis != int(millis) or not 1 <= millis <= 3600000:
                raise ValueError('Client timeouts must be whole milliseconds between 1ms and 1h')


@register_step('client/start', effect='write')
class ClientStart(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: Start) -> bool:
        """Start the same finite plan on all Clients; rate and concurrency are global budgets."""
        body = {key: value for key, value in asdict(args).items() if value is not None}
        case = ctx.runtime.case
        candidate = Path(args.requests_file)
        if candidate.parts and candidate.parts[0] == 'data':
            path = (case.shared_root / candidate).resolve()
            if not path.is_relative_to((case.shared_root / 'data').resolve()):
                raise ExecutionError(f'Client requests_file escapes shared data: {candidate}')
        else:
            path = (case.directory / candidate).resolve()
        if not path.is_file():
            raise ExecutionError(f'Client requests_file is not a regular file: {path}')
        body['requests_file'] = str(path)
        body['request_timeout_ms'] = int(duration(body.pop('request_timeout')) * 1000)
        body['drain_timeout_ms'] = int(duration(body.pop('drain_timeout')) * 1000)
        return await ctx.runtime.clients.start_playback(body, ctx.query_time, ctx.timeout)
