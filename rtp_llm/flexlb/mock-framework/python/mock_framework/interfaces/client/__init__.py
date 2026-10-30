"""One current playback across all Clients; Java owns playback and coordination."""
import asyncio
from dataclasses import asdict, dataclass
import math
import threading
from pathlib import Path
from typing import Annotated

from ._records import RequestRecord, read_requests
from ...execution.client_group import ClientResult
from ...config.registry import Bounds, case_interface
from ...config.values import duration
from ...execution.transport import ExecutionError


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

    def __post_init__(self):
        if self.masters is not None and (not self.masters or len(set(self.masters)) != len(self.masters) or any(not v.strip() for v in self.masters)):
            raise ValueError('masters must be nonempty and unique when supplied')
        rates = [v for v in (self.qps, self.replay_speed) if v is not None]
        if len(rates) != 1 or not math.isfinite(rates[0]) or rates[0] <= 0:
            raise ValueError('Supply exactly one positive finite qps or replay_speed')
        if not self.requests_file.strip() or '://' in self.requests_file:
            raise ValueError('requests_file must be a local Protobuf+Zstd input path')
        for value in (self.request_timeout, self.drain_timeout):
            millis = duration(value, positive=True) * 1000
            if millis != int(millis) or not 1 <= millis <= 3600000:
                raise ValueError('Client timeouts must be whole milliseconds between 1ms and 1h')


@dataclass(frozen=True)
class Empty:
    pass


@case_interface('client/start', effect='write')
async def start(ctx, args: Start) -> bool:
    """Start the same finite plan on all Clients; rate and concurrency are global budgets."""
    body = {key: value for key, value in asdict(args).items() if value is not None}
    path = (ctx.runtime.case.directory / Path(args.requests_file)).resolve()
    if not path.is_file():
        raise ExecutionError(f'Client requests_file is not a regular file: {path}')
    body['requests_file'] = str(path)
    body['request_timeout_ms'] = int(duration(body.pop('request_timeout')) * 1000)
    body['drain_timeout_ms'] = int(duration(body.pop('drain_timeout')) * 1000)
    return await ctx.runtime.clients.start(body, ctx.query_time, ctx.timeout)


@case_interface('client/stop', effect='write')
async def stop(ctx, args: Empty) -> dict[str, ClientResult]:
    """Stop all Clients and collect bounded cleanup; natural completion returns its saved result."""
    return await ctx.runtime.clients.collect(ctx.query_time, ctx.timeout)


@dataclass(frozen=True)
class Requests:
    targets: list[str] | None = None
    playback: Annotated[int, Bounds(minimum=1)] | None = None
    request_ids: list[Annotated[int, Bounds(minimum=1)]] | None = None
    outcome: str | None = None
    since_ms: Annotated[int, Bounds(minimum=0)] | None = None
    until_ms: Annotated[int, Bounds(minimum=0)] | None = None
    limit: Annotated[int, Bounds(minimum=0, maximum=1000)] = 20

    def __post_init__(self):
        if self.outcome is not None and self.outcome not in {'success', 'error', 'timeout', 'cancelled'}:
            raise ValueError('Unknown Client outcome')
        if self.since_ms is not None and self.until_ms is not None and self.since_ms > self.until_ms:
            raise ValueError('since_ms must not exceed until_ms')
        if self.request_ids is not None and (not self.request_ids or len(set(self.request_ids)) != len(self.request_ids)):
            raise ValueError('request_ids must be nonempty and unique')


@dataclass(frozen=True)
class RequestsResult:
    matched: int
    outcomes: dict[str, int]
    records: list[RequestRecord]
    truncated: bool


def playback_directories(root, topology, targets=None, playback=None):
    clients = {node['rpc']: node for node in topology if node['role'] == 'client'}
    chosen = list(clients) if targets is None else targets
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - clients.keys():
        raise ExecutionError('Expected nonempty, unique Client addresses')
    if playback is None:
        rounds = root / 'components' / 'client-0' / 'coordinator'
        available = [int(path.name) for path in rounds.iterdir() if path.is_dir() and path.name.isdecimal()] if rounds.exists() else []
        if not available:
            raise ExecutionError('No Client playback exists')
        playback = max(available)
    return [root / 'components' / clients[target]['id'] / 'playback' / str(playback) for target in chosen]



@case_interface('client/requests', effect='read')
async def requests(ctx, args: Requests) -> RequestsResult:
    """Read a sealed playback; filters limit returned details, never completeness validation."""
    directories = playback_directories(ctx.records.directory, ctx.runtime.topology.active(), args.targets, args.playback)
    ids = None if args.request_ids is None else set(args.request_ids)

    cancelled = threading.Event()

    def read():
        details, outcomes = [], {key: 0 for key in ('success', 'error', 'timeout', 'cancelled')}
        matched = 0
        for directory in directories:
            for record in read_requests(directory, cancelled):
                if (ids is not None and record.request_id not in ids
                        or args.outcome is not None and record.outcome != args.outcome
                        or args.since_ms is not None and record.sent_at_ms < args.since_ms
                        or args.until_ms is not None and record.sent_at_ms >= args.until_ms):
                    continue
                matched += 1
                outcomes[record.outcome] += 1
                if len(details) < args.limit:
                    details.append(record)
        return RequestsResult(matched, outcomes, details, matched > len(details))

    try:
        return await asyncio.to_thread(read)
    finally:
        cancelled.set()
