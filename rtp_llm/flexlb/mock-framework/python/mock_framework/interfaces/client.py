"""One current playback across all Clients; Java owns playback and coordination."""
import asyncio
from dataclasses import asdict, dataclass
import json
import math
import threading
from pathlib import Path
from typing import Annotated

from ..evidence import RequestRecord, playback_directories, read_requests, verify_playback
from ..config.registry import Bounds, case_interface
from ..config.values import duration
from ..execution.transport import ExecutionError, post


@dataclass(frozen=True)
class Request:
    token_ids: list[Annotated[int, Bounds(minimum=0, maximum=2147483647)]]
    max_new_tokens: Annotated[int, Bounds(minimum=1, maximum=2147483647)]
    priority: Annotated[int, Bounds(minimum=0, maximum=2147483647)] = 50

    def __post_init__(self):
        if not self.token_ids:
            raise ValueError('token_ids must not be empty')


@dataclass(frozen=True)
class Start:
    qps: float
    concurrency: Annotated[int, Bounds(minimum=1, maximum=2147483647)]
    requests: list[Request] | None = None
    requests_file: str | None = None
    masters: list[str] | None = None
    queue_capacity: Annotated[int, Bounds(minimum=1, maximum=100000)] = 16
    request_timeout: Annotated[str, 'duration'] = '30s'
    drain_timeout: Annotated[str, 'duration'] = '30s'

    def __post_init__(self):
        if self.masters is not None and (not self.masters or len(set(self.masters)) != len(self.masters) or any(not v.strip() for v in self.masters)):
            raise ValueError('masters must be nonempty and unique when supplied')
        if (self.requests is None) == (self.requests_file is None):
            raise ValueError('Specify exactly one of requests or requests_file')
        if not math.isfinite(self.qps) or self.qps <= 0:
            raise ValueError('Positive finite QPS is required')
        if self.requests is not None and (not self.requests or len(self.requests) / self.qps > 86400):
            raise ValueError('A nonempty text request list with at most 24h playback is required')
        if self.requests_file is not None and (not self.requests_file.strip() or '://' in self.requests_file):
            raise ValueError('requests_file must be a local file path')
        for value in (self.request_timeout, self.drain_timeout):
            millis = duration(value, positive=True) * 1000
            if millis != int(millis) or not 1 <= millis <= 3600000:
                raise ValueError('Client timeouts must be whole milliseconds between 1ms and 1h')


@dataclass(frozen=True)
class Empty:
    pass


@dataclass(frozen=True)
class ClientResult:
    done: bool
    stopped: bool
    produced: Annotated[int, Bounds(minimum=0)]
    issued: Annotated[int, Bounds(minimum=0)]
    finished: Annotated[int, Bounds(minimum=0)]
    unsent: Annotated[int, Bounds(minimum=0)]
    queued: Annotated[int, Bounds(minimum=0)]
    inflight: Annotated[int, Bounds(minimum=0)]
    outcomes: dict[str, Annotated[int, Bounds(minimum=0)]]
    errors: list[str]


def coordinator(processes):
    nodes = [node for node in processes.topology if node['role'] == 'client' and node['id'] == 'client-0']
    if len(nodes) != 1:
        raise ExecutionError('Expected exactly one client-0 coordinator')
    return nodes[0]['control']


async def control(processes, operation, args, query_time, timeout):
    envelope = {'interface': operation, 'args': args, 'context': {'run': 'foreground',
                'query_time': query_time, 'observation': None, 'timeout_ms': max(1, int(timeout * 1000))}}
    if len(json.dumps(envelope).encode()) > 1024 * 1024:
        raise ExecutionError('Inline Client requests exceed the 1 MiB control limit; use requests_file for larger inputs')
    return await post(coordinator(processes), envelope)


@case_interface('client/start', effect='write')
async def start(ctx, args: Start) -> bool:
    """Start the same finite plan on all Clients; rate and concurrency are global budgets."""
    clients = [node for node in ctx.processes.topology if node['role'] == 'client']
    if not clients or args.concurrency < len(clients) or args.requests is not None and len(args.requests) < len(clients):
        raise ExecutionError('Need at least one request and concurrency slot per Client')
    available = {node['rpc'] for node in ctx.processes.topology if node['role'] == 'master'}
    if not available or args.masters is not None and set(args.masters) - available:
        raise ExecutionError('Unknown Master address')
    if ctx.processes.client_pending:
        raise ExecutionError('Collect previous playback with client/stop before starting again')
    body = {key: value for key, value in asdict(args).items() if value is not None}
    if args.requests_file is not None:
        path = (ctx.processes.case.directory / Path(args.requests_file)).resolve()
        if not path.is_file():
            raise ExecutionError(f'Client requests_file is not a regular file: {path}')
        body['requests_file'] = str(path)
    body['request_timeout_ms'] = int(duration(body.pop('request_timeout')) * 1000)
    body['drain_timeout_ms'] = int(duration(body.pop('drain_timeout')) * 1000)
    # A lost start response can still leave Java sending: teardown must always attempt stop.
    ctx.processes.client_pending = True
    ctx.processes.client_stop_timeout = float(duration(args.drain_timeout)) + 15 + len(clients) * 6
    result = await control(ctx.processes, 'client/start', body, ctx.query_time, ctx.timeout)
    if result != {'accepted': True}:
        raise ExecutionError('Invalid Client start acknowledgement')
    return True


async def collect(processes, records, query_time, timeout):
    result = await control(processes, 'client/stop', {}, query_time, timeout)
    records.emit(event='client_result', result=result)
    if not isinstance(result, dict) or set(result) != {'clients', 'errors'}:
        raise ExecutionError('Invalid Client result')
    from ..config.registry import schema
    from ..config.validation import validate
    validate(result['clients'], schema(dict[str, ClientResult]), 'client/stop.result')
    validate(result['errors'], schema(list[str]), 'client/stop.errors')
    if not result['clients'] and not result['errors'] and not processes.client_pending:
        return {}
    expected = {node['control'] for node in processes.topology if node['role'] == 'client'}
    if result['errors'] or set(result['clients']) != expected or any(
            not value['done'] or value['errors'] or value['inflight'] or value['queued']
            or value['issued'] != value['finished'] or sum(value['outcomes'].values()) != value['finished']
            or set(value['outcomes']) != {'success', 'error', 'timeout', 'cancelled'} for value in result['clients'].values()):
        raise ExecutionError(f'Client playback failed or results incomplete: {result}')
    cancelled = threading.Event()
    try:
        await asyncio.to_thread(verify_playback, records.directory, processes.topology, result['clients'], cancelled)
    finally:
        cancelled.set()
    processes.client_pending = False
    by_address = {node['control']: node['rpc'] for node in processes.topology if node['role'] == 'client'}
    return {by_address[url]: value for url, value in result['clients'].items()}


@case_interface('client/stop', effect='write')
async def stop(ctx, args: Empty) -> dict[str, ClientResult]:
    """Stop all Clients and collect bounded cleanup; natural completion returns its saved result."""
    return await collect(ctx.processes, ctx.records, ctx.query_time, ctx.timeout)


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


@case_interface('client/requests', effect='read')
async def requests(ctx, args: Requests) -> RequestsResult:
    """Read a sealed playback; filters limit returned details, never completeness validation."""
    directories = playback_directories(ctx.records.directory, ctx.processes.topology, args.targets, args.playback)
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
