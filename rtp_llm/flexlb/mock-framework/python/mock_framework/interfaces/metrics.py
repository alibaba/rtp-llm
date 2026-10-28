"""Common Mock metric semantics. No request is routed through an Engine JVM."""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import json
import math
import re
from typing import Annotated
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..config.registry import case_interface
from ..config.values import duration
from ..execution.transport import ExecutionError
from .master import Targets


@dataclass(frozen=True)
class MetricArgs(Targets):
    range: Annotated[str | list[str], 'range'] = 'latest'


@dataclass(frozen=True)
class Promql:
    promql: str

    def __post_init__(self):
        if not self.promql.strip():
            raise ValueError('promql must not be blank')


@dataclass(frozen=True)
class QuerySample:
    labels: dict[str, str]
    value: float
    sample_time: float


@dataclass(frozen=True)
class QueryResult:
    samples: list[QuerySample]
    query_time: str


def epoch(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ExecutionError('Query time requires timezone')
    return parsed.timestamp()


def fetch(url, expression, at, timeout):
    body = urlencode({'query': expression, 'time': at, 'timeout': f'{max(1, int(timeout * 1000))}ms'}).encode()
    request = Request(url.rstrip('/') + '/api/v1/query', body, {'Content-Type': 'application/x-www-form-urlencoded'})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ExecutionError('Prometheus response exceeds limit')
    data = json.loads(raw, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    if data.get('status') != 'success' or data.get('warnings'):
        raise ExecutionError('Prometheus failed or returned partial/warning results')
    data = data['data']
    kind = data['resultType']
    if kind not in ('scalar', 'vector'):
        raise ExecutionError('Expected scalar or float instant vector')
    rows = [{'metric': {}, 'value': data['result']}] if kind == 'scalar' else data['result']
    result = []
    for item in rows:
        labels, pair = item['metric'], item['value']
        if not isinstance(labels, dict) or any(type(k) is not str or type(v) is not str for k, v in labels.items()) \
                or not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) not in (int, float) or type(pair[1]) is not str:
            raise ExecutionError('Invalid Prometheus sample')
        value, sample_time = float(pair[1]), float(pair[0])
        if not math.isfinite(value) or not math.isfinite(sample_time):
            raise ExecutionError('Nonfinite Prometheus sample; missing observations or zero denominator')
        result.append(QuerySample(labels, value, sample_time))
    return result


async def instant(ctx, expression, at):
    monitor = ctx.processes.prometheus
    if monitor is None or monitor.url is None:
        raise ExecutionError('This Case needs a runner-managed Prometheus')
    # A cancelled read can outlive its await only until the same bounded HTTP timeout.
    try:
        result = await asyncio.to_thread(fetch, monitor.url, expression, at, ctx.timeout)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ExecutionError(f'Prometheus query failed: {error}') from error
    ctx.records.emit(event='metric_query', query_time=ctx.query_time, at=at, promql=expression)
    return result


@case_interface('prometheus/query', effect='read')
async def query(ctx, args: Promql) -> QueryResult:
    """Execute raw PromQL at the fixed step time without rewriting it."""
    return QueryResult(await instant(ctx, args.promql, epoch(ctx.query_time)), ctx.query_time)


def window(ctx, args):
    end = epoch(ctx.query_time)
    if args.range == 'latest':
        return end, 60000, 'irate'
    start, stop = (duration(v, negative=True) for v in args.range)
    millis = (stop - start) * 1000
    if millis < 1 or millis != int(millis):
        raise ExecutionError('Metric window must contain integral milliseconds')
    return end + float(stop), int(millis), 'increase'


def scope(ctx, args, role):
    at, millis, _ = window(ctx, args)
    available = {item['rpc'] for item in ctx.processes.history if item['role'] == role
                 and epoch(item['started_at']) <= at
                 and ('stopped_at' not in item or epoch(item['stopped_at']) > at - millis / 1000)}
    targets = sorted(available) if args.targets is None else args.targets
    if not targets or set(targets) - available:
        raise ExecutionError(f'No known {role} metric targets or unknown address')
    expression = '|'.join(re.escape(address) for address in targets)
    return f'role={json.dumps(role)},instance=~{json.dumps(expression)}', targets


async def metric(ctx, args, role, numerator, denominator=None, *, kind=None):
    at, millis, operation = window(ctx, args)
    labels, targets = scope(ctx, args, role)
    # Every selected instance must contribute the counter history; aggregate alone hides missing peers.
    for name in [numerator] + ([denominator] if denominator else []):
        check_labels = labels + (f',kind={json.dumps(kind)}' if name == numerator and kind else '')
        counts = await instant(ctx, f'count_over_time({name}{{{check_labels}}}[{millis}ms])', at)
        if len(counts) != len(targets) or {s.labels.get('instance') for s in counts} != set(targets) or any(s.value < 2 for s in counts):
            raise ExecutionError('Incomplete metric coverage for selected instances')
    def aggregate(name, selector):
        return f'sum({operation}({name}{{{selector}}}[{millis}ms]))'
    expression = aggregate(numerator, labels + (f',kind={json.dumps(kind)}' if kind else ''))
    if denominator:
        expression += ' / ' + aggregate(denominator, labels)
    elif operation == 'increase':
        expression += ' / ' + str(Decimal(millis) / 1000)
    values = await instant(ctx, expression, at)
    if len(values) != 1:
        raise ExecutionError('Metric query must return one finite sample')
    return values[0].value


@case_interface('engine/cache/hit_rate', effect='read', unit='ratio')
async def hit_rate(ctx, args: MetricArgs) -> float:
    return await metric(ctx, args, 'prefill', 'mock_engine_cache_hit_tokens_total', 'mock_engine_cache_lookup_tokens_total')


@case_interface('engine/prefill/tps', effect='read', unit='tokens/s')
async def prefill_tps(ctx, args: MetricArgs) -> float:
    return await metric(ctx, args, 'prefill', 'mock_engine_forward_tokens_total', 'mock_engine_forward_seconds_sum', kind='compute')


@case_interface('engine/prefill/tps_with_cache', effect='read', unit='tokens/s')
async def cached_tps(ctx, args: MetricArgs) -> float:
    return await metric(ctx, args, 'prefill', 'mock_engine_forward_tokens_total', 'mock_engine_forward_seconds_sum', kind='input')


@case_interface('engine/decode/tps', effect='read', unit='tokens/s')
async def decode_tps(ctx, args: MetricArgs) -> float:
    return await metric(ctx, args, 'decode', 'mock_engine_forward_tokens_total', kind='output')
