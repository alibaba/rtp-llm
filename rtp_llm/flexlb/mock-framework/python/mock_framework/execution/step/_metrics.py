"""Common Mock metric semantics. No request is routed through an Engine JVM."""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import json
import math
import re
from typing import Annotated, Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from ...config.validation import duration
from .. import ExecutionError
from ._targets import Targets


@dataclass(frozen=True)
class MetricArgs(Targets):
    range: Annotated[list[str], 'range'] = field(default_factory=lambda: ['-60s', '0s'])
    mode: Literal['interval', 'series'] = 'interval'


@dataclass(frozen=True)
class QuerySample:
    labels: dict[str, str]
    value: float
    sample_time: float


def epoch(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ExecutionError('Query time requires timezone')
    return parsed.timestamp()


def _query(url, path, parameters, timeout):
    body = urlencode(parameters).encode()
    request = Request(url.rstrip('/') + path, body, {'Content-Type': 'application/x-www-form-urlencoded'})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ExecutionError('Prometheus response exceeds limit')
    data = json.loads(raw, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    if data.get('status') != 'success' or data.get('warnings'):
        raise ExecutionError('Prometheus failed or returned partial/warning results')
    return data['data']


def fetch(url, expression, at, timeout):
    data = _query(url, '/api/v1/query',
                  {'query': expression, 'time': at, 'timeout': f'{max(1, int(timeout * 1000))}ms'}, timeout)
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
    monitor = ctx.runtime.monitoring
    if monitor is None or monitor.url is None:
        raise ExecutionError('This Case needs a runner-managed Prometheus')
    # A cancelled read can outlive its await only until the same bounded HTTP timeout.
    try:
        result = await asyncio.to_thread(fetch, monitor.url, expression, at, ctx.timeout)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ExecutionError(f'Prometheus query failed: {error}') from error
    ctx.records.emit(event='metric_query', query_time=ctx.query_time, at=at, promql=expression)
    return result


def window(ctx, args):
    end = epoch(ctx.query_time)
    start, stop = (duration(v, negative=True) for v in args.range)
    millis = (stop - start) * 1000
    if millis < 1 or millis != int(millis):
        raise ExecutionError('Metric window must contain integral milliseconds')
    return end + float(stop), int(millis), 'increase'


def scope(ctx, args, role):
    available = {item['rpc'] for item in ctx.runtime.topology.active(role)}
    targets = sorted(available) if args.targets is None else args.targets
    if not targets or set(targets) - available:
        raise ExecutionError(f'No known {role} metric targets or unknown address')
    expression = '|'.join(re.escape(address) for address in targets)
    return f'role={json.dumps(role)},instance=~{json.dumps(expression)}', targets


def history_range(ctx, name, labels, millis):
    if labels is not None:
        return f'{name}{{{labels}}}[{millis}ms]'
    # A derived worker gauge: sum priority at each scrape before averaging over time.
    step = max(1, round(ctx.runtime.monitoring.interval * 1000))
    return f'({name})[{millis}ms:{step}ms]'


async def counter_history(ctx, name, labels, targets, at, millis):
    counts = await instant(ctx, f'count_over_time({history_range(ctx, name, labels, millis)})', at)
    if len(counts) != len(targets) or {s.labels.get('instance') for s in counts} != set(targets) or any(s.value < 2 for s in counts):
        raise ExecutionError('Incomplete metric coverage for selected instances')


async def metric(ctx, args, role, numerator, denominator=None, *, kind=None):
    at, millis, operation = evaluation_window(ctx, args)
    labels, targets = scope(ctx, args, role)
    ctx.records.emit(event='metric_window', query_time=ctx.query_time, start=at - millis / 1000, end=at)
    # Every selected instance must contribute the counter history; aggregate alone hides missing peers.
    for name in [numerator] + ([denominator] if denominator else []):
        check_labels = labels + (f',kind={json.dumps(kind)}' if name == numerator and kind else '')
        await counter_history(ctx, name, check_labels, targets, at, millis)
    def aggregate(name, selector):
        return f'sum({operation}({name}{{{selector}}}[{millis}ms]))'
    expression = aggregate(numerator, labels + (f',kind={json.dumps(kind)}' if kind else ''))
    if denominator:
        expression += ' / ' + aggregate(denominator, labels)
    elif operation == 'increase':
        expression += ' / ' + str(Decimal(millis) / 1000)
    histories = [(name, labels + (f',kind={json.dumps(kind)}' if name == numerator and kind else ''), targets)
                 for name in [numerator] + ([denominator] if denominator else [])]
    return await evaluate(ctx, args, expression, histories)


def fetch_series(url, expression, start, end, step, timeout):
    data = _query(url, '/api/v1/query_range',
                  {'query': expression, 'start': start, 'end': end, 'step': step,
                   'timeout': f'{max(1, int(timeout * 1000))}ms'}, timeout)
    if data['resultType'] != 'matrix':
        raise ExecutionError('Expected a Prometheus range matrix')
    series = []
    for item in data['result']:
        labels = item['metric']
        if not isinstance(labels, dict) or any(type(k) is not str or type(v) is not str for k, v in labels.items()):
            raise ExecutionError('Invalid Prometheus labels')
        samples = []
        for pair in item['values']:
            if not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) not in (int, float) or type(pair[1]) is not str:
                raise ExecutionError('Invalid Prometheus range sample')
            point, value = float(pair[0]), float(pair[1])
            if not math.isfinite(point) or not math.isfinite(value):
                raise ExecutionError('Nonfinite Prometheus range sample')
            samples.append(QuerySample(labels, value, point))
        expected = int(round((end - start) / step)) + 1
        if len(samples) != expected or any(abs(s.sample_time - (start + i * step)) > .001 for i, s in enumerate(samples)):
            raise ExecutionError('Incomplete metric range coverage')
        series.append(samples)
    return series


def sample_period(ctx):
    # Like Grafana's minimum rate interval: enough scrapes for a stable counter delta.
    monitor = ctx.runtime.monitoring
    if monitor is None:
        raise ExecutionError('This Case needs a runner-managed Prometheus')
    return max(.001, monitor.interval) * 4


def evaluation_window(ctx, args):
    at, millis, operation = window(ctx, args)
    if args.mode == 'series':
        millis = max(1, round(sample_period(ctx) * 1000))
        operation = 'increase'
    return at, millis, operation


async def series_query(ctx, expression, start, end, step):
    monitor = ctx.runtime.monitoring
    if monitor is None or monitor.url is None:
        raise ExecutionError('This Case needs a runner-managed Prometheus')
    try:
        values = await asyncio.to_thread(fetch_series, monitor.url, expression, start, end, step, ctx.timeout)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ExecutionError(f'Prometheus range query failed: {error}') from error
    ctx.records.emit(event='metric_query', query_time=ctx.query_time, start=start, end=end, step=step, promql=expression)
    return values


async def evaluate(ctx, args, expression, histories=()):
    at, millis, _ = window(ctx, args)
    if args.mode == 'interval':
        values = await instant(ctx, expression, at)
        if len(values) != 1 or not math.isfinite(values[0].value):
            raise ExecutionError('Metric query must return one finite sample')
        return values[0].value
    step = sample_period(ctx) / 4
    count = int(millis / 1000 / step)
    if not 1 <= count <= 10000:
        raise ExecutionError('series requires 1–10000 sampling points')
    start = at - (count - 1) * step
    period = max(1, round(sample_period(ctx) * 1000))
    for name, labels, targets in histories:
        coverage = await series_query(ctx, f'count_over_time({history_range(ctx, name, labels, period)})', start, at, step)
        if len(coverage) != len(targets) or {s[0].labels.get('instance') for s in coverage} != set(targets) or any(
                point.value < 2 for samples in coverage for point in samples):
            raise ExecutionError('Incomplete metric coverage for selected instances')
    values = await series_query(ctx, expression, start, at, step)
    if len(values) != 1:
        raise ExecutionError('Metric query must return one complete series')
    points = values[0]
    ctx.records.emit(event='metric_series', samples=[{'time': p.sample_time, 'value': p.value} for p in points])
    return [p.value for p in points]


async def gauge(ctx, args, role, name):
    at, millis, _ = window(ctx, args)
    labels, targets = scope(ctx, args, role)
    worker = f'sum by(instance)({name}{{{labels}}})'
    await counter_history(ctx, worker, None, targets, at, millis)
    expression = f'sum(avg_over_time({history_range(ctx, worker, None, millis)}))' if args.mode == 'interval' else f'sum({worker})'
    return await evaluate(ctx, args, expression, [(worker, None, targets)])


async def latency(ctx, args, name):
    at, millis, operation = evaluation_window(ctx, args)
    labels, targets = scope(ctx, args, 'client')
    histories = [(name + suffix, labels, targets) for suffix in ('_sum', '_count')]
    for metric_name, selector, addresses in histories:
        await counter_history(ctx, metric_name, selector, addresses, at, millis)
    if args.statistic == 'mean':
        expression = f'sum({operation}({name}_sum{{{labels}}}[{millis}ms])) / sum({operation}({name}_count{{{labels}}}[{millis}ms]))'
    else:
        quantile = {'p95': .95, 'p99': .99}[args.statistic]
        expression = f'histogram_quantile({quantile}, sum by(le)({operation}({name}_bucket{{{labels}}}[{millis}ms])))'
    return await evaluate(ctx, args, expression, histories)
