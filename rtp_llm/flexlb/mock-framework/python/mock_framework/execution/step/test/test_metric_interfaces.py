from mock_framework.execution.step.client_success_rate import SuccessRateArgs
from mock_framework.execution.step.client_success_rate import ClientSuccessRate
from mock_framework.execution.step.engine_prefill_tps_with_cache import EnginePrefillTpsWithCache
from mock_framework.execution.step.engine_prefill_tps import EnginePrefillTps
from mock_framework.execution.step.engine_cache_hit_rate import EngineCacheHitRate
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from urllib.parse import parse_qs
import pytest
from mock_framework.execution.step import _metrics as metrics
from mock_framework.execution import ExecutionError


@pytest.fixture
def prometheus():
    state = {'body': {'status': 'success', 'data': {'resultType': 'vector', 'result': []}}, 'requests': []}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state['requests'].append(parse_qs(self.rfile.read(int(self.headers['Content-Length'])).decode()))
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(state['body']).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    state['url'] = 'http://127.0.0.1:' + str(server.server_port)
    try:
        yield state
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_raw_promql_is_sent_unchanged_at_fixed_time_and_empty_vector_stays_empty(prometheus):
    assert metrics.fetch(prometheus['url'], 'sum(rate(x[2m]))', 123.25, 2) == []
    assert prometheus['requests'] == [{'query': ['sum(rate(x[2m]))'], 'time': ['123.25'], 'timeout': ['2000ms']}]
    prometheus['body']['data'] = {'resultType': 'scalar', 'result': [123.25, '4']}
    assert metrics.fetch(prometheus['url'], '4', 123.25, 2) == [metrics.QuerySample({}, 4.0, 123.25)]


@pytest.mark.parametrize('change', [
    lambda b: b.update(warnings=['partial result']),
    lambda b: b.update(status='error'),
    lambda b: b.update(data={'resultType': 'matrix', 'result': []}),
    lambda b: b.update(data={'resultType': 'scalar', 'result': [1, 'NaN']}),
    lambda b: b.update(data={'resultType': 'scalar', 'result': [1, '+Inf']}),
    lambda b: b.update(data={'resultType': 'scalar', 'result': [True, '1']}),
    lambda b: b.update(data={'resultType': 'vector', 'result': [{'metric': {'bad': 1}, 'value': [1, '2']}]}),
])
def test_backend_errors_nonfinite_and_malformed_results_are_not_valid_observations(prometheus, change):
    change(prometheus['body'])
    with pytest.raises(ExecutionError):
        metrics.fetch(prometheus['url'], 'x', 1, 2)


def context():
    nodes = [
        {'role': 'prefill', 'rpc': '127.0.0.1:1001'},
        {'role': 'prefill', 'rpc': '127.0.0.1:1002'}]
    return SimpleNamespace(query_time='2026-10-28T00:00:00Z', nodes=nodes,
        records=SimpleNamespace(emit=lambda **event: None),
        runtime=SimpleNamespace(topology=SimpleNamespace(active=lambda role=None: [node for node in nodes if role is None or node['role'] == role])))


def test_hit_rate_aggregates_counters_before_division_and_uses_one_query_time(monkeypatch):
    calls = []
    async def instant(ctx, expression, at):
        calls.append((expression, at))
        if expression.startswith('count_over_time'):
            return [metrics.QuerySample({'instance': f'127.0.0.1:{port}'}, 4, at) for port in (1001, 1002)]
        return [metrics.QuerySample({}, .76, at)]
    monkeypatch.setattr(metrics, 'instant', instant)
    assert asyncio.run(EngineCacheHitRate.execute(context(), metrics.MetricArgs(range=['-90s', '-30s']))) == .76
    assert len(calls) == 3 and len({at for _, at in calls}) == 1
    assert calls[0][1] == metrics.epoch(context().query_time) - 30
    assert 'sum(increase(rtp_llm_cache_hit_tokens_total' in calls[2][0]
    assert ' / sum(increase(rtp_llm_cache_lookup_tokens_total' in calls[2][0]
    assert '[60000ms]' in calls[2][0]
    assert 'node=' not in calls[2][0] and 'engine=' not in calls[2][0]


@pytest.mark.parametrize('ports,counts', [([1001], [4]), ([1001, 1002], [4, 1]), ([1001, 1001, 1002], [4, 4, 4])])
def test_missing_insufficient_or_duplicate_instance_history_fails_before_aggregation(monkeypatch, ports, counts):
    calls = []
    async def instant(ctx, expression, at):
        calls.append(expression)
        return [metrics.QuerySample({'instance': f'127.0.0.1:{port}'}, count, at) for port, count in zip(ports, counts)]
    monkeypatch.setattr(metrics, 'instant', instant)
    with pytest.raises(ExecutionError, match='coverage'):
        asyncio.run(EngineCacheHitRate.execute(context(), metrics.MetricArgs()))
    assert len(calls) == 1


def test_explicit_target_scope_and_default_sixty_second_interval():
    ctx = context()
    selector, targets = metrics.scope(ctx, metrics.MetricArgs(targets=['127.0.0.1:1002']), 'prefill')
    assert targets == ['127.0.0.1:1002'] and '1001' not in selector
    assert metrics.window(ctx, metrics.MetricArgs()) == (metrics.epoch(ctx.query_time), 60000, 'increase')
    with pytest.raises(ExecutionError):
        metrics.scope(ctx, metrics.MetricArgs(targets=['old:1']), 'prefill')
    with pytest.raises(ValueError):
        metrics.MetricArgs(targets=[])


def test_metric_scope_uses_current_instances_and_rejects_removed_addresses():
    ctx = context()
    ctx.nodes.pop(0)
    args = metrics.MetricArgs(range=['-60s', '0s'])
    assert metrics.scope(ctx, args, 'prefill')[1] == ['127.0.0.1:1002']
    with pytest.raises(ExecutionError):
        metrics.scope(ctx, metrics.MetricArgs(targets=['127.0.0.1:1001'], range=['-60s', '0s']), 'prefill')


def test_success_rate_aggregates_all_terminals_without_a_request_count_floor(monkeypatch):
    calls = []
    ctx = context()
    events = []
    ctx.records.emit = lambda **event: events.append(event)
    for node in ctx.nodes:
        node['role'] = 'client'
    async def instant(ctx, expression, at):
        calls.append((expression, at))
        if expression.startswith('count_over_time'):
            return [metrics.QuerySample({'instance': f'127.0.0.1:{port}'}, 2, at) for port in (1001, 1002)]
        # Even one successful terminal in the interval is sufficient.
        return [metrics.QuerySample({}, 1.0, at)]
    monkeypatch.setattr(metrics, 'instant', instant)
    assert asyncio.run(ClientSuccessRate.execute(ctx, SuccessRateArgs(range=['-20s', '0s']))) == 1
    assert len(calls) == 5 and len({at for _, at in calls}) == 1
    assert all('outcome=' + json.dumps(outcome) in calls[i][0]
               for i, outcome in enumerate(('success', 'error', 'timeout', 'cancelled')))
    expression = calls[-1][0]
    numerator, denominator = expression.split(' / ')
    assert 'outcome="success"' in numerator and 'outcome=' not in denominator
    assert 'sum(increase(' in expression and '[20000ms]' in expression
    assert events == [{'event': 'metric_window', 'query_time': ctx.query_time,
                       'start': calls[-1][1] - 20, 'end': calls[-1][1]}]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1])
def test_success_rate_rejects_zero_terminal_nan_and_invalid_ratios(monkeypatch, value):
    ctx = context()
    for node in ctx.nodes:
        node['role'] = 'client'
    async def instant(ctx, expression, at):
        if expression.startswith('count_over_time'):
            return [metrics.QuerySample({'instance': f'127.0.0.1:{port}'}, 2, at) for port in (1001, 1002)]
        return [metrics.QuerySample({}, value, at)]
    monkeypatch.setattr(metrics, 'instant', instant)
    with pytest.raises(ExecutionError, match='terminal observations|finite sample'):
        asyncio.run(ClientSuccessRate.execute(ctx, SuccessRateArgs(range=['-20s', '0s'])))


def test_success_rate_rejects_a_missing_outcome_series(monkeypatch):
    ctx = context()
    for node in ctx.nodes:
        node['role'] = 'client'
    async def instant(ctx, expression, at):
        ports = (1001,) if 'outcome="timeout"' in expression else (1001, 1002)
        return [metrics.QuerySample({'instance': f'127.0.0.1:{port}'}, 2, at) for port in ports]
    monkeypatch.setattr(metrics, 'instant', instant)
    with pytest.raises(ExecutionError, match='coverage'):
        asyncio.run(ClientSuccessRate.execute(ctx, SuccessRateArgs(range=['-20s', '0s'])))


@pytest.mark.parametrize('args', [{}, {'range': 'latest'}, {'range': 'all'},
    {'range': ['0s', '0s']}, {'range': ['-20s', '1s']}, {'range': ['-20s']}])
def test_success_rate_requires_an_explicit_valid_window(args):
    from mock_framework.execution.step.registry import Registry
    from mock_framework.config import ConfigError
    with pytest.raises(ConfigError):
        Registry.standard().get('client/success_rate').arguments(args)


def test_success_rate_window_keeps_selected_clients_and_fixed_end(monkeypatch):
    from mock_framework.execution.step.registry import Registry
    ctx = context()
    for node in ctx.nodes:
        node['role'] = 'client'
    calls = []
    async def instant(ctx, expression, at):
        calls.append((expression, at))
        if expression.startswith('count_over_time'):
            return [metrics.QuerySample({'instance': '127.0.0.1:1002'}, 2, at)]
        return [metrics.QuerySample({}, .75, at)]
    monkeypatch.setattr(metrics, 'instant', instant)
    result = asyncio.run(Registry.standard().get('client/success_rate').invoke(ctx,
        {'targets': ['127.0.0.1:1002'], 'range': ['-40s', '-20s']}))
    assert result == .75
    assert all(at == metrics.epoch(ctx.query_time) - 20 for _, at in calls)
    assert all('1002' in expression and '1001' not in expression for expression, _ in calls)
    assert '[20000ms]' in calls[-1][0] and 'irate' not in calls[-1][0]


@pytest.mark.parametrize('method,name', [
    (EnginePrefillTps.execute, 'rtp_llm_context_tps'),
    (EnginePrefillTpsWithCache.execute, 'rtp_llm_context_tps_with_cache'),
])
def test_prefill_throughput_reads_the_real_named_gauge(monkeypatch, method, name):
    calls = []
    async def gauge(ctx, args, role, metric_name):
        calls.append((role, metric_name))
        return 42
    monkeypatch.setattr(metrics, 'gauge', gauge)
    assert asyncio.run(method(context(), metrics.MetricArgs())) == 42
    assert calls == [('prefill', name)]


def test_series_rejects_missing_points_and_nonfinite_samples(prometheus):
    prometheus['body']['data'] = {'resultType': 'matrix', 'result': [
        {'metric': {}, 'values': [[100, '0.8'], [101, '0.9'], [102, '1']]}]}
    samples = metrics.fetch_series(prometheus['url'], 'ratio', 100, 102, 1, 2)
    assert [s.value for s in samples[0]] == [.8, .9, 1]
    assert prometheus['requests'][0]['step'] == ['1']
    prometheus['body']['data']['result'][0]['values'].pop(1)
    with pytest.raises(ExecutionError, match='coverage'):
        metrics.fetch_series(prometheus['url'], 'ratio', 100, 102, 1, 2)
    prometheus['body']['data']['result'][0]['values'] = [[100, 'NaN'], [101, '1'], [102, '1']]
    with pytest.raises(ExecutionError, match='Nonfinite'):
        metrics.fetch_series(prometheus['url'], 'ratio', 100, 102, 1, 2)


def test_series_checks_every_point_and_each_instances_history(monkeypatch):
    from mock_framework.execution.step.assertions import matches_expectation
    ctx = context()
    ctx.runtime.monitoring = SimpleNamespace(interval=1)
    async def instant(ctx, expression, at):
        return [metrics.QuerySample({'instance': f'127.0.0.1:{p}'}, 4, at) for p in (1001, 1002)]
    async def series(ctx, expression, start, end, step):
        if expression.startswith('count_over_time'):
            return [[metrics.QuerySample({'instance': f'127.0.0.1:{p}'}, 4, t) for t in (end - 1, end)] for p in (1001, 1002)]
        return [[metrics.QuerySample({}, v, t) for v, t in zip((.9, .7), (end - 1, end))]]
    monkeypatch.setattr(metrics, 'instant', instant)
    monkeypatch.setattr(metrics, 'series_query', series)
    values = asyncio.run(EngineCacheHitRate.execute(ctx, metrics.MetricArgs(range=['-2s', '0s'], mode='series')))
    assert values == [.9, .7]
    assert not matches_expectation(values, {'gte': .8})
    assert matches_expectation(values, {'gte': .6, 'lte': 1})
    assert not matches_expectation([], {'gte': 0})


def test_latency_interval_uses_request_weights_and_merges_histogram_buckets(monkeypatch):
    from mock_framework.execution.step.client_ttft import ClientTtft, LatencyArgs
    ctx = context()
    for node in ctx.nodes:
        node['role'] = 'client'
    expressions = []
    async def instant(ctx, expression, at):
        expressions.append(expression)
        if expression.startswith('count_over_time'):
            return [metrics.QuerySample({'instance': f'127.0.0.1:{p}'}, 4, at) for p in (1001, 1002)]
        return [metrics.QuerySample({}, .25, at)]
    monkeypatch.setattr(metrics, 'instant', instant)
    assert asyncio.run(ClientTtft.execute(ctx, LatencyArgs(range=['-30s', '0s']))) == .25
    assert 'sum(increase(mock_client_ttft_seconds_sum' in expressions[-1]
    assert ' / sum(increase(mock_client_ttft_seconds_count' in expressions[-1]
    asyncio.run(ClientTtft.execute(ctx, LatencyArgs(range=['-30s', '0s'], statistic='p95')))
    assert 'histogram_quantile(0.95, sum by(le)(increase(' in expressions[-1]


@pytest.mark.parametrize('mode', ['interval', 'series'])
def test_prefill_priority_is_summed_before_time_average_and_coverage_is_per_worker(monkeypatch, mode):
    ctx = context()
    ctx.runtime.monitoring = SimpleNamespace(interval=1)
    calls = []
    async def instant(ctx, expression, at):
        calls.append(expression)
        if expression.startswith('count_over_time'):
            return [metrics.QuerySample({'instance': f'127.0.0.1:{p}'}, 4, at) for p in (1001, 1002)]
        return [metrics.QuerySample({}, 30, at)]
    async def series(ctx, expression, start, end, step):
        calls.append(expression)
        if expression.startswith('count_over_time'):
            return [[metrics.QuerySample({'instance': f'127.0.0.1:{p}'}, 4, end)] for p in (1001, 1002)]
        return [[metrics.QuerySample({}, 30, end)]]
    monkeypatch.setattr(metrics, 'instant', instant)
    monkeypatch.setattr(metrics, 'series_query', series)
    result = asyncio.run(EnginePrefillTps.execute(ctx, metrics.MetricArgs(range=['-10s', '0s'], mode=mode)))
    assert result == (30 if mode == 'interval' else [30])
    assert all('sum by(instance)(rtp_llm_context_tps{' in query for query in calls)
    if mode == 'interval':
        assert calls[-1].startswith('sum(avg_over_time((sum by(instance)(')
        assert '[10000ms:1000ms]' in calls[-1]
    else:
        assert calls[-1].startswith('sum(sum by(instance)(')


def test_prefill_gauge_rejects_a_missing_worker_after_priority_aggregation(monkeypatch):
    ctx = context()
    ctx.runtime.monitoring = SimpleNamespace(interval=1)
    async def instant(ctx, expression, at):
        return [metrics.QuerySample({'instance': '127.0.0.1:1001'}, 4, at)]
    monkeypatch.setattr(metrics, 'instant', instant)
    with pytest.raises(ExecutionError, match='coverage'):
        asyncio.run(EnginePrefillTps.execute(ctx, metrics.MetricArgs()))
