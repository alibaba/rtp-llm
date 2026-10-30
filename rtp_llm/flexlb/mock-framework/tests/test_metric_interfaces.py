import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from urllib.parse import parse_qs
import pytest
from mock_framework.interfaces import metrics
from mock_framework.execution.transport import ExecutionError


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
    assert asyncio.run(metrics.hit_rate(context(), metrics.MetricArgs(range=['-90s', '-30s']))) == .76
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
        asyncio.run(metrics.hit_rate(context(), metrics.MetricArgs()))
    assert len(calls) == 1


def test_explicit_target_scope_does_not_broadcast_and_latest_is_instant_rate():
    ctx = context()
    selector, targets = metrics.scope(ctx, metrics.MetricArgs(targets=['127.0.0.1:1002']), 'prefill')
    assert targets == ['127.0.0.1:1002'] and '1001' not in selector
    assert metrics.window(ctx, metrics.MetricArgs()) == (metrics.epoch(ctx.query_time), 60000, 'irate')
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
    assert asyncio.run(metrics.client_success_rate(ctx, metrics.SuccessRateArgs(range=['-20s', '0s']))) == 1
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
    with pytest.raises(ExecutionError, match='terminal observations'):
        asyncio.run(metrics.client_success_rate(ctx, metrics.SuccessRateArgs(range=['-20s', '0s'])))


def test_success_rate_rejects_a_missing_outcome_series(monkeypatch):
    ctx = context()
    for node in ctx.nodes:
        node['role'] = 'client'
    async def instant(ctx, expression, at):
        ports = (1001,) if 'outcome="timeout"' in expression else (1001, 1002)
        return [metrics.QuerySample({'instance': f'127.0.0.1:{port}'}, 2, at) for port in ports]
    monkeypatch.setattr(metrics, 'instant', instant)
    with pytest.raises(ExecutionError, match='coverage'):
        asyncio.run(metrics.client_success_rate(ctx, metrics.SuccessRateArgs(range=['-20s', '0s'])))


@pytest.mark.parametrize('args', [{}, {'range': 'latest'}, {'range': 'all'},
    {'range': ['0s', '0s']}, {'range': ['-20s', '1s']}, {'range': ['-20s']}])
def test_success_rate_requires_an_explicit_valid_window(args):
    from mock_framework.config import Registry
    from mock_framework.config.errors import ConfigError
    with pytest.raises(ConfigError):
        Registry.standard().get('client/success_rate').arguments(args)


def test_success_rate_window_keeps_selected_clients_and_fixed_end(monkeypatch):
    from mock_framework.config import Registry
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


@pytest.mark.parametrize('method,denominator', [
    (metrics.prefill_tps, 'rtp_llm_context_forward_seconds_total'),
    (metrics.cached_tps, 'rtp_llm_context_forward_with_cache_seconds_total'),
])
def test_prefill_throughput_uses_the_matching_nonempty_forward_duration(monkeypatch, method, denominator):
    calls = []
    async def metric(ctx, args, role, numerator, duration, **labels):
        calls.append((role, numerator, duration, labels))
        return 42
    monkeypatch.setattr(metrics, 'metric', metric)
    assert asyncio.run(method(context(), metrics.MetricArgs())) == 42
    assert calls[0][1:3] == ('rtp_llm_forward_tokens_total', denominator)
