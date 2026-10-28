import asyncio
from types import SimpleNamespace
import pytest
from mock_framework.config import Registry
from mock_framework.config.errors import ConfigError
from mock_framework.interfaces import client, metrics
from mock_framework.execution.transport import ExecutionError


def context():
    events = []
    return SimpleNamespace(processes=SimpleNamespace(client_pending=False, topology=[
        {'id': 'master-0', 'role': 'master', 'rpc': 'master:1'},
        {'id': 'client-0', 'role': 'client', 'rpc': 'c:1', 'control': 'http://c:1'},
        {'id': 'client-1', 'role': 'client', 'rpc': 'c:2', 'control': 'http://c:2'}]),
        query_time='2026-10-28T00:00:00Z', timeout=20, records=SimpleNamespace(emit=lambda **v: events.append(v)), events=events)


def args():
    return {'masters': ['master:1'], 'requests': [{'token_ids': [1, 2], 'max_new_tokens': 3}] * 2,
            'qps': 10, 'concurrency': 2}


def test_start_only_calls_client_zero_and_does_not_multiply_total_rate(monkeypatch):
    calls = []
    async def post(url, body):
        calls.append((url, body)); return {'accepted': True}
    monkeypatch.setattr(client, 'post', post)
    ctx = context()
    assert asyncio.run(Registry.standard().get('client/start').invoke(ctx, args())) is True
    assert len(calls) == 1 and calls[0][0] == 'http://c:1'
    assert calls[0][1]['args']['qps'] == 10
    assert calls[0][1]['args']['concurrency'] == 2
    assert calls[0][1]['args']['master'] == 'master:1'
    assert 'masters' not in calls[0][1]['args'] and ctx.processes.client_pending
    with pytest.raises(ExecutionError, match='previous'):
        asyncio.run(Registry.standard().get('client/start').invoke(ctx, args()))


def test_unknown_start_response_is_still_cleaned_up(monkeypatch):
    async def post(*a):
        raise OSError('connection lost')
    monkeypatch.setattr(client, 'post', post)
    ctx = context()
    with pytest.raises(OSError):
        asyncio.run(Registry.standard().get('client/start').invoke(ctx, args()))
    assert ctx.processes.client_pending


def state():
    return dict(done=True, stopped=False, produced=1, issued=1, finished=1, unsent=0, queued=0, inflight=0, outcomes={'success': 1, 'error': 0, 'timeout': 0, 'cancelled': 0}, errors=[])


def test_stop_collects_all_original_clients_without_task_id(monkeypatch):
    calls = []
    async def post(url, body):
        calls.append(body)
        return {'clients': {'http://c:1': state(), 'http://c:2': state()}, 'errors': []}
    monkeypatch.setattr(client, 'post', post)
    ctx = context(); ctx.processes.client_pending = True
    result = asyncio.run(Registry.standard().get('client/stop').invoke(ctx, {}))
    assert set(result) == {'c:1', 'c:2'} and not ctx.processes.client_pending
    assert calls[0]['args'] == {} and ctx.events[0]['event'] == 'client_result'


@pytest.mark.parametrize('bad', ['missing', 'overflow', 'unfinished'])
def test_stop_does_not_hide_missing_workers_or_execution_errors(monkeypatch, bad):
    value = {'clients': {'http://c:1': state(), 'http://c:2': state()}, 'errors': []}
    if bad == 'missing': del value['clients']['http://c:2']
    if bad == 'overflow': value['errors'] = ['queue full']
    if bad == 'unfinished': value['clients']['http://c:2']['done'] = False
    async def post(*a): return value
    monkeypatch.setattr(client, 'post', post)
    with pytest.raises(ExecutionError):
        asyncio.run(Registry.standard().get('client/stop').invoke(context(), {}))


@pytest.mark.parametrize('changes', [{'qps': 0}, {'qps': float('inf')}, {'masters': []}, {'masters': ['a', 'b']},
                                      {'queue_capacity': 0}, {'request_timeout': '0s'}, {'requests': []},
                                      {'requests': [{'token_ids': [], 'max_new_tokens': 1}]}])
def test_invalid_plans_fail_at_python_boundary(changes):
    with pytest.raises(ConfigError):
        Registry.standard().get('client/start').arguments(args() | changes)


def test_qps_reuses_metric_aggregation_and_measures_issued_not_finished(monkeypatch):
    async def metric(ctx, selection, role, numerator):
        assert role == 'client' and numerator == 'mock_client_requests_issued_total'
        return 123.0
    monkeypatch.setattr(metrics, 'metric', metric)
    assert asyncio.run(metrics.client_qps(context(), metrics.MetricArgs())) == 123.0


def test_client_zero_coordinates_two_jvms_through_real_master(tmp_path):
    import json
    import os
    from pathlib import Path
    import yaml
    from mock_framework.config import compile_case
    from mock_framework.execution import prepare_launch, run_case
    from test_master import configuration

    root = Path(__file__).resolve().parents[1]
    java = str(Path(os.environ['JAVA_HOME']) / 'bin/java')
    master_classpath = os.pathsep.join([str(root / 'java/master/target/classes'),
        (root / 'java/master/target/dependencies.txt').read_text().strip()])
    native = configuration()
    native['FLEXLB_CONFIG'] = json.dumps({'scheduler': {'type': 'DIRECT'}, 'dispatcher': {'type': 'NON_BATCH'},
        'requestLifecycle': {'request': {'timeoutMs': 5000}},
        'grpcServer': {'executorCoreSize': 4, 'executorMaxSize': 8, 'executorQueueSize': 64, 'shutdownQuietPeriodMs': 10}})
    engine = {'engine': 'rtp', 'config': {'seq_size_per_block': 16, 'test_block_num': 32,
              'reserve_block_ratio': 0, 'max_seq_len': 4096, 'compute.expression': '1 + batchSize'}}
    data = {'id': 'client_playback', 'describe': 'Client0 coordinates two real JVMs',
        'environment': {'master': {'config': native}, 'prefill': engine, 'decode': engine,
                        'client': {'deployment': {'replicas': 2}, 'config': {}}},
        'phases': [{'id': 'traffic', 'describe': 'Play and collect', 'timeout': '35s', 'steps': [
            {'id': 'start', 'describe': 'Start global request plan', 'interface': 'client/start', 'args': {
                'masters': {'ref': 'globals.masters'}, 'requests': [{'token_ids': [1, 2, 3], 'max_new_tokens': 3}] * 12,
                'qps': 4, 'concurrency': 4, 'queue_capacity': 8, 'request_timeout': '5s', 'drain_timeout': '5s'}, 'wait_after': '6s'},
            {'id': 'stop', 'describe': 'Collect every Client', 'interface': 'client/stop'}]}]}
    path = tmp_path / 'case.yaml'; path.write_text(yaml.safe_dump(data))
    registry = Registry.standard(); case = compile_case(path, registry)
    components = {'master': {'command': [java, '-XX:ActiveProcessorCount=2', '-cp', master_classpath, 'org.flexlb.MasterNode'], 'startup_timeout': '60s'},
                  'client': {'command': [java, '-jar', str(root / 'java/client/target/mock-client-1.0-SNAPSHOT-all.jar')]}}
    for role in ('prefill', 'decode'):
        components[role] = {'command': [java, '-jar', str(root / 'java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar')]}
    recipe = {'components': components, 'zookeeper': {'command': [java, '-XX:ActiveProcessorCount=2', '-cp', master_classpath, 'org.flexlb.mock.master.ZookeeperNode']}}
    if os.environ.get('PROMETHEUS_BIN'):
        recipe['prometheus'] = {'binary': os.environ['PROMETHEUS_BIN'], 'scrape_interval': '250ms'}
        data['phases'][0]['steps'].insert(1, {'id': 'qps', 'describe': 'Check aggregate issued rate',
            'interface': 'client/qps', 'args': {'range': ['-6s', '0s']}, 'expect': {'gte': 1.5, 'lte': 3}})
        path.write_text(yaml.safe_dump(data)); case = compile_case(path, registry)
    launch = prepare_launch(recipe, case, tmp_path)
    output = tmp_path / 'result'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    summary = json.loads((output / 'components/client-0/coordinator/1/result.json').read_text())
    assert not summary['errors'] and len(summary['clients']) == 2
    assert sum(v['issued'] for v in summary['clients'].values()) == 12
    assert all(v['finished'] == 6 and v['outcomes']['success'] == 6 and v['unsent'] == 0 and not v['stopped'] for v in summary['clients'].values())
    pids = [json.loads(line)['pid'] for line in (output / 'events.jsonl').read_text().splitlines()
            if json.loads(line).get('event') == 'process_started' and json.loads(line).get('node', '').startswith('client-')]
    assert len(set(pids)) == 2


@pytest.mark.parametrize('failed', [False, True])
def test_teardown_collects_unstopped_playback_before_final_scrape(monkeypatch, failed, tmp_path):
    from mock_framework.execution.processes import Processes
    order = []
    async def collect(*a):
        order.append('client')
        if failed:
            raise ExecutionError('queue full')
    async def scrape(): order.append('scrape')
    async def stop_processes(*a): order.append('processes'); return []
    monkeypatch.setattr(client, 'collect', collect)
    async def exercise():
        p = Processes(None, {}, SimpleNamespace(directory=tmp_path))
        p.client_pending = True
        p.prometheus = SimpleNamespace(final_scrape=scrape, finish=lambda errors: None)
        p._stop = stop_processes
        return await p.close()
    errors = asyncio.run(exercise())
    assert order[:2] == ['client', 'scrape']
    assert bool(errors) is failed
    if failed: assert 'queue full' in errors[0]
