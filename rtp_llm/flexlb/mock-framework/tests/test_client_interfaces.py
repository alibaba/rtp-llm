import asyncio
from types import SimpleNamespace
from pathlib import Path
import pytest
from mock_framework.config import Registry
from mock_framework.config.errors import ConfigError
from mock_framework.interfaces import metrics
from mock_framework.execution.transport import ExecutionError
from mock_framework.execution import clients as client_runtime
from mock_framework.execution.clients import ClientGroup
from helpers.input import read_inputs, write_inputs, pb


def context():
    events = []
    nodes = [
        {'id': 'master-0', 'role': 'master', 'rpc': 'master:1'},
        {'id': 'client-0', 'role': 'client', 'rpc': 'c:1', 'control': 'http://c:1'},
        {'id': 'client-1', 'role': 'client', 'rpc': 'c:2', 'control': 'http://c:2'}]
    topology = SimpleNamespace(active=lambda role=None: [item for item in nodes if role is None or item['role'] == role])
    records = SimpleNamespace(emit=lambda **v: events.append(v))
    clients = ClientGroup(component=None, recipe=None, launcher=None, supervisor=None,
        topology=topology, records=records, monitoring=None)
    return SimpleNamespace(runtime=SimpleNamespace(topology=topology, clients=clients, case=SimpleNamespace(directory=Path(__file__).resolve().parents[1])), nodes=nodes,
        query_time='2026-10-28T00:00:00Z', timeout=20, records=records, events=events)


def args():
    return {'masters': ['master:1'], 'requests_file': str(Path(__file__).resolve().parents[1] / 'cases/basic/requests.binpb.zst'),
            'qps': 10, 'concurrency': 2}


def test_start_only_calls_client_zero_and_does_not_multiply_total_rate(monkeypatch):
    calls = []
    async def post(url, body):
        calls.append((url, body)); return {'accepted': True}
    monkeypatch.setattr(client_runtime, 'post', post)
    ctx = context()
    assert asyncio.run(Registry.standard().get('client/start').invoke(ctx, args())) is True
    assert len(calls) == 1 and calls[0][0] == 'http://c:1'
    assert calls[0][1]['args']['qps'] == 10
    assert calls[0][1]['args']['concurrency'] == 2
    assert calls[0][1]['args']['masters'] == ['master:1']
    assert 'master' not in calls[0][1]['args'] and ctx.runtime.clients.pending
    with pytest.raises(ExecutionError, match='previous'):
        asyncio.run(Registry.standard().get('client/start').invoke(ctx, args()))


def test_unknown_start_response_is_still_cleaned_up(monkeypatch):
    async def post(*a):
        raise OSError('connection lost')
    monkeypatch.setattr(client_runtime, 'post', post)
    ctx = context()
    with pytest.raises(OSError):
        asyncio.run(Registry.standard().get('client/start').invoke(ctx, args()))
    assert ctx.runtime.clients.pending


def state():
    return dict(done=True, stopped=False, produced=1, issued=1, finished=1, unsent=0, queued=0, inflight=0, outcomes={'success': 1, 'error': 0, 'timeout': 0, 'cancelled': 0}, errors=[])


def test_stop_collects_all_original_clients_without_task_id(monkeypatch, tmp_path):
    calls = []
    async def post(url, body):
        calls.append(body)
        return {'clients': {'http://c:1': state(), 'http://c:2': state()}, 'errors': []}
    monkeypatch.setattr(client_runtime, 'post', post)
    ctx = context(); ctx.runtime.clients.pending = True; ctx.records.directory = tmp_path
    result = asyncio.run(Registry.standard().get('client/stop').invoke(ctx, {}))
    assert set(result) == {'c:1', 'c:2'} and not ctx.runtime.clients.pending
    assert calls[0]['args'] == {} and ctx.events[0]['event'] == 'client_result'


@pytest.mark.parametrize('bad', ['missing', 'overflow', 'unfinished', 'writer'])
def test_stop_does_not_hide_missing_workers_or_execution_errors(monkeypatch, bad):
    value = {'clients': {'http://c:1': state(), 'http://c:2': state()}, 'errors': []}
    if bad == 'missing': del value['clients']['http://c:2']
    if bad == 'overflow': value['errors'] = ['queue full']
    if bad == 'unfinished': value['clients']['http://c:2']['done'] = False
    if bad == 'writer': value['clients']['http://c:2']['errors'] = ['Client evidence write failed']
    async def post(*a): return value
    monkeypatch.setattr(client_runtime, 'post', post)
    with pytest.raises(ExecutionError):
        asyncio.run(Registry.standard().get('client/stop').invoke(context(), {}))


@pytest.mark.parametrize('changes', [{'qps': 0}, {'qps': float('inf')}, {'masters': []}, {'masters': ['a', 'a']},
                                      {'queue_capacity': 0}, {'request_timeout': '0s'}, {'requests_file': ''}, {'requests_file': None}])
def test_invalid_plans_fail_at_python_boundary(changes):
    with pytest.raises(ConfigError):
        Registry.standard().get('client/start').arguments(args() | changes)


def test_qps_reuses_metric_aggregation_and_measures_issued_not_finished(monkeypatch):
    async def metric(ctx, selection, role, numerator):
        assert role == 'client' and numerator == 'mock_client_requests_issued_total'
        return 123.0
    monkeypatch.setattr(metrics, 'metric', metric)
    assert asyncio.run(metrics.client_qps(context(), metrics.MetricArgs())) == 123.0


@pytest.mark.parametrize('replicas,large_input', [(2, False), (1, True), (2, True)], ids=['small-two', 'large-one', 'large-two'])
def test_client_playback_through_real_master(tmp_path, replicas, large_input):
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
    # Match the sender's concurrency so this test measures playback, not DIRECT admission rejection.
    native['FLEXLB_CONFIG'] = json.dumps({'scheduler': {'type': 'DIRECT'},
        'dispatcher': {'type': 'NON_BATCH', 'maxInflightPerPrefillWorker': 16},
        'requestLifecycle': {'request': {'timeoutMs': 5000}},
        'grpcServer': {'executorCoreSize': 4, 'executorMaxSize': 8, 'executorQueueSize': 64, 'shutdownQuietPeriodMs': 10}})
    engine = {'engine': 'rtp', 'config': {'seq_size_per_block': 16, 'test_block_num': 4096,
              'reserve_block_ratio': 0, 'max_seq_len': 4096, 'compute.forward.expression': '1 + batchSize'}}
    data = {'id': 'client_playback', 'describe': 'Client0 coordinates native playback',
        'environment': {'master': {'config': native}, 'prefill': engine, 'decode': engine,
                        'client': {'deployment': {'replicas': replicas}, 'config': {}}},
        'phases': [{'id': 'traffic', 'describe': 'Play and collect', 'timeout': '35s', 'steps': [
            {'id': 'start', 'describe': 'Start global request plan', 'interface': 'client/start', 'args': {
                'masters': {'ref': 'globals.masters'}, 'requests_file': 'requests.binpb.zst',
                'qps': 4, 'concurrency': 4, 'queue_capacity': 8, 'request_timeout': '5s', 'drain_timeout': '5s'}, 'wait_after': '6s'},
            {'id': 'stop', 'describe': 'Collect every Client', 'interface': 'client/stop'}]}]}
    count = 720 if large_input else 12
    source = tmp_path / 'requests.binpb.zst'
    write_inputs(source, (pb.Request(token_ids=[index + 10000] * (1024 if large_input else 3), max_new_tokens=3)
                          for index in range(count)))
    if large_input:
        assert sum(request.ByteSize() for request in read_inputs(source)) > 1024 * 1024
        start = data['phases'][0]['steps'][0]
        start['args'].update(qps=24, concurrency=16, queue_capacity=32)
        start['wait_after'] = '24s'
        data['phases'][0]['timeout'] = '60s'
    data['phases'][0]['steps'].append({'id': 'records', 'describe': 'Read sealed Client evidence',
        'interface': 'client/requests', 'args': {'limit': 2}})
    path = tmp_path / 'case.yaml'; path.write_text(yaml.safe_dump(data))
    registry = Registry.standard(); case = compile_case(path, registry)
    components = {'master': {'command': [java, '-XX:ActiveProcessorCount=2', '-cp', master_classpath, 'org.flexlb.MasterNode'], 'startup_timeout': '60s'},
                  'client': {'command': [java, '-jar', str(root / 'java/client/target/mock-client-1.0-SNAPSHOT-all.jar')]}}
    for role in ('prefill', 'decode'):
        components[role] = {'command': [java, '-cp', str(root / 'java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar'), 'org.flexlb.mock.host.EngineHost']}
    recipe = {'components': components, 'zookeeper': {'command': [java, '-XX:ActiveProcessorCount=2', '-cp', master_classpath, 'org.flexlb.mock.master.ZookeeperNode']}}
    if os.environ.get('PROMETHEUS_BIN'):
        recipe['prometheus'] = {'binary': os.environ['PROMETHEUS_BIN'], 'scrape_interval': '250ms'}
        data['phases'][0]['steps'].insert(1, {'id': 'qps', 'describe': 'Check aggregate issued rate',
            'interface': 'client/qps', 'args': {'range': ['-10s' if large_input else '-6s', '0s']},
            'expect': {'gte': 21.6, 'lte': 26.4} if large_input else {'gte': 1.5, 'lte': 3},
            'wait_after': '10s' if large_input else '0s'})
        data['phases'][0]['steps'].insert(2, {'id': 'success', 'describe': 'Check aggregate terminal success rate',
            'interface': 'client/success_rate', 'args': {'range': ['-20s', '0s']}, 'expect': {'eq': 1.0}})
        path.write_text(yaml.safe_dump(data)); case = compile_case(path, registry)
    elif large_input:
        data['phases'][0]['steps'][0]['wait_after'] = '34s'
        path.write_text(yaml.safe_dump(data)); case = compile_case(path, registry)
    launch = prepare_launch(recipe, case, tmp_path)
    output = tmp_path / 'result'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    if 'prometheus' in recipe:
        gates = [json.loads(line) for line in (output / 'gates.jsonl').read_text().splitlines()]
        success, = [gate for gate in gates if gate['interface'] == 'client/success_rate']
        assert success['actual'] == 1.0
    summary = json.loads((output / 'components/client-0/coordinator/1/result.json').read_text())
    assert not summary['errors'] and len(summary['clients']) == replicas
    assert sum(v['issued'] for v in summary['clients'].values()) == count
    assert all(v['finished'] == count // replicas and v['outcomes']['success'] == count // replicas and v['unsent'] == 0 and not v['stopped'] for v in summary['clients'].values())
    pids = [json.loads(line)['pid'] for line in (output / 'events.jsonl').read_text().splitlines()
            if json.loads(line).get('event') == 'process_started' and json.loads(line).get('node', '').startswith('client-')]
    assert len(set(pids)) == replicas
    manifest = json.loads((output / 'components/client-0/coordinator/1/input.json').read_text())
    assert manifest['count'] == count and len(manifest['shards']) == replicas
    seen = set()
    for shard in manifest['shards']:
        import hashlib
        contents = Path(shard['path']).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == shard['sha256']
        if large_input:
            for request in read_inputs(shard['path']):
                index = request.token_ids[0] - 10000
                assert index not in seen
                seen.add(index)
    if large_input: assert seen == set(range(count))
    from mock_framework.interfaces.client import playback_directories
    from mock_framework.interfaces.client._records import read_requests
    assert not (output / 'topology-history.json').exists()
    topology = [json.loads(path.read_text()) for path in sorted(output.glob('components/client-*/endpoints.json'))]
    records = [record for directory in playback_directories(output, topology) for record in read_requests(directory)]
    assert len(records) == count and {record.request_id for record in records} == set(range(1, count + 1))
    assert all(record.outcome == 'success' and record.ttft_nanos is not None and record.output_tokens == 3 for record in records)
    assert all(len(record.attempts) == 1 and record.attempts[0].grpc_code == 'OK'
               and record.attempts[0].response_success is True for record in records)
    queried, = [event['result'] for line in (output / 'events.jsonl').read_text().splitlines()
                if (event := json.loads(line)).get('event') == 'step_finished' and event.get('step') == 'records']
    assert queried['matched'] == count and queried['outcomes']['success'] == count
    assert len(queried['records']) == 2 and queried['truncated']


@pytest.mark.parametrize('failed', [False, True])
def test_teardown_collects_unstopped_playback_before_final_scrape(failed, tmp_path):
    from mock_framework.execution.environment import RunEnvironment
    order = []
    async def collect(*a):
        order.append('client')
        if failed:
            raise ExecutionError('queue full')
    async def scrape(): order.append('scrape')
    async def close_monitor(): order.append('monitor stopped')
    async def exercise():
        runtime = RunEnvironment(SimpleNamespace(environment={}), {'components': {}, 'engine_mode': 'shared'},
            SimpleNamespace(directory=tmp_path, emit=lambda **event: None))
        runtime.clients.pending = True
        runtime.clients.collect = collect
        runtime.monitoring = SimpleNamespace(final_scrape=scrape, close=close_monitor, finish=lambda errors: None)
        return await runtime.close()
    errors = asyncio.run(exercise())
    assert order == ['client', 'scrape', 'monitor stopped']
    assert bool(errors) is failed
    if failed: assert 'queue full' in errors[0]


@pytest.mark.parametrize('selection', [None, ['master:1'], ['master:1', 'master:2']])
def test_master_selection_is_forwarded_without_freezing_default_discovery(monkeypatch, selection):
    calls = []
    async def post(url, body):
        calls.append(body['args']); return {'accepted': True}
    monkeypatch.setattr(client_runtime, 'post', post)
    ctx = context()
    ctx.nodes.append({'id': 'master-1', 'role': 'master', 'rpc': 'master:2'})
    values = args(); values.pop('masters')
    if selection is not None: values['masters'] = selection
    assert asyncio.run(Registry.standard().get('client/start').invoke(ctx, values))
    assert calls[0].get('masters') == selection
    if selection is None: assert 'masters' not in calls[0]


def test_unknown_master_selection_is_rejected_before_start(monkeypatch):
    async def post(*args): raise AssertionError('must not contact Client')
    monkeypatch.setattr(client_runtime, 'post', post)
    ctx = context()
    with pytest.raises(ExecutionError, match='Unknown Master'):
        asyncio.run(Registry.standard().get('client/start').invoke(ctx, args() | {'masters': ['master:1', 'missing:2']}))
    assert not ctx.runtime.clients.pending


def test_file_input_resolves_against_case_and_only_sends_metadata(tmp_path, monkeypatch):
    source = tmp_path / 'requests.binpb.zst'
    write_inputs(source, [pb.Request(token_ids=[1], max_new_tokens=2)])
    ctx = context(); ctx.runtime.case = SimpleNamespace(directory=tmp_path)
    calls = []
    async def post(url, body): calls.append(body['args']); return {'accepted': True}
    monkeypatch.setattr(client_runtime, 'post', post)
    value = args(); value['requests_file'] = source.name
    assert asyncio.run(Registry.standard().get('client/start').invoke(ctx, value))
    assert calls[0]['requests_file'] == str(source) and 'requests' not in calls[0]


@pytest.mark.parametrize('source', ['', 'https://example.com/data', None])
def test_missing_or_invalid_input_source_fails_contract(source):
    value = args(); value['requests_file'] = source
    with pytest.raises(ConfigError): Registry.standard().get('client/start').arguments(value)


def test_removed_inline_input_is_rejected():
    with pytest.raises(ConfigError): Registry.standard().get('client/start').arguments(args() | {'requests': [{'token_ids': [1], 'max_new_tokens': 2}]})


def test_non_file_source_is_rejected_before_marking_client_pending(tmp_path):
    ctx = context(); ctx.runtime.case = SimpleNamespace(directory=tmp_path)
    value = args(); value['requests_file'] = str(tmp_path)
    with pytest.raises(ExecutionError, match='regular file'):
        asyncio.run(Registry.standard().get('client/start').invoke(ctx, value))
    assert not ctx.runtime.clients.pending


def test_replay_speed_replaces_qps_and_is_forwarded_without_timestamp_conversion(monkeypatch):
    ctx = context()
    calls = []
    async def start(body, query_time, timeout):
        calls.append(body)
        return True
    monkeypatch.setattr(ctx.runtime.clients, 'start', start)
    values = args(); values.pop('qps'); values['replay_speed'] = 5
    assert asyncio.run(Registry.standard().get('client/start').invoke(ctx, values))
    assert calls[0]['replay_speed'] == 5 and 'qps' not in calls[0]


@pytest.mark.parametrize('rate', [{}, {'qps': 1, 'replay_speed': 5}, {'replay_speed': 0},
    {'replay_speed': float('inf')}, {'replay_speed': float('nan')}])
def test_playback_requires_exactly_one_finite_positive_rate(rate):
    values = args(); values.pop('qps'); values.update(rate)
    with pytest.raises(ConfigError):
        Registry.standard().get('client/start').arguments(values)
