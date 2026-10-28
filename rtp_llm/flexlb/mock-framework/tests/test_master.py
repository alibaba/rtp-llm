import json
from pathlib import Path

import pytest

from mock_framework.execution.master import master_environment
from mock_framework.execution.transport import ExecutionError


def configuration():
    return {'FLEXLB_CONFIG': '{ "requestLifecycle": {"request": {"timeoutMs": 60000}} }',
            'MODEL_SERVICE_CONFIG': json.dumps({'service_id': 'aigc.text-generation.generation.mock',
                'role_endpoints': [{'group': 'group', 'prefill_endpoint': {'address': 'p-domain', 'protocol': 'grpc'},
                                   'decode_endpoint': {'address': 'd-domain', 'protocol': 'grpc'}}],
                'hosts': {'p-domain': ['old-host:9000']}})}


def topology():
    return [{'id': 'prefill-0', 'role': 'prefill', 'rpc': '127.0.0.1:30001'},
            {'id': 'decode-0', 'role': 'decode', 'rpc': '127.0.0.1:40001'}]


def test_binds_only_run_owned_discovery_and_preserves_business_document(tmp_path):
    config = configuration()
    env = master_environment(config, topology(), tmp_path)
    assert env['FLEXLB_CONFIG'] == config['FLEXLB_CONFIG']
    assert 'old-host' in config['MODEL_SERVICE_CONFIG']
    route = json.loads(env['MODEL_SERVICE_CONFIG'])
    assert 'hosts' not in route
    assert json.loads(Path(route['discovery_file']).read_text()) == {
        'p-domain': ['127.0.0.1:30001'], 'd-domain': ['127.0.0.1:40001']}
    assert json.loads((tmp_path / 'native-environment.json').read_text()) == env


@pytest.mark.parametrize('change', [
    lambda c: c.update(FLEXLB_CONFIG={}),
    lambda c: c.update(MODEL_SERVICE_CONFIG='not-json'),
    lambda c: c.update(MODEL_SERVICE_CONFIG='{"role_endpoints":[]}'),
])
def test_invalid_native_settings_fail_instead_of_starting_a_different_configuration(tmp_path, change):
    config = configuration(); change(config)
    with pytest.raises(ExecutionError):
        master_environment(config, topology(), tmp_path)


def test_missing_running_role_is_rejected(tmp_path):
    with pytest.raises(ExecutionError, match='No running Engines'):
        master_environment(configuration(), topology()[:1], tmp_path)


def test_generic_transport_probe_does_not_require_a_native_master(tmp_path):
    assert master_environment({'a': {'b': 2}}, [], tmp_path) == {}


@pytest.mark.parametrize('batched', [False, True], ids=['single-nonbatch', 'fixed-window-batch'])
def test_real_master_routes_to_isolated_rtp_processes(tmp_path, batched):
    import asyncio
    import os
    import subprocess
    import urllib.request
    import yaml
    from mock_framework.config import Registry, compile_case
    from mock_framework.execution import prepare_launch, run_case

    root = Path(__file__).resolve().parents[1]
    java = Path(os.environ['JAVA_HOME']) / 'bin/java'
    master_classpath = os.pathsep.join([str(root / 'java/master/target/classes'),
        (root / 'java/master/target/dependencies.txt').read_text().strip()])
    client_classpath = os.pathsep.join([str(root / 'java/client/target/test-classes'), str(root / 'java/client/target/classes'),
        (root / 'java/engine/rtp/target/rtp-classpath.txt').read_text().strip()])
    native = {'scheduler': {'type': 'QUEUE' if batched else 'DIRECT'},
              'dispatcher': {'type': 'BATCH' if batched else 'NON_BATCH'},
              'requestLifecycle': {'request': {'timeoutMs': 60000}},
              'grpcServer': {'executorCoreSize': 4, 'executorMaxSize': 8, 'executorQueueSize': 64, 'shutdownQuietPeriodMs': 10}}
    if batched:
        native['scheduler']['decision'] = {'type': 'FIXED_WINDOW', 'maxRequests': 2, 'maxCollectionWaitMs': 20}
    master_config = configuration(); master_config['FLEXLB_CONFIG'] = json.dumps(native)
    engine_config = {'seq_size_per_block': 16, 'test_block_num': 9, 'reserve_block_ratio': 0,
                     'max_seq_len': 4096, 'compute.expression': '1 + batchSize'}
    document = {'id': 'production_master_probe', 'describe': '真实Master与新P/D及Client接线验证',
        'environment': {'master': {'config': master_config},
                        'prefill': {'engine': 'rtp', 'config': engine_config},
                        'decode': {'engine': 'rtp', 'config': engine_config}},
        'phases': [{'id': 'probe', 'describe': '逐Master读取并由外部Client发请求', 'timeout': '30s', 'steps': [
            {'check': {'id': 'prefills', 'describe': '实际登记P集合', 'interface': 'master/engine/get',
                       'args': {'role': 'prefill'}, 'expect': {'eq': {'instances': [{'id': 'prefill-0'}]}}}},
            {'check': {'id': 'decodes', 'describe': '实际登记D集合', 'interface': 'master/engine/get',
                       'args': {'role': 'decode'}, 'expect': {'eq': {'instances': [{'id': 'decode-0'}]}}}},
            {'check': {'id': 'leader', 'describe': 'standalone没有选举声明', 'interface': 'master/leadership/get', 'expect': {'count': 0}}},
            {'call': {'id': 'wait', 'describe': '供外部Client取流', 'interface': 'runtime/wait', 'args': {'duration': '12s'}}}]}]}
    case_file = tmp_path / 'case.yaml'; case_file.write_text(yaml.safe_dump(document, allow_unicode=True))
    registry = Registry.load(root / 'build/interfaces.json'); case = compile_case(case_file, registry)
    components = {'master': {'command': [str(java), '-XX:ActiveProcessorCount=2', '-cp', master_classpath, 'org.flexlb.MasterNode'], 'startup_timeout': '60s'}}
    for role in ('prefill', 'decode'):
        components[role] = {'command': [str(java), '-jar', str(root / 'java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar')]}
    launch = prepare_launch({'components': components, 'bindings': {'master/': 'master-0'}}, case, tmp_path)
    output = tmp_path / 'result'

    async def exercise():
        task = asyncio.create_task(run_case(case, registry, launch, output))
        try:
            ready = output / 'components/master-0/ready.json'
            while not ready.exists():
                if task.done():
                    raise AssertionError(await task)
                await asyncio.sleep(.02)
            endpoints = json.loads((ready.parent / 'endpoints.json').read_text())
            actual = tmp_path / 'client.json'
            probe = await asyncio.to_thread(subprocess.run,
                [str(java), '-cp', client_classpath, 'org.flexlb.mock.client.MasterProbe', endpoints['rpc'], str(batched).lower(), str(actual)],
                capture_output=True, text=True, timeout=20)
            assert probe.returncode == 0, probe.stderr
            result = json.loads(actual.read_text())
            assert result['enqueued_by_master'] is batched and result['outcome'] == 'success'
            assert result['tokens'] == 3 and 0 < result['ttft_ns'] <= result['elapsed_ns']
            with urllib.request.urlopen(endpoints['metrics'], timeout=3) as response:
                metrics = response.read().decode('utf-8')
                assert response.status == 200 and 'jvm_memory_used_bytes' in metrics and 'flexlb_' in metrics
            (ready.parent / 'metrics.prom').write_text(metrics)
            assert (await task)['status'] == 'PASS'
        finally:
            if not task.done():
                task.cancel(); await asyncio.gather(task, return_exceptions=True)
    asyncio.run(exercise())
    effective = json.loads((output / 'components/master-0/effective-config.json').read_text())
    assert effective['FLEXLB_CONFIG'] == master_config['FLEXLB_CONFIG']
    events = [json.loads(row) for row in (output / 'events.jsonl').read_text().splitlines()]
    started = [event for event in events if event['event'] == 'process_started']
    assert [event['node'] for event in started] == ['prefill-0', 'decode-0', 'master-0']
    for event in started:
        with pytest.raises(ProcessLookupError):
            os.kill(event['pid'], 0)
