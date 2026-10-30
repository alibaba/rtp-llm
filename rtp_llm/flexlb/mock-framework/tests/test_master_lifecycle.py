import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import pytest
import yaml

from mock_framework.config import Registry, compile_case
from mock_framework.config.registry import case_interface
from mock_framework.execution import prepare_launch, run_case
from mock_framework.execution.processes import ProcessSupervisor
from mock_framework.execution.master import master_port
from mock_framework.execution.launch import local_launch
from mock_framework.execution.transport import ExecutionError, post


def step(name, interface, **values):
    return dict(id=name, describe=name, interface=interface, **values)


def logs(output):
    return [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]


def assert_reaped(output):
    for item in logs(output):
        if item['event'] == 'process_started':
            with pytest.raises(ProcessLookupError):
                os.kill(item['pid'], 0)


def generic(tmp_path, steps, replicas=2, fail_index=-1):
    peer = tmp_path / 'peer.py'
    peer.write_text('''import os, json, pathlib, time, sys
node = os.environ['MOCK_NODE_ID']
index = int(node.split('-')[1])
if index == FAIL: sys.exit(19)
directory = pathlib.Path(os.environ['MOCK_OUTPUT_DIR'])
endpoint = {'id': node, 'role': 'master', 'rpc': f'127.0.0.1:{12000+index}', 'control': f'http://127.0.0.1:{13000+index}'}
(directory / 'endpoints.json').write_text(json.dumps(endpoint))
pathlib.Path(os.environ['MOCK_READY_FILE']).write_text(json.dumps({'url': endpoint['control']}))
time.sleep(120)
'''.replace('FAIL', str(fail_index)))
    document = dict(id='master_lifecycle', describe='Master lifecycle',
        environment={'master': {'deployment': {'replicas': replicas}, 'config': {}}},
        phases=[dict(id='lifecycle', describe='lifecycle', timeout='15s', steps=steps)])
    path = tmp_path / 'case.yaml'; path.write_text(yaml.safe_dump(document))
    registry = Registry.standard(); case = compile_case(path, registry)
    launch = prepare_launch({'components': {'master': {'command': [sys.executable, str(peer)]}}}, case, tmp_path)
    return case, registry, launch


def test_stop_selection_and_replacement_preserve_identity_and_artifacts(tmp_path):
    case, registry, launch = generic(tmp_path, [
        step('one', 'master/stop', args={'targets': ['127.0.0.1:12000']}),
        step('added', 'master/start', args={'count': 2}),
        step('all', 'master/stop')])
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    completed = {v['step']: v['result'] for v in logs(output) if v['event'] == 'step_finished'}
    assert completed == {'one': ['127.0.0.1:12000'], 'added': ['127.0.0.1:12002', '127.0.0.1:12003'],
                         'all': ['127.0.0.1:12001', '127.0.0.1:12002', '127.0.0.1:12003']}
    endpoints = [json.loads(path.read_text()) for path in sorted(output.glob('components/master-*/endpoints.json'))]
    assert len(endpoints) == 4 and len({v['rpc'] for v in endpoints}) == 4
    assert json.loads((output / 'topology.json').read_text()) == []
    assert len({v['pid'] for v in logs(output) if v['event'] == 'process_started'}) == 4
    assert_reaped(output)


def test_stale_reference_cannot_stop_replacement(tmp_path):
    case, registry, launch = generic(tmp_path, [step('old', 'master/stop'), step('new', 'master/start'),
        step('stale', 'master/stop', args={'targets': {'ref': 'steps.old'}})], replicas=1)
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'ERROR' and 'active addresses' in result['error']
    assert [v['node'] for v in logs(output) if v['event'] == 'process_started'] == ['master-0', 'master-1']
    assert_reaped(output)


def test_partial_start_failure_is_not_retried_and_reclaims_started_jvms(tmp_path):
    case, registry, launch = generic(tmp_path, [step('add', 'master/start', args={'count': 2})], replicas=1, fail_index=2)
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'ERROR' and 'unexpectedly' in result['error']
    assert [v['node'] for v in logs(output) if v['event'] == 'process_started'] == ['master-0', 'master-1', 'master-2']
    assert_reaped(output)


def test_partial_stop_failure_aborts_and_keeps_completed_operations(tmp_path, monkeypatch):
    original = ProcessSupervisor.stop_expected
    async def fail_second(self, node, grace=3):
        errors = await original(self, node, grace)
        return [*errors, 'injected stop failure'] if node == 'master-1' else errors
    monkeypatch.setattr(ProcessSupervisor, 'stop_expected', fail_second)
    case, registry, launch = generic(tmp_path, [step('stop', 'master/stop'), step('never', 'master/start')])
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'ERROR' and 'injected stop failure' in result['error']
    assert not any(v.get('step') == 'never' for v in logs(output))
    assert len([v for v in logs(output) if v['event'] == 'process_stopped']) == 2
    assert_reaped(output)


def test_master_cleanup_waits_for_peer_exit_and_slow_graceful_shutdown(tmp_path):
    from mock_framework.execution.master import MasterGroup
    from mock_framework.execution.records import Records
    case, _, launch = generic(tmp_path, [step('masters', 'master/get')])
    output = tmp_path / 'run'
    peer = tmp_path / 'shutdown.py'
    peer.write_text('''import os, pathlib, signal, sys, time
root = pathlib.Path(os.environ['MOCK_OUTPUT_DIR'])
name = os.environ['MOCK_NODE_ID']
def stop(*args):
    if name == 'master-0':
        while not (root.parent / 'master-1' / 'exited').exists(): time.sleep(.01)
    time.sleep(3.2)
    (root / 'exited').touch()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
(root / 'ready').touch()
while True: time.sleep(1)
''')
    async def exercise():
        records = Records(output, case, launch)
        supervisor = ProcessSupervisor(records)
        group = MasterGroup(component=None, recipe=None, election={}, launcher=None,
            supervisor=supervisor, topology=None, records=records, monitoring=None)
        try:
            for name in ('master-0', 'master-1'):
                directory = output / name
                directory.mkdir()
                await supervisor.spawn(name, [sys.executable, str(peer)], tmp_path,
                    {**os.environ, 'MOCK_OUTPUT_DIR': str(directory), 'MOCK_NODE_ID': name}, directory)
                group._owned.append(name)
                await supervisor.guard(_wait_file(directory / 'ready'), 5)
            async with asyncio.timeout(15):
                assert await group.close() == []
            assert all((output / name / 'exited').exists() for name in group._owned)
            assert not supervisor.children
        finally:
            await supervisor.close()
            records.close({'status': 'PASS'})
    asyncio.run(exercise())
    assert_reaped(output)


async def _wait_file(path):
    while not path.exists():
        await asyncio.sleep(.01)


@dataclass(frozen=True)
class Empty:
    pass


@case_interface('test/client_progress', effect='read')
async def progress(ctx, args: Empty) -> int:
    successes = 0
    for node in ctx.runtime.topology.active():
        if node['role'] == 'client':
            value = await post(node['control'], {'interface': 'worker/state', 'args': {'sequence': 1},
                'context': {'run': 'foreground', 'query_time': ctx.query_time, 'observation': None, 'timeout_ms': 1000}})
            successes += value['outcomes']['success']
    return successes


@pytest.mark.parametrize('batched', [False, True], ids=['direct', 'batch'])
def test_real_master_replacement_with_uninterrupted_client_plan(tmp_path, batched):
    from test_master import configuration
    root = Path(__file__).resolve().parents[1]
    java = str(Path(os.environ['JAVA_HOME']) / 'bin/java')
    classpath = os.pathsep.join([str(root / 'java/master/target/classes'), (root / 'java/master/target/dependencies.txt').read_text().strip()])
    config = configuration()
    config['FLEXLB_CONFIG'] = json.dumps({'scheduler': {'type': 'QUEUE' if batched else 'DIRECT'},
        'dispatcher': {'type': 'BATCH' if batched else 'NON_BATCH'},
        'requestLifecycle': {'request': {'timeoutMs': 5000}},
        'grpcServer': {'executorCoreSize': 4, 'executorMaxSize': 8, 'executorQueueSize': 64, 'shutdownQuietPeriodMs': 10}})
    if batched:
        native = json.loads(config['FLEXLB_CONFIG'])
        native['scheduler']['decision'] = {'type': 'FIXED_WINDOW', 'maxRequests': 2, 'maxCollectionWaitMs': 20}
        config['FLEXLB_CONFIG'] = json.dumps(native)
    engine = {'engine': 'rtp', 'config': {'seq_size_per_block': 16, 'test_block_num': 256,
        'reserve_block_ratio': 0, 'max_seq_len': 4096, 'compute.forward.expression': '1 + batchSize'}}
    from mock_framework.input import write_inputs, pb
    source = tmp_path / 'traffic.binpb.zst'
    write_inputs(source, (pb.Request(token_ids=[1, 2, 3], max_new_tokens=3) for _ in range(800)))
    steps = [step('leader', 'master/get', args={'is_leader': True}, expect={'count': 1}),
        step('traffic', 'client/start', args={'requests_file': source.name,
            'qps': 8, 'concurrency': 32, 'queue_capacity': 64, 'request_timeout': '5s', 'drain_timeout': '5s'}, wait_after='4s'),
        step('before', 'test/client_progress', expect={'gte': 1}),
        step('stop_primary', 'master/stop', args={'targets': {'ref': 'steps.leader'}}, wait_after='10s'),
        step('survivor', 'master/get', args={'is_leader': True}, expect={'count': 1}),
        step('after_first', 'test/client_progress', expect={'gte': {'ref': 'steps.before'}}),
        step('replacement', 'master/start', within='60s', wait_after='3s'),
        step('before_second', 'test/client_progress'),
        step('stop_survivor', 'master/stop', args={'targets': {'ref': 'steps.survivor'}}, wait_after='10s'),
        step('new_leader', 'master/get', args={'is_leader': True}, expect={'eq': {'ref': 'steps.replacement'}}),
        step('after_second', 'test/client_progress', expect={'gte': {'ref': 'steps.before_second'}}, wait_after='3s'),
        step('steady', 'test/client_progress', expect={'gte': {'ref': 'steps.after_second'}}),
        step('engines', 'master/engine/get'), step('collected', 'client/stop')]
    document = dict(id='master_ha', describe='Native Master replacement during one Client plan',
        environment={'master': {'deployment': {'replicas': 2}, 'config': config}, 'prefill': engine, 'decode': engine,
                     'client': {'deployment': {'replicas': 2}, 'config': {}}},
        phases=[dict(id='replace', describe='replace', timeout='150s', steps=steps)])
    path = tmp_path / 'case.yaml'; path.write_text(yaml.safe_dump(document))
    registry = Registry.standard(); registry.register(progress); case = compile_case(path, registry)
    try:
        master_port(3)
    except ExecutionError as error:
        if 'enable loopback aliases' in str(error): pytest.skip(str(error))
        raise
    components = {'master': {'command': [java, '-XX:ActiveProcessorCount=2', '-cp', classpath, 'org.flexlb.MasterNode'], 'startup_timeout': '60s'},
        'client': {'command': [java, '-XX:ActiveProcessorCount=2', '-jar', str(root / 'java/client/target/mock-client-1.0-SNAPSHOT-all.jar')]}}
    for role in ('prefill', 'decode'):
        components[role] = {'command': [java, '-XX:ActiveProcessorCount=2', '-cp', str(root / 'java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar'), 'org.flexlb.mock.host.EngineHost']}
    recipe = {'components': components, 'zookeeper': {'command': [java, '-XX:ActiveProcessorCount=2', '-cp', classpath, 'org.flexlb.mock.master.ZookeeperNode']}}
    if os.environ.get('PROMETHEUS_BIN'):
        recipe['prometheus'] = {'binary': os.environ['PROMETHEUS_BIN'], 'scrape_interval': '250ms'}
    output = tmp_path / 'result'
    result = asyncio.run(run_case(case, registry, prepare_launch(recipe, case, tmp_path), output))
    assert result['status'] == 'PASS', result
    values = {v['step']: v['result'] for v in logs(output) if v['event'] == 'step_finished'}
    assert values['leader'] != values['survivor'] != values['replacement']
    assert values['before'] < values['after_first'] <= values['before_second'] < values['after_second'] < values['steady']
    assert values['engines'] == {values['replacement'][0]: [json.loads((output / 'components/prefill-0/endpoints.json').read_text())['rpc']]}
    assert all(v['done'] and not v['errors'] and v['issued'] == v['finished'] for v in values['collected'].values())
    masters = [json.loads(path.read_text()) for path in sorted(output.glob('components/master-*/endpoints.json'))]
    assert len(masters) == 3 and len({v['rpc'] for v in masters}) == 3 and len({v['control'] for v in masters}) == 3
    native_envs = [json.loads((output / f"components/{v['id']}/native-environment.json").read_text()) for v in masters]
    assert len({v['HIPPO_ROLE'] for v in native_envs}) == 1
    assert len({v['FLEXLB_SYNC_CONSISTENCY_CONFIG'] for v in native_envs}) == 1
    assert len({v['MOCK_MASTER_PORT'] for v in native_envs}) == 1
    assert json.loads((output / 'topology.json').read_text()) == []
    assert all(not (output / f'components/{node}/ready.json').exists() for node in ('master-0', 'master-1'))
    if 'prometheus' in recipe:
        collection = json.loads((output / 'prometheus/collection.json').read_text())
        assert not {'master-0', 'master-1'} & collection['targets'].keys()
        assert 'master-2' in collection['targets'] and collection['complete']
    assert_reaped(output)


def test_kill_and_restore_keep_addresses_and_preserve_each_launch_artifact(tmp_path):
    case, registry, launch = generic(tmp_path, [
        step('killed', 'master/stop', args={'targets': ['127.0.0.1:12000'], 'mode': 'kill'}),
        step('restored', 'master/restart', args={'targets': {'ref': 'steps.killed'}}),
        step('killed_again', 'master/stop', args={'targets': {'ref': 'steps.restored'}, 'mode': 'kill'}),
        step('restored_again', 'master/restart', args={'targets': {'ref': 'steps.killed'}}),
        step('all', 'master/stop')])
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    values = {v['step']: v['result'] for v in logs(output) if v['event'] == 'step_finished'}
    assert all(values[key] == ['127.0.0.1:12000'] for key in ('killed', 'restored', 'killed_again', 'restored_again'))
    assert values['all'] == ['127.0.0.1:12000', '127.0.0.1:12001']
    directory = output / 'components/master-0'
    endpoints = [json.loads((path / 'endpoints.json').read_text()) for path in
                 (directory, directory / 'restart-1', directory / 'restart-2')]
    assert endpoints[0] == endpoints[1] == endpoints[2]
    assert all((path / 'stdout.log').is_file() for path in (directory, directory / 'restart-1', directory / 'restart-2'))
    started = [v for v in logs(output) if v['event'] == 'process_started']
    assert len(started) == 4 and len({v['pid'] for v in started}) == 4
    killed = [v for v in logs(output) if v['event'] == 'process_stopped' and v['code'] == -9]
    assert len(killed) == 2
    assert not (output / 'topology-history.json').exists()
    assert_reaped(output)


@pytest.mark.skipif(not os.environ.get('JAVA_HOME') or not os.environ.get('PROMETHEUS_BIN'),
                    reason='Java artifacts, JAVA_HOME and PROMETHEUS_BIN required')
def test_native_restart_preserves_address_and_refreshes_only_current_discovery(tmp_path):
    from test_master import configuration
    root = Path(__file__).resolve().parents[1]
    config = configuration()
    config['FLEXLB_CONFIG'] = json.dumps(json.loads(config['FLEXLB_CONFIG']) | {
        'grpcServer': {'shutdownQuietPeriodMs': 5000}})
    engine = {'engine': 'rtp', 'config': {'seq_size_per_block': 16, 'test_block_num': 256,
        'max_seq_len': 4096, 'compute.forward.expression': '1'}}
    document = dict(id='restart', describe='Native restart and current discovery',
        environment={'master': {'config': config}, 'prefill': engine, 'decode': engine},
        phases=[dict(id='lifecycle', describe='restart', timeout='120s', steps=[
            step('a', 'master/get', args={'is_leader': True}, expect={'count': 1}),
            step('killed', 'master/stop', args={'targets': {'ref': 'steps.a'}, 'mode': 'kill'}),
            step('restored', 'master/restart', args={'targets': {'ref': 'steps.a'}}, within='60s', wait_after='2s'),
            step('leader', 'master/get', args={'is_leader': True}, expect={'eq': {'ref': 'steps.a'}}),
            step('added', 'engine/scale_out', args={'role': 'prefill', 'count': 1}, expect={'count': 1})])])
    source = tmp_path / 'case.yaml'; source.write_text(yaml.safe_dump(document))
    registry = Registry.standard(); case = compile_case(source, registry)
    launch = local_launch(case, Path(os.environ['JAVA_HOME']) / 'bin/java', os.environ['PROMETHEUS_BIN'], framework=root)
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    values = {item['step']: item['result'] for item in logs(output) if item['event'] == 'step_finished'}
    assert values['a'] == values['restored']
    first = output / 'components/master-0'
    current = first / 'restart-1'
    original = json.loads((first / 'endpoints.json').read_text())
    restored = json.loads((current / 'endpoints.json').read_text())
    assert original['rpc'] == restored['rpc'] and original['http'] == restored['http']
    assert len(json.loads((first / 'discovery.json').read_text())['p-domain']) == 1
    assert len(json.loads((current / 'discovery.json').read_text())['p-domain']) == 2
    process = [item for item in logs(output) if item['event'] == 'process_started' and item['node'].startswith('master-')]
    assert len(process) == 2 and process[0]['pid'] != process[1]['pid']
    assert_reaped(output)


def test_restart_does_not_hide_unexpected_exit_of_new_process(tmp_path):
    case, registry, launch = generic(tmp_path, [
        step('killed', 'master/stop', args={'targets': ['127.0.0.1:12000'], 'mode': 'kill'}),
        step('restored', 'master/restart', args={'targets': {'ref': 'steps.killed'}}, wait_after='1s'),
        step('never', 'master/start')])
    peer = tmp_path / 'peer.py'
    peer.write_text(peer.read_text().replace('time.sleep(120)',
        "time.sleep(.1) if 'restart-' in str(directory) else time.sleep(120)\nif 'restart-' in str(directory): sys.exit(19)"))
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'ERROR' and 'master-0-restart-1 exited unexpectedly with code 19' in result['error']
    assert not any(item.get('step') == 'never' for item in logs(output))
    assert_reaped(output)


@pytest.mark.parametrize('steps', [
    [step('bad', 'master/restart', args={'targets': ['127.0.0.1:12000']})],
    [step('bad', 'master/restart', args={'targets': ['missing:1']})]])
def test_restart_requires_owned_stopped_masters(tmp_path, steps):
    case, registry, launch = generic(tmp_path, steps)
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'ERROR' and 'stopped addresses' in result['error']
    assert_reaped(output)
