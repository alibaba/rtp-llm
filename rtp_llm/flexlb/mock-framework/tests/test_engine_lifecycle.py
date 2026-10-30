import asyncio
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import socket
import urllib.parse
import urllib.request
from collections import deque
import pytest
import yaml
from mock_framework.config import Registry, compile_case
from mock_framework.config.registry import case_interface
from mock_framework.execution import prepare_launch, run_case, ExecutionError
from mock_framework.execution.records import timestamp
from mock_framework.interfaces.engines import ScaleIn, ScaleOut, scale_in, scale_out
from mock_framework.interfaces.metrics import Promql, query

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Args:
    pass


def setup_case(tmp_path, operation, mode='shared', monitor=False, invalid=False, master=False):
    registry = Registry.standard(); registry.register(operation)
    config = {'seq_size_per_block': 16, 'test_block_num': 9, 'reserve_block_ratio': 0, 'max_seq_len': 4096,
              'compute.forward.expression': '1'}
    env = {role: {'engine': 'rtp', 'deployment': {'replicas': 2 if role == 'prefill' else 1}, 'config': dict(config)}
           for role in ('prefill', 'decode')}
    if invalid:
        env['decode']['config']['seq_size_per_block'] = 0
    if master:
        from test_master import configuration
        config = configuration()
        config['FLEXLB_CONFIG'] = json.dumps({'scheduler': {'type': 'DIRECT'}, 'dispatcher': {'type': 'NON_BATCH'},
            'requestLifecycle': {'request': {'timeoutMs': 3000}},
            'grpcServer': {'executorCoreSize': 4, 'executorMaxSize': 8, 'executorQueueSize': 64, 'shutdownQuietPeriodMs': 10}})
        env['master'] = {'config': config}
    source = tmp_path / 'case.yaml'
    source.write_text(yaml.safe_dump({'id': 'lifecycle', 'describe': 'host lifecycle', 'environment': env,
        'phases': [{'id': 'phase', 'describe': 'instance operations', 'timeout': '60s', 'steps': [
            {'id': 'exercise', 'describe': 'exercise', 'interface': operation.case_interface.name}]}]}))
    case = compile_case(source, registry)
    java = Path(os.environ['JAVA_HOME']) / 'bin/java'
    command = [str(java), '-cp', str(ROOT / 'java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar'), 'org.flexlb.mock.host.EngineHost']
    raw = {'engine_mode': mode, 'components': {role: {'command': command} for role in env if role != 'master'}}
    if master:
        classpath = os.pathsep.join([str(ROOT / 'java/master/target/classes'),
            (ROOT / 'java/master/target/dependencies.txt').read_text().strip()])
        prefix = [str(java), '-XX:ActiveProcessorCount=2', '-cp', classpath]
        raw['components']['master'] = {'command': prefix + ['org.flexlb.MasterNode'], 'startup_timeout': '60s'}
        raw['zookeeper'] = {'command': prefix + ['org.flexlb.mock.master.ZookeeperNode']}
    if monitor:
        raw['prometheus'] = {'binary': os.environ['PROMETHEUS_BIN'], 'scrape_interval': '100ms'}
    return case, registry, prepare_launch(raw, case, tmp_path)


def no_children(output):
    assert not (output / 'topology-history.json').exists()
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    for event in events:
        if event['event'] == 'process_started':
            with pytest.raises(ProcessLookupError):
                os.kill(event['pid'], 0)
    return events


@pytest.mark.parametrize('mode', ['shared', 'isolated'])
def test_scale_in_selects_once_and_instances_keep_separate_artifacts_and_metrics(tmp_path, mode, monkeypatch):
    @case_interface('probe/lifecycle', effect='write')
    async def exercise(ctx, args: Args) -> list[str]:
        owner = ctx.runtime
        assert len(owner.engines.hosts) == (1 if mode == 'shared' else 3)
        original = ctx.globals()['prefill']
        first = next(item for item in owner.topology.active('prefill') if item['rpc'] == original[0])
        original_files = {name: (ctx.records.directory / 'components' / first['id'] / name).read_bytes()
                          for name in ('config.json', 'effective-config.json', 'endpoints.json')}
        removed = await scale_in(ctx, ScaleIn(count=1))
        assert len(removed) == 1 and len(ctx.globals()['prefill']) == 1
        with pytest.raises(ExecutionError):
            await scale_in(ctx, ScaleIn(targets=removed))
        # Force the allocator to encounter retired RPC and HTTP ports before fresh ports.
        candidates = deque([int(first['rpc'].rsplit(':', 1)[1]), urllib.parse.urlsplit(first['control']).port])
        real_socket = socket.socket
        class ReusedPortSocket(real_socket):
            def bind(self, address):
                if address == ('127.0.0.1', 0) and candidates:
                    self.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    address = (address[0], candidates.popleft())
                return super().bind(address)
        with monkeypatch.context() as patch:
            patch.setattr(socket, 'socket', ReusedPortSocket)
            created = await scale_out(ctx, ScaleOut(count=2))
        assert not candidates
        for name, content in original_files.items():
            assert (ctx.records.directory / 'components' / first['id'] / name).read_bytes() == content
        if owner.monitoring:
            # Keep an obsolete discovery target present while new instances are already serving.
            from mock_framework.execution.records import write_json
            from mock_framework.execution.monitoring import _get
            prom = owner.monitoring
            retired = {'targets': [urllib.parse.urlsplit(first['metrics']).netloc], 'labels': {'instance': first['rpc'], '__metrics_path__': '/metrics'}}
            write_json(prom.directory / 'targets.json', list(prom.targets.values()) + [{key: retired[key] for key in ('targets', 'labels')}])
            expression = 'up{job="mock",instance="' + first['rpc'] + '"}'
            async with asyncio.timeout(5):
                while True:
                    value = await asyncio.to_thread(_get, prom.url + '/api/v1/query?' + urllib.parse.urlencode({'query': expression}))
                    samples = value['data']['result']
                    if samples and samples[0]['value'][1] == '0':
                        break
                    await asyncio.sleep(.05)
            # A stale control URL must not reach a replacement host or Engine either.
            with pytest.raises(OSError):
                await asyncio.to_thread(urllib.request.urlopen, first['metrics'], timeout=1)
            write_json(prom.directory / 'targets.json', list(prom.targets.values()))
        assert len(ctx.globals()['prefill']) == 3 and not set(created) & set(original)
        assert await scale_in(ctx, ScaleIn(targets=created[:1])) == created[:1]
        assert len(ctx.globals()['prefill']) == 2
        endpoints = [json.loads(path.read_text()) for path in sorted(ctx.records.directory.glob('components/*/endpoints.json'))]
        for item in endpoints:
            folder = ctx.records.directory / 'components' / item['id']
            assert json.loads((folder / 'endpoints.json').read_text())['rpc'] == item['rpc']
            assert (folder / 'ready.json').exists() == (item['id'] in {node['id'] for node in owner.topology.active()})
        # Exercise real TCP request traffic after removal and subsequent creation.
        classpath = os.pathsep.join([str(ROOT / 'java/backend/rtp/target/test-classes'), str(ROOT / 'java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar')])
        probe = await asyncio.to_thread(subprocess.run, [str(Path(os.environ['JAVA_HOME']) / 'bin/java'),
            '-cp', classpath, 'org.flexlb.mock.engine.rtp.NativeProbe', ctx.globals()['decode'][0],
            str(tmp_path / 'traffic.json'), ctx.globals()['prefill'][0]], capture_output=True, text=True, timeout=15)
        assert probe.returncode == 0, probe.stderr
        if owner.monitoring:
            samples = await query(replace(ctx, query_time=timestamp()), Promql('count_over_time(rtp_llm_requests_started_total[1m])'))
            assert {item.labels['instance'] for item in samples.samples} == {item['rpc'] for item in endpoints}
        assert len(await scale_in(ctx, ScaleIn())) == 2
        assert ctx.globals()['prefill'] == []
        assert len(await scale_out(ctx, ScaleOut())) == 1
        return removed + created
    monitor = bool(os.environ.get('PROMETHEUS_BIN'))
    case, registry, launch = setup_case(tmp_path, exercise, mode, monitor)
    output = tmp_path / 'result'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    events = no_children(output)
    assert len([event for event in events if event['event'] == 'instance_stopped']) == 4
    assert json.loads((output / 'topology.json').read_text()) == []
    endpoints = [json.loads(path.read_text()) for path in sorted(output.glob('components/*/endpoints.json'))]
    assert len(endpoints) == 6 and len({item['rpc'] for item in endpoints}) == 6
    listeners = [urllib.parse.urlsplit('//'+item['rpc']).port for item in endpoints]
    listeners += [urllib.parse.urlsplit(item['control']).port for item in endpoints]
    listeners += [urllib.parse.urlsplit(event['url']).port for event in events
                  if event['event'] == 'process_ready' and event['node'].startswith('engine-')]
    assert len(listeners) == len(set(listeners))
    if monitor:
        metadata = json.loads((output / 'prometheus/collection.json').read_text())
        assert metadata['complete'] and len(metadata['targets']) == 2


@pytest.mark.parametrize('invalid', [False, True])
def test_host_failure_or_partial_startup_fails_the_run_and_cleans_every_process(tmp_path, invalid):
    @case_interface('probe/crash', effect='write')
    async def crash(ctx, args: Args) -> bool:
        next(iter(ctx.runtime.engines.hosts.values()))['child'].kill()
        await asyncio.sleep(20)
        return True
    case, registry, launch = setup_case(tmp_path, crash, invalid=invalid)
    output = tmp_path / 'result'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'ERROR'
    events = no_children(output)
    assert len([event for event in events if event['event'] == 'process_started']) == 1
    if not invalid:
        assert all(node in result['error'] for node in ('prefill-0', 'prefill-1', 'decode-0'))


def test_shared_host_rejects_conflicting_process_environment_and_bad_mode(tmp_path):
    @case_interface('probe/noop', effect='read')
    async def noop(ctx, args: Args) -> bool:
        return True
    case, _, launch = setup_case(tmp_path, noop)
    components = {role: {'command': recipe['command'], 'env': {'ROLE_SETTING': role}} for role, recipe in launch['components'].items()}
    with pytest.raises(ExecutionError, match='identical'):
        prepare_launch({'components': components}, case, tmp_path)
    assert prepare_launch({'components': components, 'engine_mode': 'isolated'}, case, tmp_path)['engine_mode'] == 'isolated'
    with pytest.raises(ExecutionError, match='engine_mode'):
        prepare_launch({'components': components, 'engine_mode': 'arbitrary'}, case, tmp_path)


def test_allocator_exhaustion_fails_without_reusing_a_retired_listener(monkeypatch):
    from mock_framework.execution.engines import EngineHosts
    owner = EngineHosts(supervisor=None, topology=None, records=None, environment={}, recipes={}, mode='shared', monitoring=None)
    owner.used_ports.add(12345)
    class Probe:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, address): pass
        def getsockname(self): return ('127.0.0.1', 12345)
    monkeypatch.setattr(socket, 'socket', Probe)
    with pytest.raises(ExecutionError, match='fresh Engine listener'):
        owner.port()


@pytest.mark.parametrize('operation', ['create', 'stop'])
def test_partial_scale_failure_is_not_replayed_or_reported_as_success(tmp_path, operation):
    @case_interface('probe/partial', effect='write')
    async def exercise(ctx, args: Args) -> bool:
        original = ctx.runtime.engines.call
        calls = []
        async def fail_second(host, name, body):
            if name == operation:
                calls.append(body['id'])
                if len(calls) == 2:
                    ctx.records.emit(event='injected_failure', operation=name, calls=calls)
                    body = {**body, **({'controlPort': body['rpcPort']} if name == 'create' else {'id': 'prefill-999999'})}
            return await original(host, name, body)
        ctx.runtime.engines.call = fail_second
        if operation == 'create': await scale_out(ctx, ScaleOut(count=2))
        else: await scale_in(ctx, ScaleIn(count=2))
        raise AssertionError('Partial failure was swallowed')
    case, registry, launch = setup_case(tmp_path, exercise)
    output = tmp_path / 'result'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'ERROR', result
    events = no_children(output)
    injected = [e for e in events if e['event'] == 'injected_failure']
    assert len(injected) == 1 and len(injected[0]['calls']) == 2
    assert len([e for e in events if e['event'] == 'instance_started']) == (4 if operation == 'create' else 3)
    assert len([e for e in events if e['event'] == 'instance_stopped']) == (1 if operation == 'stop' else 0)
    assert json.loads((output / 'topology.json').read_text()) == []


@pytest.mark.parametrize('mode', ['shared', 'isolated'])
def test_live_traffic_survives_scale_and_real_master_eventually_retires_old_address(tmp_path, mode):
    if not os.environ.get('PROMETHEUS_BIN'):
        pytest.skip('PROMETHEUS_BIN required for live lifecycle and historical metric verification')
    @case_interface('probe/live_scale', effect='write')
    async def exercise(ctx, args: Args) -> bool:
        from datetime import datetime
        from mock_framework.interfaces.master import engines, EngineSelection
        from mock_framework.execution.records import write_json
        from mock_framework.execution.monitoring import _get
        directory = ctx.records.directory / 'traffic'
        directory.mkdir()
        classpath = os.pathsep.join([str(ROOT / 'java/client/target/test-classes'), str(ROOT / 'java/client/target/classes'),
            str(ROOT / 'java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar')])
        master = ctx.globals()['masters'][0]
        def rows():
            path = directory / 'requests.jsonl'
            return [json.loads(line) for line in path.read_text().splitlines() if line.endswith('}')] if path.exists() else []
        async def until(predicate, timeout=20):
            async with asyncio.timeout(timeout):
                while not await predicate():
                    if child.returncode is not None:
                        raise AssertionError((directory / 'stderr.log').read_text())
                    await asyncio.sleep(.05)
        with (directory / 'stdout.log').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr:
            child = await asyncio.create_subprocess_exec(str(Path(os.environ['JAVA_HOME']) / 'bin/java'), '-cp', classpath,
                'org.flexlb.mock.client.LifecycleProbe', master, str(directory), stdout=stdout, stderr=stderr)
            try:
                async def healthy(): return len([r for r in rows() if r['outcome'] == 'success']) >= 5
                await until(healthy)
                began = timestamp()
                removed = await scale_in(ctx, ScaleIn(count=1))
                created = await scale_out(ctx, ScaleOut())
                expected = sorted(ctx.globals()['prefill'])
                observations = []
                async def converged():
                    value = await engines(ctx, EngineSelection())
                    observations.append({'at': timestamp(), 'observed': value})
                    return value == {master: expected}
                await until(converged)
                converged_at = timestamp()
                async def recovered():
                    subsequent = [r for r in rows() if datetime.fromisoformat(r['sent_at']) >= datetime.fromisoformat(converged_at)]
                    return len(subsequent) >= 5 and all(r['outcome'] == 'success' and r['tokens'] == 3 for r in subsequent[-5:])
                await until(recovered, 10)
                prom = ctx.runtime.monitoring
                async def new_engine_received_requests():
                    expr = 'rtp_llm_requests_started_total{instance="' + created[0] + '"}'
                    value = await asyncio.to_thread(_get, prom.url + '/api/v1/query?' + urllib.parse.urlencode({'query': expr}))
                    return any(float(r['value'][1]) > 0 for r in value['data']['result'])
                await until(new_engine_received_requests, 10)
                historical = await query(replace(ctx, query_time=timestamp()), Promql('count_over_time(rtp_llm_requests_started_total[1m])'))
                assert set(removed + created) <= {sample.labels['instance'] for sample in historical.samples}
                all_rows = rows()
                transition = [r for r in all_rows if datetime.fromisoformat(began) <= datetime.fromisoformat(r['sent_at']) < datetime.fromisoformat(converged_at)]
                assert transition, 'No requests were issued during topology propagation'
                write_json(directory / 'lifecycle.json', {'removed': removed, 'created': created,
                    'began': began, 'converged_at': converged_at, 'master_observations': observations,
                    'transition_outcomes': {outcome: sum(r['outcome'] == outcome for r in transition)
                        for outcome in {r['outcome'] for r in transition}}})
                return True
            finally:
                (directory / 'stop').touch()
                try:
                    await asyncio.wait_for(child.wait(), 5)
                except TimeoutError:
                    child.kill(); await child.wait()
                assert child.returncode == 0, (directory / 'stderr.log').read_text()
    case, registry, launch = setup_case(tmp_path, exercise, mode, monitor=True, master=True)
    output = tmp_path / 'result'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    no_children(output)
    assert json.loads((output / 'prometheus/collection.json').read_text())['complete']
