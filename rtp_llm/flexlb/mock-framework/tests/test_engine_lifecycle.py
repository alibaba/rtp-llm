import asyncio
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
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


def setup_case(tmp_path, operation, mode='shared', monitor=False, invalid=False):
    registry = Registry.standard(); registry.register(operation)
    config = {'seq_size_per_block': 16, 'test_block_num': 9, 'reserve_block_ratio': 0, 'max_seq_len': 4096,
              'compute.expression': '1'}
    env = {role: {'engine': 'rtp', 'deployment': {'replicas': 2 if role == 'prefill' else 1}, 'config': dict(config)}
           for role in ('prefill', 'decode')}
    if invalid:
        env['decode']['config']['seq_size_per_block'] = 0
    source = tmp_path / 'case.yaml'
    source.write_text(yaml.safe_dump({'id': 'lifecycle', 'describe': 'host lifecycle', 'environment': env,
        'phases': [{'id': 'phase', 'describe': 'instance operations', 'timeout': '60s', 'steps': [
            {'id': 'exercise', 'describe': 'exercise', 'interface': operation.case_interface.name}]}]}))
    case = compile_case(source, registry)
    java = Path(os.environ['JAVA_HOME']) / 'bin/java'
    command = [str(java), '-jar', str(ROOT / 'java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar')]
    raw = {'engine_mode': mode, 'components': {role: {'command': command} for role in env}}
    if monitor:
        raw['prometheus'] = {'binary': os.environ['PROMETHEUS_BIN'], 'scrape_interval': '100ms'}
    return case, registry, prepare_launch(raw, case, tmp_path)


def no_children(output):
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    for event in events:
        if event['event'] == 'process_started':
            with pytest.raises(ProcessLookupError):
                os.kill(event['pid'], 0)
    return events


@pytest.mark.parametrize('mode', ['shared', 'isolated'])
def test_scale_in_selects_once_and_instances_keep_separate_artifacts_and_metrics(tmp_path, mode):
    @case_interface('probe/lifecycle', effect='write')
    async def exercise(ctx, args: Args) -> list[str]:
        owner = ctx.processes
        assert len(owner.engines.hosts) == (1 if mode == 'shared' else 3)
        original = ctx.globals()['prefill']
        removed = await scale_in(ctx, ScaleIn(count=1))
        assert len(removed) == 1 and len(ctx.globals()['prefill']) == 1
        with pytest.raises(ExecutionError):
            await scale_in(ctx, ScaleIn(targets=removed))
        created = await scale_out(ctx, ScaleOut(count=2))
        assert len(ctx.globals()['prefill']) == 3 and not set(created) & set(original)
        assert await scale_in(ctx, ScaleIn(targets=created[:1])) == created[:1]
        assert len(ctx.globals()['prefill']) == 2
        for item in owner.history:
            folder = ctx.records.directory / 'components' / item['id']
            assert json.loads((folder / 'endpoints.json').read_text())['rpc'] == item['rpc']
            assert (folder / 'ready.json').exists() == ('stopped_at' not in item)
        # Exercise real TCP request traffic after removal and subsequent creation.
        classpath = (ROOT / 'java/engine/rtp/target/rtp-classpath.txt').read_text().strip()
        probe = await asyncio.to_thread(subprocess.run, [str(Path(os.environ['JAVA_HOME']) / 'bin/java'),
            '-cp', classpath, 'org.flexlb.mock.engine.rtp.NativeProbe', ctx.globals()['decode'][0],
            str(tmp_path / 'traffic.json'), ctx.globals()['prefill'][0]], capture_output=True, text=True, timeout=15)
        assert probe.returncode == 0, probe.stderr
        if owner.prometheus:
            samples = await query(replace(ctx, query_time=timestamp()), Promql('count_over_time(mock_engine_requests_started_total[1m])'))
            assert {item.labels['instance'] for item in samples.samples} == {item['rpc'] for item in owner.history}
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
    history = json.loads((output / 'topology-history.json').read_text())
    assert len(history) == 6 and len({item['rpc'] for item in history}) == 6
    if monitor:
        metadata = json.loads((output / 'prometheus/collection.json').read_text())
        assert metadata['complete'] and len(metadata['retired_targets']) == 4


@pytest.mark.parametrize('invalid', [False, True])
def test_host_failure_or_partial_startup_fails_the_run_and_cleans_every_process(tmp_path, invalid):
    @case_interface('probe/crash', effect='write')
    async def crash(ctx, args: Args) -> bool:
        next(iter(ctx.processes.engines.hosts.values()))['child'].kill()
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
