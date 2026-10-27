import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import subprocess

import pytest

from mock_framework.config import Registry, compile_case
from mock_framework.config.model import Reference
from mock_framework.execution import ExecutionError, prepare_launch, run_case
from mock_framework.execution.assertions import compare
from mock_framework.execution.runner import observation, resolve

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'java/runtime/target'


@pytest.fixture
def probe(tmp_path):
    import yaml
    registry = Registry.load(RUNTIME / 'probe-catalog.json')
    java = str(Path(os.environ.get('JAVA_HOME', '/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home')) / 'bin/java')
    classpath = (RUNTIME / 'probe-classpath.txt').read_text().strip()

    def make(steps, *, timeout='3s', replicas=1, extra_args=()):
        path = tmp_path / 'case.yaml'
        raw = {'id': 'probe_case', 'describe': '仅验证传输与执行器',
               'environment': {'master': {'deployment': {'replicas': replicas}, 'config': {'a.b': 2}}},
               'phases': [{'id': 'phase', 'describe': '顺序调用', 'timeout': timeout, 'steps': steps}]}
        path.write_text(yaml.safe_dump(raw, allow_unicode=True))
        case = compile_case(path, registry)
        recipe = {'components': {'master': {'command': [java, '-cp', classpath, 'org.flexlb.mock.runtime.TransportProbe', *extra_args],
                                             'startup_timeout': '5s'}},
                  'bindings': {'probe/': 'master-0'}}
        return case, registry, prepare_launch(recipe, case, tmp_path)
    return make


def call(id='action', **kwargs):
    return {'call': {'id': id, 'describe': '调用测试端', 'interface': 'probe/action', **kwargs}}


def check(id='metric', interface='probe/metric', expect=None, args=None):
    return {'check': {'id': id, 'describe': '检查测试端', 'interface': interface,
                      'args': args or {}, 'expect': expect or {'lte': '1s'}}}


def execute(probe, tmp_path, steps, **kwargs):
    case, registry, launch = probe(steps, **kwargs)
    directory = tmp_path / 'output'
    return asyncio.run(run_case(case, registry, launch, directory)), directory


def events(directory, name='events'):
    return [json.loads(line) for line in (directory / f'{name}.jsonl').read_text().splitlines()]


def assert_cleaned(directory):
    started = [e for e in events(directory) if e['event'] == 'process_started']
    stopped = [e for e in events(directory) if e['event'] == 'process_stopped']
    assert len(started) == len(stopped) and started
    for event in started:
        with pytest.raises(ProcessLookupError):
            os.kill(event['pid'], 0)


def test_real_jvm_calls_typed_refs_gates_and_records(probe, tmp_path):
    result, out = execute(probe, tmp_path, [
        call(run='background'),
        check('masters', 'probe/masters', {'eq': {'ref': 'action.affected'}}),
        check('membership', 'probe/engines', {'eq': {'instances': [{'id': 'prefill-0'}, {'id': 'prefill-1'}]}}),
        check(), check('promql', 'probe/query', {'gte': 1, 'lte': 2}),
        {'call': {'id': 'wait', 'describe': '固定等待', 'interface': 'runtime/wait', 'args': {'duration': '1ms'}}},
    ], replicas=2)
    assert result['status'] == 'PASS'
    assert [g['status'] for g in events(out, 'gates')] == ['PASS'] * 4
    starts = [e['step'] for e in events(out) if e['event'] == 'step_started']
    assert starts == ['action', 'masters', 'membership', 'metric', 'promql', 'wait']
    assert json.loads((out / 'components/master-0/config.json').read_text())['component']['config'] == {'a': {'b': 2}}
    assert (out / 'case.yaml').is_file() and (out / 'interfaces.json').is_file() and (out / 'code.json').is_file()
    assert_cleaned(out)


@pytest.mark.parametrize('args,expect,status', [({}, {'lte': '1ms'}, 'FAIL'),
                                             ({'invalid_result': True}, {'lte': '1s'}, 'ERROR')])
def test_gate_failure_or_invalid_result_skips_later_steps_and_cleans(probe, tmp_path, args, expect, status):
    result, out = execute(probe, tmp_path, [check(args=args, expect=expect), call()])
    assert result['status'] == status
    assert [e['step'] for e in events(out) if e['event'] == 'step_started'] == ['metric']
    assert events(out, 'gates')[0]['status'] == status
    assert_cleaned(out)


def test_empty_prometheus_result_is_error_not_vacuous_pass(probe, tmp_path):
    result, out = execute(probe, tmp_path, [check('query', 'probe/query', {'gte': 0}, {'empty': True})])
    assert result['status'] == 'ERROR'
    assert 'no samples' in result['error']
    assert_cleaned(out)


def test_within_timeout_does_not_retry_mutation(probe, tmp_path):
    result, out = execute(probe, tmp_path, [call(args={'delay_ms': 1000}, within='40ms'), check()])
    assert result['status'] == 'ERROR'
    ends = [e for e in events(out) if e['event'] == 'step_finished']
    assert len(ends) == 1 and ends[0]['outcome_unknown']
    assert_cleaned(out)


def test_phase_deadline_includes_previous_steps(probe, tmp_path):
    result, out = execute(probe, tmp_path, [
        {'call': {'id': 'wait', 'describe': '超出阶段预算', 'interface': 'runtime/wait', 'args': {'duration': '1s'}}}, call()
    ], timeout='40ms')
    assert result['status'] == 'ERROR'
    assert [e['step'] for e in events(out) if e['event'] == 'step_started'] == ['wait']
    assert_cleaned(out)


def test_step_timeout_is_limited_by_remaining_phase_budget(probe, tmp_path):
    # Consume the first budget locally so JVM first-call warmup cannot fail step one.
    result, out = execute(probe, tmp_path, [
        {'call': {'id': 'first', 'describe': '消耗阶段预算', 'interface': 'runtime/wait',
                  'args': {'duration': '1s'}}},
        call('second', args={'delay_ms': 1500}, within='3s'),
    ], timeout='2s')
    assert result['status'] == 'ERROR'
    assert [e['status'] for e in events(out) if e['event'] == 'step_finished'] == ['PASS', 'ERROR']
    assert_cleaned(out)


def test_resolved_observation_range_reaches_java_context(probe, tmp_path):
    result, out = execute(probe, tmp_path, [check('window', 'probe/range_metric', {'eq': 60}, {'range': ['-60s', '0s']})])
    assert result['status'] == 'PASS'
    gate = events(out, 'gates')[0]
    assert gate['observation']['end'] - gate['observation']['start'] == 60
    assert_cleaned(out)


def test_component_crash_aborts_run(probe, tmp_path):
    result, out = execute(probe, tmp_path, [call(args={'crash': True}), check()])
    assert result['status'] == 'ERROR'
    assert_cleaned(out)


def test_early_process_exit_during_setup_is_error(probe, tmp_path):
    result, out = execute(probe, tmp_path, [check()], extra_args=['exit'])
    assert result['status'] == 'ERROR' and '19' in result['error']
    assert not [e for e in events(out) if e['event'] == 'step_started']
    assert_cleaned(out)


def test_cancel_during_step_records_cancelled_and_cleans(probe, tmp_path):
    case, registry, launch = probe([call(args={'delay_ms': 2000})])
    out = tmp_path / 'cancelled'
    async def run():
        task = asyncio.create_task(run_case(case, registry, launch, out))
        for _ in range(500):
            if (out / 'events.jsonl').is_file() and any(e['event'] == 'step_started' for e in events(out)):
                break
            await asyncio.sleep(.01)
        else:
            pytest.fail('Probe did not reach the step')
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert json.loads((out / 'result.json').read_text())['status'] == 'CANCELLED'
    assert_cleaned(out)


def test_startup_timeout_kills_owned_process(probe, tmp_path):
    case, registry, launch = probe([check()])
    launch['components']['master']['command'] = [sys.executable, '-c', 'import time; time.sleep(30)']
    launch['components']['master']['startup_timeout'] = '50ms'
    out = tmp_path / 'timeout'
    result = asyncio.run(run_case(case, registry, launch, out))
    assert result['status'] == 'ERROR'
    assert_cleaned(out)


def test_no_binding_fails_before_output_or_process_start(probe, tmp_path):
    case, registry, launch = probe([check()])
    launch['bindings'] = {}
    with pytest.raises(ExecutionError, match='No process binding'):
        asyncio.run(run_case(case, registry, launch, tmp_path / 'missing'))
    assert not (tmp_path / 'missing').exists()


def test_existing_output_is_never_overwritten(probe, tmp_path):
    case, registry, launch = probe([check()])
    with pytest.raises(FileExistsError):
        asyncio.run(run_case(case, registry, launch, tmp_path))


@pytest.mark.parametrize('expect,exit_code', [({'lte': '1s'}, 0), ({'lte': '1ms'}, 1)])
def test_cli_run_records_success_and_gate_failure(probe, tmp_path, expect, exit_code):
    import yaml
    case, registry, launch = probe([check(expect=expect)])
    launch_file = tmp_path / 'launch.yaml'
    launch_file.write_text(yaml.safe_dump(launch))
    output = tmp_path / 'cli-output'
    result = subprocess.run([sys.executable, '-m', 'mock_framework.cli', 'run', str(case.source),
                             '--catalog', str(RUNTIME / 'probe-catalog.json'), '--launch', str(launch_file),
                             '--output', str(output)], capture_output=True, text=True, timeout=20,
                            env={**os.environ, 'PYTHONPATH': str(ROOT / 'python')})
    assert result.returncode == exit_code, result.stderr
    assert json.loads(result.stdout)['status'] == ('PASS' if exit_code == 0 else 'FAIL')
    assert_cleaned(output)


def test_range_resolution_uses_fixed_check_time_and_actual_start_receipt():
    now = '2026-10-27T10:01:00+00:00'
    epoch = datetime.fromisoformat(now).timestamp()
    assert observation({'range': ['-60s', '0s']}, now, None) == {
        'selector': ['-60s', '0s'], 'start': epoch - 60, 'end': epoch}
    assert observation({'range': 'latest'}, now, None) == {'selector': 'latest', 'at': now}
    assert observation({'range': 'all'}, now, '2026-10-27T10:00:30Z')['start'] == epoch - 30
    assert observation({'promql': 'rate(example[5m])'}, now, None) is None
    with pytest.raises(ExecutionError, match='actual traffic start'):
        observation({'range': 'all'}, now, None)


def test_collection_comparison_is_per_master_and_order_independent():
    spec = {'returns': {'name': 'PerMasterEngines'}}
    expected = {'eq': {'instances': [{'id': 'p1'}, {'id': 'p0'}]}}
    assert compare({'instances': {'m1': {'instances': [{'id': 'p0'}, {'id': 'p1'}]}}}, expected, spec)
    assert not compare({'instances': {'m0': {'instances': [{'id': 'p0'}]}, 'm1': {'instances': [{'id': 'p1'}]}}}, expected, spec)
    with pytest.raises(ExecutionError, match='No Master'):
        compare({'instances': {}}, expected, spec)
    with pytest.raises(ExecutionError, match='Duplicate'):
        compare({'instances': {'m0': {'instances': [{'id': 'p0'}, {'id': 'p0'}]}}}, expected, spec)


def test_reference_resolves_only_actual_returned_fields():
    assert resolve({'x': Reference('stop', ('affected',))}, {'stop': {'affected': {'instances': []}}}) == {'x': {'instances': []}}
    with pytest.raises(ExecutionError, match='Cannot resolve'):
        resolve(Reference('missing', ()), {})
