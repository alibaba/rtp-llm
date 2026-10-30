import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import subprocess
from typing import Annotated
import pytest
import yaml

from mock_framework.config import Registry, compile_case
from mock_framework.config.registry import case_interface, Bounds
from mock_framework.config.model import Reference
from mock_framework.execution import ExecutionError, prepare_launch, run_case
from mock_framework.execution.runner import resolve


@dataclass(frozen=True)
class Args:
    value: Annotated[int, Bounds(0, 10)] = 2
    delay: float = 0
    invalid: bool = False
    crash: bool = False


@case_interface('probe/read', effect='read')
async def read(ctx, args: Args) -> int:
    await asyncio.sleep(args.delay)
    return 'invalid' if args.invalid else args.value


@case_interface('probe/write', effect='write')
async def write(ctx, args: Args) -> int:
    ctx.records.emit(event='mutation')
    if args.crash:
        ctx.runtime.supervisor.children[0][1].kill()
    await asyncio.sleep(args.delay)
    return args.value


def step(name='read', **kw):
    return {'id': name, 'describe': name, 'interface': 'probe/' + name, **kw}


@pytest.fixture
def setup(tmp_path):
    def make(steps, timeout='2s', child='ready'):
        registry = Registry.standard()
        registry.register(read); registry.register(write)
        # Real subprocess ownership and teardown, independent of any Java Case catalog.
        script = tmp_path / 'peer.py'
        script.write_text('import os,json,pathlib,time,sys\n' +
            ('sys.exit(19)\n' if child == 'exit' else
             'pathlib.Path(os.environ["MOCK_READY_FILE"]).write_text(json.dumps({"url":"http://127.0.0.1:1"}))\ntime.sleep(30)\n' if child == 'ready' else 'time.sleep(30)\n'))
        document = {'id': 'case', 'describe': 'runtime', 'environment': {'master': {'config': {}}},
            'phases': [{'id': 'phase', 'describe': 'phase', 'timeout': timeout, 'steps': steps}]}
        path = tmp_path / 'case.yaml'; path.write_text(yaml.safe_dump(document))
        case = compile_case(path, registry)
        launch = prepare_launch({'components': {'master': {'command': [sys.executable, str(script)],
            'startup_timeout': '150ms' if child != 'ready' else '3s'}}}, case, tmp_path)
        return case, registry, launch
    return make


def events(out, name='events'):
    return [json.loads(line) for line in (out / (name + '.jsonl')).read_text().splitlines()]


def cleaned(out):
    starts = [e for e in events(out) if e['event'] == 'process_started']
    stops = [e for e in events(out) if e['event'] == 'process_stopped']
    assert len(starts) == len(stops) == 1
    for e in starts:
        with pytest.raises(ProcessLookupError):
            os.kill(e['pid'], 0)


def run(setup, tmp_path, steps, **kw):
    case, registry, launch = setup(steps, **kw)
    out = tmp_path / 'out'
    result = asyncio.run(run_case(case, registry, launch, out))
    cleaned(out)
    return result, out


def test_reads_writes_refs_and_gates_share_one_path(setup, tmp_path):
    result, out = run(setup, tmp_path, [step(), step('write', args={'value': {'ref': 'steps.read'}}, expect={'eq': 2})])
    assert result['status'] == 'PASS'
    assert len([e for e in events(out) if e['event'] == 'mutation']) == 1
    assert events(out, 'gates')[0]['actual'] == 2
    assert (out / 'interfaces.json').is_file()


@pytest.mark.parametrize('args,expect,status', [({}, {'eq': 3}, 'FAIL'), ({'invalid': True}, {'eq': 2}, 'ERROR')])
def test_bad_result_or_gate_stops_following_steps(setup, tmp_path, args, expect, status):
    result, out = run(setup, tmp_path, [step(args=args, expect=expect), step('write')])
    assert result['status'] == status
    assert not [e for e in events(out) if e['event'] == 'mutation']


def test_timed_out_write_is_not_retried(setup, tmp_path):
    result, out = run(setup, tmp_path, [step('write', args={'delay': 1}, within='30ms')])
    assert result['status'] == 'ERROR'
    assert len([e for e in events(out) if e['event'] == 'mutation']) == 1
    assert next(e for e in events(out) if e['event'] == 'step_finished')['outcome_unknown']


def test_failed_write_assertion_is_not_replayed_or_marked_unknown(setup, tmp_path):
    result, out = run(setup, tmp_path, [step('write', expect={'eq': 3})])
    assert result['status'] == 'FAIL'
    assert len([e for e in events(out) if e['event'] == 'mutation']) == 1
    assert not next(e for e in events(out) if e['event'] == 'step_finished')['outcome_unknown']


def test_phase_budget_includes_earlier_steps(setup, tmp_path):
    result, out = run(setup, tmp_path, [step(args={'delay': .08}), step('write', args={'delay': .2}, within='5s')], timeout='150ms')
    assert result['status'] == 'ERROR'
    assert [e['status'] for e in events(out) if e['event'] == 'step_finished'] == ['PASS', 'ERROR']


@pytest.mark.parametrize('child', ['exit', 'hang'])
def test_setup_failure_reclaims_process(setup, tmp_path, child):
    result, out = run(setup, tmp_path, [step()], child=child)
    assert result['status'] == 'ERROR'
    assert not [e for e in events(out) if e['event'] == 'step_started']


def test_child_exit_interrupts_operation(setup, tmp_path):
    result, _ = run(setup, tmp_path, [step('write', args={'crash': True, 'delay': 1})])
    assert result['status'] == 'ERROR' and 'unexpectedly' in result['error']


def test_cancel_records_and_reclaims(setup, tmp_path):
    case, registry, launch = setup([step(args={'delay': 10})])
    out = tmp_path / 'out'
    async def exercise():
        task = asyncio.create_task(run_case(case, registry, launch, out))
        for _ in range(500):
            if (out / 'events.jsonl').exists() and any(e['event'] == 'step_started' for e in events(out)):
                break
            await asyncio.sleep(.01)
        else:
            pytest.fail('Step did not start')
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(exercise()); cleaned(out)
    assert json.loads((out / 'result.json').read_text())['status'] == 'CANCELLED'


def test_existing_output_is_not_overwritten(setup, tmp_path):
    case, registry, launch = setup([step()])
    with pytest.raises(FileExistsError):
        asyncio.run(run_case(case, registry, launch, tmp_path))


def test_refs_do_not_eval_and_require_actual_fields():
    assert resolve(Reference('x', ('value',)), {'x': {'value': 2}}, {}) == 2
    assert resolve(Reference('globals', ('masters',)), {}, {'masters': ['a:1']}) == ['a:1']
    with pytest.raises(ExecutionError):
        resolve(Reference('x', ('missing',)), {'x': {}}, {})


def test_cli_compiles_without_java_catalog(setup, tmp_path):
    case, _, _ = setup([{'id': 'leader', 'describe': 'leader', 'interface': 'master/get', 'args': {'is_leader': True}, 'wait_after': '1ms'}])
    output = tmp_path / 'compiled.json'
    result = subprocess.run([sys.executable, '-m', 'mock_framework.cli', 'compile', str(case.source),
        '--output', str(output)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['phases'][0]['steps'][0]['wait_after'] == '1ms'


def test_wait_after_preserves_refs_and_also_runs_on_last_step(setup, tmp_path):
    from datetime import datetime
    result, out = run(setup, tmp_path, [step(wait_after='40ms', expect={'eq': 2}),
        step('write', args={'value': {'ref': 'steps.read'}}, wait_after='40ms')])
    assert result['status'] == 'PASS'
    log = events(out)
    for name in ('read', 'write'):
        waiting = next(e for e in log if e['event'] == 'wait_started' and e['step'] == name)
        finished = next(e for e in log if e['event'] == 'step_finished' and e['step'] == name)
        assert finished['result'] == 2
        assert (datetime.fromisoformat(finished['recorded_at']) - datetime.fromisoformat(waiting['recorded_at'])).total_seconds() >= .035
    assert [e['event'] for e in log if e.get('step') == 'read'] == ['step_started', 'wait_started', 'step_finished']


@pytest.mark.parametrize('args,expect,status', [({}, {'eq': 3}, 'FAIL'), ({'invalid': True}, None, 'ERROR')])
def test_failed_call_or_assertion_does_not_wait(setup, tmp_path, args, expect, status):
    options = {'expect': expect} if expect else {}
    result, out = run(setup, tmp_path, [step(args=args, wait_after='10s', **options)], timeout='1s')
    assert result['status'] == status
    assert not any(e['event'] == 'wait_started' for e in events(out))


@pytest.mark.parametrize('within,timeout', [('150ms', '2s'), (None, '150ms')])
def test_wait_timeout_preserves_successful_write_and_stops_next_step(setup, tmp_path, within, timeout):
    result, out = run(setup, tmp_path, [step('write', args={'delay': .1}, wait_after='100ms', within=within), step()], timeout=timeout)
    assert result['status'] == 'ERROR'
    finished = [e for e in events(out) if e['event'] == 'step_finished']
    assert len(finished) == 1 and finished[0]['error_stage'] == 'wait_after'
    assert finished[0]['result'] == 2 and not finished[0]['outcome_unknown']
    assert len([e for e in events(out) if e['event'] == 'mutation']) == 1


@pytest.mark.parametrize('interrupt', ['cancel', 'process_exit'])
def test_wait_is_interruptible_without_losing_completed_action(setup, tmp_path, interrupt):
    case, registry, launch = setup([step('write', wait_after='10s'), step()], timeout='20s')
    out = tmp_path / 'out'
    async def exercise():
        task = asyncio.create_task(run_case(case, registry, launch, out))
        async with asyncio.timeout(5):
            while not (out / 'events.jsonl').exists() or not any(e['event'] == 'wait_started' for e in events(out)):
                await asyncio.sleep(.01)
            if interrupt == 'cancel':
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                import signal
                pid = next(e['pid'] for e in events(out) if e['event'] == 'process_started')
                os.kill(pid, signal.SIGTERM)
                result = await task
                assert result['status'] == 'ERROR' and 'unexpectedly' in result['error']
    asyncio.run(exercise()); cleaned(out)
    finished = [e for e in events(out) if e['event'] == 'step_finished']
    assert len(finished) == 1 and finished[0]['error_stage'] == 'wait_after'
    assert finished[0]['result'] == 2 and not finished[0]['outcome_unknown']


def test_multiple_masters_always_use_distinct_processes(setup, tmp_path):
    case, registry, launch = setup([step()])
    case.environment['master']['deployment']['replicas'] = 2
    output = tmp_path / 'out'
    assert asyncio.run(run_case(case, registry, launch, output))['status'] == 'PASS'
    starts = [e for e in events(output) if e['event'] == 'process_started']
    assert {e['node'] for e in starts} == {'master-0', 'master-1'}
    assert len({e['pid'] for e in starts}) == 2
    for event in starts:
        with pytest.raises(ProcessLookupError):
            os.kill(event['pid'], 0)


@pytest.mark.parametrize('exit_code', [None, 0])
def test_intentional_stop_retains_unreaped_process_for_final_cleanup(exit_code, monkeypatch):
    from types import SimpleNamespace
    from mock_framework.execution.processes import ProcessSupervisor
    async def exercise():
        p = ProcessSupervisor(SimpleNamespace(emit=lambda **event: None))
        async def wait():
            child.returncode = exit_code
            if exit_code is None:
                raise TimeoutError()
        child = SimpleNamespace(returncode=None, pid=123, wait=wait)
        p._children['engine-prefill-0'] = child
        monkeypatch.setattr('mock_framework.execution.processes.os.killpg',
                            lambda *args: setattr(child, 'returncode', exit_code))
        original = p.stop
        async def stop(names, grace):
            return await original(names, 0)
        p.stop = stop
        errors = await p.stop_expected('engine-prefill-0')
        assert bool(p.children) == (exit_code is None)
        assert bool(errors) == (exit_code is None)
        assert 'engine-prefill-0' in p._expected
    asyncio.run(exercise())


def test_master_ports_avoid_ephemeral_range_and_retry_busy_pair(monkeypatch):
    import socket
    from mock_framework.execution import master as master_runtime
    first = master_runtime.master_port(1)
    assert 10000 <= first < 30000
    with socket.socket() as busy:
        busy.bind(('127.0.0.1', first + 2))
        second = master_runtime.master_port(1)
        choices = iter([first - 10000, second - 10000])
        monkeypatch.setattr(master_runtime.secrets, 'randbelow', lambda bound: next(choices))
        assert master_runtime.master_port(1) == second
