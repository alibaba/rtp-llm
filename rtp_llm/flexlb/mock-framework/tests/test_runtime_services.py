import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

from mock_framework.execution.environment import RunEnvironment
from mock_framework.execution.processes import ProcessSupervisor
from mock_framework.execution.topology import Topology
from mock_framework.execution.transport import ExecutionError


def test_topology_readers_and_publishers_cannot_mutate_current_or_retired_instances(tmp_path):
    topology = Topology(tmp_path)
    received = []
    topology.add_publisher(lambda nodes: nodes[0].update(rpc='corrupted') if nodes else None)
    topology.add_publisher(lambda nodes: received.append(nodes))
    endpoint = {'rpc': '127.0.0.1:1001', 'control': 'http://127.0.0.1:1002', 'extra': {'value': 1}}
    topology.register('prefill-0', 'prefill', endpoint['control'], endpoint)
    endpoint['extra']['value'] = 2
    topology.active()[0]['extra']['value'] = 3
    topology.history()[0]['rpc'] = 'changed'
    assert topology.active('decode') == []
    assert topology.active()[0]['extra']['value'] == 1
    assert received[-1][0]['rpc'] == '127.0.0.1:1001'
    topology.retire('prefill-0')
    assert topology.active() == []
    history = topology.history()
    assert history[0]['stopped_at'] >= history[0]['started_at']
    assert json.loads((tmp_path / 'topology.json').read_text()) == []
    assert json.loads((tmp_path / 'topology-history.json').read_text()) == history
    with pytest.raises(ExecutionError, match='identity'):
        topology.register('prefill-0', 'prefill', 'http://127.0.0.1:1002', endpoint)
    with pytest.raises(ExecutionError, match='RPC address'):
        topology.register('prefill-1', 'prefill', endpoint['control'], endpoint)
    assert topology.active() == [] and topology.history() == history


def test_discovery_failure_after_launch_still_reaps_process_and_retains_history(tmp_path, monkeypatch):
    from mock_framework.execution import run_case
    from mock_framework.execution.master import MasterGroup
    from test_master_lifecycle import generic, logs, assert_reaped, step

    def publish(self, nodes):
        if nodes:
            raise OSError('discovery unavailable')
    monkeypatch.setattr(MasterGroup, 'publish_discovery', publish)
    case, registry, launch = generic(tmp_path, [step('never', 'master/stop')])
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'ERROR' and 'discovery unavailable' in result['error']
    events = logs(output)
    assert len([e for e in events if e['event'] == 'process_started']) == 1
    assert not any(e['event'] == 'step_started' for e in events)
    assert_reaped(output)
    assert json.loads((output / 'topology.json').read_text()) == []
    history = json.loads((output / 'topology-history.json').read_text())
    assert len(history) == 1 and 'stopped_at' in history[0]


def test_cancelled_spawn_remains_owned_until_final_cleanup(tmp_path, monkeypatch):
    async def exercise():
        supervisor = ProcessSupervisor(SimpleNamespace(emit=lambda **event: None))
        creating, release = asyncio.Event(), asyncio.Event()
        original = asyncio.create_subprocess_exec
        async def spawn(*args, **kwargs):
            child = await original(*args, **kwargs)
            creating.set()
            await release.wait()
            return child
        monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
        task = asyncio.create_task(supervisor.spawn('peer', [sys.executable, '-c', 'import time; time.sleep(30)'],
            str(tmp_path), dict(os.environ), tmp_path))
        await asyncio.wait_for(creating.wait(), 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        child = supervisor.child('peer')
        assert child.returncode is None
        assert await supervisor.close() == []
        assert supervisor.children == () and child.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(child.pid, 0)
    asyncio.run(exercise())


def test_cleanup_continues_in_order_after_domain_failures(tmp_path):
    async def exercise():
        runtime = RunEnvironment(SimpleNamespace(environment={}), {'components': {}, 'engine_mode': 'shared'},
            SimpleNamespace(directory=tmp_path, emit=lambda **event: None))
        order = []
        def operation(name, *, raises=False, errors=None):
            async def run():
                order.append(name)
                if raises:
                    raise ExecutionError(name + ' failed')
                return errors
            return run
        runtime.engines.stop_monitor = operation('health')
        runtime.clients.drain = operation('drain', raises=True)
        runtime.clients.close = operation('clients', errors=['client cleanup failed'])
        runtime.masters.close = operation('masters')
        runtime.engines.close = operation('engines')
        runtime.masters.close_election = operation('election')
        runtime.supervisor.close = operation('fallback')
        finished = []
        runtime.monitoring = SimpleNamespace(final_scrape=operation('scrape', raises=True),
            close=operation('prometheus'), finish=lambda errors: finished.extend(errors))
        errors = await runtime.close()
        assert order == ['health', 'drain', 'scrape', 'clients', 'masters', 'engines', 'election', 'prometheus', 'fallback']
        assert len(errors) == 3 and finished == errors
        assert 'drain failed' in errors[0] and 'scrape failed' in errors[1]
        assert errors[2] == 'client cleanup failed'
    asyncio.run(exercise())


def test_exit_before_cleanup_is_not_reclassified_as_normal_stop(tmp_path):
    async def exercise():
        supervisor = ProcessSupervisor(SimpleNamespace(emit=lambda **event: None))
        child = await supervisor.spawn('peer', [sys.executable, '-c', 'pass'], str(tmp_path), dict(os.environ), tmp_path)
        assert await child.wait() == 0
        errors = await supervisor.close()
        assert len(errors) == 1 and 'exited unexpectedly before stop with code 0' in errors[0]
        assert supervisor.children == ()
    asyncio.run(exercise())


def test_health_failure_at_teardown_still_fails_without_a_process_exit(tmp_path):
    async def exercise():
        runtime = RunEnvironment(SimpleNamespace(environment={}), {'components': {}, 'engine_mode': 'shared'},
            SimpleNamespace(directory=tmp_path, emit=lambda **event: None))
        runtime.supervisor.fail('Engine membership differs')
        assert await runtime.close() == ['Engine membership differs']
    asyncio.run(exercise())
