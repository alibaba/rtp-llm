"""Startup acknowledges owned resources and cannot replenish the stage budget."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from runtime.deadline import Deadline, StageTimeout
from runtime.startup import wait_ready, wait_flow_started, startup_phase


class Clock:
    def __init__(self):
        self.now = 0
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds
    def deadline(self, seconds=2):
        return Deadline(seconds, self, self.sleep)


def test_shared_deadline_is_not_reset_between_phases():
    clock = Clock()
    deadline = clock.deadline()
    wait_ready(deadline, 'first', lambda: clock.now >= 1)
    with pytest.raises(TimeoutError, match='second.*budget expired'):
        wait_ready(deadline, 'second', lambda: False)
    assert clock.now == pytest.approx(2)


def test_dead_process_fails_before_probe_or_wait():
    clock, probe = Clock(), Mock()
    process = NS(alive=lambda: False, tail_log=lambda: 'owned failure')
    with pytest.raises(RuntimeError, match='owned failure'):
        wait_ready(clock.deadline(), 'http', probe, process=process)
    probe.assert_not_called()
    assert clock.now == 0


def test_stability_requires_continuous_success_and_records_phase_cost(tmp_path):
    clock = Clock()
    # One good tick followed by a bad tick must restart the stable window.
    with startup_phase(tmp_path, 'topology', clock.deadline(4)):
        wait_ready(clock.deadline(4), 'topology', lambda: clock.now != .5, stable_s=1)
    assert clock.now == pytest.approx(2)
    row, = json.loads((tmp_path / 'startup.json').read_text())
    assert row['status'] == 'PASS' and row['duration_s'] == pytest.approx(2)


def test_client_creation_does_not_count_as_sending_acknowledgement(tmp_path):
    clock = Clock()
    identity = dict(run_id='r', group_id='g', phase_id='p')
    process = NS(alive=lambda: True)
    with pytest.raises(TimeoutError, match='sending acknowledgement'):
        wait_flow_started(tmp_path, identity, process, clock.deadline())
    (tmp_path / 'status.json').write_text(json.dumps(dict(identity, state='SENDING')))
    assert wait_flow_started(tmp_path, identity, process, clock.deadline(3))['state'] == 'SENDING'
    (tmp_path / 'status.json').write_text(json.dumps(dict(identity, run_id='foreign', state='SENDING')))
    with pytest.raises(ValueError, match='identity mismatch'):
        wait_flow_started(tmp_path, identity, process, clock.deadline(3))


def test_cancelled_startup_does_not_probe():
    cancelled, probe = threading.Event(), Mock()
    cancelled.set()
    with pytest.raises(TimeoutError):
        wait_ready(Deadline(10, lambda: 0, cancelled=cancelled), 'cancelled', probe)
    probe.assert_not_called()


def test_http_helpers_do_not_swallow_stage_interrupt():
    from runtime.network import http_get_json, http_post_json, wait_for
    with patch('runtime.network.urllib.request.urlopen', side_effect=StageTimeout('stage expired')):
        with pytest.raises(StageTimeout):
            http_get_json('http://unused')
        with pytest.raises(StageTimeout):
            http_post_json('http://unused', {})
    with pytest.raises(StageTimeout):
        wait_for(Mock(side_effect=StageTimeout('stage expired')), 10)


def test_single_and_dual_master_share_topology_wait(tmp_path):
    from adapters.master_lifecycle import MasterLifecycle
    clock = Clock()
    env = NS(run_dir=tmp_path, spec=NS(n_prefill=2, n_decode=1, master_stable_window_s=1, zk_consistency=None))
    proc = NS(alive=lambda: True)
    data = dict(ready=True, worker_summary={'PREFILL': dict(discovered=2, alive=2),
                                           'DECODE': dict(discovered=1, alive=1)})
    lifecycle = MasterLifecycle(Mock())
    with patch('adapters.master_lifecycle.port_accepting', return_value=True), patch(
            'adapters.master_lifecycle.http_post_json', return_value=(200, data)) as request:
        lifecycle._wait_master(env, proc, 'A', '127.0.0.2', 18080, clock.deadline())
    assert clock.now == pytest.approx(1)
    assert all(call.kwargs['timeout'] <= 2 for call in request.call_args_list)
    assert '127.0.0.2' in request.call_args.args[0]


def test_zk_runtime_only_reads_verified_prepared_artifacts(tmp_path):
    from adapters.zk_helper import ZkHelperOps, ZK_LAUNCHER_CLASS
    target = tmp_path / 'flexlb-sync' / 'target'
    runtime = target / 'online-eval-runtime'
    runtime.mkdir(parents=True)
    jar = tmp_path / 'dependency.jar'
    jar.write_bytes(b'dependency')
    launcher = target / 'test-classes' / (ZK_LAUNCHER_CLASS.replace('.', '/') + '.class')
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b'launcher')
    classes = target / 'classes'
    classes.mkdir()
    (classes / 'implementation.class').write_bytes(b'implementation')
    with patch('adapters.zk_helper.FLEXLB_DIR', tmp_path), patch.object(ZkHelperOps, '_mvnw') as maven:
        with pytest.raises(RuntimeError, match='not prepared'):
            ZkHelperOps.default_launch_argv('java')
        cp = runtime / 'classpath.txt'
        cp.write_text(str(jar))
        (runtime / 'manifest.json').write_text(json.dumps({'files': ZkHelperOps._fingerprints(ZkHelperOps._classpath_paths(cp))}))
        argv = ZkHelperOps.default_launch_argv('java')
        assert ZK_LAUNCHER_CLASS in argv
        jar.write_bytes(b'changed')
        with pytest.raises(RuntimeError, match='changed after preparation'):
            ZkHelperOps.default_launch_argv('java')
        maven.assert_not_called()


def test_missing_zk_preparation_fails_before_starting_mock(tmp_path):
    from adapters.environment import EnvManager
    from adapters.environment_config import EnvSpec
    from adapters.zk_helper import ZkHelperOps
    from adapters.zookeeper_config import ZooKeeperConfig
    spec = EnvSpec(label='fixture', n_prefill=1, n_decode=1, perf={},
                   masters=[NS(name='A')], zk_consistency=ZooKeeperConfig())
    manager = EnvManager(tmp_path, verbose=False)
    with patch.object(manager, '_start_mock') as start, patch('adapters.environment.resolve_java21', return_value='java'), patch.object(
            ZkHelperOps, 'default_launch_argv', side_effect=RuntimeError('not prepared')):
        with pytest.raises(RuntimeError, match='not prepared'):
            manager.ensure(spec, deadline=Clock().deadline())
    start.assert_not_called()
    assert manager.current is None


def test_acknowledged_short_flow_may_already_have_exited_successfully(tmp_path):
    identity = dict(run_id='r', group_id='g', phase_id='p')
    process = NS(alive=lambda: False, proc=NS(poll=lambda: 0), tail_log=lambda: 'client failure')
    (tmp_path / 'status.json').write_text(json.dumps(dict(identity, state='DRAINED')))
    assert wait_flow_started(tmp_path, identity, process, Clock().deadline())['state'] == 'DRAINED'
    process.proc.poll = lambda: 1
    with pytest.raises(RuntimeError, match='before sending acknowledgement'):
        wait_flow_started(tmp_path, identity, process, Clock().deadline())


def test_process_death_during_probe_cannot_be_mistaken_for_readiness():
    process = NS(alive=lambda: True, tail_log=lambda: 'owned process died')
    def foreign_port():
        process.alive = lambda: False
        return True
    with pytest.raises(RuntimeError, match='owned process died'):
        wait_ready(Clock().deadline(), 'http', foreign_port, process=process)


def test_elected_master_invokes_production_online_hook(tmp_path):
    from adapters.master_lifecycle import MasterLifecycle
    from adapters.zookeeper_config import ZooKeeperConfig
    clock = Clock()
    env = NS(run_dir=tmp_path, spec=NS(n_prefill=1, n_decode=1,
        master_stable_window_s=0, zk_consistency=ZooKeeperConfig()))
    proc = NS(alive=lambda: True)
    response = Mock(status=200)
    response.read.return_value = b'success'
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    with patch('adapters.master_lifecycle.port_accepting', return_value=True), patch(
            'adapters.master_lifecycle.http_post_json', return_value=(200, {'ready': True})), patch(
            'adapters.master_lifecycle.urllib.request.urlopen', return_value=response) as online:
        MasterLifecycle(Mock())._wait_master(env, proc, 'B', '127.0.0.2', 18080, clock.deadline())
    assert online.call_args.args[0] == 'http://127.0.0.2:18080/hook/after_start'
    assert online.call_args.kwargs['timeout'] <= 2


def test_zookeeper_helper_keeps_stdin_open(tmp_path):
    from adapters.environment import EnvManager
    from adapters.zookeeper_config import ZooKeeperConfig
    proc = NS(pid=100, alive=lambda: True)
    env = NS(zk_helper=None, run_dir=tmp_path, zk_launch_argv=['java', 'launcher'])
    with patch('adapters.environment.ProcessOps.start', return_value=proc) as start, patch(
            'adapters.environment.ZkHelperOps.find_ready_connect_string', return_value='127.0.0.1:2181'):
        EnvManager(tmp_path, verbose=False).start_zk_helper(env, deadline=Clock().deadline())
    assert start.call_args.kwargs['keep_stdin'] is True
    assert env.zk_connect_string == '127.0.0.1:2181'
