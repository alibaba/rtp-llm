"""Failure injection and real process witnesses for final resource ownership."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from runtime.deadline import Deadline
from scenario.resources import ResourceScope
from scenario.stage_execution import _core_action


def test_explicit_teardown_keeps_failed_owner_for_final_retry():
    ctx = ResourceScope(time.monotonic, time.sleep, True)
    callback = Mock(side_effect=[RuntimeError('transient'), None])
    ctx.add_cleanup('resource', callback)
    with pytest.raises(RuntimeError, match='teardown'):
        _core_action('teardown', ctx, {}, Deadline(time.monotonic()+1))
    assert ctx._cleanup[0].state == 'CLOSE_FAILED'
    assert ctx.cleanup(1)[0]['status'] == 'PASS'
    assert callback.call_count == 2
    assert ctx.cleanup_results[0]['status'] == 'ERROR'


def test_expired_budget_does_not_enter_uncooperative_callback():
    ctx = ResourceScope(time.monotonic, time.sleep, True)
    callback = Mock(side_effect=lambda d: time.sleep(60))
    ctx.add_cleanup('resource', callback)
    assert ctx.cleanup(0)[0]['status'] == 'TIMEOUT'
    callback.assert_not_called()
    assert ctx._cleanup


def test_all_stop_requests_precede_any_join_and_retries_keep_order():
    ctx = ResourceScope(time.monotonic, time.sleep, True)
    events = []
    for name in ('one', 'two'):
        ctx.add_cleanup(name, lambda d, name=name: events.append('join-'+name),
                        request_stop=lambda name=name: events.append('stop-'+name))
    assert all(row['status'] == 'PASS' for row in ctx.cleanup(1))
    assert events == ['stop-two', 'stop-one', 'join-two', 'join-one']


def test_probe_partial_close_remains_retryable():
    from monitoring.probe import PrometheusEvidence
    probe = PrometheusEvidence.__new__(PrometheusEvidence)
    probe.closed, probe.stopping, probe.error, probe.started = False, False, None, None
    probe.ended = None
    probe.lock = threading.Lock()
    probe.server, probe.thread = Mock(), Mock()
    server = probe.server
    server.shutdown.side_effect = [TimeoutError('interrupted'), None]
    probe.thread.is_alive.return_value = False
    with pytest.raises(TimeoutError):
        probe.stop(deadline=Deadline(time.monotonic()+1))
    assert not probe.closed
    probe.stop(deadline=Deadline(time.monotonic()+1))
    assert server.shutdown.call_count == 2
    assert probe.closed and probe.server is None


def test_channel_failure_does_not_prevent_pid_reclamation(tmp_path):
    from scenario.backend import JavaMockBackend
    backend = JavaMockBackend({})
    backend.raw_ops = Mock()
    backend.raw_ops.close.side_effect = RuntimeError('channel close')
    process = Mock(pid=12345)
    process.alive.side_effect = [True, False]
    process.proc.wait.return_value = 0
    backend.owned_processes[12345] = process
    ctx = NS(clock=time.monotonic, artifact_dir=tmp_path, env_epoch=1)
    with patch('scenario.backend.os.kill'), pytest.raises(RuntimeError, match='channel close'):
        backend.teardown(ctx, Deadline(time.monotonic()+1))
    process.proc.terminate.assert_called_once()
    assert json.loads((tmp_path/'process-cleanup.json').read_text())['remaining_pids'] == []


def test_retired_case_api_is_not_public():
    from case_api.builder import CaseBuilder
    from scenario import compiler
    builder = CaseBuilder({}, {}, number_rules={})
    assert not hasattr(builder, 'inputs') and not hasattr(builder, 'environment')
    assert not hasattr(compiler, 'environment')


def test_author_documentation_does_not_recommend_retired_inputs():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1]/'docs/development/adding-cases.md').read_text()
    assert 'case.inputs(' not in text and 'ProgramInputs' not in text
    assert 'NUMERIC_PARAMETERS' not in text


def test_parent_sigkill_closes_owner_pipe_and_reclaims_workload_group(tmp_path):
    witness = tmp_path/'witness.json'
    payload = "import os,json,time;from pathlib import Path;Path("+repr(str(witness))+").write_text(json.dumps([os.getpid(),os.getpgrp()]));time.sleep(60)"
    script = "from scenario.instance_runner import ChildProcesses;from pathlib import Path;import os,sys;ChildProcesses().run([sys.executable,'-c',"+repr(payload)+"],dict(os.environ),Path("+repr(str(tmp_path/'worker.log'))+"))"
    parent = subprocess.Popen([sys.executable, '-c', script])
    group = None
    try:
        end = time.monotonic()+10
        while not witness.exists():
            assert parent.poll() is None
            assert time.monotonic() < end
            time.sleep(.02)
        pid, group = json.loads(witness.read_text())
        parent.kill()
        parent.wait(timeout=5)
        while time.monotonic() < end:
            status = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
            if not status or status.startswith('Z'):
                break
            time.sleep(.02)
        else:
            pytest.fail('workload survived loss of its owner')
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if group is not None:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize('callback,stop', [(lambda:None,None), (lambda d:None,lambda d:None)])
def test_cleanup_signatures_are_rejected_at_registration_without_acquisition(callback,stop):
    ctx=ResourceScope(time.monotonic,time.sleep,False)
    with pytest.raises(ValueError,match='positional arguments'):
        ctx.add_cleanup('bad',callback,request_stop=stop)
    assert ctx._cleanup==[]
    with pytest.raises(ValueError,match='positional arguments'):
        ctx.register_resource('bad',object(),cleanup=callback,request_stop=stop)
    assert ctx.resource_count==0 and ctx._cleanup==[]
