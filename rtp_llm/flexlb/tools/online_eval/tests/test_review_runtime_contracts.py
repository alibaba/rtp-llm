"""Real failure counterexamples for process, evidence and execution admission."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from runtime.cleanup import stop_process
from runtime.deadline import Deadline, StageTimeout, interruptible
from runtime.sample_worker import SampleWorker
from monitoring.contracts import MetricContractError, MetricUnavailable
from monitoring.metric_store import MetricStore


def test_expired_guard_restores_the_installed_signal_handler():
    handler = signal.getsignal(signal.SIGALRM)
    timer = signal.getitimer(signal.ITIMER_REAL)
    with pytest.raises(StageTimeout):
        with interruptible(Deadline(time.monotonic() - 1), enabled=True):
            pytest.fail('expired guard yielded')
    assert signal.getsignal(signal.SIGALRM) == handler
    assert signal.getitimer(signal.ITIMER_REAL) == timer


def test_expired_cleanup_reaps_a_real_term_ignoring_child(tmp_path):
    ready = tmp_path / 'ready'
    child = subprocess.Popen([sys.executable, '-c',
        'import signal,time,pathlib,sys; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
        'pathlib.Path(sys.argv[1]).touch(); time.sleep(30)', str(ready)])
    try:
        until = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < until:
            time.sleep(.01)
        assert ready.exists()
        resource = SimpleNamespace(proc=child, alive=lambda: child.poll() is None)
        with pytest.raises(StageTimeout):
            stop_process(resource, Deadline(time.monotonic() - 1))
        assert child.poll() == -signal.SIGKILL
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


@pytest.mark.parametrize('ending', ['cancel', 'window', 'unexpected'])
def test_sampling_distinguishes_cooperative_end_from_transport_timeout(ending):
    now = [0.0]
    def capture(deadline):
        if ending == 'cancel':
            worker.stop_event.set()
        elif ending == 'window':
            now[0] = worker.end
        raise TimeoutError('capture stopped')
    worker = SampleWorker(capture, lambda: now[0],
        dict(duration_s=10, interval_s=.001, max_samples=10, max_bytes=1000))
    worker._run()
    assert worker.done_event.is_set()
    assert (worker.error is not None) == (ending == 'unexpected')


def test_series_labels_are_not_split_at_slashes_inside_json():
    from reporting.workload_panels import _identity
    labels = {'path': '/api/{request}/result'}
    identity, actual = _identity('1/client/client/count/' + json.dumps(labels))
    assert identity == ('1', 'client', 'client/count')
    assert actual == labels


@pytest.mark.parametrize('color, canonical', [('red', '#ff0000'), ('#f00', '#ff0000'),
                                           ('rebeccapurple', '#663399'), ('#abcd', '#aabbcc')])
def test_comparison_accepts_view_color_keywords_and_short_hex(color, canonical):
    from reporting.comparison_colors import run_color
    assert run_color(color, 0, 2) == run_color(canonical, 0, 2)


def test_bad_metric_definitions_fail_at_admission_and_nonfinite_series_fail_at_read():
    document = dict(metrics_schema_version=1, definitions={'x': {}}, metrics={})
    with pytest.raises(MetricContractError, match='unit'):
        MetricStore(document)
    document['definitions']['x'] = {'unit': 'count'}
    document['metrics']['x'] = [dict(metric_id='x', source='mock', epoch='1', status='PASS',
        labels={}, provenance={}, points=[[1, float('nan')]])]
    store = MetricStore(document)
    with pytest.raises(MetricUnavailable, match='non-finite'):
        store.series(0)


def test_ambiguous_archive_epochs_are_rejected(tmp_path):
    from monitoring.archive import export_metrics
    (tmp_path / 'queries.json').write_text('{}')
    (tmp_path / 'telemetry/1').mkdir(parents=True)
    (tmp_path / 'telemetry/1/queries.json').write_text('{}')
    with pytest.raises(MetricContractError, match='ambiguous epoch 1'):
        export_metrics(tmp_path)


@pytest.mark.parametrize('case', ['cache_scale_in', 'master_performance', 'master_ha_failover'])
@pytest.mark.parametrize('incompatibility', ['aggregate', 'no_monitor'])
def test_workload_evidence_prerequisites_fail_before_process_launch(case, incompatibility, tmp_path):
    import yaml
    from scenario import load_scenarios
    from document_io import DocumentError
    root = Path(__file__).resolve().parents[1]
    document = yaml.safe_load((root / 'config/scenarios' / (case + '.yaml')).read_text())
    if incompatibility == 'aggregate':
        document.setdefault('execution', {})['collection'] = 'aggregate'
    else:
        document.setdefault('execution', {}).setdefault('monitoring', {})['capture_metrics'] = False
    path = tmp_path / 'case.yaml'
    path.write_text(yaml.safe_dump(document))
    with pytest.raises((DocumentError, ValueError), match=r'execution\.'):
        load_scenarios(path)


def test_shared_proto_cache_survives_concurrent_generators_and_repairs_damage(tmp_path):
    code = 'from adapters.proto_utils import ensure_proto_modules; ensure_proto_modules()'
    # Use a fresh interpreter per consumer so import caches cannot mask corruption.
    env = dict(os.environ, FLEXLB_EVAL_PROTO_OUT=str(tmp_path), PYTHONPATH='src:.')
    children = [subprocess.Popen([sys.executable, '-c', code], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
    for child in children:
        out, err = child.communicate(timeout=30)
        assert child.returncode == 0, (out, err)
    manifest, = tmp_path.glob('*/generated.json')
    hashes = json.loads(manifest.read_text())
    generated = manifest.parent / next(name for name in hashes if name.endswith('_pb2.py'))
    original = generated.read_bytes()
    generated.write_text('not python')
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert generated.read_bytes() == original


def test_live_zero_warmup_starts_with_a_real_covered_sample(tmp_path):
    from unittest.mock import patch
    from cases.master_performance.actions import observe
    from scenario.context import RuntimeContext
    from test_performance_gate import evidence
    now = [1.0]
    def status():
        now[0] += .1  # Real control reads take time; the first is before the anchor.
        return dict(state='SENDING', process_returncode=None)
    flow = SimpleNamespace(control_status=status)
    ctx = RuntimeContext(dict(id='performance', profile='single-nonbatch'), None,
        tmp_path, lambda: now[0], lambda seconds: now.__setitem__(0, now[0] + seconds),
        wall_clock=lambda: 100)
    ctx.env_epoch = 1
    ctx.resource = lambda name, kind: flow
    ctx.register_resource = lambda kind, value, **kwargs: value
    criteria = evidence()['criteria']
    assert criteria['warmup_s'] == 0
    params = dict(flow='flow', criteria=criteria, gate_input=None,
        observation=dict(capture=dict(max_samples=100, max_bytes=100000), windows={}))
    deadline = Deadline(100, lambda: now[0], ctx.sleeper)
    with patch('cases.master_performance.actions.provenance', return_value={}):
        observe(ctx, params, deadline)
    frozen = json.loads((tmp_path / 'performance-gate-evidence.json').read_text())
    assert frozen['errors'] == []
    assert frozen['samples'][0]['epoch_ms'] == frozen['window']['start_epoch_ms']
    assert frozen['samples'][-1]['epoch_ms'] >= frozen['window']['end_epoch_ms']


def test_bundled_ha_prefix_covers_its_declared_one_pass_window():
    import yaml
    from traffic.datasets import CapturedDataset
    root = Path(__file__).resolve().parents[1]
    path = root / 'config/scenarios/master_ha_failover.yaml'
    traffic = yaml.safe_load(path.read_text())['parameters']['traffic']
    capture = CapturedDataset.read(path.parent / traffic['source']['parameters']['path'])
    prefix = capture.model.events[:traffic['max_requests']]
    assert len(prefix) == traffic['max_requests']
    span = (prefix[-1].timestamp_ms - prefix[0].timestamp_ms) / 1000
    assert span / traffic['replay_speed'] >= traffic['duration_s']
