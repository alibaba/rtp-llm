"""Gate completeness follows raw scrapes, independently of slow control polling."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cases.cache_scale_in.analysis import analyze
from cases.cache_scale_in.observation import measurement_samples
from runtime.deadline import Deadline


def fixture():
    from test_cache_scale_gate import CacheGateTest
    # Existing gate semantics and thresholds are reused without patching analyze.
    return CacheGateTest().evidence()


def prepare(hit=.8):
    e = fixture()
    if hit != .8:
        from test_cache_scale_gate import CacheGateTest
        e = CacheGateTest().evidence(hit=hit)
    e['clock'] = dict(origin_epoch_s=1000, origin_monotonic_s=50)
    dense = copy.deepcopy(e['samples'])
    fields = {key: dict(metric='mock/'+key, labels={'role': 'prefill'}) for key in dense[0]['engines']['p0'] if key != 'grpc_addr'}
    e['gate_input'] = dict(fields=fields)
    # The control loop stalls for six seconds; the TSDB retains each real scrape.
    e['samples'] = [r for r in e['samples'] if not 40 <= r['t'] <= 45]
    def chunks(metric_id, *, source, start, end, deadline, labels):
        assert source == 'mock' and labels == {'role': 'prefill'}
        field = metric_id.split('/')[1]
        rows = []
        for row in dense:
            if not start < row['epoch_s'] <= end:
                continue
            for name, engine in row['engines'].items():
                if field in engine:
                    rows.append(dict(metric=dict(role='prefill', engine_name=name,
                        engine_incarnation=engine.get('engine_incarnation', 'one'),
                        engine_ip=name, grpc_port='7'), values=[[row['epoch_s'], str(engine[field])]]))
        return [rows]
    monitor = SimpleNamespace(client=SimpleNamespace(metric_chunks=Mock(side_effect=chunks),
        metric_definition=lambda identity: dict(unit='number', labels=['engine_name', 'engine_incarnation', 'engine_ip', 'grpc_port'])))
    return e, dense, monitor


def acquire(e, monitor):
    rows = measurement_samples(monitor, e, Deadline(float('inf')),
                               dict(max_samples=1000, max_bytes=1_000_000))
    e['samples'] = rows
    from cases.cache_scale_in.analysis import align_send_counters
    align_send_counters(e, [dict(send_start_epoch_ms=1_000_000+(i+.5)*100) for i in range(800)])
    return rows


def test_dense_scrape_ledger_remains_valid_when_control_poll_is_slow():
    e, _, monitor = prepare()
    assert analyze(e)['verdict'] == 'INVALID'
    rows = acquire(e, monitor)
    assert all(row['epoch_s'] == row['t']+1000 for row in rows)
    assert analyze(e)['verdict'] == 'PASS'


def test_raw_window_uses_frozen_provenance_epoch_without_defaulting(tmp_path):
    import json
    e, _, monitor = prepare()
    e['provenance']['env_epoch'] = 7
    directory = tmp_path/'capture'
    measurement_samples(monitor, e, Deadline(float('inf')),
                        dict(max_samples=1000, max_bytes=1_000_000), directory=directory)
    raw = json.loads((directory/'metrics.json').read_text())
    assert raw['capture']['epoch'] == '7'
    assert {row['epoch'] for rows in raw['metrics'].values() for row in rows} == {'7'}
    del e['provenance']['env_epoch']
    with pytest.raises(KeyError, match='env_epoch'):
        acquire(e, monitor)


def test_dense_scrapes_preserve_real_collapse_failure():
    e, _, monitor = prepare(hit=.3)
    acquire(e, monitor)
    assert analyze(e)['verdict'] == 'FAIL'


@pytest.mark.parametrize('fault', ['missing_scrapes', 'missing_counter', 'reset', 'incarnation'])
def test_real_data_faults_are_not_repaired_or_interpolated(fault):
    e, dense, monitor = prepare()
    if fault == 'missing_scrapes':
        dense[:] = [r for r in dense if not 40 <= r['t'] <= 45]
    elif fault == 'missing_counter':
        del dense[40]['engines']['p0']['context_requests_total']
    elif fault == 'reset':
        dense[40]['engines']['p0']['context_requests_total'] = 0
    else:
        dense[40]['engines']['p0']['engine_incarnation'] = 'new'
    acquire(e, monitor)
    assert analyze(e)['verdict'] == 'INVALID'


def test_measurement_capture_keeps_sample_budget_and_deadline():
    e, _, monitor = prepare()
    with pytest.raises(ValueError, match='budget'):
        measurement_samples(monitor, e, Deadline(float('inf')), dict(max_samples=1, max_bytes=100000))
    with pytest.raises(TimeoutError):
        measurement_samples(monitor, e, Deadline(0), dict(max_samples=1000, max_bytes=100000))


@pytest.mark.parametrize('raw_error', [False, True])
def test_gate_action_freezes_control_evidence_and_never_falls_back_to_poll_success(tmp_path, raw_error):
    import json
    from cases.cache_scale_in.actions import check
    from cases.cache_scale_in.program import GATE
    from artifacts.gate import load_gate
    e, dense, monitor = prepare()
    e['capture_limits'] = dict(max_samples=1000, max_bytes=1_000_000)
    e['errors'] = []
    if raw_error:
        e['samples'] = copy.deepcopy(dense)
        assert analyze(e)['verdict'] == 'PASS'
        monitor.client.metric_chunks.side_effect = RuntimeError('TSDB read failed')
    controls = copy.deepcopy(e['samples'])
    issued = [dict(send_start_epoch_ms=1_000_000+(i+.5)*100) for i in range(800)]
    snapshot = dict(issued=issued, records=[], status=dict(submitted=800), complete=True)
    flow = SimpleNamespace(evidence_snapshot=lambda: snapshot, directory=tmp_path/'flow')
    ctx = SimpleNamespace(artifact_dir=tmp_path, monitor=monitor,
        resource=lambda ref, kind: e if kind=='gate_evidence' else flow,
        services=SimpleNamespace(gate=lambda name: GATE))
    check(ctx, dict(evidence='e', flow='f'), Deadline(float('inf')))
    frozen, result = load_gate(tmp_path, 'cache')
    control = json.loads((tmp_path/'cache-control-evidence.json').read_text())
    assert control['samples'] == controls
    assert frozen['criteria'] == control['criteria']
    assert result['verdict'] == ('INVALID' if raw_error else 'PASS')
    if raw_error:
        assert 'TSDB read failed' in ' '.join(result['errors'])
    else:
        assert frozen['gate_sample_source'] == 'prometheus_raw_scrape'
