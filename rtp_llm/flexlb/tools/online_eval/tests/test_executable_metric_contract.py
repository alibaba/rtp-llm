"""Metric calculation parameters execute; descriptions cannot override algorithms."""

import copy
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from analysis.request_metrics import RequestLedger, describe_calculation
from cases.master_performance.analysis import analyze
from cases.master_performance.metrics import produce, values
from monitoring.metric_store import MetricContractError, MetricStore, publish, series_row
from monitoring.query_plan import definitions, load_plan
from scenario.loader import ScenarioError

ROOT = Path(__file__).resolve().parents[1]


def config(**changes):
    result = dict(calculator='token_throughput', window='measurement',
                  selection=dict(time_basis='completion', status='ok'), token_field='input_len', bucket_s=1)
    result.update(changes)
    return result


def calculation_spec():
    return dict(producer='performance_requests', calculation=config(),
                **describe_calculation(config(), windows={'measurement'}))


def ledger():
    return RequestLedger([
        dict(rid='before', send_start_epoch_ms=9500, total_ms=1000, status='ok',
             input_len=10, observed_output_tokens=2, ttft_ms=10),
        dict(rid='delayed', send_start_epoch_ms=10500, total_ms=1000, status='ok',
             input_len=20, observed_output_tokens=3, ttft_ms=20),
        dict(rid='failed', send_start_epoch_ms=10500, total_ms=250, status='exception', input_len=100),
        dict(rid='tail', send_start_epoch_ms=11500, total_ms=500, status='ok',
             input_len=30, observed_output_tokens=2, ttft_ms=30),
    ])


def test_selection_and_bucket_width_change_the_actual_result():
    rows, windows = ledger(), {'measurement': (10000, 12500)}
    assert rows.series(config(), windows) == [[10, 10], [11, 20], [12, 60]]
    assert rows.series(config(selection=dict(time_basis='arrival', status='ok')), windows) == [
        [10, 20], [11, 30], [12, 0]]
    assert rows.series(config(selection=dict(time_basis='completion', status='non_ok')), windows) == [
        [10, 100], [11, 0], [12, 0]]
    assert rows.series(config(bucket_s=2), windows) == [[10, 15], [12, 60]]
    assert rows.series(config(token_field='observed_output_tokens'), windows) == [[10, 2], [11, 3], [12, 4]]
    shifted = dict(other=(11000, 12000))
    assert rows.series(config(window='other'), shifted) == [[11, 20]]


def test_calculator_choice_and_percentile_execute():
    rows, windows = ledger(), {'measurement': (10000, 12500)}
    selection = dict(time_basis='arrival', status='all')
    rate = dict(calculator='request_rate', window='measurement', selection=selection, bucket_s=1)
    assert rows.series(rate, windows) == [[10, 2], [11, 1], [12, 0]]
    assert rows.series(dict(rate, calculator='success_share'), windows) == [[10, .5], [11, 1], [12, None]]
    assert rows.series(dict(rate, calculator='mean', field='input_len'), windows) == [[10, 60], [11, 30], [12, None]]
    q = dict(rate, calculator='quantile', field='ttft_ms', percentile=.99,
             selection=dict(time_basis='arrival', status='ok'), bucket_s=2)
    assert rows.series(q, windows) == [[10, 30], [12, None]]
    assert rows.series(dict(q, percentile=.5), windows) == [[10, 20], [12, None]]
    inflight = dict(rate, calculator='inflight', selection=dict(time_basis='lifetimes', status='all'))
    assert rows.series(inflight, windows) == [[10, 1], [11, 1], [12, 0]]


@pytest.mark.parametrize('change', [
    dict(calculator='unknown'), dict(window='missing'), dict(bucket_s=0), dict(bucket_s=True),
    dict(bucket_s=math.inf), dict(unknown=1), dict(token_field='unknown'),
    dict(selection=dict(time_basis='terminal_fallback', status='ok')),
    dict(selection=dict(time_basis='completion', status='typo')),
    dict(selection=dict(time_basis='completion')), dict(selection=[]),
])
def test_invalid_calculation_is_rejected_at_load(tmp_path, change):
    spec = calculation_spec()
    del spec['measurement']
    spec['calculation'].update(change)
    path = tmp_path/'case.yaml'
    path.write_text(json.dumps(dict(metric_plan_schema_version=5, produced={'request/input_tps': spec})))
    with patch('monitoring.query_plan.CATALOG', tmp_path), pytest.raises(ScenarioError):
        load_plan('case.yaml')


@pytest.mark.parametrize('field,value', [('unit', 'ms'), ('source_type', 'prometheus'),
                                        ('value_kind', 'scalar'), ('labels', ['role'])])
def test_declared_metadata_must_match_the_calculator(tmp_path, field, value):
    spec = calculation_spec()
    del spec['measurement']
    spec[field] = value
    (tmp_path/'case.yaml').write_text(json.dumps(dict(metric_plan_schema_version=5, produced={'request/input_tps': spec})))
    with patch('monitoring.query_plan.CATALOG', tmp_path), pytest.raises(ScenarioError, match='mismatch'):
        load_plan('case.yaml')


def test_produced_metadata_is_generated_and_cannot_be_authored(tmp_path):
    import yaml
    for path in (ROOT/'config/monitoring').glob('*.yaml'):
        data = yaml.safe_load(path.read_text())
        assert all('measurement' not in spec for spec in data.get('produced', {}).values())
        assert all('measurement' in spec for spec in load_plan(path.name)['produced'].values())
    descriptor = describe_calculation(config(), windows={'measurement'})
    assert descriptor['measurement'] == dict(method='token_throughput', population='measurement:completion:ok',
                                             accuracy='request_ledger', requires_request_identity=True)
    spec = calculation_spec()
    (tmp_path/'case.yaml').write_text(json.dumps(dict(metric_plan_schema_version=5, produced={'request/input_tps': spec})))
    with patch('monitoring.query_plan.CATALOG', tmp_path), pytest.raises(ScenarioError, match='invalid produced metric'):
        load_plan('case.yaml')


def test_case_calculation_contract_rejects_wrong_source_and_unknown_output(tmp_path):
    spec = copy.deepcopy(load_plan('cache_scale_in.yaml')['produced']['derived/survivor_hit_ratio'])
    del spec['measurement']
    spec['source_type'] = 'client_journal'
    (tmp_path/'case.yaml').write_text(json.dumps(dict(metric_plan_schema_version=5, produced={'derived/survivor_hit_ratio': spec})))
    with patch('monitoring.query_plan.CATALOG', tmp_path), pytest.raises(ScenarioError, match='source_type mismatch'):
        load_plan('case.yaml')
    spec['source_type'] = 'prometheus'
    (tmp_path/'case.yaml').write_text(json.dumps(dict(metric_plan_schema_version=5, produced={'cache_gate/unknown': spec})))
    with patch('monitoring.query_plan.CATALOG', tmp_path), pytest.raises(ScenarioError, match='unknown cache metric'):
        load_plan('case.yaml')


def test_new_request_projection_needs_no_case_code():
    from test_performance_gate import evidence
    data = evidence()
    definition = calculation_spec()
    actual = values(data, {'request/custom_input_rate': definition})
    assert list(actual) == ['request/custom_input_rate']
    assert actual['request/custom_input_rate']


def test_publication_rejects_forged_measurement_and_preserves_frozen_calculation(tmp_path):
    from test_performance_gate import evidence
    data = evidence()
    produce(tmp_path, data, analyze(data))
    store = MetricStore.read(tmp_path)
    metric = 'request/ttft_p99_ms'
    row = store.document['metrics'][metric][0]
    assert row['provenance']['calculation_module'] == 'analysis.request_metrics'
    assert len(row['provenance']['calculation_sha256']) == 64
    assert row['provenance']['calculation']['selection']['status'] == 'ok'
    original = row['points']
    with patch('monitoring.query_plan.load_plan', side_effect=AssertionError('must read frozen metrics')):
        assert MetricStore.read(tmp_path).document['metrics'][metric][0]['points'] == original
    store.document['definitions'][metric]['measurement']['population'] = 'forged'
    with pytest.raises(MetricContractError, match='measurement does not match'):
        publish(store, metric, store.document['definitions'][metric],
                [series_row([[10, 1]], epoch=1, source='client', labels={})],
                producer='performance_requests', evidence={})


@pytest.mark.parametrize('change', [dict(rid=None), dict(rid='duplicate'),
                                  dict(send_start_epoch_ms=math.nan), dict(total_ms=math.inf),
                                  dict(status='scheduled')])
def test_incomplete_or_duplicate_journal_never_becomes_a_zero_metric(change):
    row = dict(rid='duplicate', send_start_epoch_ms=1000, total_ms=100, status='ok', input_len=10)
    with pytest.raises(ValueError):
        RequestLedger([row, dict(row, **change)])


def test_old_plan_shape_is_rejected(tmp_path):
    (tmp_path/'case.yaml').write_text('metric_plan_schema_version: 3\nsources: {}\n')
    with patch('monitoring.query_plan.CATALOG', tmp_path), pytest.raises(ScenarioError, match='header'):
        load_plan('case.yaml')


def test_invalid_journal_publishes_missing_samples_without_zero_fallback(tmp_path):
    from test_performance_gate import evidence
    data = evidence()
    data['flow']['complete'] = False
    result = analyze(data)
    assert result['verdict'] == 'INVALID'
    produce(tmp_path, data, result)
    row = MetricStore.read(tmp_path).document['metrics']['request/ttft_p99_ms'][0]
    assert row['status'] == 'ABSENT'
    assert row['points'] == []
