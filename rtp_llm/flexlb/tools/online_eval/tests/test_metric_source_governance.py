"""Public measurements have consumers; sampled trends cannot replace exact gates."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from cases.master_ha_failover.analysis import HA_METRICS, measure_client_metric
from cases.master_performance.analysis import analyze
from cases.master_performance.metrics import produce
from cases.master_performance.panels import prepare_curves, report_panels
from monitoring.metric_store import MetricContractError, MetricStore
from monitoring.query_plan import load_plan, queries_for_targets
from scenario.loader import ScenarioError
from reporting.view_config import view
from test_performance_gate import evidence

ROOT = Path(__file__).resolve().parents[1]
CLIENT_CURVES = {'client/actual_send_qps': 7.5, 'client/success_qps': 6, 'client/error_qps': 1.5}


def archive_client(directory):
    archive = directory/'telemetry/1/queries.json'
    archive.parent.mkdir(parents=True)
    plan = load_plan('master_performance.yaml')
    archive.write_text(json.dumps(dict(start=100, end=110, step=1, targets={}, queries={
        'client-flow/'+identity.partition('/')[2]: dict(
            promql=plan['sources']['client'][identity.partition('/')[2]]['promql'],
            result=[dict(metric={}, values=[[100, value], [101, value]])])
        for identity, value in CLIENT_CURVES.items()})))


def test_ha_gate_capabilities_equal_the_actual_scene_checks():
    scenario = yaml.safe_load((ROOT/'config/scenarios/master_ha_failover.yaml').read_text())
    consumed = {rule['metric'] for rule in scenario['parameters']['checks'].values()}
    assert consumed == {'ha_gate/'+name for name in HA_METRICS}
    declared = load_plan('master_ha_failover.yaml')['produced']
    assert consumed == {identity for identity in declared if identity.startswith('ha_gate/')}


@pytest.mark.parametrize('name', ['sample_count', 'target_count', 'route_share', 'wrong_error_code',
                                  'prefill_peak_skew', 'failover_count', 'failed_rate_above_one',
                                  'business_rate_above_one', 'visible_terminal_count', 'error_kind_count'])
def test_retired_ha_capabilities_fail_instead_of_returning_zero(name):
    with pytest.raises(ValueError, match='unknown HA metric'):
        measure_client_metric({'metric': 'ha_gate/'+name}, [])


def test_performance_public_scalars_have_checks_or_coverage_consumers():
    scenario = yaml.safe_load((ROOT/'config/scenarios/master_performance.yaml').read_text())
    checks = scenario['parameters']['checks']
    consumed = {criterion['metric'] for key, criterion in checks.items() if key != 'engine_tps'}
    consumed |= {criterion['metric'] for criterion in checks['engine_tps'].values()}
    # Observed incarnation counts explain coverage; they are not business floors.
    coverage = {'performance_gate/'+metric+'_engine_count' for metric in checks['engine_tps']}
    produced = load_plan('master_performance.yaml')['produced']
    assert {identity for identity, spec in produced.items() if spec['value_kind'] == 'scalar'} == consumed | coverage
    presentation = view('master_performance.yaml')
    plotted = {spec['metric_id'] for spec in presentation['charts']['curves'].values()}
    assert {identity for identity, spec in produced.items() if spec['value_kind'] == 'gauge'} <= plotted


def test_client_trends_are_real_queries_without_journal_twins():
    plan = load_plan('master_performance.yaml')
    queries, _ = queries_for_targets(plan, {'client-flow': ''}, lambda _: '{job="client-flow"}', 1)
    for identity in CLIENT_CURVES:
        assert 'client-flow/'+identity.partition('/')[2] in queries
    assert {identity for identity in plan['produced'] if identity.startswith('request/')} == {'request/ttft_p99_ms'}


def test_sampled_curve_values_and_provenance_do_not_change_the_exact_verdict(tmp_path):
    data = evidence()
    frozen = analyze(data)
    archive_client(tmp_path)
    produce(tmp_path, data, frozen)
    curves, _ = prepare_curves(tmp_path, data, view('master_performance.yaml'))
    for identity, value in CLIENT_CURVES.items():
        curve = next(curve for curve in curves if curve['metric_id'] == identity)
        assert curve['points'][0]['y'] == value
        assert curve['provenance']['source_type'] == 'prometheus'
    assert frozen['metrics']['sent_qps'] == 10
    assert analyze(data) == frozen
    stored = MetricStore.read(tmp_path).document['metrics']
    assert stored['performance_gate/cohort_requests'][0]['points'][0][1] == 100
    assert 'performance_gate/sent_qps' not in stored
    assert frozen['metrics']['sent_qps'] == 10  # retained explanatory evidence


def test_missing_prometheus_keeps_qps_empty_even_with_a_complete_journal(tmp_path):
    data = evidence()
    produce(tmp_path, data, analyze(data))
    presentation = view('master_performance.yaml')
    curves, _ = prepare_curves(tmp_path, data, presentation)
    panels = report_panels(curves, data['criteria'], presentation)
    qps = next(panel for panel in panels if panel['id'] == 'client-qps')
    assert qps['series'] == []
    assert qps['caption'] == presentation['charts']['panels'][1]['empty_caption']
    assert any(curve['metric_id'] == 'request/ttft_p99_ms' for curve in curves)
    assert analyze(data)['verdict'] == 'PASS'


def test_complete_prometheus_does_not_rescue_an_incomplete_request_ledger(tmp_path):
    archive_client(tmp_path)
    data = evidence()
    data['flow']['complete'] = False
    invalid = analyze(data)
    assert invalid['verdict'] == 'INVALID'
    produce(tmp_path, data, invalid)
    store = MetricStore.read(tmp_path)
    assert store.select('client/actual_send_qps')[0]['points']
    assert store.document['metrics']['request/ttft_p99_ms'][0]['status'] == 'ABSENT'


def test_missing_required_gate_result_is_a_contract_error(tmp_path):
    data = evidence()
    frozen = copy.deepcopy(analyze(data))
    del frozen['metrics']['goodput_rps']
    with pytest.raises(MetricContractError, match='missing declared performance result'):
        produce(tmp_path, data, frozen)


@pytest.mark.parametrize('selector,value', [('target', 'A'), ('route', 'master'),
                                           ('error_kind', 'business'), ('code', 503)])
def test_ha_rejects_selectors_that_have_no_active_measurement(selector, value):
    from cases.master_ha_failover.inputs import validate_client_criterion
    params = dict(metric='ha_gate/success_rate', op='ge', expected=.95, min_samples=10)
    with pytest.raises((ValueError, ScenarioError), match='does not consume|unknown configuration fields'):
        validate_client_criterion(dict(params, **{selector: value}))
