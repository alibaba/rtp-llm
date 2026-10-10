"""Metric bindings, typed thresholds and traffic semantics fail before execution."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cases.config import configure_program
from cases.metric_inputs import metric_fields
from monitoring.metric_store import MetricStore, export_metrics, publish
from monitoring.producers import produce
from monitoring.query_plan import definitions, load_plan, validate_export_filter
from scenario.loader import load_document, ScenarioError

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return load_document(ROOT / 'config/scenarios' / (name + '.yaml'))


@pytest.mark.parametrize('name,criterion', [('master_performance', 'ttft'),
    ('cache_scale_in', 'baseline_hit'), ('master_ha_failover', 'outage_failures')])
@pytest.mark.parametrize('field,value', [('metric', 'missing/id'), ('unit', 'seconds'),
                                       ('windows', ['typo'])])
def test_check_identity_unit_and_window_are_compile_time_contracts(name, criterion, field, value):
    data = config(name)
    data['parameters']['checks'][criterion][field] = value
    with pytest.raises((ScenarioError, ValueError)):
        configure_program(data, name + '.yaml')


def test_impossible_offered_load_requires_an_explicit_threshold_review():
    data = config('master_performance')
    data['parameters']['traffic']['client']['playback']['qps'] /= 2
    with pytest.raises(ValueError, match='offered-load envelope'):
        configure_program(data, 'case.yaml')
    checks = data['parameters']['checks']
    checks['requests']['expected'] /= 2
    checks['requests']['expected'] = int(checks['requests']['expected'])
    checks['goodput']['expected'] /= 2
    plan = configure_program(data, 'case.yaml')
    criteria = next(s for s in plan['variants'][0]['stages'] if s['id'] == 'measure')['params']['criteria']
    assert criteria['min_input_tps'] == checks['input_tps']['expected']
    assert criteria['qps'] == 768


def test_profile_threshold_typo_is_rejected_even_for_an_unselected_profile():
    data = config('master_performance')
    data['profiles'] = ['batch-window']
    item = data['parameters']['checks']['engine_tps']['rtp_llm_context_tps']
    item['expected_by_profile'] = {'single-nonbacth': 45000}
    with pytest.raises(ValueError, match='registered profiles'):
        configure_program(data, 'case.yaml')


@pytest.mark.parametrize('name', ['master_performance', 'cache_scale_in', 'master_ha_failover'])
def test_source_priority_is_the_client_priority_authority(name):
    data = config(name)
    traffic = data['parameters']['traffic']
    traffic['source']['parameters']['priority'] = 60
    plan = configure_program(data, 'case.yaml')
    stage = next(s for s in plan['variants'][0]['stages'] if s['action'] in ('java_flow_start', 'master_client_start'))
    assert stage['params']['source']['parameters']['priority'] == 60
    if name != 'master_ha_failover':
        assert stage['params']['client']['PRIORITY'] == '60'
        traffic['client']['PRIORITY'] = '50'
        with pytest.raises(ValueError, match='conflicts'):
            configure_program(data, 'case.yaml')


@pytest.mark.parametrize('name', ['request_completion', 'master_performance', 'cache_scale_in', 'master_ha_failover'])
def test_driver_is_required_and_cannot_select_another_field_contract(name):
    data = config(name)
    data['parameters']['traffic']['kind'] = 'typo'
    with pytest.raises(ValueError, match='traffic.kind'):
        configure_program(data, 'case.yaml')


def test_numeric_schema_does_not_hide_unused_program_fields():
    data = config('request_completion')
    data['parameters']['traffic']['unused'] = 1
    data['parameter_schema'] = {'traffic.unused': dict(minimum=0, integer=True)}
    with pytest.raises(ScenarioError, match='unknown configuration fields'):
        configure_program(data, 'case.yaml')
    data = config('master_performance')
    data['parameters']['observation']['windows']['measurement']['until']['offset_s'] = -1
    with pytest.raises(ScenarioError, match='observation.windows.measurement'):
        configure_program(data, 'case.yaml')


def test_metric_projections_have_one_direction_and_no_duplicate_identity():
    fields = dict(a=dict(metric='mock/running', labels=dict(role='prefill')),
                  b=dict(metric='mock/running', labels=dict(role='decode')))
    assert metric_fields(dict(fields=fields)) == fields
    fields['b']['labels']['role'] = 'prefill'
    with pytest.raises(ValueError, match='duplicate'):
        metric_fields(dict(fields=fields))


def test_export_allowlist_and_query_selection_are_independent_but_consistent():
    plan = load_plan('master_performance.yaml')
    validate_export_filter(plan, dict(metric_whitelist='flexlb_'))
    with pytest.raises(ScenarioError, match='excluded by metric_whitelist'):
        validate_export_filter(plan, dict(metric_whitelist='flexlb_auto_tpm_'))


def test_unused_query_twins_are_excluded_without_merging_different_estimands():
    perf = definitions(load_plan('master_performance.yaml'))
    assert 'client/ttft_p99_seconds' not in perf
    assert perf['request/ttft_p99_ms']['measurement']['accuracy'] == 'request_ledger'
    assert perf['performance_gate/rtp_llm_context_tps_scrape_engine_mean']['source_type'] == 'prometheus'
    assert perf['mock/rtp_llm_context_tps_engine_mean']['measurement']['method'] == 'promql_evaluation'
    cache = definitions(load_plan('cache_scale_in.yaml'))
    assert 'mock/cache_hit_ratio' in cache and 'derived/survivor_hit_ratio' in cache


def test_origin_does_not_control_producer_execution_or_retention(tmp_path):
    plan = load_plan('cache_scale_in.yaml')
    store = export_metrics(tmp_path, plan)
    metric = 'derived/survivor_hit_ratio'
    definition = store.document['definitions'][metric]
    publish(store, metric, definition,
            [dict(epoch='1', source='cache_gate', labels={}, points=[[1, .5]])],
            producer='cache_windows', evidence=dict(survivors=['P0']))
    store.save(tmp_path)
    archive = tmp_path / 'telemetry/1/queries.json'
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps(dict(start=1, end=2, step=1, targets={}, queries={},
                                      query_plan='cache_scale_in.yaml', metric_plan=plan)))
    assert export_metrics(tmp_path, plan).reduce(metric, op='last') == .5
    frozen = MetricStore.read(tmp_path).document['metrics'][metric][0]
    assert frozen['provenance']['measurement']['population'] == 'surviving_prefill_engines'
    # Finalize selection uses producer registration, not an origin != prometheus test.
    ha = load_plan('master_ha_failover.yaml')
    for spec in ha['produced'].values():
        spec['source_type'] = 'prometheus'
    export_metrics(tmp_path / 'ha', ha)
    from dataclasses import replace
    from cases import registry
    from case_registry_fixtures import snapshot
    from unittest.mock import Mock
    producer = Mock(return_value={})
    case = registry.registry().cases['master_ha_failover']
    producers = dict(case.definition.producers)
    producers['ha_evidence'] = replace(producers['ha_evidence'], execute=producer)
    changed = replace(case, definition=replace(case.definition, producers=producers))
    with patch.object(registry, '_snapshot', snapshot(changed, include_existing=True)):
        produce(tmp_path / 'ha', {})
    producer.assert_called_once()


@pytest.mark.parametrize('name,criterion', [
    ('cache_scale_in', 'offered_load'), ('master_performance', 'ttft'),
    ('master_ha_failover', 'b_success'), ('request_completion', 'completed'),
])
@pytest.mark.parametrize('windows', [[], 'baseline_and_post', ['typo'], [None],
                                     ['baseline', 'baseline']])
def test_check_windows_reject_ambiguous_or_invalid_references(name, criterion, windows):
    data = config(name)
    data['parameters']['checks'][criterion]['windows'] = windows
    with pytest.raises((ScenarioError, ValueError), match='window'):
        configure_program(data, 'case.yaml')


def test_composite_check_requires_both_declared_windows_and_has_no_order_semantics():
    data = config('cache_scale_in')
    expected = configure_program(data, 'case.yaml')['variants']
    check = data['parameters']['checks']['offered_load']
    check['windows'] = ['post', 'baseline']
    assert configure_program(data, 'case.yaml')['variants'] == expected
    check['windows'] = ['baseline']
    with pytest.raises(ValueError, match='measurement contract'):
        configure_program(data, 'case.yaml')


@pytest.mark.parametrize('name,criterion,windows', [
    ('master_ha_failover', 'b_success', ['b_only', 'both']),
    ('request_completion', 'completed', ['terminal', 'terminal']),
])
def test_single_window_measurements_do_not_concatenate_scopes(name, criterion, windows):
    data = config(name)
    data['parameters']['checks'][criterion]['windows'] = windows
    with pytest.raises(ScenarioError, match='window'):
        configure_program(data, 'case.yaml')


@pytest.mark.parametrize('name,policy', [('cache_scale_in', 'collapse'),
                                        ('master_performance', 'slo')])
def test_analysis_policies_are_separate_from_observation_and_still_typed(name, policy):
    data = config(name)
    data['parameters']['analysis'][policy][next(iter(data['parameters']['analysis'][policy]))] = True
    with pytest.raises(ScenarioError, match='analysis'):
        configure_program(data, 'case.yaml')
    data = config(name)
    data['parameters']['observation'][policy] = data['parameters']['analysis'].pop(policy)
    with pytest.raises(ScenarioError):
        configure_program(data, 'case.yaml')


@pytest.mark.parametrize('name,input_group', [('cache_scale_in', 'engine_counters'),
                                             ('master_performance', 'engine_tps')])
def test_metric_bindings_cannot_select_a_reader_backend(name, input_group):
    data = config(name)
    data['parameters']['observation']['inputs'][input_group]['source'] = 'metric_store'
    with pytest.raises(ValueError, match='program owns the reader'):
        configure_program(data, 'case.yaml')
