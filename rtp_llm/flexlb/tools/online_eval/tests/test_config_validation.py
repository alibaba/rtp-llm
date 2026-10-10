"""Configuration boundaries reject invalid declarations without runtime exceptions."""

import copy
import json
from unittest.mock import patch

import pytest

from cases.master_ha_failover.analysis import HA_METRICS, measure_client_metric
from monitoring.query_plan import load_plan
from reporting.view_config import view
from scenario.loader import ScenarioError


@pytest.mark.parametrize('filename', ['cache_scale_in_overview.yaml', 'master_ha_core.yaml'])
def test_view_loads_its_metric_plan_once(filename):
    with patch('monitoring.query_plan.load_plan', wraps=load_plan) as loader:
        view(filename)
    # Recursive include loads are separate files; each file is read once.
    names = [call.args[0] for call in loader.call_args_list]
    assert len(names) == len(set(names))


@pytest.mark.parametrize('filename', ['cache_scale_in_overview.yaml', 'master_ha_core.yaml'])
@pytest.mark.parametrize('metric_id', [None, [], {}, 'mock/does_not_exist'])
def test_invalid_metric_binding_reports_configuration_error(filename, metric_id):
    data = copy.deepcopy(view(filename))
    identity = next(iter(data['curves']))
    style = data['curves'][identity]
    if metric_id is None:
        del style['metric_id']
    else:
        style['metric_id'] = metric_id
    # HA requires an exact field set; produced views check the shared binding.
    message = 'invalid HA metric presentation' if filename == 'master_ha_core.yaml' and metric_id is None else 'curve must bind'
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError, match=message):
            view(filename)


def test_curves_require_an_explicit_plan():
    data = copy.deepcopy(view('cache_scale_in_overview.yaml'))
    del data['monitoring_query_plan']
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError, match='curves require monitoring_query_plan'):
            view('cache_scale_in_overview.yaml')


@pytest.mark.parametrize('change, message', [
    ({'hidden': 1}, 'invalid metric visibility'),
    ({'scale': True}, 'invalid metric scale'),
    ({'scale': 0}, 'invalid metric scale'),
    ({'scale': 2, 'source_unit': None}, 'invalid metric unit or color'),
    ({'unknown': True}, 'invalid metric presentation'),
])
def test_curve_styles_keep_strict_types_and_whitelists(change, message):
    data = copy.deepcopy(view('cache_scale_in_overview.yaml'))
    next(iter(data['curves'].values())).update(change)
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError, match=message):
            view('cache_scale_in_overview.yaml')


@pytest.mark.parametrize('change, message', [
    ({'mode': []}, 'invalid timestamp mode'),
    ({'value_kind': {}}, 'metric requires unit'),
    ({'required': 1}, 'required must be boolean'),
    ({'labels': ['role', 'role']}, 'metric requires unit'),
    ({'unknown': True}, 'invalid query definition'),
    ({'promql': 'metric${other}'}, 'invalid PromQL placeholders'),
    ({'mode': 'scrape', 'promql': 'sum(metric${selector})'}, 'raw metric selector'),
])
def test_query_definition_errors_are_strict_and_path_aware(tmp_path, change, message):
    spec = dict(promql='metric${selector}', unit='count', value_kind='gauge', labels=[])
    spec.update(change)
    filename = tmp_path/'test.yaml'
    filename.write_text(json.dumps(dict(metric_plan_schema_version=2, sources={'mock': {'metric': spec}})))
    with patch('monitoring.query_plan.CATALOG', tmp_path):
        with pytest.raises(ScenarioError, match=message) as error:
            load_plan('test.yaml')
    assert str(filename) in str(error.value)


@pytest.mark.parametrize('change, message', [
    ({'source_type': []}, 'invalid produced metric'),
    ({'value_kind': {}}, 'metric requires unit'),
    ({'producer': 'missing_producer'}, 'unknown metric producer'),
])
def test_produced_definition_errors_are_configuration_errors(tmp_path, change, message):
    spec = dict(producer='ha_evidence', source_type='derived', unit='count', value_kind='scalar', labels=[])
    spec.update(change)
    (tmp_path/'test.yaml').write_text(json.dumps(dict(metric_plan_schema_version=2, produced={'derived/count': spec})))
    with patch('monitoring.query_plan.CATALOG', tmp_path):
        with pytest.raises(ScenarioError, match=message):
            load_plan('test.yaml')


@pytest.mark.parametrize('name', sorted(HA_METRICS - {'prefill_max_share', 'prefill_peak_skew'}))
def test_empty_ha_windows_retain_zero_result(name):
    assert measure_client_metric({'metric': 'ha_gate/'+name}, []) == 0


def test_ha_count_and_rate_boundaries():
    rows = [dict(rid='same', failover=1, route_path='failed', error_kind='business',
                 error='1503') for _ in range(3)]
    assert measure_client_metric({'metric': 'ha_gate/duplicate_ids'}, rows) == 1
    assert measure_client_metric({'metric': 'ha_gate/failover_count'}, rows) == 0
    assert measure_client_metric({'metric': 'ha_gate/failed_rate_above_one'}, rows[:1]) == 0
    assert measure_client_metric({'metric': 'ha_gate/failed_rate_above_one'}, rows) == 1
    assert measure_client_metric({'metric': 'ha_gate/business_rate_above_one'}, rows) == 1
    assert measure_client_metric({'metric': 'ha_gate/wrong_error_code', 'code': 503}, rows) == 0
    with pytest.raises(ValueError, match='unknown HA metric'):
        measure_client_metric({'metric': 'ha_gate/unknown'}, rows)


@pytest.mark.parametrize('filename', ['cache_scale_in_overview.yaml', 'master_ha_core.yaml'])
def test_bad_panel_identity_or_binding_has_configuration_error(filename):
    data = copy.deepcopy(view(filename))
    for field, value, message in [('id', [], 'invalid.*panel'),
                                  ('curve_ids', ['missing'], 'invalid.*panel')]:
        broken = copy.deepcopy(data)
        broken['panels'][0][field] = value
        with patch('reporting.view_config.load_document', return_value=broken):
            with pytest.raises(ScenarioError, match=message):
                view(filename)
