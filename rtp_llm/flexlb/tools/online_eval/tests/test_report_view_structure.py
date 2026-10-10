"""All views share a bounded hierarchy and section presentation contract."""

import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from reporting.renderer import render_sections
from reporting.view_config import view
from reporting.view_sections import view_details, view_table
from scenario.loader import ScenarioError

ROOT = Path(__file__).resolve().parents[1]
VIEWS = sorted(path.name for path in (ROOT / 'config/report_views').glob('*.yaml'))


@pytest.mark.parametrize('filename', VIEWS)
def test_top_level_is_only_shared_core_blocks(filename):
    data = view(filename)
    assert list(data) == [key for key in (
        'report_view_schema_version', 'kind', 'report', 'metrics', 'charts', 'sections',
    ) if key in data]
    assert not {'meta', 'kpis', 'audit_columns', 'criteria_columns', 'time_origin'} & data.keys()


@pytest.mark.parametrize('filename', VIEWS)
@pytest.mark.parametrize('key', ['time_origin', 'meta', 'kpis', 'audit_columns', 'custom_case_setting'])
def test_flat_or_private_top_level_fields_are_rejected(filename, key):
    data = copy.deepcopy(view(filename))
    data[key] = 'unexpected'
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError, match='invalid report view fields'):
            view(filename)


@pytest.mark.parametrize('block,key', [('report', 'meta'), ('report', 'kpis'),
                                      ('metrics', 'fallback'), ('charts', 'audit_columns')])
def test_nested_blocks_are_not_an_unvalidated_options_bag(block, key):
    filename = 'cache_scale_in.yaml'
    data = copy.deepcopy(view(filename))
    data[block][key] = 'unexpected'
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError):
            view(filename)


@pytest.mark.parametrize('header', [{'schema_version': 1}, {'report_spec_schema_version': 1},
                                     {'report_view_schema_version': True},
                                     {'report_view_schema_version': 1.0}])
def test_view_loader_rejects_other_version_one_formats(header):
    data = copy.deepcopy(view('default.yaml'))
    data.pop('report_view_schema_version')
    data.update(header)
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError, match='version'):
            view('default.yaml')


@pytest.mark.parametrize('filename, key', [
    ('cache_scale_in.yaml', 'audit'), ('master_performance.yaml', 'checks'),
    ('master_ha_failover.yaml', 'sources'),
])
@pytest.mark.parametrize('mutation', ['missing', 'unknown', 'wrong_opened', 'wrong_columns'])
def test_case_sections_fail_at_load_instead_of_report_generation(filename, key, mutation):
    data = copy.deepcopy(view(filename))
    sections = data['sections']
    if mutation == 'missing':
        sections.pop(key)
    elif mutation == 'unknown':
        sections['arbitrary'] = {'title': 'unknown', 'opened': False}
    elif mutation == 'wrong_opened':
        sections[key]['opened'] = 1
    elif mutation == 'wrong_columns':
        sections[key]['columns'] = ['only one column']
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError, match='sections'):
            view(filename)


def test_all_section_labels_and_open_states_use_common_components():
    data = copy.deepcopy(view('cache_scale_in.yaml'))
    data['sections']['audit'].update(title='YAML audit', columns=['a', 'b', 'c', 'd'], opened=False)
    data['sections']['measurement'].update(title='YAML evidence', opened=True)
    table = view_table(data, 'audit', (row for row in [[1, 2, 3, 4]]))
    detail = view_details(data, 'measurement', {'source': 'frozen'})
    assert table['title'] == 'YAML audit'
    assert table['columns'] == ['a', 'b', 'c', 'd']
    assert table['opened'] is False
    assert detail['opened'] is True
    html = render_sections([table, detail])
    assert '<details class="report-block attachment"><summary>YAML audit' in html
    assert '<details class="report-block attachment" open><summary>YAML evidence' in html
    with pytest.raises(ValueError, match='does not match declared columns'):
        view_table(data, 'audit', [[1, 2]])
    with pytest.raises(ValueError, match='cannot declare columns'):
        view_details(data, 'audit', {})


def test_missing_case_chart_label_fails_before_rendering():
    data = copy.deepcopy(view('cache_scale_in.yaml'))
    del data['charts']['time_origin_label']
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError, match='requires time_origin_label'):
            view('cache_scale_in.yaml')


@pytest.mark.parametrize('filename', VIEWS)
def test_report_title_cannot_override_runtime_identity(filename):
    data = copy.deepcopy(view(filename))
    assert 'title' not in data['report']
    data['report']['title'] = 'arbitrary case title'
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError, match='invalid report fields'):
            view(filename)
