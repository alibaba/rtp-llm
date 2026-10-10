"""New case folders register complete capabilities without editing common lists."""
import importlib
import sys
from pathlib import Path
from dataclasses import replace
from unittest import mock

import pytest

from cases import registry
from scenario.environment_config import environment
from scenario.backend import make_env_spec
from scenario.environment_snapshot import EnvironmentSnapshot
from cases.config_data import merge_environment
from reporting.run_context import checks_section, provenance_from
from reporting.comparison_model import measurement_signature


def test_new_directory_is_discovered_without_framework_registration(tmp_path):
    package = tmp_path / 'discovery_fixture'
    case = package / 'new_case'
    case.mkdir(parents=True)
    (case / 'program.py').write_text('from cases.registry import CaseDefinition\ndef build(case): pass\nCASE = CaseDefinition({"default": build})\n')
    (package / 'helpers.py').write_text('raise AssertionError("helpers must not be imported")')
    sys.path.insert(0, str(tmp_path))
    try:
        importlib.invalidate_caches()
        snapshot = registry.discover(package, package='discovery_fixture')
        assert list(snapshot.cases) == ['new_case']
        assert callable(snapshot.cases['new_case'].definition.builders['default'])
        with pytest.raises(TypeError):
            snapshot.cases['new_case'] = None
    finally:
        sys.path.remove(str(tmp_path))
        for name in list(sys.modules):
            if name.startswith('discovery_fixture'):
                del sys.modules[name]


def test_conflicting_producers_and_sources_are_rejected():
    original = registry.registry().cases['master_ha_failover']
    producer = original.definition.producers['ha_gates']
    other = replace(original, name='another', definition=replace(original.definition,
        producers={'ha_gates': replace(producer, execute=lambda directory: None)}))
    with pytest.raises(ValueError, match='conflicting registered producer'):
        registry.CaseRegistry.build([original, other])
    source = original.definition.sources['master_inflight']
    other = replace(original, name='another', definition=replace(original.definition,
        sources={'master_inflight': replace(source, factory=lambda *args: None)}))
    with pytest.raises(ValueError, match='conflicting registered source'):
        registry.CaseRegistry.build([original, other])


def test_profile_patch_inherits_other_profiles_and_fields():
    base = {'profile_overrides': {'batch-window': {'status_rpc_ms': 1000},
                                 'single-nonbatch': {'status_rpc_ms': 2000}}}
    result = merge_environment(base, {'profile_overrides': {'batch-window': {'cleanup_interval_ms': 5000}}})
    assert result['profile_overrides'] == {'batch-window': {'status_rpc_ms': 1000, 'cleanup_interval_ms': 5000},
                                           'single-nonbatch': {'status_rpc_ms': 2000}}
    assert base['profile_overrides']['batch-window'] == {'status_rpc_ms': 1000}


def test_startup_consumes_snapshot_without_reading_preset_again():
    plan = environment({}, 'fixture', 'batch-window')
    frozen = EnvironmentSnapshot.read(plan['rendered'])
    with mock.patch('scenario.backend.load_preset', side_effect=AssertionError('runtime preset read')):
        spec = make_env_spec(plan, 'batch-window', {'master_base': 18080})
    assert spec.raw_config == frozen.master_json
    corrupt = dict(plan['rendered'], master_json='{}')
    with pytest.raises(ValueError, match='integrity'):
        EnvironmentSnapshot.read(corrupt)


def test_yaml_preemption_omit_matches_renderer_capability():
    plan = environment({'config_overrides': {'ordering': 'priority', 'preemption': {'omit': True}}},
                       'fixture', 'batch-window')
    assert 'preemption' not in plan['resolved_config']['scheduler']['ordering']


def test_report_preserves_inherited_provenance_and_failed_parent():
    meta = provenance_from({'id': 'fixture'}, {'configuration': {'declared': {'known': 1}, 'sha256': 'known'}})
    assert meta['configuration']['sha256'] == 'known'
    rows = checks_section({'id': 'fixture', 'checks': [dict(stage='s', id='parent', status='FAIL', evidence={'checks': []})]})['rows']
    assert rows == [['s/parent', 'FAIL', None, None]]


def test_pairing_uses_measurement_not_display_names():
    panel = dict(id='latency', axes={'y': {}}, timeX=True,
        series=[dict(name='TTFT', metric_id='request/ttft', unit='ms', points=[],
                     provenance={'measurement': {'population': 'arrival:ok'}})])
    renamed = dict(panel, series=[dict(panel['series'][0], name='显示名称变化')])
    completion = dict(panel, series=[dict(panel['series'][0], provenance={'measurement': {'population': 'completion:ok'}})])
    assert measurement_signature(panel) == measurement_signature(renamed)
    assert measurement_signature(panel) != measurement_signature(completion)
