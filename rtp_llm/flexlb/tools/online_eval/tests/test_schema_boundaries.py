"""Same numeric versions must not make independent formats interchangeable."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from cases.config import configure_program
from monitoring.metric_store import MetricContractError, MetricStore
from monitoring.query_plan import load_plan
from reporting import load_analysis, read_bundle, write_bundle
from reporting.spec import validate
from runtime.instance_plan import InstancePlanError, parse_catalog
from scenario import ScenarioError, compile_scenarios

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('header', [
    {'schema_version': 2}, {'metric_plan_schema_version': 2},
    {'case_schema_version': 2.0}, {'case_schema_version': True},
    {'case_schema_version': 3}, {},
    {'case_schema_version': 2, 'metric_plan_schema_version': 2},
])
def test_case_reader_rejects_foreign_legacy_mixed_and_invalid_versions(header):
    document = yaml.safe_load((ROOT / 'config/scenarios/request_completion.yaml').read_text())
    document.pop('case_schema_version')
    document.update(header)
    with pytest.raises(ScenarioError):
        configure_program(document, 'input.yaml')


def test_external_case_and_internal_program_have_different_contracts():
    document = yaml.safe_load((ROOT / 'config/scenarios/request_completion.yaml').read_text())
    program = configure_program(document, 'input.yaml')
    assert program['program_schema_version'] == 1
    assert 'case_schema_version' not in program
    program['case_schema_version'] = program.pop('program_schema_version')
    with pytest.raises(ScenarioError, match='program_schema_version'):
        compile_scenarios([('input.yaml', program)])


@pytest.mark.parametrize('header', [
    {'schema_version': 2}, {'case_schema_version': 2},
    {'metric_plan_schema_version': 2.0}, {'metric_plan_schema_version': True},
    {'metric_plan_schema_version': 2, 'case_schema_version': 2},
])
def test_query_plan_reader_checks_included_format_too(tmp_path, monkeypatch, header):
    from monitoring import query_plan
    monkeypatch.setattr(query_plan, 'CATALOG', tmp_path)
    document = {**header, 'sources': {'mock': {'count': {
        'promql': 'count${selector}', 'unit': 'count', 'value_kind': 'gauge', 'labels': [],
    }}}}
    (tmp_path / 'wrong.yaml').write_text(yaml.safe_dump(document))
    (tmp_path / 'outer.yaml').write_text(yaml.safe_dump({
        'metric_plan_schema_version': 4, 'include': ['wrong.yaml'],
    }))
    with pytest.raises(ScenarioError, match='invalid query plan header'):
        load_plan('outer.yaml')


@pytest.mark.parametrize('header', [
    {'schema_version': 1}, {'lease_schema_version': 1},
    {'instance_catalog_schema_version': True}, {'instance_catalog_schema_version': 1.0},
    {'instance_catalog_schema_version': 1, 'lease_schema_version': 1},
])
def test_instance_inventory_cannot_be_a_different_version_one_format(header):
    with pytest.raises(InstancePlanError, match='instance_catalog_schema_version'):
        parse_catalog({**header, 'instances': []}, source='yaml', profile='batch-window')


@pytest.mark.parametrize('header', [
    {'schema_version': 1}, {'report_spec_schema_version': 1},
    {'metrics_schema_version': True}, {'metrics_schema_version': 1.0},
    {'metrics_schema_version': 1, 'report_spec_schema_version': 1},
])
def test_metric_store_rejects_other_artifacts_even_with_valid_empty_inventory(header):
    with pytest.raises(MetricContractError, match='schema'):
        MetricStore({**header, 'definitions': {}, 'metrics': {}})


@pytest.mark.parametrize('filename,field,foreign', [
    ('manifest.json', 'report_manifest_schema_version', 'report_spec_schema_version'),
    ('report-spec.json', 'report_spec_schema_version', 'report_analysis_schema_version'),
    ('analysis.json', 'report_analysis_schema_version', 'report_manifest_schema_version'),
])
@pytest.mark.parametrize('replacement', ['foreign', 'legacy', 'bool', 'float', 'mixed', 'missing'])
def test_report_reader_rejects_wrong_formats_with_valid_checksums(tmp_path, filename, field, foreign, replacement):
    directory = write_bundle(tmp_path, 'run', 'identity', {}, {'panels': []})
    path = directory / filename
    document = json.loads(path.read_text())
    value = document.pop(field)
    if replacement == 'foreign':
        document[foreign] = value
    elif replacement == 'legacy':
        document['schema_version'] = value
    elif replacement == 'bool':
        document[field] = True
    elif replacement == 'float':
        document[field] = float(value)
    elif replacement == 'mixed':
        document.update({field: value, foreign: value})
    raw = json.dumps(document).encode()
    path.write_bytes(raw)
    if filename != 'manifest.json':
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files'][filename]['sha256'] = hashlib.sha256(raw).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='version'):
        read_bundle(directory)
    if filename == 'analysis.json':
        with pytest.raises(ValueError, match='version'):
            load_analysis(path)


def test_bundle_writer_does_not_retag_a_foreign_spec(tmp_path):
    spec = {'report_analysis_schema_version': 1, 'panels': []}
    original = copy.deepcopy(spec)
    with pytest.raises(ValueError, match='version'):
        write_bundle(tmp_path, 'run', 'wrong', {}, spec)
    assert spec == original
    assert not (tmp_path / 'reports').exists()


def test_in_memory_specs_can_be_unversioned_but_frozen_specs_cannot():
    validate({'panels': []})
    with pytest.raises(ValueError, match='version'):
        validate({'panels': []}, versioned=True)


def test_bundle_reader_validates_nested_run_metadata(tmp_path):
    directory = write_bundle(tmp_path, 'run', 'identity', {}, {'panels': []})
    path = directory / 'report-spec.json'
    document = json.loads(path.read_text())
    document['run_meta']['metrics_schema_version'] = document['run_meta'].pop('run_meta_schema_version')
    raw = json.dumps(document).encode()
    path.write_bytes(raw)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['report-spec.json']['sha256'] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='run metadata version'):
        read_bundle(directory)


@pytest.mark.parametrize('header', [
    {'schema_version': 1}, {'metrics_schema_version': 1},
    {'mode_profiles_schema_version': True}, {'mode_profiles_schema_version': 1.0},
    {'mode_profiles_schema_version': 1, 'metrics_schema_version': 1},
])
def test_standalone_mode_table_uses_its_own_strict_header(tmp_path, header):
    from mode_profiles import load_mode_tables
    document = yaml.safe_load((ROOT / 'mode_profiles.yaml').read_text())
    document.pop('mode_profiles_schema_version')
    document.update(header)
    path = tmp_path / 'mode.yaml'
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match='schema'):
        load_mode_tables(path)


def test_analysis_loader_cannot_treat_a_report_spec_as_native_analysis(tmp_path):
    path = tmp_path / 'analysis.json'
    path.write_text(json.dumps({'report_spec_schema_version': 1, 'panels': []}))
    with pytest.raises(ValueError, match='analysis version'):
        load_analysis(path)
