"""New authors can compile the documented example without framework edits."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from case_api import registry as module
from case_api.registry import CaseRegistry
from case_api.registry import CaseDefinition
from case_api.registry import MetricProducer
from case_api.registry import ProducerPhase
from case_api.registry import RegisteredCase
from case_api.registry import ReportView
from case_api.registry import EvidenceSource
from scenario.case_validation import validate_cases

ROOT = Path(__file__).resolve().parents[1]


def entry(definition):
    return RegisteredCase('example', 'example.program', Path(__file__), definition)


@pytest.mark.parametrize('definition, location', [
    (CaseDefinition(builders={'default': lambda: None}), 'builders.default'),
    (CaseDefinition(builders={'default': lambda case: None}, actions=(object(),)), 'actions'),
    (CaseDefinition(builders={'default': lambda case: None}, producers={
        'metric': MetricProducer(lambda a, b, c: {}, lambda a, b: None, ProducerPhase.GATE)}),
        'producers.metric.execute'),
    (CaseDefinition(builders={'default': lambda case: None}, producers={
        'metric': MetricProducer(lambda a, b, c: {}, lambda a: None, ProducerPhase.FINALIZE)}),
        'producers.metric.execute'),
    (CaseDefinition(builders={'default': lambda case: None}, report_views={
        'example.yaml': ReportView(lambda a, b, c: None, lambda a, b: None)}),
        'views.example.yaml.renderer'),
    (CaseDefinition(builders={'default': lambda case: None}, sources={
        'example': EvidenceSource(lambda env: {}, frozenset({'value'}), 'example')}),
        'sources.example.factory'),
])
def test_capability_signature_errors_are_located_before_execution(definition, location):
    with pytest.raises(ValueError, match=re.escape(location)):
        CaseRegistry.build([entry(definition)])


def test_registration_does_not_execute_capabilities():
    def forbidden(*args):
        raise AssertionError('registration executed a capability')
    CaseRegistry.build([entry(CaseDefinition(builders={'default': forbidden}, producers={
        'gate': MetricProducer(forbidden, forbidden, ProducerPhase.GATE),
        'finalize': MetricProducer(forbidden, forbidden, ProducerPhase.FINALIZE),
    }))])


def test_import_failure_identifies_case_and_preserves_cause(tmp_path, monkeypatch):
    package = tmp_path / 'broken_cases'
    (package / 'broken').mkdir(parents=True)
    (package / '__init__.py').touch()
    (package / 'broken' / '__init__.py').touch()
    (package / 'broken' / 'program.py').write_text('raise RuntimeError("missing dependency")')
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match='broken_cases.broken.program: case import failed') as error:
        module.discover(package, package='broken_cases')
    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.fixture
def documented_case(tmp_path, monkeypatch):
    document = (ROOT / 'docs/development/adding-cases.md').read_text()
    python = re.findall(r'```python\n(.*?)\n```', document, re.S)[0]
    yaml = re.findall(r'```yaml\n(.*?)\n```', document, re.S)[0]
    package = tmp_path / 'author_cases'
    directory = package / 'protocol_terminal'
    directory.mkdir(parents=True)
    (package / '__init__.py').touch()
    (directory / '__init__.py').touch()
    (directory / 'program.py').write_text(python)
    monkeypatch.syspath_prepend(str(tmp_path))
    original = module.registry()
    discovered = module.discover(package, package='author_cases')
    monkeypatch.setattr(module, '_snapshot', CaseRegistry.build([
        *original.cases.values(), *discovered.cases.values()]))
    source = tmp_path / 'protocol_terminal.yaml'
    source.write_text(yaml)
    yield source
    for name in list(sys.modules):
        if name == 'author_cases' or name.startswith('author_cases.'):
            monkeypatch.delitem(sys.modules, name)


def test_documented_case_is_discovered_and_validated(documented_case):
    result = validate_cases(documented_case)
    assert result['runtime_validation'] == 'NOT_RUN'
    instance, = result['instances']
    assert instance['id'] == 'protocol_terminal::default::single-nonbatch'
    assert instance['stages'][3]['check_ids'] == ['comparison']
    assert instance['reports'] == ['default.yaml']
    assert instance['metrics']['definitions'] == {}
    assert instance['numeric_parameters']['traffic.count']['integer'] is True


def test_documented_case_rejects_unknown_and_invalid_inputs(documented_case):
    import yaml
    original = yaml.safe_load(documented_case.read_text())
    for key, value, message in [('typo', 1, 'unknown'), ('count', 0, 'count')]:
        data = json.loads(json.dumps(original))
        data['parameters']['traffic'][key] = value
        documented_case.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match=message):
            validate_cases(documented_case)


def test_validation_command_works_from_other_directory(tmp_path):
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts/commands/validate_case.py'),
        '--source', str(ROOT / 'config/scenarios/request_completion.yaml'), '--json'],
        cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert len(report['instances']) == 4
    assert report['scope'] == 'static_compilation'
    failed = subprocess.run([sys.executable, str(ROOT / 'scripts/commands/validate_case.py'),
        '--source', str(tmp_path / 'missing.yaml'), '--json'],
        cwd=tmp_path, capture_output=True, text=True)
    assert failed.returncode == 2
    assert json.loads(failed.stdout)['status'] == 'ERROR'


def test_full_inventory_explains_metric_consumers_and_profile_contracts(tmp_path):
    import os
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts/commands/validate_case.py'),
        '--source', str(ROOT / 'config/scenarios'), '--json'], cwd=tmp_path,
        env=dict(os.environ, FLEXLB_FT_WORKER_PORT_CAPACITY='2048'),
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert len(report['instances']) == 18
    for instance in report['instances']:
        metrics = instance['metrics']
        assert set().union(*map(set, metrics['consumers'].values())) == set(metrics['definitions'])
        assert instance['reports']
        assert instance['numeric_parameters']
        assert any(stage['check_ids'] for stage in instance['stages'])
