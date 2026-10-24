"""判定统一，结果引用可校验，归档搬迁后报告明细不丢失。"""
import copy
import shutil
from dataclasses import asdict

import pytest

from analysis.checks import CheckResult
from analysis.gates import gate_result
from artifacts.experiment import ExperimentReader, publish_run
from artifacts.gate import freeze_gate, gate_check
from monitoring.metric_store import MetricStore
from reporting.run_model import RunPresentation
from reporting.view_config import view
from reporting.workload import build_spec
from reporting import render
from test_experiment_container import definition


@pytest.mark.parametrize('status,verdict,validity', [
    ('PASS', 'PASS', 'VALID'), ('FAIL', 'FAIL', 'VALID'),
    ('ERROR', 'INVALID', 'INVALID'), ('SKIP', 'INVALID', 'INVALID'),
    ('WARNING', 'PASS', 'VALID'),
])
def test_common_gate_summary_preserves_status_semantics(status, verdict, validity):
    summary = gate_result([CheckResult('business', status)])
    assert summary['verdict'] == verdict and summary['validity'] == validity


def test_requirement_failure_dominates_business_failure_without_mutating_checks():
    checks = [CheckResult('business', 'FAIL', actual=1, expected=2),
              CheckResult('coverage', 'FAIL', actual=1, expected=3)]
    summary = gate_result(checks, requirements={'coverage'})
    assert summary['verdict'] == summary['validity'] == 'INVALID'
    assert [row['status'] for row in summary['checks']] == ['FAIL', 'ERROR']
    assert checks[1].status == 'FAIL'
    with pytest.raises(ValueError, match='duplicate'):
        gate_result([checks[0], checks[0]])


def frozen_check(directory):
    decision = dict(gate_result([CheckResult('threshold', 'PASS', actual=3, expected=2)]), errors=[])
    path = freeze_gate(directory, 'example', {}, decision)
    check = gate_check('summary', decision, path, 'contract')
    return decision, check


def test_reference_preserves_report_rows_and_legacy_inline_checks(tmp_path):
    decision, check = frozen_check(tmp_path)
    assert set(check.evidence) == {'result', 'sha256'}
    analysis = dict(id='case', checks=[dict(stage='gate', **asdict(check))])
    run = RunPresentation.read(analysis, directory=tmp_path)
    old = copy.deepcopy(analysis)
    old['checks'][0]['evidence']['checks'] = decision['checks']
    legacy = RunPresentation.read(old)
    assert list(run.checks[0].rows()) == list(legacy.checks[0].rows())
    assert len(list(run.checks[0].rows())) == 2
    with pytest.raises(ValueError, match='experiment source'):
        RunPresentation.read(analysis)
    (tmp_path / 'example-gate-result.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        RunPresentation.read(analysis, directory=tmp_path)


def test_inconsistent_validity_is_rejected_before_any_gate_file_is_written(tmp_path):
    with pytest.raises(ValueError, match='validity'):
        freeze_gate(tmp_path, 'bad', {}, dict(gate_result([CheckResult('x', 'PASS')]), validity='INVALID'))
    assert not list(tmp_path.iterdir())


def test_compact_archive_reconstructs_exact_analysis_and_renders_offline(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    _, check = frozen_check(root)
    stage = dict(id='gate', checks=[asdict(check)])
    execution = dict(id='case', status='PASS', test_kind='workload', stages=[stage],
        cleanup=[], implementation=dict(configuration={'traffic': {'count': 1}}, configuration_sha256='a'*64),
        workload=dict(runtime_validity='VALID'))
    analysis = dict(id=execution['id'], status=execution['status'], stages=execution['stages'],
        implementation=execution['implementation'], workload=execution['workload'],
        clock_anchor={'epoch_s': 100}, series={}, statistic_sources={}, phases=[], events=[])
    MetricStore(dict(metrics_schema_version=1, definitions={}, metrics={})).save(root)
    frozen = definition()
    frozen['views'] = {'default.yaml': view('default.yaml').to_dict()}
    run = RunPresentation.read(analysis, directory=root)
    expected = render(build_spec(analysis, root, run=run))
    archived = publish_run(root, frozen, execution, analysis=analysis)
    destination = tmp_path / 'moved.eval.zip'
    shutil.copyfile(archived['experiment'], destination)
    shutil.rmtree(root)
    with ExperimentReader(destination) as reader:
        reader.verify()
        assert 'run/result.json' not in reader.entries
        assert reader.read_json('run/result.json') == reader.read_json('execution.json') == execution
        saved = reader.read_json('analysis.json')
        assert not {'stages', 'implementation', 'id', 'status', 'workload'} & saved['metadata'].keys()
        assert set(saved['execution_fields']) == {'stages', 'implementation', 'id', 'status', 'workload'}
        assert reader.analysis() == analysis
        reader.render('default.yaml', tmp_path / 'report.html')
    assert (tmp_path / 'report.html').read_text() == expected


def test_archive_keeps_intentionally_distinct_presentation_metadata(tmp_path):
    MetricStore(dict(metrics_schema_version=1, definitions={}, metrics={})).save(tmp_path)
    execution = dict(id='execution', status='ERROR', workload={}, stages=[], test_kind='functional')
    analysis = dict(id='presentation', status='FAIL', workload={'report_status': 'PASS'},
                    clock_anchor={'epoch_s': 1}, series={}, statistic_sources={})
    published = publish_run(tmp_path, definition(), execution, analysis=analysis)
    with ExperimentReader(published['experiment']) as reader:
        assert reader.analysis() == analysis
        assert reader.read_json('analysis.json')['metadata']['id'] == 'presentation'


def test_version_one_analysis_remains_readable(tmp_path):
    from artifacts.experiment import ArtifactRole, ExperimentWriter
    path = tmp_path / 'legacy.eval.zip'
    metadata = dict(id='old', status='FAIL', clock_anchor={'epoch_s': 1})
    with ExperimentWriter(path).publication() as writer:
        writer.add_json('experiment.json', definition(), ArtifactRole.DEFINITION)
        writer.add_json('execution.json', {}, ArtifactRole.EXECUTION)
        writer.add_json('completeness.json', {}, ArtifactRole.EXECUTION)
        writer.add_json('run/metrics.json', dict(metrics_schema_version=1, definitions={}, metrics={}),
                        ArtifactRole.OBSERVATIONS)
        writer.add_json('analysis.json', dict(experiment_analysis_schema_version=1,
            metadata=metadata, metrics='run/metrics.json', anchor_epoch_s=1), ArtifactRole.ANALYSIS)
    with ExperimentReader(path) as reader:
        reader.verify()
        assert reader.analysis() == dict(metadata, series={}, statistic_sources={})


def test_default_live_report_expands_references_without_saving_nested_checks(tmp_path):
    from reporting.workload import write_report
    from reporting import load_analysis
    _, check = frozen_check(tmp_path)
    payload = dict(id='live', status='PASS', workload={}, checks=[dict(stage='gate', **asdict(check))],
                   series={}, statistic_sources={}, clock_anchor={'epoch_s': 1})
    bundle = write_report(tmp_path, payload)
    saved = load_analysis(bundle)
    assert 'checks' not in saved['checks'][0]['evidence']
    import json
    spec = json.loads((bundle / 'report-spec.json').read_text())
    section = next(section for section in spec['sections'] if section['id'] == 'run.checks')
    assert 'threshold' in str(section)


@pytest.mark.parametrize('value', [True, 1.0])
def test_metadata_deduplication_preserves_nested_json_scalar_types(tmp_path, value):
    MetricStore(dict(metrics_schema_version=1, definitions={}, metrics={})).save(tmp_path)
    execution = dict(id='same', workload={'count': 1}, test_kind='functional')
    analysis = dict(id='same', workload={'count': value}, clock_anchor={'epoch_s': 1},
                    series={}, statistic_sources={})
    published = publish_run(tmp_path, definition(), execution, analysis=analysis)
    with ExperimentReader(published['experiment']) as reader:
        assert type(reader.analysis()['workload']['count']) is type(value)
        assert 'workload' in reader.read_json('analysis.json')['metadata']
