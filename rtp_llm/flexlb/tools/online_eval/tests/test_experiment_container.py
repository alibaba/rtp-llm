import copy
import gzip
import json
import shutil
import zipfile
from unittest.mock import patch
from pathlib import Path


import pytest

from artifacts.experiment import ArtifactRole, ExperimentReader, MetricDescriptor, publish_run
from artifacts.archive import create_archive
from monitoring.metric_store import MetricStore
from reporting import write_bundle, render
from scenario.runtime import execute_instance
from test_scenario_runtime import source, Backend, Clock
from scenario import compile_scenarios


def definition():
    return dict(experiment_definition_schema_version=1, plan={'id': 'case::default::profile'},
                dependencies={'renderer.py': 'a' * 64}, views={})


def execution():
    return dict(id='case::default::profile', test_kind='functional', status='PASS',
                outcome=dict(gate='PASS', validity='VALID', delivery='PASS'), stages=[], cleanup=[])


def frozen_spec():
    points = [dict(x=-1.25, y=None), dict(x=0, y=1 / 3), dict(x=1e6, y=0)]
    return dict(title='Frozen case', timeAxis=dict(min=-2, max=1e6), panels=[dict(
        id='metric', axes={'y': {'title': 'count'}}, timeX=True, series=[dict(
            name='A', points=points, statistics_points=points), dict(name='B', points=points)])])


def test_portable_single_file_render_and_lossless_separate_raw(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    requests = b'{"rid":1}\n' * 300000
    (root / 'client_events.jsonl').write_bytes(requests)
    logs = b'original log\n' * 300000
    (root / 'engine.log').write_bytes(logs)
    compressed = gzip.compress(logs)
    (root / 'engine.log.gz').write_bytes(compressed)
    (root / 'api_token.txt').write_text('not archived')
    spec = frozen_spec()
    bundle = write_bundle(root, 'run', 'case', {}, spec)
    original_spec = json.loads((bundle / 'report-spec.json').read_text())
    original_html = (bundle / 'report.html').read_bytes()
    result = execution()
    result['workload'] = dict(reports={'case.yaml': str(bundle / 'report.html')})
    published = publish_run(root, definition(), result)
    primary = tmp_path / 'moved.eval.zip'
    raw = tmp_path / 'moved.raw.zip'
    shutil.copyfile(published['experiment'], primary)
    shutil.copyfile(published['raw'], raw)
    shutil.rmtree(root)
    with ExperimentReader(primary) as reader:
        reader.verify()
        assert reader.view_spec('case.yaml') == original_spec
        reader.render('case.yaml', tmp_path / 'rerendered.html')
        assert (tmp_path / 'rerendered.html').read_bytes() == original_html
        with reader.open('run/client_events.jsonl') as stream:
            assert stream.read() == requests
        assert reader.entries['run/client_events.jsonl'].role is ArtifactRole.REQUESTS
        datasets = [entry for name, entry in reader.entries.items() if '/datasets/' in name]
        assert len(datasets) == 3 and len({entry.blob for entry in datasets}) == 1
        assert reader.read_json('execution.json') == result
        assert reader.read_json('raw-attachment.json')['files']['run/engine.log']['size'] == len(logs)
        assert reader.omitted[0]['reason'] == 'secret-like filename'
    with ExperimentReader(raw) as reader:
        reader.verify()
        with reader.open('run/engine.log') as stream:
            assert stream.read() == logs
        with reader.open('run/engine.log.gz') as stream:
            assert stream.read() == compressed
        entry = reader.entries['run/engine.log.gz']
        assert reader.archive.getinfo(entry.blob).compress_type == zipfile.ZIP_STORED


def test_legacy_archive_preserves_large_requests_and_precompressed_evidence(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    for name in ('requests.jsonl', 'series-large.json.gz'):
        (root / name).write_bytes(b'a' * (2 * 1024 * 1024 + 1))
    create_archive(tmp_path / 'legacy.zip', {'run': root}, kind='scenario')
    with zipfile.ZipFile(tmp_path / 'legacy.zip') as archive:
        for path in root.iterdir():
            assert archive.read('run/' + path.name) == path.read_bytes()
        assert archive.getinfo('run/series-large.json.gz').compress_type == zipfile.ZIP_STORED


def rewrite_zip(path, change):
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    change(files)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in files.items():
            archive.writestr(name, value)


def test_checksum_invalid_inventory_and_paths_rejected(tmp_path):
    published = publish_run(tmp_path, definition(), execution())
    path = published['experiment']
    def corrupt(files):
        blob = next(name for name in files if name.startswith('blobs/'))
        files[blob] += b'bad'
    rewrite_zip(path, corrupt)
    with ExperimentReader(path) as reader, pytest.raises(ValueError, match='checksum'):
        reader.verify()
    publish_run(tmp_path, definition(), execution())
    def escape(files):
        manifest = json.loads(files['manifest.json'])
        manifest['files']['execution.json']['path'] = '../escape'
        files['manifest.json'] = json.dumps(manifest).encode()
    rewrite_zip(path, escape)
    with pytest.raises(ValueError, match='entry'):
        ExperimentReader(path)


def test_metric_classification_keeps_processing_definitions():
    for definition_value, expected in [({'mode': 'scrape'}, 'raw_observation'),
                                      ({'promql': 'rate(counter[1m])'}, 'query_result'),
                                      ({'producer': 'case', 'calculation': {'op': 'quantile'}}, 'derived')]:
        descriptor = MetricDescriptor.from_definition('id', definition_value)
        assert descriptor.kind == expected and descriptor.definition == definition_value


def test_analysis_restored_from_canonical_metrics_without_duplicate_series(tmp_path):
    key = '1/mock/count/{}'
    document = dict(metrics_schema_version=1, definitions={'mock/count': {'unit': 'count', 'mode': 'scrape'}},
        metrics={'mock/count': [dict(series_key=key, metric_id='mock/count', epoch='1', source='mock', status='PRESENT',
        labels={}, points=[[100, None], [101, 1 / 3]], provenance={'source_type': 'prometheus'})]})
    store = MetricStore(document)
    store.save(tmp_path)
    series, sources, _, _ = store.series(100)
    analysis = dict(id='case', clock_anchor={'epoch_s': 100}, series=series, statistic_sources=sources,
                    status='PASS', workload={})
    original = copy.deepcopy(analysis)
    published = publish_run(tmp_path, definition(), execution(), analysis=analysis)
    with ExperimentReader(published['experiment']) as reader:
        reader.verify()
        assert reader.analysis() == original == analysis
        assert 'series' not in reader.read_json('analysis.json')['metadata']
        assert reader.read_json('datasets/metric-catalog.json')[0]['kind'] == 'raw_observation'


def test_automatic_archive_freezes_plan_and_preserves_failed_execution(tmp_path):
    clock = Clock()
    instance = compile_scenarios([('fixture', source())])[0]
    result = execute_instance(instance, Backend(clock), artifact_dir=tmp_path, clock=clock, sleeper=clock.sleep)
    assert result['finalization'][-1]['phase'] == 'archive'
    assert result['finalization'][-1]['status'] == 'PASS'
    with ExperimentReader(result['artifacts']['experiment']) as reader:
        reader.verify()
        assert reader.read_json('experiment.json')['plan'] == instance
        saved = reader.read_json('execution.json')
        assert saved['stages'] == result['stages']
        assert saved['outcome']['gate'] == result['outcome']['gate']
        assert not any(row['phase'] == 'archive' for row in saved['finalization'])


def test_archive_failure_has_separate_delivery_and_preserves_gate(tmp_path):
    clock = Clock()
    instance = compile_scenarios([('fixture', source())])[0]
    with patch('artifacts.experiment.publish_run', side_effect=OSError('disk full')):
        result = execute_instance(instance, Backend(clock), artifact_dir=tmp_path, clock=clock, sleeper=clock.sleep)
    assert result['outcome']['gate'] == 'PASS'
    assert result['outcome']['validity'] == 'VALID'
    assert result['outcome']['delivery'] == 'ERROR'
    assert result['finalization'][-1]['phase'] == 'archive'
    assert result['finalization'][-1]['status'] == 'ERROR'
    assert json.loads((tmp_path / 'result.json').read_text()) == result
    assert not (tmp_path / 'experiment.eval.zip').exists()


def test_failed_publication_preserves_existing_primary_and_removes_temporary(tmp_path):
    original = publish_run(tmp_path, definition(), execution())
    previous = Path(original['experiment']).read_bytes()
    with patch('artifacts.experiment.ExperimentWriter.add_view', side_effect=ValueError('bad view')):
        write_bundle(tmp_path, 'run', 'case', {}, frozen_spec())
        with pytest.raises(ValueError, match='bad view'):
            publish_run(tmp_path, definition(), execution())
    assert Path(original['experiment']).read_bytes() == previous
    assert not list(tmp_path.glob('.experiment-*'))


def test_unrendered_default_view_uses_frozen_declaration_and_metrics(tmp_path):
    from reporting.view_config import view
    from reporting.workload import build_spec
    from test_workload_report_views import payload
    from reporting import run_meta

    key = '1/mock/count/{}'
    store = MetricStore(dict(metrics_schema_version=1,
        definitions={'mock/count': {'unit': 'count', 'mode': 'scrape'}},
        metrics={'mock/count': [dict(series_key=key, metric_id='mock/count', epoch='1', source='mock', status='PRESENT',
            labels={}, points=[[10, 1], [11, None], [12, 2]], provenance={'source_type': 'prometheus'})]}))
    store.save(tmp_path)
    series, sources, _, _ = store.series(10)
    analysis = payload(series)
    analysis['statistic_sources'] = sources
    frozen = definition()
    frozen['views'] = {'default.yaml': view('default.yaml').to_dict()}
    expected = render(build_spec(analysis, tmp_path))
    published = publish_run(tmp_path, frozen, execution(), analysis=analysis)
    with ExperimentReader(published['experiment']) as reader, \
            patch('reporting.workload.view', side_effect=AssertionError('current YAML must not be loaded')):
        assert reader.available_views == ['default.yaml']
        reader.render('default.yaml', tmp_path / 'default.html')
        assert (tmp_path / 'default.html').read_text() == expected


def test_raw_attachment_checked_against_primary(tmp_path):
    (tmp_path / 'engine.log').write_text('original')
    published = publish_run(tmp_path, definition(), execution())
    with ExperimentReader(published['experiment']) as reader:
        reader.verify_raw(published['raw'])
        with Path(published['raw']).open('ab') as stream:
            stream.write(b'bad')
        with pytest.raises(ValueError, match='raw attachment'):
            reader.verify_raw(published['raw'])


def test_gate_manifest_checks_exact_frozen_inputs(tmp_path):
    import hashlib
    evidence = tmp_path / 'case-gate-evidence.json'
    result = tmp_path / 'case-gate-result.json'
    evidence.write_text('{"used":1}')
    result.write_text('{"verdict":"PASS"}')
    manifest = dict(gate_result_schema_version=1,
        evidence=dict(file=evidence.name, sha256=hashlib.sha256(evidence.read_bytes()).hexdigest()),
        result=dict(file=result.name, sha256=hashlib.sha256(result.read_bytes()).hexdigest()))
    (tmp_path / 'case-gate-manifest.json').write_text(json.dumps(manifest))
    published = publish_run(tmp_path, definition(), execution())
    with ExperimentReader(published['experiment']) as reader:
        reader.verify()
    evidence.write_text('{"used":2}')
    with pytest.raises(ValueError, match='gate evidence'):
        publish_run(tmp_path, definition(), execution())
    with ExperimentReader(published['experiment']) as reader:
        reader.verify()


def test_archive_deadline_preserves_gate_and_has_its_own_budget(tmp_path):
    clock = Clock()
    instance = compile_scenarios([('fixture', source())])[0]
    instance['execution']['archive_timeout_s'] = .5
    def delayed(directory, definition, execution, *, deadline, analysis=None):
        deadline.sleep(10)
    with patch('artifacts.experiment.publish_run', side_effect=delayed):
        result = execute_instance(instance, Backend(clock), artifact_dir=tmp_path,
                                  clock=clock, sleeper=clock.sleep)
    assert result['outcome']['gate'] == 'PASS'
    assert result['outcome']['validity'] == 'VALID'
    assert result['outcome']['delivery'] == 'TIMEOUT'
    assert result['finalization'][-1]['duration_ms'] == 500
    assert result['finalization'][-1]['status'] == 'TIMEOUT'


def test_preferred_view_selects_case_before_default(tmp_path):
    first = write_bundle(tmp_path, 'run', 'default', {}, frozen_spec())
    second = write_bundle(tmp_path, 'run', 'selected', {}, frozen_spec())
    frozen = definition()
    frozen['plan']['reports'] = ['default.yaml', 'case.yaml']
    result = execution()
    result['workload'] = {'reports': {'default.yaml': str(first / 'report.html'),
                                    'case.yaml': str(second / 'report.html')}}
    published = publish_run(tmp_path, frozen, result)
    with ExperimentReader(published['experiment']) as reader:
        assert reader.preferred_view == 'case.yaml'
        reader.render(reader.preferred_view, tmp_path / 'selected.html')
        assert (tmp_path / 'selected.html').read_bytes() == (second / 'report.html').read_bytes()


def test_failed_republish_preserves_previous_raw_attachment(tmp_path):
    from artifacts.experiment import ExperimentWriter
    (tmp_path / 'engine.log').write_text('previous log')
    previous = publish_run(tmp_path, definition(), execution())
    primary = Path(previous['experiment']).read_bytes()
    (tmp_path / 'engine.log').write_text('new log')
    original = ExperimentWriter.add_json
    def fail_commit(self, name, *args, **kwargs):
        if name == 'raw-attachment.json':
            raise OSError('primary commit failed')
        return original(self, name, *args, **kwargs)
    with patch.object(ExperimentWriter, 'add_json', fail_commit), pytest.raises(OSError):
        publish_run(tmp_path, definition(), execution())
    assert Path(previous['experiment']).read_bytes() == primary
    with ExperimentReader(previous['experiment']) as reader:
        reader.verify_raw(previous['raw'])
    with ExperimentReader(previous['raw']) as reader, reader.open('run/engine.log') as stream:
        assert stream.read() == b'previous log'
