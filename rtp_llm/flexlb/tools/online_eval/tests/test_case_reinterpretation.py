"""Offline adjudication cannot mutate or depend on the historical archive."""

import hashlib
import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

from cases.master_performance.analysis import analyze
from cases.master_performance.publication import publish_performance
from cases.master_performance.replay import performance_main
from cases.master_performance.report import render_view
from reporting.view_config import view
from test_performance_gate import evidence


def run_cli(source, destination, *options):
    with mock.patch.object(sys, 'argv', ['performance', str(source), '--output', str(destination), *options]):
        return performance_main()


def archive_bytes(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*') if p.is_file()}


def test_html_reinterpretation_is_self_contained_and_source_is_read_only(tmp_path):
    source = tmp_path / 'archive'
    e = evidence()
    from metric_fixtures import freeze_metrics
    freeze_metrics(source, "master_performance")
    publish_performance(source, e, analyze(e))
    from monitoring.metric_store import MetricStore
    store = MetricStore.read(source)
    identity = 'mock/rtp_llm_context_tps'
    raw = dict(metric_id=identity, epoch='1', source='mock', labels=dict(role='prefill'),
               series_key='1/mock/rtp_llm_context_tps/' + json.dumps(dict(role='prefill'), sort_keys=True),
               points=[[100, 1024]], status='PRESENT',
               provenance=dict(source_type='prometheus', promql='rtp_llm_context_tps'))
    store.document['metrics'][identity] = [raw]
    store.document['collection_gaps'] = {'1/mock/collection': [101]}
    store.document['errors'] = [dict(source='1/mock', error='historical gap')]
    store.save(source)
    for row in e['flow']['records']:
        row['ttft_ms'] = 75
    input_path = source / 'replay-input.json'
    input_path.write_text(json.dumps(e))
    before = archive_bytes(source)
    destination = tmp_path / 'new'
    assert run_cli(input_path, destination, '--reinterpret') == 0
    assert archive_bytes(source) == before
    metrics = json.loads((destination / 'metrics.json').read_text())
    assert metrics['metrics']['performance_gate/ttft_p99_ms'][0]['points'][0][1] == 75
    assert metrics['metrics'][identity] == [raw]
    assert metrics['collection_gaps'] == {'1/mock/collection': [101]}
    assert metrics['errors'] == [dict(source='1/mock', error='historical gap')]
    saved = json.loads((destination / 'performance-gate-evidence.json').read_text())
    assert saved['reinterpretation']['source_sha256'] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    from cases.master_performance import analysis
    assert saved['reinterpretation']['analyzer_sha256'] == hashlib.sha256(Path(analysis.__file__).read_bytes()).hexdigest()
    provenance = metrics['metrics']['performance_gate/ttft_p99_ms'][0]['provenance']
    assert provenance['evidence']['path'] == str(destination / 'performance-gate-evidence.json')
    shutil.rmtree(source)
    render_view(destination, None, view("master_performance.yaml"))
    assert (destination / 'reports/run/master-performance/report.html').is_file()


@pytest.mark.parametrize('json_only', [False, True])
def test_reinterpretation_guards_apply_to_both_outputs(tmp_path, json_only):
    source = tmp_path / 'archive'
    source.mkdir()
    input_path = source / 'evidence.json'
    input_path.write_text(json.dumps(evidence()))
    options = ['--json-only'] if json_only else []
    output = tmp_path / 'new'
    with pytest.raises(SystemExit):
        run_cli(input_path, output, *options)
    assert not output.exists()
    for destination in [source, source / 'nested']:
        with pytest.raises(SystemExit):
            run_cli(input_path, destination, '--reinterpret', *options)
    alias = tmp_path / 'alias'
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(SystemExit):
        run_cli(input_path, alias, '--reinterpret', *options)
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('frozen')
    with pytest.raises(SystemExit):
        run_cli(input_path, output, '--reinterpret', *options)
    assert sentinel.read_text() == 'frozen'
    file_output = tmp_path / 'file'
    file_output.write_text('keep')
    with pytest.raises(SystemExit):
        run_cli(input_path, file_output, '--reinterpret', *options)
    assert file_output.read_text() == 'keep'


def test_json_only_freezes_reinterpretation_metadata(tmp_path):
    source = tmp_path / 'archive'
    source.mkdir()
    input_path = source / 'evidence.json'
    input_path.write_text(json.dumps(evidence()))
    frozen = archive_bytes(source)
    destination = tmp_path / 'new'
    assert run_cli(input_path, destination, '--reinterpret', '--json-only') == 0
    assert archive_bytes(source) == frozen
    saved = json.loads((destination / 'performance-gate-evidence.json').read_text())
    assert saved['reinterpretation']['source'] == str(input_path)
    assert json.loads((destination / 'analysis.json').read_text())['verdict'] == 'PASS'
    with pytest.raises(SystemExit):
        run_cli(input_path, destination, '--reinterpret', '--json-only')


def test_raw_query_archives_are_copied_into_new_output(tmp_path):
    source = tmp_path / 'archive'
    telemetry = source / 'telemetry' / '2'
    telemetry.mkdir(parents=True)
    payload = dict(start=100, end=110, step=1, targets=[], queries={}, errors=[], query_plan='master_performance.yaml')
    (telemetry / 'queries.json').write_text(json.dumps(payload))
    input_path = source / 'evidence.json'
    input_path.write_text(json.dumps(evidence()))
    input_alias = tmp_path / 'evidence-link.json'
    input_alias.symlink_to(input_path)
    before = archive_bytes(source)
    destination = tmp_path / 'new'
    assert run_cli(input_alias, destination, '--reinterpret') == 0
    assert archive_bytes(source) == before
    assert (destination / 'telemetry/2/queries.json').read_bytes() == (telemetry / 'queries.json').read_bytes()
    shutil.rmtree(source)
    render_view(destination, None, view("master_performance.yaml"))
