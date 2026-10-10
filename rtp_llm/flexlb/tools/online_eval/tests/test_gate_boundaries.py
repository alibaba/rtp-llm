"""Frozen gate analysis and presentation have separate side-effect boundaries."""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from cases.master_ha_failover.analysis import HA_METRICS, measure_client_metric
from cases.master_performance.analysis import analyze as analyze_performance
from cases.cache_scale_in.analysis import analyze as analyze_cache
from cases.master_performance.report import write_report as performance_report
from cases.cache_scale_in.report import write_report as cache_report
from cases.master_performance.publication import publish_performance
from cases.cache_scale_in.publication import publish_cache
from test_performance_gate import evidence as performance_evidence
import test_cache_scale_gate as cache_fixtures
from reporting import load_analysis

ROOT = Path(__file__).resolve().parents[1]


def test_analyzers_run_without_io_or_execution_and_presentation_imports():
    documents = [performance_evidence(), cache_fixtures.CacheGateTest().evidence(.8)]
    script = '''
import builtins, json, socket, sys
from pathlib import Path
from unittest.mock import patch
from cases.master_performance.analysis import analyze as performance
from cases.cache_scale_in.analysis import analyze as cache
from cases.master_ha_failover.analysis import measure_client_metric
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'reporting', 'runtime'}:
        raise AssertionError('analysis imported execution/presentation: ' + name)
    return original_import(name, *args, **kwargs)
performance_input, cache_input = json.loads(sys.stdin.read())
with patch('builtins.__import__', side_effect=guarded_import), \\
     patch('builtins.open', side_effect=AssertionError('file IO')), \\
     patch.object(Path, 'open', side_effect=AssertionError('path IO')), \\
     patch.object(socket, 'create_connection', side_effect=AssertionError('network IO')):
    assert performance(performance_input)['verdict'] == 'PASS'
    assert cache(cache_input)['verdict'] == 'PASS'
    assert measure_client_metric({'metric':'ha_gate/sample_count'}, [{}]) == 1
assert not any(name.split('.')[0] in {'reporting', 'runtime'} or name.startswith('cases.') and name.rsplit('.', 1)[-1] in {'program', 'actions', 'publication', 'report', 'metrics', 'panels', 'runtime'} for name in sys.modules)
'''
    process = subprocess.run([sys.executable, '-c', script], input=json.dumps(documents),
                             text=True, cwd=ROOT, capture_output=True)
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize('kind', ['cache', 'performance'])
def test_report_uses_frozen_verdict_without_republishing_metrics(kind):
    evidence = (cache_fixtures.CacheGateTest().evidence(.8) if kind == 'cache'
                else performance_evidence())
    analyze = analyze_cache if kind == 'cache' else analyze_performance
    publish = publish_cache if kind == 'cache' else publish_performance
    render = cache_report if kind == 'cache' else performance_report
    result = analyze(evidence)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        publish(root, evidence, result)
        before = {name: (root/name).read_bytes() for name in
                  ('metrics.json', kind+'-gate-evidence.json')}
        changed = copy.deepcopy(evidence)
        if kind == 'performance':
            changed['criteria']['min_input_tps'] = 1e9
        else:
            changed.setdefault('errors', []).append('would invalidate a new analysis')
        with mock.patch(('cases.cache_scale_in' if kind == 'cache' else 'cases.master_performance')+'.analysis.analyze', side_effect=AssertionError('reanalysis')), \
             mock.patch(('cases.cache_scale_in' if kind == 'cache' else 'cases.master_performance')+'.metrics.produce', side_effect=AssertionError('republish')):
            render(root, changed, result)
        bundle = root/'reports/run'/('cache-scale-in' if kind == 'cache' else 'master-performance')
        assert load_analysis(bundle) == result
        spec = json.loads((bundle/'report-spec.json').read_text())
        identity = evidence['provenance']['instance']
        assert spec['run_id'] == identity
        assert spec['title'] == ' : '.join(identity.split('::'))
        meta = spec['run_meta']
        assert meta['identity']['id'] == identity
        assert before == {name: (root/name).read_bytes() for name in before}
        with pytest.raises(TypeError):
            render(root, evidence)


def test_ha_metrics_use_explicit_targets_and_topology():
    rows = [dict(rid='r'+str(i), status='ok' if i < 5 else 'exception',
                 route_path='master' if i < 5 else 'failed',
                 error_kind='' if i < 5 else 'business',
                 error='' if i < 5 else 'error 503', failover=i == 4,
                 master_target='B' if i < 5 else 'A',
                 prefill='p0' if i < 4 else 'p1', send_start_epoch_ms=(100+i)*1000)
            for i in range(6)]
    expected = dict(sample_count=6, success_rate=5/6, non_ok_count=1,
                    target_share=5/6, target_count=5, route_share=5/6, route_count=5,
                    failover_count=1, duplicate_ids=0, error_kind_count=1,
                    wrong_error_code=5, failed_count=1, failed_rate_above_one=0,
                    business_rate_above_one=0, visible_terminal_count=6,
                    visible_terminal_share=1, prefill_max_share=4/5, prefill_peak_skew=8/5)
    assert set(expected) == HA_METRICS
    for name, value in expected.items():
        params = dict(metric='ha_gate/'+name, route='master', error_kind='business',
                      code=503, min_samples=1)
        assert measure_client_metric(params, rows, target='B', prefill_pool=['p0','p1']) == value
    with pytest.raises(ValueError, match='unknown Prefill'):
        measure_client_metric(dict(metric='ha_gate/prefill_peak_skew', min_samples=1),
                              rows, prefill_pool=['p0'])
    with pytest.raises(ValueError, match='unknown HA metric'):
        measure_client_metric(dict(metric='ha_gate/misspelled'), rows)


def test_performance_cli_replays_evidence_explicitly():
    evidence = performance_evidence()
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory)/'input.json'
        source.write_text(json.dumps(evidence))
        original = source.read_bytes()
        output = Path(directory)/'output'
        process = subprocess.run([sys.executable, str(ROOT/'scripts/commands/analyze_performance.py'),
                                  str(source), '--output', str(output), '--json-only'],
                                 text=True, capture_output=True)
        assert process.returncode == 0, process.stderr
        assert json.loads(process.stdout) == analyze_performance(evidence)
        assert json.loads((output/'analysis.json').read_text()) == analyze_performance(evidence)
        assert source.read_bytes() == original


def test_report_finalizer_is_a_registered_program_capability():
    from types import SimpleNamespace
    from cases import registry

    finalizer = mock.Mock()
    with mock.patch.dict(registry.PROGRAMS, {'another_case': 'another_case.program'}), \
         mock.patch('importlib.import_module', return_value=SimpleNamespace(REPORT_FINALIZER=finalizer)):
        registry.finalize_reports('another_case', '/unused/run')
    finalizer.assert_called_once_with('/unused/run')
    with mock.patch('importlib.import_module', return_value=SimpleNamespace(REPORT_FINALIZER='not callable')):
        with pytest.raises(ValueError, match='must be callable'):
            registry.finalize_reports('master_performance', '/unused/run')


def test_registered_view_extension_needs_no_workload_case_branch():
    from cases.registry import VIEW_RENDERERS, VIEW_VALIDATORS
    from workload.report import write_views

    render = mock.Mock(return_value=Path('/unused/new/report.html'))
    validator = mock.Mock()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        from reporting.view_config import view
        import yaml
        definition = view('master_ha_failover.yaml')
        (root/'another.yaml').write_text(yaml.safe_dump(definition))
        with mock.patch.dict(VIEW_RENDERERS, {'another.yaml': 'test_extension.render'}), \
             mock.patch.dict(VIEW_VALIDATORS, {'another.yaml': 'test_extension.validate'}), \
             mock.patch('reporting.view_config.VIEWS', root), \
             mock.patch('reporting.view_config.load_capability', return_value=validator), \
             mock.patch('workload.report.load_capability', return_value=render):
            assert write_views(root, {'status': 'PASS'}, ['another.yaml']) == {
                'another.yaml': Path('/unused/new/report.html')}
        validator.assert_called_once()
        assert render.call_args.args[1] == {'status': 'PASS'}


def test_shared_runtime_and_analysis_do_not_import_case_implementations():
    import ast

    for package in ('runtime', 'analysis'):
        for path in (ROOT/'src'/package).glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                         else [alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [])
                assert not any(name.startswith('cases.') for name in names), path
