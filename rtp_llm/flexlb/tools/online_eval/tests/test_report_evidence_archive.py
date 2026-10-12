import copy
import gzip
import json
import shutil
from unittest.mock import patch

import pytest

from monitoring.metric_store import MetricStore, frozen_metric_reads
from reporting import load_analysis, read_bundle, write_bundle
from artifacts.archive import file_digest
from reporting.evidence_archive import EvidenceReference, archive_session, restore


def evidence():
    return dict(status='PASS', series={'引擎/A': [[-1.25, None], [0, 1 / 3], [1e6, 0.0]]},
                statistic_sources={'引擎/A': dict(unit='count', labels={'role': 'prefill'})})


def publish(root, identity, analysis):
    return write_bundle(root, 'run', identity, analysis, dict(title='冻结证据', panels=[]))


def test_lossless_nested_roundtrip_and_portable_bundle(tmp_path):
    source = dict(evidence(), run=evidence())
    original = copy.deepcopy(source)
    with archive_session():
        first = publish(tmp_path / 'original', 'first', source)
        second = publish(tmp_path / 'original', 'second', source)
    assert source == original
    assert load_analysis(first) == original
    archived = list(first.glob('series-*.json.gz'))
    assert len(archived) == 1
    assert archived[0].stat().st_ino == (second / archived[0].name).stat().st_ino
    wire = json.loads((first / 'analysis.json').read_text())['result']
    assert 'series' not in wire and 'series' not in wire['run']
    shutil.copytree(first, tmp_path / 'moved')
    shutil.rmtree(tmp_path / 'original')
    assert read_bundle(tmp_path / 'moved') == tmp_path / 'moved'
    assert load_analysis(tmp_path / 'moved') == original
    assert '冻结证据' in (tmp_path / 'moved' / 'report.html').read_text()


def test_same_evidence_compressed_once_per_publication_and_copy_fallback(tmp_path):
    source = evidence()
    with archive_session(), patch('reporting.evidence_archive.gzip.GzipFile', wraps=gzip.GzipFile) as encoder:
        first = publish(tmp_path, 'one', source)
        with patch('reporting.evidence_archive.os.link', side_effect=OSError('no hard links')):
            second = publish(tmp_path, 'two', dict(source, status='FAIL'))
        assert encoder.call_count == 1
    assert load_analysis(first) == source
    assert load_analysis(second) == dict(source, status='FAIL')
    with archive_session(), patch('reporting.evidence_archive.gzip.GzipFile', wraps=gzip.GzipFile) as encoder:
        publish(tmp_path, 'three', source)
        assert encoder.call_count == 1  # The previous session retains no object cache.
    assert len(list((tmp_path / 'reports/evidence').glob('*.json.gz'))) == 1


def test_corrupt_or_missing_archive_cannot_be_loaded(tmp_path):
    bundle = publish(tmp_path, 'example', evidence())
    archive = next(bundle.glob('series-*.json.gz'))
    archive.write_bytes(archive.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_analysis(bundle)
    archive.unlink()
    with pytest.raises(FileNotFoundError):
        load_analysis(bundle)


@pytest.mark.parametrize('change', [
    {'path': '../escape.json.gz'}, {'sha256': 'bad'}, {'fields': []},
    {'fields': ['series', 'series']}, {'fields': [{}]}, {'fields': ['other']},
    {'path': 'analysis.json'}, {'extra': 1},
])
def test_invalid_references_rejected_at_boundary(change):
    reference = dict(path='series-' + 'a' * 64 + '.json.gz', sha256='a' * 64, fields=['series'])
    reference.update(change)
    with pytest.raises(ValueError, match='reference'):
        EvidenceReference.parse(reference, {})


def test_null_reference_inline_collision_and_unmanifested_reference(tmp_path):
    with pytest.raises(ValueError, match='reference'):
        restore(tmp_path, {'series_evidence': None})
    reference = dict(path='series-' + 'a' * 64 + '.json.gz', sha256='a' * 64, fields=['series'])
    with pytest.raises(ValueError, match='reference'):
        EvidenceReference.parse(reference, {'series': {}})
    bundle = publish(tmp_path, 'example', evidence())
    manifest = json.loads((bundle / 'manifest.json').read_text())
    del manifest['files'][next(bundle.glob('series-*.json.gz')).name]
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest'):
        read_bundle(bundle)


def test_archive_document_requires_inventory_contract(tmp_path):
    content = gzip.compress(json.dumps(dict(series_evidence_schema_version=1, values={'series': []})).encode())
    digest_path = tmp_path / 'temporary'
    digest_path.write_bytes(content)
    digest = file_digest(digest_path)
    name = 'series-' + digest + '.json.gz'
    digest_path.rename(tmp_path / name)
    with pytest.raises(ValueError, match='document'):
        restore(tmp_path, dict(series_evidence=dict(path=name, sha256=digest, fields=['series'])))


def test_nonfinite_evidence_does_not_commit_bundle_or_leave_temporary(tmp_path):
    with pytest.raises(ValueError):
        publish(tmp_path, 'invalid', {'series': {'a': [[1, float('nan')]]}})
    assert not list(tmp_path.rglob('manifest.json'))
    assert not list(tmp_path.rglob('*.tmp'))


def metric_document(value):
    return dict(metrics_schema_version=1, definitions={}, metrics={}, run={'value': value})


def test_metric_reads_scoped_to_publication_and_invalidated_by_atomic_save(tmp_path):
    MetricStore(metric_document(1)).save(tmp_path)
    with patch('monitoring.metric_store.json.load', wraps=json.load) as decoder:
        with frozen_metric_reads():
            first = MetricStore.read(tmp_path)
            assert MetricStore.read(tmp_path) is first
            assert decoder.call_count == 1
            MetricStore(metric_document(2)).save(tmp_path)
            second = MetricStore.read(tmp_path)
            assert second is not first and second.document['run']['value'] == 2
            assert decoder.call_count == 2
            with frozen_metric_reads():
                assert MetricStore.read(tmp_path) is not second
            assert MetricStore.read(tmp_path) is second
        assert MetricStore.read(tmp_path) is not second
        assert MetricStore.read(tmp_path) is not second
        assert decoder.call_count == 5


def test_metric_scope_resets_on_exception_and_separates_directories(tmp_path):
    for name, value in [('a', 1), ('b', 2)]:
        MetricStore(metric_document(value)).save(tmp_path / name)
    with pytest.raises(RuntimeError):
        with frozen_metric_reads():
            first = MetricStore.read(tmp_path / 'a')
            assert MetricStore.read(tmp_path / 'b').document['run']['value'] == 2
            raise RuntimeError('renderer failed')
    assert MetricStore.read(tmp_path / 'a') is not first


def test_comparison_run_collection_archives_and_restores_each_run(tmp_path):
    source = dict(purpose='OBSERVATION_ONLY', runs={'A': evidence(), 'B': dict(run=evidence())})
    original = copy.deepcopy(source)
    bundle = write_bundle(tmp_path, 'comparison', 'runs', source, dict(panels=[]))
    assert load_analysis(bundle) == source == original
    wire = json.loads((bundle / 'analysis.json').read_text())['result']
    assert 'series_evidence' in wire['runs']['A']
    assert 'series_evidence' in wire['runs']['B']['run']


def test_write_views_shares_metric_reads_and_archives_across_renderers(tmp_path):
    from types import SimpleNamespace
    from reporting.workload import write_views

    source = dict(evidence(), status='PASS')
    MetricStore(metric_document(1)).save(tmp_path)

    def renderer(identity):
        def render(directory, analysis, presentation):
            assert MetricStore.read(directory).document['run']['value'] == 1
            assert MetricStore.read(directory).document['run']['value'] == 1
            return publish(directory, identity, dict(run=analysis)) / 'report.html'
        return render

    capabilities = {name: SimpleNamespace(renderer=renderer(name)) for name in ('a.yaml', 'b.yaml')}
    with patch('reporting.workload.view', return_value={}), \
            patch('reporting.workload.view_capabilities', return_value=capabilities), \
            patch('monitoring.metric_store.json.load', wraps=json.load) as decoder, \
            patch('reporting.evidence_archive.gzip.GzipFile', wraps=gzip.GzipFile) as encoder:
        paths = write_views(tmp_path, source, names=['a.yaml', 'b.yaml'])
        assert list(paths) == ['a.yaml', 'b.yaml']
        assert decoder.call_count == encoder.call_count == 1
    assert load_analysis(paths['b.yaml'].parent)['run'] == source
