import time
from pathlib import Path

import pytest

from artifacts.experiment import ExperimentReader, publish_run
from runtime.deadline import Deadline
from artifacts.retention import prune_tsdb
from test_experiment_container import definition, execution


def archived(tmp_path):
    data = tmp_path / 'telemetry/1/data'
    data.mkdir(parents=True)
    (data / 'wal').write_bytes(b'raw WAL' * 1000)
    (data / 'meta.json').write_text('{"immutable":true}')
    artifacts = publish_run(tmp_path, definition(), execution())
    return data, artifacts


def test_verified_archive_releases_tsdb_and_retains_lossless_payload(tmp_path):
    data, artifacts = archived(tmp_path)
    prune_tsdb(tmp_path, artifacts, Deadline(time.monotonic() + 10))
    assert not data.exists()
    with ExperimentReader(artifacts['raw']) as reader:
        with reader.open('run/telemetry/1/data/wal') as stream:
            assert stream.read() == b'raw WAL' * 1000


def test_changed_source_is_not_deleted(tmp_path):
    data, artifacts = archived(tmp_path)
    (data / 'wal').write_bytes(b'changed')
    with pytest.raises(ValueError, match='source differs'):
        prune_tsdb(tmp_path, artifacts, Deadline(time.monotonic() + 10))
    assert (data / 'meta.json').exists()
    assert (data / 'wal').read_bytes() == b'changed'


def test_corrupt_attachment_preserves_original_tsdb(tmp_path):
    data, artifacts = archived(tmp_path)
    with Path(artifacts['raw']).open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        prune_tsdb(tmp_path, artifacts, Deadline(time.monotonic() + 10))
    assert data.exists()


def test_expired_budget_and_foreign_directory_cannot_delete_data(tmp_path):
    data, artifacts = archived(tmp_path)
    with pytest.raises(TimeoutError):
        prune_tsdb(tmp_path, artifacts, Deadline(time.monotonic()))
    assert data.exists()
    external = tmp_path.parent / (tmp_path.name + '-foreign')
    external.mkdir()
    foreign = tmp_path / 'telemetry/2'
    foreign.symlink_to(external, target_is_directory=True)
    (external / 'data').mkdir()
    with pytest.raises(ValueError, match='owned directories'):
        prune_tsdb(tmp_path, artifacts, Deadline(time.monotonic() + 10))
    assert (external / 'data').exists()
