import asyncio
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard

from mock_framework import client_evidence_pb2 as pb
from mock_framework.config import Registry
from mock_framework.evidence import read_requests, playback_directories, main
from mock_framework.execution.transport import ExecutionError
from mock_framework.execution import clients as client_runtime


def frame(batch):
    raw = batch.SerializeToString()
    size, prefix = len(raw), bytearray()
    while size > 127:
        prefix.append((size & 127) | 128); size >>= 7
    prefix.append(size)
    encoded = zstandard.ZstdCompressor(write_checksum=True).compress(bytes(prefix) + raw)
    return struct.pack('>I', len(encoded)) + encoded


@pytest.fixture
def evidence(tmp_path):
    def create(name='client-0', playback=1, first=9007199254740993, index=0, stride=1, outcome='success'):
        directory = tmp_path / 'components' / name / 'playback' / str(playback)
        directory.mkdir(parents=True)
        plan = dict(sequence=playback, first_request_id=first, first_index=index, stride=stride, count=1, concurrency=1)
        (directory / 'plan.json').write_text(json.dumps(plan))
        batch = pb.EventBatch(version=1, client=name, playback=playback, frame=1,
                              first_request_id=first, first_index=index, stride=stride, planned=1, concurrency=1)
        for kind in [pb.Event.ISSUED, pb.Event.SCHEDULE_START, pb.Event.SCHEDULE_RESULT, pb.Event.FIRST_TOKEN, pb.Event.TERMINAL]:
            n = len(batch.events) + 1
            event = batch.events.add(sequence=n, request_id=first + index, epoch_ms=1000 + n, elapsed_nanos=n * 10, kind=kind)
            if kind in (pb.Event.SCHEDULE_START, pb.Event.SCHEDULE_RESULT): event.attempt = 1; event.master = 'm:1'
            if kind == pb.Event.SCHEDULE_RESULT: event.grpc_code = 'OK'; event.error_code = 200
            if kind == pb.Event.TERMINAL:
                event.outcome = outcome; event.grpc_code = 'OK'; event.ttft_nanos = 40; event.output_tokens = 3
        batch.seal.issued = batch.seal.finished = 1
        result = dict(done=True, issued=1, finished=1, unsent=0, outcomes={v: int(v == outcome) for v in ('success', 'error', 'timeout', 'cancelled')})
        (directory / 'result.json').write_text(json.dumps(result))
        (directory / 'requests.binpb.zst').write_bytes(frame(batch))
        (tmp_path / 'components/client-0/coordinator' / str(playback)).mkdir(parents=True, exist_ok=True)
        return directory, batch
    return create


def test_roundtrip_preserves_int64_identity_and_reconciles_summary(evidence):
    directory, batch = evidence()
    result, = read_requests(directory)
    assert result.request_id == 9007199254740993
    assert result.attempts[0].master == 'm:1' and result.output_tokens == 3
    assert result.ttft_nanos == 40 and result.elapsed_nanos == 50


@pytest.mark.parametrize('damage', ['truncated', 'checksum', 'missing_seal', 'sequence', 'duplicate', 'frame_gap',
    'trailing', 'unfinished', 'identity', 'summary', 'attempt', 'first_token', 'unknown_kind', 'oversized'])
def test_damaged_evidence_never_passes(evidence, damage):
    directory, batch = evidence()
    path = directory / 'requests.binpb.zst'
    if damage == 'truncated': path.write_bytes(path.read_bytes()[:-2])
    elif damage == 'checksum':
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
    elif damage == 'trailing': path.write_bytes(path.read_bytes() + frame(batch))
    elif damage == 'oversized': path.write_bytes(struct.pack('>I', 3 * 1024 * 1024))
    elif damage == 'summary': (directory / 'result.json').write_text('{"done": false, "issued": 2}')
    else:
        if damage == 'missing_seal': batch.ClearField('seal')
        if damage == 'sequence': batch.events[2].sequence = 99
        if damage == 'duplicate': batch.events[1].kind = pb.Event.ISSUED
        if damage == 'frame_gap': batch.frame = 2
        if damage == 'unfinished': del batch.events[-1]
        if damage == 'identity': batch.client = 'client-7'
        if damage == 'attempt': batch.events[2].attempt = 2
        if damage == 'first_token': batch.events[-1].ClearField('ttft_nanos')
        if damage == 'unknown_kind': batch.events[2].kind = 99
        path.write_bytes(frame(batch))
    with pytest.raises(ExecutionError): list(read_requests(directory))


def test_query_defaults_to_all_clients_and_limits_details_not_counts(evidence, tmp_path):
    evidence(index=0, stride=2)
    evidence('client-1', index=1, stride=2, outcome='timeout')
    topology = [dict(id=f'client-{i}', role='client', rpc=f'c:{i}', control=f'http://c:{i}') for i in range(2)]
    ctx = SimpleNamespace(records=SimpleNamespace(directory=tmp_path), runtime=SimpleNamespace(topology=SimpleNamespace(active=lambda: topology)))
    query = Registry.standard().get('client/requests')
    result = asyncio.run(query.invoke(ctx, {'limit': 1}))
    assert result['matched'] == 2 and result['truncated'] and len(result['records']) == 1
    assert result['outcomes']['timeout'] == 1
    result = asyncio.run(query.invoke(ctx, {'targets': ['c:1'], 'outcome': 'timeout', 'since_ms': 1001, 'until_ms': 1002}))
    assert result['matched'] == 1 and result['records'][0]['request_id'] == 9007199254740994
    assert asyncio.run(query.invoke(ctx, {'request_ids': [1]}))['matched'] == 0
    with pytest.raises(ExecutionError): asyncio.run(query.invoke(ctx, {'targets': ['missing']}))
    path = tmp_path / 'components/client-1/playback/1/requests.binpb.zst'
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ExecutionError): asyncio.run(query.invoke(ctx, {'limit': 0, 'request_ids': [1]}))


def test_collect_rejects_missing_evidence_even_when_rpc_summary_passes(monkeypatch, tmp_path):
    from test_client_interfaces import context, state
    ctx = context(); ctx.records.directory = tmp_path; ctx.runtime.clients.pending = True
    async def post(*args): return {'clients': {'http://c:1': state(), 'http://c:2': state()}, 'errors': []}
    monkeypatch.setattr(client_runtime, 'post', post)
    with pytest.raises(ExecutionError, match='No Client playback'):
        asyncio.run(Registry.standard().get('client/stop').invoke(ctx, {}))
    assert ctx.runtime.clients.pending


def test_cli_partial_output_is_nonzero_and_explicit(evidence, tmp_path, monkeypatch, capsys):
    directory, batch = evidence()
    batch.ClearField('seal'); (directory / 'requests.binpb.zst').write_bytes(frame(batch))
    (tmp_path / 'topology-history.json').write_text(json.dumps([dict(id='client-0', role='client', rpc='c:0')]))
    monkeypatch.setattr('sys.argv', ['mock-requests', str(tmp_path)])
    with pytest.raises(SystemExit) as error: main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert '9007199254740993' in captured.out and 'recovered prefix' in captured.err


def test_killed_java_writer_leaves_readable_frames_but_no_complete_evidence(tmp_path):
    import os
    import shutil
    import subprocess
    import time
    root = Path(__file__).resolve().parents[1]
    classpath = os.pathsep.join(str(root / path) for path in [
        'java/client/target/test-classes', 'java/client/target/mock-client-1.0-SNAPSHOT-all.jar',
        'java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar'])
    java = str(Path(os.environ['JAVA_HOME']) / 'bin/java') if os.environ.get('JAVA_HOME') else shutil.which('java')
    with (tmp_path / 'probe.log').open('wb') as log:
        process = subprocess.Popen([java, '-Xmx256m', '-cp', classpath,
            'org.flexlb.mock.client.EvidenceCrashProbe', str(tmp_path)], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while not (tmp_path / 'written').exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            assert (tmp_path / 'written').exists(), (tmp_path / 'probe.log').read_text()
        finally:
            process.kill(); process.wait(timeout=10)
    assert process.returncode != 0
    directory = tmp_path / 'components/client-0/playback/1'
    assert (directory / 'requests.binpb.zst').stat().st_size > 0
    with pytest.raises(ExecutionError, match='Missing evidence seal'):
        list(read_requests(directory))


def test_history_selection_does_not_overwrite_or_mix_playbacks(evidence, tmp_path):
    first, _ = evidence()
    second, _ = evidence(playback=2, first=9007199254741000)
    topology = [dict(id='client-0', role='client', rpc='c:0', control='http://c:0')]
    assert playback_directories(tmp_path, topology) == [second]
    assert playback_directories(tmp_path, topology, playback=1) == [first]
    records = list(read_requests(first)) + list(read_requests(second))
    assert [record.request_id for record in records] == [9007199254740993, 9007199254741000]


def test_cancelled_scan_stops_before_reading_next_frame(evidence):
    import threading
    directory, _ = evidence()
    cancelled = threading.Event(); cancelled.set()
    with pytest.raises(ExecutionError, match='scan cancelled'): list(read_requests(directory, cancelled))


def test_zero_issued_round_can_seal_without_fabricating_terminal_events(evidence):
    directory, batch = evidence()
    del batch.events[:]
    batch.seal.issued = batch.seal.finished = 0
    (directory / 'requests.binpb.zst').write_bytes(frame(batch))
    (directory / 'result.json').write_text(json.dumps(dict(done=True, issued=0, finished=0, unsent=1,
        outcomes={v: 0 for v in ('success', 'error', 'timeout', 'cancelled')})))
    assert list(read_requests(directory)) == []


def test_small_compressed_frame_cannot_allocate_unbounded_decompressed_output(evidence):
    directory, _ = evidence()
    encoded = zstandard.ZstdCompressor().compress(b'x' * (3 * 1024 * 1024))
    (directory / 'requests.binpb.zst').write_bytes(struct.pack('>I', len(encoded)) + encoded)
    with pytest.raises(ExecutionError, match='Oversized decompressed'): list(read_requests(directory))
