from mock_framework.execution.test.helpers import FRAMEWORK
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mock_framework.execution.step.registry import Registry
from mock_framework.execution.step.client_requests import playback_directories
from mock_framework.execution.evidence.client import read_requests
from mock_framework.execution import ExecutionError
from mock_framework.execution.roles import client as client_runtime


def write_rows(directory, rows):
    (directory / 'requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.fixture
def evidence(tmp_path):
    def create(name='client-0', playback=1, first=9007199254740993, index=0, stride=1, outcome='success'):
        directory = tmp_path / 'components' / name / 'playback' / str(playback)
        directory.mkdir(parents=True)
        plan = dict(sequence=playback, first_request_id=first, first_index=index, stride=stride, count=1, concurrency=1)
        (directory / 'plan.json').write_text(json.dumps(plan))
        header = dict(version=1, client=name, playback=str(playback), first_request_id=str(first),
                      first_index=str(index), stride=stride, planned='1', concurrency=1)
        events = []
        for n, kind in enumerate(('ISSUED', 'SCHEDULE_START', 'SCHEDULE_RESULT', 'FIRST_TOKEN', 'TERMINAL'), 1):
            event = dict(sequence=str(n), request_id=str(first + index), epoch_ms=str(1000 + n),
                         elapsed_nanos=str(n * 10), kind=kind)
            if kind in ('SCHEDULE_START', 'SCHEDULE_RESULT'):
                event.update(attempt=1, master='m:1')
            if kind == 'SCHEDULE_RESULT':
                event.update(grpc_code='OK', error_code='200')
            if kind == 'TERMINAL':
                event.update(outcome=outcome, grpc_code='OK', ttft_nanos='40', output_tokens=3)
            events.append(event)
        rows = [header, *events, {'seal': {'issued': '1', 'finished': '1'}}]
        result = dict(done=True, issued=1, finished=1, unsent=0, outcomes={v: int(v == outcome) for v in ('success', 'error', 'timeout', 'cancelled')})
        (directory / 'result.json').write_text(json.dumps(result))
        write_rows(directory, rows)
        (tmp_path / 'components/client-0/coordinator' / str(playback)).mkdir(parents=True, exist_ok=True)
        return directory, rows
    return create


def test_roundtrip_preserves_int64_identity_and_reconciles_summary(evidence):
    directory, _ = evidence()
    result, = read_requests(directory)
    assert result.request_id == 9007199254740993
    assert result.attempts[0].master == 'm:1' and result.output_tokens == 3
    assert result.ttft_nanos == 40 and result.elapsed_nanos == 50


@pytest.mark.parametrize('damage', ['missing_seal', 'sequence', 'duplicate_field', 'trailing', 'truncated'])
def test_invalid_jsonl_is_rejected(evidence, damage):
    directory, rows = evidence()
    rows = [json.dumps(row) for row in rows]
    if damage == 'missing_seal':
        rows.pop()
    elif damage == 'sequence':
        event = json.loads(rows[1])
        rows[1] = json.dumps({**event, 'sequence': '99'})
    elif damage == 'duplicate_field':
        rows[1] = rows[1][:-1] + ', "sequence": "1"}'
    elif damage == 'trailing':
        rows.append('{}')
    newline = '' if damage == 'truncated' else '\n'
    (directory / 'requests.jsonl').write_text('\n'.join(rows) + newline)
    with pytest.raises(ExecutionError, match=r'requests\.jsonl'):
        list(read_requests(directory))


@pytest.mark.parametrize('damage', ['duplicate', 'unfinished', 'identity', 'summary',
                                   'attempt', 'first_token', 'unknown_kind'])
def test_damaged_evidence_never_passes(evidence, damage):
    directory, rows = evidence()
    if damage == 'duplicate':
        rows[2]['kind'] = 'ISSUED'
    elif damage == 'unfinished':
        del rows[-2]
    elif damage == 'identity':
        rows[0]['client'] = 'client-7'
    elif damage == 'summary':
        (directory / 'result.json').write_text('{"done": false, "issued": 2}')
    elif damage == 'attempt':
        rows[3]['attempt'] = 2
    elif damage == 'first_token':
        del rows[-2]['ttft_nanos']
    elif damage == 'unknown_kind':
        rows[3]['kind'] = 'UNKNOWN'
    write_rows(directory, rows)
    with pytest.raises(ExecutionError):
        list(read_requests(directory))


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
    path = tmp_path / 'components/client-1/playback/1/requests.jsonl'
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ExecutionError): asyncio.run(query.invoke(ctx, {'limit': 0, 'request_ids': [1]}))


def test_stop_uses_rpc_summary_while_request_query_requires_files(monkeypatch, tmp_path):
    from mock_framework.execution.step.test.test_client_interfaces import context, state
    ctx = context(); ctx.records.directory = tmp_path; ctx.runtime.clients.pending = True
    async def post(*args): return {'clients': {'http://c:1': state(), 'http://c:2': state()}, 'errors': []}
    monkeypatch.setattr(client_runtime, 'post', post)
    result = asyncio.run(Registry.standard().get('client/stop').invoke(ctx, {}))
    assert set(result) == {'c:1', 'c:2'} and not ctx.runtime.clients.pending
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ExecutionError, match='No Client playback'):
        asyncio.run(Registry.standard().get('client/requests').invoke(ctx, {}))


def test_killed_java_writer_leaves_readable_lines_but_no_complete_evidence(tmp_path):
    import os
    import shutil
    import subprocess
    import time
    root = FRAMEWORK
    classpath = os.pathsep.join(str(root / path) for path in [
        'java/client/target/test-classes', 'java/client/target/mock-client-1.0-SNAPSHOT-all.jar',
        'java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar'])
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
    assert any(json.loads(line).get('kind') == 'FIRST_TOKEN'
               for line in (directory / 'requests.jsonl').read_text().splitlines())
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


def test_cancelled_scan_stops_before_reading_next_line(evidence):
    import threading
    directory, _ = evidence()
    cancelled = threading.Event(); cancelled.set()
    with pytest.raises(ExecutionError, match='scan cancelled'): list(read_requests(directory, cancelled))


def test_zero_issued_round_can_seal_without_fabricating_terminal_events(evidence):
    directory, rows = evidence()
    write_rows(directory, [rows[0], {'seal': {}}])
    (directory / 'result.json').write_text(json.dumps(dict(done=True, issued=0, finished=0, unsent=1,
        outcomes={v: 0 for v in ('success', 'error', 'timeout', 'cancelled')})))
    assert list(read_requests(directory)) == []
