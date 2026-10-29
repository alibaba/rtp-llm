import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mock_framework.config import Registry, compile_case
from mock_framework.execution.records import Records
from mock_framework.view.service import handler, serve
from mock_framework.view.grafana import dashboard, provision
from mock_framework.view.records import experiment

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def completed(tmp_path):
    source = tmp_path / 'experiment'
    write(source / 'result.json', {'status': 'ERROR', 'started_at': '2026-10-29T00:00:00Z',
        'finished_at': '2026-10-29T00:00:20Z', 'error': 'saved error', 'cleanup_errors': []})
    write(source / 'configuration.json', {'case': {'id': 'basic', 'name': '保存的实验', 'describe': '仅查看',
        'phases': [{'steps': [{'id': 'qps', 'describe': 'QPS'}, {'id': 'raw', 'describe': 'PromQL'}]}]}})
    (source / 'case.yaml').write_text('id: basic\n')
    write(source / 'code.json', [{'commit': 'saved-sha', 'diff': 'saved working tree diff'}])
    write(source / 'components/prefill-0/effective-config.json', {'device_blocks': 4095})
    write(source / 'prometheus/collection.json', {'complete': False, 'errors': ['scrape unavailable']})
    gates = [
        {'step': 'qps', 'interface': 'client/qps', 'query_time': '2026-10-29T00:00:10Z',
         'args': {'range': ['-4s', '0s']}, 'expect': {'gte': 2}, 'actual': 1, 'status': 'FAIL'},
        {'step': 'raw', 'interface': 'prometheus/query', 'query_time': '2026-10-29T00:00:11Z',
         'args': {'promql': 'rate(x[1h])'}, 'expect': {'count': 1}, 'status': 'ERROR', 'error': 'saved error'}]
    (source / 'gates.jsonl').write_text(''.join(json.dumps(gate) + '\n' for gate in gates))
    events = [
        {'event': 'phase_started', 'phase': 'traffic', 'recorded_at': '2026-10-29T00:00:02Z'},
        {'event': 'step_started', 'step': 'qps', 'recorded_at': '2026-10-29T00:00:09Z'},
        {'event': 'metric_window', 'query_time': gates[0]['query_time'],
         'start': 1793232005, 'end': 1793232010, 'recorded_at': '2026-10-29T00:00:10Z'},
        {'event': 'step_finished', 'step': 'qps', 'status': 'FAIL', 'recorded_at': '2026-10-29T00:00:12Z'}]
    (source / 'events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
    write(source / 'view.json', {'title': 'Default', 'panels': []})
    return source


def test_view_uses_saved_values_windows_versions_and_failed_results(completed):
    value = experiment(completed)
    assert value['result']['status'] == 'ERROR'
    assert value['gates'][0]['status'] == 'FAIL' and value['gates'][0]['actual'] == 1
    # The actual recorded window wins over a reconstructed interpretation of args.range.
    assert value['gates'][0]['from'] == 1793232005000
    assert value['gates'][0]['to'] == 1793232010000
    assert value['gates'][1]['from'] is None and value['gates'][1]['to'] is None
    assert 'actual' not in value['gates'][1] and value['gates'][1]['status'] == 'ERROR'
    assert value['versions'][0]['commit'] == 'saved-sha'
    assert value['effective']['prefill-0']['device_blocks'] == 4095
    assert value['collection']['complete'] is False
    assert any(event['timeEnd'] == 1793232010000 for event in value['events'] if event.get('timeEnd'))


def test_running_output_is_rejected(completed):
    write(completed / 'result.json', {'status': 'PASS', 'started_at': '2026-10-29T00:00:00Z'})
    with pytest.raises(ValueError, match='completed'):
        experiment(completed)


def test_case_saves_its_view_without_retaining_a_dependency_on_source(tmp_path):
    case = compile_case(ROOT / 'cases/basic/case.yaml', Registry.standard())
    view = tmp_path / 'view.json'
    original = {'title': 'Original', 'panels': []}
    write(view, original)
    records = Records(tmp_path / 'result', replace(case, view=view), {})
    records.close({'status': 'PASS'})
    write(view, {'title': 'Changed', 'panels': []})
    assert json.loads((records.directory / 'view.json').read_text()) == original


def test_invalid_native_dashboard_is_rejected_before_component_start(tmp_path):
    case = compile_case(ROOT / 'cases/basic/case.yaml', Registry.standard())
    view = tmp_path / 'view.json'
    write(view, {'title': 'Invalid', 'panels': 'wrong'})
    with pytest.raises(ValueError, match='Grafana'):
        Records(tmp_path / 'result', replace(case, view=view), {})


def test_dashboard_rebinds_nested_panels_without_mutating_the_source_or_future_views(tmp_path):
    source = {'uid': 'original', 'title': 'Original', 'panels': [{'type': 'row', 'panels': [
        {'type': 'timeseries', 'datasource': {'uid': 'old'}, 'targets': [{'expr': 'up', 'datasource': {'uid': 'old'}}]}]}]}
    original = deepcopy(source)
    adapted = dashboard(source)
    panel = adapted['panels'][0]['panels'][0]
    assert panel['targets'][0]['datasource']['uid'] == 'mock-prometheus'
    panel['datasource']['uid'] = 'mutated'
    assert dashboard(source)['panels'][0]['panels'][0]['datasource']['uid'] == 'mock-prometheus'
    assert source == original
    provision(tmp_path, source, 'http://127.0.0.1:1234', 1793232000000, 1793232020000)
    saved = json.loads((tmp_path / 'dashboards/default.json').read_text())
    assert saved['annotations']['list'][0]['enable'] is True
    assert saved['time']['from'].startswith('2026-10-29')
    assert saved['refresh'] == ''


def test_view_http_server_only_serves_fixed_read_only_resources(completed):
    body = experiment(completed)
    body['name'] = '<script>must not execute</script>'
    body['metrics'] = {'status': 'starting', 'url': None, 'error': None}
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(body))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(url + '/api/experiment') as response:
            assert response.headers['Content-Type'].startswith('application/json')
            assert json.load(response)['name'] == body['name']
        with urlopen(url + '/') as response:
            assert b'Grafana' in response.read()
        with pytest.raises(HTTPError) as rejected:
            urlopen(Request(url + '/api/experiment', headers={'Host': 'untrusted.example'}))
        assert rejected.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(url + '/../../configuration.json')
        assert error.value.code == 404
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_view_refuses_to_write_inside_experiment_outputs(completed):
    with pytest.raises(ValueError, match='outside'):
        asyncio.run(serve(completed, completed / 'view', Path('missing-prometheus'), Path('missing-grafana'), 0))
    assert not (completed / 'view').exists()


@pytest.mark.parametrize('name', ['configuration.json', 'case.yaml', 'code.json', 'interfaces.json',
                                  'prometheus/collection.json', 'events.jsonl', 'gates.jsonl'])
def test_missing_optional_artifact_preserves_saved_verdict(completed, name):
    (completed / name).unlink(missing_ok=True)
    body = experiment(completed)
    assert body['result']['status'] == 'ERROR'
    assert any(name in warning for warning in body['warnings'])
    if name == 'configuration.json':
        assert body['configuration'] is None
        assert body['gates'][0]['title'] == 'qps'


def test_damaged_optional_artifact_and_partial_journal_are_reported(completed):
    (completed / 'code.json').write_text('{broken')
    with (completed / 'gates.jsonl').open('a') as stream:
        stream.write('{"step":\n')
        stream.write(json.dumps({'status': 'PASS'}) + '\n')
    with (completed / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps({'event': 'metric_window', 'start': 'wrong', 'end': 1}) + '\n')
    body = experiment(completed)
    assert body['versions'] is None
    assert len(body['gates']) == 2
    assert body['result']['status'] == 'ERROR'
    assert any('code.json' in warning for warning in body['warnings'])
    assert any('gates.jsonl:3' in warning for warning in body['warnings'])
    assert any('gates.jsonl:4' in warning for warning in body['warnings'])
    assert any('events.jsonl:5' in warning for warning in body['warnings'])


@pytest.mark.parametrize('value', [[], {'status': 'UNKNOWN', 'started_at': '2026-10-29T00:00:00Z',
                                      'finished_at': '2026-10-29T00:00:20Z'},
                                  {'status': 'PASS', 'started_at': None, 'finished_at': 'invalid'}])
def test_invalid_core_result_names_its_artifact(completed, value):
    write(completed / 'result.json', value)
    with pytest.raises(ValueError, match='result.json'):
        experiment(completed)


def test_metrics_endpoint_reflects_service_changes(completed):
    body = experiment(completed)
    body['metrics'] = {'status': 'starting', 'url': None, 'error': None}
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(body))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/api/metrics'
        assert json.load(urlopen(url))['status'] == 'starting'
        body['metrics'] = {'status': 'unavailable', 'url': None, 'error': 'saved service error'}
        assert json.load(urlopen(url))['error'] == 'saved service error'
    finally:
        server.shutdown(); server.server_close(); thread.join()


async def until(check):
    async with asyncio.timeout(10):
        while not check():
            await asyncio.sleep(.02)


@pytest.mark.parametrize('missing', ['tsdb', 'binary', 'dashboard'])
def test_metrics_failure_keeps_http_page_and_result(completed, tmp_path, missing):
    async def scenario():
        if missing != 'tsdb':
            (completed / 'prometheus/data').mkdir(parents=True)
        if missing == 'dashboard':
            (completed / 'view.json').unlink()
        output = tmp_path / 'viewer'
        task = asyncio.create_task(serve(completed, output, Path('missing'), Path('missing'), 0))
        try:
            await until(lambda: (output / 'session.json').exists())
            url = json.loads((output / 'session.json').read_text())['url']
            def fetch():
                with urlopen(url + '/api/experiment', timeout=1) as response:
                    return json.load(response)
            async with asyncio.timeout(5):
                while (body := await asyncio.to_thread(fetch))['metrics']['status'] == 'starting':
                    await asyncio.sleep(.02)
            assert body['result']['status'] == 'ERROR'
            assert body['gates'][0]['actual'] == 1
            assert body['metrics']['status'] == 'unavailable' and body['metrics']['error']
            assert not task.done()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    asyncio.run(scenario())


@pytest.fixture
def metrics_binaries(tmp_path):
    import sys
    script = '''
import configparser, json, os, signal, sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
prom = any(arg.startswith('--web.listen-address=') for arg in sys.argv)
name = 'prometheus' if prom else 'grafana'
signal.signal(signal.SIGTERM, lambda *_: sys.exit(7 if prom and os.environ.get('VIEW_TEST_FAIL') == 'shutdown' else 0))
Path(name + '.pid').write_text(str(os.getpid()))
if os.environ.get('VIEW_TEST_FAIL') == name:
    sys.exit(9)
if prom:
    port = int(next(arg.rsplit(':', 1)[1] for arg in sys.argv if arg.startswith('--web.listen-address=')))
else:
    cfg = configparser.ConfigParser()
    cfg.read(sys.argv[sys.argv.index('--config') + 1])
    port = cfg.getint('server', 'http_port')
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.do_GET()
HTTPServer(('127.0.0.1', port), Handler).serve_forever()
'''
    prom, grafana = tmp_path / 'bin/prometheus', tmp_path / 'grafana/bin/grafana'
    for binary in (prom, grafana):
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text('#!' + sys.executable + '\n' + script)
        binary.chmod(0o755)
    return prom, grafana.parent.parent


def assert_exited(output):
    import os
    for path in output.glob('*.pid'):
        with pytest.raises(ProcessLookupError):
            os.kill(int(path.read_text()), 0)


@pytest.mark.parametrize('fault', ['startup', 'running', 'cancel', 'cleanup'])
def test_metrics_processes_are_reaped_after_failure_or_cancel(completed, tmp_path, metrics_binaries, monkeypatch, fault):
    import os
    import signal
    from mock_framework.view.service import metrics
    async def scenario():
        data = completed / 'prometheus/data'
        data.mkdir(parents=True)
        (data / 'source').write_text('immutable data')
        output = tmp_path / 'session'
        output.mkdir()
        body = experiment(completed)
        body['metrics'] = {'status': 'starting', 'url': None, 'error': None}
        if fault in ('startup', 'cleanup'):
            monkeypatch.setenv('VIEW_TEST_FAIL', 'grafana' if fault == 'startup' else 'shutdown')
        task = asyncio.create_task(metrics(body, completed, output, *metrics_binaries))
        try:
            if fault == 'startup':
                await asyncio.wait_for(task, 10)
                assert '9' in body['metrics']['error']
            else:
                await until(lambda: body['metrics']['status'] == 'ready' or task.done())
                assert body['metrics']['status'] == 'ready'
                if fault == 'running':
                    os.kill(int((output / 'prometheus.pid').read_text()), signal.SIGKILL)
                    await asyncio.wait_for(task, 10)
                    assert 'unexpectedly' in body['metrics']['error']
                else:
                    task.cancel()
                    with pytest.raises(RuntimeError if fault == 'cleanup' else asyncio.CancelledError):
                        await task
            assert body['metrics']['status'] == 'unavailable'
            assert body['result']['status'] == 'ERROR'
            assert (data / 'source').read_text() == 'immutable data'
            assert list(data.iterdir()) == [data / 'source']
            assert_exited(output)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('number', ['NaN', 'Infinity', '1e999'])
def test_nonfinite_optional_json_is_reported_instead_of_breaking_http(completed, number):
    (completed / 'code.json').write_text('[{"value":' + number + '}]')
    body = experiment(completed)
    assert body['versions'] is None
    json.dumps(body, allow_nan=False)
    assert any('Non-finite' in warning for warning in body['warnings'])


def test_cli_sigterm_stops_http_and_owned_processes(completed, tmp_path, metrics_binaries):
    import os
    import signal
    import subprocess
    import sys
    import time
    (completed / 'prometheus/data').mkdir(parents=True)
    output = tmp_path / 'cli-session'
    prom, grafana_home = metrics_binaries
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'python')}
    with (tmp_path / 'cli.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'mock_framework.view', str(completed),
            '--output', str(output), '--prometheus', str(prom), '--grafana-home', str(grafana_home)],
            env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 10
            url = None
            while time.monotonic() < deadline:
                assert process.poll() is None
                if (output / 'session.json').exists():
                    url = json.loads((output / 'session.json').read_text())['url']
                    with urlopen(url + '/api/metrics', timeout=1) as response:
                        if json.load(response)['status'] == 'ready':
                            break
                time.sleep(.02)
            else:
                pytest.fail('CLI did not become ready')
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) == 0
            assert_exited(output)
            with pytest.raises(OSError):
                urlopen(url, timeout=1)
            log.seek(0)
            assert 'Traceback' not in log.read()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_busy_http_port_starts_no_metrics_processes(completed, tmp_path, metrics_binaries):
    import socket
    async def scenario():
        output = tmp_path / 'busy'
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0))
            occupied.listen()
            with pytest.raises(OSError):
                await serve(completed, output, *metrics_binaries, occupied.getsockname()[1])
        assert not list(output.glob('*.pid'))
        assert not (output / 'session.json').exists()
    asyncio.run(scenario())
