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
from mock_framework.view.records import experiment, gate_text

ROOT = Path(__file__).resolve().parents[1]


def test_page_initializes_filters_before_the_first_grafana_load():
    import shutil
    import subprocess
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is required to execute the View page script')
    subprocess.run([node, str(ROOT / 'tests/view_page_load.cjs'),
                    str(ROOT / 'python/mock_framework/view/page.html')], check=True, timeout=10)


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
    write(source / 'interfaces.json', {'interfaces': {'client/qps': {'unit': 'requests/s'}}})
    write(source / 'code.json', [{'commit': 'saved-sha', 'diff': 'saved working tree diff'}])
    write(source / 'components/prefill-0/effective-config.json', {'device_blocks': 4095})
    write(source / 'prometheus/collection.json', {'complete': False, 'errors': ['scrape unavailable'], 'recipe': {'scrape_interval': '100ms'}})
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
    provision(tmp_path, source, 'http://127.0.0.1:1234', 1793232000000, 1793232020000, scrape_interval='100ms', filters={})
    saved = json.loads((tmp_path / 'dashboards/default.json').read_text())
    assert saved['annotations']['list'][0]['enable'] is True
    assert saved['annotations']['list'][0]['hide'] is True
    assert saved['time']['from'].startswith('2026-10-29')
    assert saved['refresh'] == ''


@pytest.mark.parametrize('name', ['engine_instance', 'prefill_instance', 'decode_instance', 'vit_instance', 'master_instance', 'client_instance'])
def test_saved_address_filters_use_prometheus_interpolation_including_nested_rows(name):
    expression = 'sum(metric{instance=~"${' + name + ':regex}",pool=~"${pool:regex}"})'
    source = {'title': 'Saved dashboard', 'panels': [{'type': 'row', 'panels': [
        {'type': 'timeseries', 'targets': [{'expr': expression}]}]}]}
    adapted = dashboard(source)
    target = adapted['panels'][0]['panels'][0]['targets'][0]
    assert target['expr'] == expression.replace('${' + name + ':regex}', '${' + name + '}')
    assert target['datasource']['type'] == 'prometheus'
    assert source['panels'][0]['panels'][0]['targets'][0]['expr'] == expression


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


def test_gate_annotations_include_saved_conditions_values_and_errors(completed):
    body = experiment(completed)
    texts = [event['text'] for event in body['events'] if event['text'].startswith('门禁 ')]
    assert '门禁 qps · client/qps · FAIL\n预期：≥ 2 requests/s\n实际：1 requests/s' in texts
    assert '门禁 raw · prometheus/query · ERROR\n预期：数量 = 1\n实际：未取得结果\n错误：saved error' in texts


def test_gate_annotation_formats_range_count_ratio_and_structured_results():
    gate = {'step': 'qps', 'interface': 'client/qps', 'status': 'PASS', 'unit': 'requests/s',
            'expect': {'gte': 1.5, 'lte': 2.5}, 'actual': 2}
    assert '预期：≥ 1.5 requests/s 且 ≤ 2.5 requests/s\n实际：2 requests/s' in gate_text(gate)
    gate.update(unit='ratio', expect={'gte': .95}, actual=.98)
    assert '预期：≥ 95%\n实际：98%' in gate_text(gate)
    gate.update(unit='', expect={'count': 1}, actual=[{'address': 'localhost'}])
    assert '预期：数量 = 1\n实际：数量 1' in gate_text(gate)
    gate.update(expect={'eq': {'success': 20}}, actual={'success': 20})
    assert '预期：= {"success": 20}\n实际：{"success": 20}' in gate_text(gate)


def test_annotation_html_escapes_saved_values_and_preserves_line_breaks(monkeypatch):
    import io
    from threading import Event
    from mock_framework.view import grafana
    sent = []
    def post(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(b'{}')
    monkeypatch.setattr(grafana, 'urlopen', post)
    grafana.annotations('http://localhost', 'unused', [
        {'step': 'qps', 'time': 10, 'timeEnd': 20, 'text': '门禁 · FAIL\n实际：<img src=x onerror=alert(1)>'}], Event())
    assert sent[0]['text'] == '门禁 · FAIL<br/>实际：&lt;img src=x onerror=alert(1)&gt;'
    assert sent[0]['time'] == 10 and sent[0]['timeEnd'] == 20
    assert len(sent) == 1 and 'dashboardUID' not in sent[0]
    assert sent[0]['tags'] == ['experiment', 'step:qps']


def test_annotations_merge_steps_and_gates_without_filtering_execution_records(completed):
    body = experiment(completed)
    assert len(body['events']) == 5
    assert [event['step'] for event in body['annotations']] == ['qps', 'raw']
    assert body['annotations'][0]['timeEnd'] == body['gates'][0]['to']
    assert '实际：1 requests/s' in body['annotations'][0]['text']
    with (completed / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps({'event': 'step_started', 'step': 'stop',
                                 'recorded_at': '2026-10-29T00:00:15Z'}) + '\n')
        stream.write(json.dumps({'event': 'step_finished', 'step': 'stop', 'status': 'PASS',
                                 'recorded_at': '2026-10-29T00:00:16Z'}) + '\n')
    stop = experiment(completed)['annotations'][-1]
    assert stop == {'step': 'stop', 'time': 1793232015000, 'text': 'stop · PASS'}


def test_annotation_defaults_can_include_steps_not_reached_by_a_failed_case(completed, tmp_path):
    configuration = json.loads((completed / 'configuration.json').read_text())
    configuration['case']['phases'][0]['steps'].append({'id': 'not_reached'})
    write(completed / 'configuration.json', configuration)
    body = experiment(completed)
    assert 'not_reached' in body['annotation_steps']
    assert all(event['step'] != 'not_reached' for event in body['annotations'])
    selected = provision(tmp_path / 'grafana', {'title': 'HA', 'panels': [], 'annotation_steps': ['not_reached']},
                         'http://localhost', body['start'], body['end'], scrape_interval='1s',
                         filters={}, steps=body['annotation_steps'])
    assert selected['selected'] == ['not_reached']


@pytest.mark.parametrize('selected', [None, [], ['kill']])
def test_annotation_selection_defaults_and_native_grafana_filter(tmp_path, selected):
    view = {'title': 'HA', 'panels': []}
    if selected is not None:
        view['annotation_steps'] = selected
    state = provision(tmp_path, view, 'http://localhost', 1000, 2000, scrape_interval='1s',
                      filters={}, steps=['traffic', 'kill'])
    assert state == {'steps': ['traffic', 'kill'], 'selected': ['traffic', 'kill'] if selected is None else selected}
    saved = json.loads((tmp_path / 'dashboards/default.json').read_text())
    assert 'annotation_steps' not in saved
    variable = saved['templating']['list'][-1]
    assert variable['current']['value'] == (['step:' + step for step in state['selected']] or ['__none__'])
    assert saved['annotations']['list'][0]['tags'] == ['$annotation_steps']
    embedded = json.loads((tmp_path / 'dashboards/embedded.json').read_text())
    assert embedded['templating']['list'][-1]['hide'] == 2
    assert view.get('annotation_steps') == selected


@pytest.mark.parametrize('selected', ['kill', [3], ['missing'], ['kill', 'kill']])
def test_invalid_annotation_selection_is_reported(tmp_path, selected):
    with pytest.raises(ValueError, match='annotation_steps'):
        provision(tmp_path, {'title': 'HA', 'panels': [], 'annotation_steps': selected},
                  'http://localhost', 1000, 2000, scrape_interval='1s', filters={}, steps=['kill'])


def test_invalid_unit_metadata_does_not_hide_a_saved_gate(completed):
    write(completed / 'interfaces.json', {'interfaces': {'client/qps': {'unit': None}}})
    body = experiment(completed)
    assert len(body['gates']) == 2
    assert body['gates'][0]['actual'] == 1
    assert any('unit must be a string' in warning for warning in body['warnings'])


def test_default_dashboard_variables_bind_and_freeze_full_experiment_choices(tmp_path):
    source = json.loads((ROOT / 'views/default.json').read_text())
    original = deepcopy(source)
    filters = {'prefill_instance': ['127.0.0.1:1001', '127.0.0.1:1002'], 'decode_instance': ['127.0.0.1:1003'], 'vit_instance': [],
               'master_instance': ['127.0.0.1:2001'], 'client_instance': ['127.0.0.1:3001']}
    provision(tmp_path, source, 'http://127.0.0.1:9090', 1000, 2000, scrape_interval='250ms', filters=filters)
    saved = json.loads((tmp_path / 'dashboards/default.json').read_text())
    for variable in saved['templating']['list']:
        if variable['name'] in ('prefill_group', 'decode_group', 'vit_group', 'master_group', 'client_group'):
            assert variable['type'] == 'query' and variable['hide'] == 2
            assert '${' + variable['name'].replace('_group', '_instance') + '}' in variable['query']
            continue
        assert variable['type'] == 'custom'  # Narrowing a gate window must not lose a retired Engine option.
        assert variable['multi'] and variable['includeAll']
        if variable['name'] in filters:
            assert [option['value'] for option in variable['options']] == filters[variable['name']]
    embedded = json.loads((tmp_path / 'dashboards/embedded.json').read_text())
    assert embedded['uid'] != saved['uid']
    assert all(variable['hide'] == 2 for variable in embedded['templating']['list'])
    assert all(variable.get('hide', 0) == (2 if variable['name'] in ('prefill_group', 'decode_group', 'vit_group', 'master_group', 'client_group', 'vit_instance') else 0)
               for variable in saved['templating']['list'])
    assert source == original
    import yaml
    datasource = yaml.safe_load((tmp_path / 'provisioning/datasources/default.yaml').read_text())['datasources'][0]
    assert datasource['jsonData']['timeInterval'] == '250ms'


@pytest.mark.parametrize('include_vit', [False, True])
def test_instance_choices_use_full_saved_series_and_separate_master_reporter_from_target(monkeypatch, include_vit):
    import io
    from mock_framework.view import grafana
    urls = []
    series = [
        {'__name__': 'rtp_llm_requests_started_total', 'instance': '10.0.1.1:9000', 'role': 'prefill'},
        {'__name__': 'rtp_llm_framework_requests_total', 'instance': '10.0.1.1:9000', 'role': 'prefill'},
        {'__name__': 'rtp_llm_requests_started_total', 'instance': '10.0.1.2:9000', 'role': 'decode'},
        {'__name__': 'flexlb_app_flexlb_batcher_queue_size', 'instance': '10.0.2.1:9000', 'role': 'PREFILL', 'engineIp': '10.0.1.1'},
        {'__name__': 'jvm_memory_used_bytes', 'instance': '10.0.2.1:9000', 'application': 'flexlb-api'},
        {'__name__': 'mock_client_requests_issued_total', 'instance': '10.0.3.1:9000', 'role': 'client'},
        {'__name__': 'up', 'instance': 'unclassified:9000'}]
    if include_vit:
        series.append({'__name__': 'py_rtp_vit_qps', 'instance': '10.0.1.3:9000', 'role': 'vit'})
    def fetch(url, timeout):
        urls.append(url)
        return io.BytesIO(json.dumps({'status': 'success', 'data': series}).encode())
    monkeypatch.setattr(grafana, 'urlopen', fetch)
    view = json.loads((ROOT / 'views/default.json').read_text())
    options = grafana.instance_filters('http://localhost', 10000, 20000, view)
    assert options == {'prefill_instance': ['10.0.1.1:9000'], 'decode_instance': ['10.0.1.2:9000'],
                       'vit_instance': ['10.0.1.3:9000'] if include_vit else [],
                       'master_instance': ['10.0.2.1:9000'], 'client_instance': ['10.0.3.1:9000']}
    from urllib.parse import parse_qs, urlsplit
    assert parse_qs(urlsplit(urls[0]).query) == {'match[]': ['{job="mock"}'], 'start': ['10.0'], 'end': ['20.0']}


def test_partial_series_enumeration_is_not_treated_as_complete_filter_options(monkeypatch):
    import io
    from mock_framework.view import grafana
    monkeypatch.setattr(grafana, 'urlopen', lambda *a, **kw: io.BytesIO(b'{"status":"success","data":[],"warnings":["partial"]}'))
    with pytest.raises(ValueError, match='enumerate'):
        grafana.instance_filters('http://localhost', 1000, 2000, {'templating': {'list': [{'name': 'engine_instance'}]}})


def test_embedded_view_keeps_unmanaged_custom_variables_visible(tmp_path):
    view = {'title': 'Custom', 'panels': [], 'templating': {'list': [
        {'name': 'model', 'type': 'custom', 'query': 'a,b', 'hide': 0}]}}
    provision(tmp_path, view, 'http://localhost:9090', 1000, 2000, scrape_interval='1s', filters={})
    saved = json.loads((tmp_path / 'dashboards/embedded.json').read_text())
    assert saved['templating']['list'][0]['hide'] == 0
