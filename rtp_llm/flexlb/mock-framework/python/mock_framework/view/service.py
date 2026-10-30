"""Own a read-only HTTP page and its isolated metrics processes."""
import asyncio
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace
from urllib.parse import urlsplit
from urllib.request import urlopen

from ..execution.processes import ProcessSupervisor
from .grafana import annotations, instance_filters, provision
from .records import experiment, read_json


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


async def ready(url):
    def fetch():
        with urlopen(url, timeout=1) as response:
            response.read()
    while True:
        try:
            await asyncio.to_thread(fetch)
            return
        except OSError:
            await asyncio.sleep(.1)


def handler(body):
    page = Path(__file__).with_name('page.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
                self.send_error(400)
                return
            path = urlsplit(self.path).path
            if path == '/':
                content, kind = page, 'text/html; charset=utf-8'
            elif path == '/api/metrics':
                content, kind = json.dumps(body['metrics'], ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8'
            elif path == '/api/experiment':
                content, kind = json.dumps(body, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
    return Handler


async def metrics(body, result, output, prometheus, grafana_home):
    supervisor = ProcessSupervisor(SimpleNamespace(emit=lambda **event: logging.debug('%s', event)))
    cancelled = Event()
    failure = None
    try:
        view = read_json(result / 'view.json')
        data = result / 'prometheus/data'
        if not data.is_dir():
            raise ValueError('The experiment has no retained Prometheus TSDB')
        grafana = grafana_home / 'bin/grafana'
        for binary in (prometheus, grafana):
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise ValueError(f'Missing executable: {binary}')
        for name in ('prometheus', 'grafana'):
            (output / name).mkdir()
        # Recovery writes belong to a disposable copy, never to the experiment.
        copying = asyncio.create_task(asyncio.to_thread(shutil.copytree, data, output / 'prometheus/data'))
        try:
            await asyncio.shield(copying)
        except asyncio.CancelledError:
            await copying
            raise
        (output / 'prometheus/config.yaml').write_text('scrape_configs: []\n', encoding='utf-8')
        prom_port, grafana_port = free_port(), free_port()
        while grafana_port == prom_port:
            grafana_port = free_port()
        prom_url, grafana_url = f'http://127.0.0.1:{prom_port}', f'http://127.0.0.1:{grafana_port}'
        password = secrets.token_urlsafe(24)
        config = output / 'grafana/grafana.ini'
        config.write_text(f'''[server]
http_addr = 127.0.0.1
http_port = {grafana_port}
[paths]
data = {output / 'grafana/data'}
logs = {output / 'grafana/logs'}
plugins = {output / 'grafana/plugins'}
provisioning = {output / 'grafana/provisioning'}
[database]
wal = true
[security]
allow_embedding = true
[auth.anonymous]
enabled = true
org_role = Viewer
[auth]
disable_login_form = true
[plugins]
preinstall_disabled = true
[analytics]
reporting_enabled = false
check_for_updates = false
check_for_plugin_updates = false
''', encoding='utf-8')
        await supervisor.spawn('prometheus', [str(prometheus), '--config.file=' + str(output / 'prometheus/config.yaml'),
            '--storage.tsdb.path=' + str(output / 'prometheus/data'), '--web.listen-address=' + prom_url.removeprefix('http://')],
            str(output), dict(os.environ), output / 'prometheus', zero_exit=True)
        await supervisor.guard(ready(prom_url + '/-/ready'), 60)
        filters = await supervisor.guard(asyncio.to_thread(instance_filters, prom_url, body['start'], body['end'], view), 15)
        interval = (body['collection'] or {}).get('recipe', {}).get('scrape_interval')
        if not isinstance(interval, str) or not interval:
            raise ValueError('Missing recorded Prometheus scrape_interval')
        annotation_selection = provision(output / 'grafana', view, prom_url, body['start'], body['end'],
                  scrape_interval=interval, filters=filters,
                  steps=body['annotation_steps'])
        env = {key: value for key, value in os.environ.items() if not key.startswith('GF_')}
        env['GF_SECURITY_ADMIN_PASSWORD'] = password
        await supervisor.spawn('grafana', [str(grafana), 'server', '--homepath', str(grafana_home), '--config', str(config)],
            str(output), env, output / 'grafana')
        await supervisor.guard(ready(grafana_url + '/api/health'), 60)
        await supervisor.guard(asyncio.to_thread(annotations, grafana_url, password, body['annotations'], cancelled), 60)
        body['metrics'] = {'status': 'ready', 'url': grafana_url + '/d/mock-experiment-embedded/experiment',
                           'open_url': grafana_url + '/d/mock-experiment/experiment', 'error': None,
                           'filters': filters, 'annotations': annotation_selection}
        await supervisor.guard(asyncio.Event().wait(), None)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        logging.error('Metrics unavailable: %s; logs: %s', failure, output)
    finally:
        cancelled.set()
        errors = await supervisor.close()
        if errors:
            failure = '; '.join(filter(None, [failure, 'cleanup: ' + '; '.join(errors)]))
            logging.error('%s', failure)
        body['metrics'] = {'status': 'unavailable', 'url': None, 'error': failure or 'Metrics services stopped'}
        if errors and asyncio.current_task().cancelling():
            raise RuntimeError(failure)


async def serve(result, output, prometheus, grafana_home, port):
    result, output, prometheus, grafana_home = (Path(path).resolve() for path in (result, output, prometheus, grafana_home))
    if result == output or result in output.parents:
        raise ValueError('View output must be outside the completed experiment directory')
    body = experiment(result)
    body['metrics'] = {'status': 'starting', 'url': None, 'error': None}
    output.mkdir(parents=True, mode=0o700)
    server = ThreadingHTTPServer(('127.0.0.1', port), handler(body))
    thread = None
    task = None
    stopped = asyncio.get_running_loop().create_future()
    loop = asyncio.get_running_loop()

    def run_http():
        try:
            server.serve_forever()
        finally:
            loop.call_soon_threadsafe(stopped.set_result, None)

    try:
        thread = Thread(target=run_http, daemon=True)
        thread.start()
        address = f'http://127.0.0.1:{server.server_port}'
        (output / 'session.json').write_text(json.dumps({'url': address, 'source': str(result)}), encoding='utf-8')
        print(address, flush=True)
        task = asyncio.create_task(metrics(body, result, output, prometheus, grafana_home))
        done, _ = await asyncio.wait((task, stopped), return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            task.result()
            await asyncio.shield(stopped)
        raise RuntimeError('View HTTP server stopped unexpectedly')
    finally:
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(server.shutdown)
            await asyncio.to_thread(thread.join)
        server.server_close()
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
