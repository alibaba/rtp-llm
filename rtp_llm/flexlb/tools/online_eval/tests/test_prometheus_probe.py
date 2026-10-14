"""Real SDK + Prometheus proves demand scheduling, raw timestamps and outages."""
import json
import time
from types import SimpleNamespace as NS

import pytest


def test_real_prometheus_retries_an_occupied_port_within_the_same_budget(tmp_path, monkeypatch):
    import threading
    import urllib.parse
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import monitoring.session as implementation
    from runtime.deadline import Deadline
    import time

    class Occupant(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(404)
            self.end_headers()
        def log_message(self, *args):
            pass
    occupied = ThreadingHTTPServer(('127.0.0.1', 0), Occupant)
    thread = threading.Thread(target=occupied.serve_forever, daemon=True)
    thread.start()
    available_port = implementation._available_port
    allocations = []
    def choose():
        port = occupied.server_port if not allocations else available_port()
        allocations.append(port)
        return port
    monkeypatch.setattr(implementation, '_available_port', choose)
    plan = select_plan(load_plan('master_ha_failover.yaml'), {'ha/http_up': {}}, [])
    session = PrometheusSession(tmp_path / 'telemetry/1', {}, interval_s=.1,
        metric_plan=plan, target_kinds={})
    try:
        session.start(deadline=Deadline(time.monotonic() + 10))
        assert len(allocations) == 2
        assert urllib.parse.urlparse(session.client.url).port != occupied.server_port
        assert 'address already in use' in (session.directory / 'prometheus.log').read_text().lower()
        assert session.process.poll() is None
    finally:
        session.stop(export=False)
        occupied.shutdown()
        occupied.server_close()
        thread.join(timeout=2)

from monitoring.collection_plan import select_plan
from monitoring.query_plan import load_plan
from monitoring.session import PrometheusSession
from monitoring.probe import evidence_collector


def wait_for(predicate):
    end = time.monotonic() + 10
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError('Prometheus did not scrape the probe')


def test_standard_probe_archives_scrape_times_and_preserves_master_outage(tmp_path):
    from monitoring.probe import PrometheusEvidence
    # Only produced measurements selected: this must still start the shared TSDB.
    plan = select_plan(load_plan('master_ha_failover.yaml'), {'ha/http_up': {}}, [])
    session = PrometheusSession(tmp_path/'telemetry/1', {}, interval_s=.1, metric_plan=plan, target_kinds={})
    session.start()
    calls = []
    state = {'up': 1}
    def read(**kwargs):
        calls.append(kwargs)
        return {'master': 'A', 'http_up': state['up']}
    probe = PrometheusEvidence(tmp_path/'states.jsonl', {'A': read}, fields={'http_up'},
                label='master', source='master_inflight', session=session,
                limits=dict(max_samples=1000, max_bytes=100000), client=session.client)
    try:
        probe.start()
        wait_for(lambda: len(calls) >= 3)
        state['up'] = 0
        wait_for(lambda: any(float(row['value'][1]) == 0 for row in
                            session.client.query('online_eval_master_inflight_http_up{job="probe-master_inflight"}')))
        probe.stop()
        rows = [json.loads(line) for line in probe.path.read_text().splitlines()]
        assert {row['http_up'] for row in rows} == {0, 1}
        assert all(row['master'] == 'A' for row in rows)
        frozen = json.loads(probe.path.with_suffix('.prometheus.json').read_text())
        assert frozen['transport'] == 'prometheus' and frozen['error'] is None
        raw = [point for query in frozen['queries'] for series in query['result']
               if series['metric']['__name__'] == probe.names['http_up'] for point in series['values']]
        assert [row['epoch_s'] for row in rows] == sorted({point[0] for point in raw})
        count = len(calls)
        time.sleep(.2)
        assert len(calls) == count  # No adapter polling thread remains.
        assert session.process.poll() is None
    finally:
        session.stop()
    assert probe.server is None and not probe.thread.is_alive()
    assert session.process.poll() is not None


def test_standard_probe_bad_payload_survives_sdk_http_error_as_run_failure(tmp_path):
    from monitoring.probe import PrometheusEvidence
    plan = select_plan(load_plan('master_ha_failover.yaml'), {'ha/http_up': {}}, [])
    session = PrometheusSession(tmp_path/'telemetry/1', {}, interval_s=.1, metric_plan=plan, target_kinds={})
    session.start()
    def invalid(**kw):
        raise ValueError('missing required debug field')
    probe = PrometheusEvidence(tmp_path/'bad.jsonl', {'A': invalid}, fields={'http_up'},
            label='master', source='master_inflight', session=session,
            limits=dict(max_samples=100, max_bytes=10000), client=session.client)
    try:
        with pytest.raises(RuntimeError, match="missing required debug field"):
            probe.start()
        assert probe.error is not None
        with pytest.raises(RuntimeError, match='missing required debug field'):
            probe.stop()
        assert probe.path.read_text() == ''
        assert probe.server is None
    finally:
        with pytest.raises(RuntimeError, match='missing required debug field'):
            session.stop()
    assert session.process.poll() is not None and probe.server is None


def test_registered_http_projection_is_scraped_and_only_selected_fields_are_required(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from cases.master_ha_failover.metrics import _state_series

    requests = []
    class Master(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200 if self.server.healthy else 503)
            self.end_headers()
            # No P/D ledger: the selected metric needs only scheduler_inflight.
            self.wfile.write(b'{"scheduler_inflight":7}')
        def log_message(self, *args):
            pass
    servers, threads = [], []
    for healthy in (True, False):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Master)
        server.healthy = healthy
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        threads.append(thread)
    plan = select_plan(load_plan('master_ha_failover.yaml'), {'ha/scheduler_inflight': {}}, [])
    session = PrometheusSession(tmp_path/'telemetry/1', {}, interval_s=.1, metric_plan=plan, target_kinds={})
    probe = None
    try:
        session.start()
        env = NS(master_specs={name: NS(bind_ip='127.0.0.1', http_port=server.server_port)
                               for name, server in zip(('A', 'B'), servers)})
        probe = evidence_collector(plan, 'master_inflight', env, tmp_path/'states.jsonl',
                    session=session, limits=dict(max_samples=100, max_bytes=10000))
        probe.start()
        wait_for(lambda: probe.budget.count >= 6)
        session.stop()  # The session also owns unfinished probe cleanup/export.
        rows = [json.loads(line) for line in probe.path.read_text().splitlines()]
        series = _state_series(rows, 0, ['scheduler_inflight'])
        assert {point['y'] for point in series['A', 'scheduler_inflight']} == {7}
        assert {point['y'] for point in series['B', 'scheduler_inflight']} == {None}
        assert all(path == '/rtp_llm/inflight_status' for path in requests)
        assert set(probe.fields) == {'http_up', 'scheduler_inflight'}
        config = json.loads((session.directory/'prometheus.json').read_text())
        assert [job['job_name'] for job in config['scrape_configs']] == ['probe-master_inflight']
        assert 'prefill_inflight_requests' not in config['scrape_configs'][0]['metric_relabel_configs'][0]['regex']
    finally:
        session.stop(export=False)
        for server, thread in zip(servers, threads):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
