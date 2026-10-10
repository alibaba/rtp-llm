"""Registered protocol adapters scraped on demand by the owned Prometheus.

The official SDK serves exposition; Prometheus owns scheduling and samples.
The evidence artifact is exported from raw TSDB timestamps, never a second
polling loop, lookback evaluation, or a private time-series database.
"""

import json
import math
import threading
import time
from pathlib import Path

from runtime.observation import SampleBudget


class PrometheusEvidence:
    def __init__(self, path, adapters, *, fields, label, source, session, limits,
                 timeout_s=.4):
        if not adapters or not fields or not 0 < timeout_s < float('inf'):
            raise ValueError('invalid probe adapters or budget')
        self.path, self.adapters = Path(path), dict(adapters)
        self.fields, self.label, self.source = sorted(fields), label, source
        self.session = session
        self.timeout_s = min(timeout_s, session.interval / (len(adapters) + 1))
        self.budget = SampleBudget(limits)
        self.error = None
        self.server = self.thread = None
        self.started = self.ended = None
        self.closed = False
        self.lock = threading.Lock()
        self.names = {field: 'online_eval_' + source + '_' + field for field in self.fields}
        self.job = 'probe-' + source
        self.url = None

    def describe(self):
        # Registration must not perform an HTTP read.
        return []

    def collect(self):
        from prometheus_client.core import GaugeMetricFamily
        with self.lock:
            if self.closed:
                return
            if self.error is not None:
                raise RuntimeError('probe failed') from self.error
            try:
                families = {field: GaugeMetricFamily(name, 'Observed ' + field,
                            labels=[self.label]) for field, name in self.names.items()}
                for name, read in self.adapters.items():
                    row = read(timeout=self.timeout_s)
                    if (not isinstance(row, dict) or row.get(self.label) != name
                            or set(row) - {self.label} - set(self.fields)):
                        raise ValueError('probe adapter returned invalid fields or identity')
                    self.budget.append(dict(row, epoch_s=time.time()))
                    for field, value in row.items():
                        if field == self.label:
                            continue
                        if type(value) not in (int, float) or not math.isfinite(value):
                            raise ValueError('probe requires finite numeric samples')
                        families[field].add_metric([name], value)
                # Buffer all families: malformed data cannot expose half a scrape.
                yield from families.values()
            except Exception as exc:
                self.error = exc
                raise

    def start(self):
        if self.started is not None or self.closed or self.path.exists():
            raise ValueError('probe requires a fresh resource and artifact')
        from prometheus_client import CollectorRegistry, start_http_server
        registry = CollectorRegistry(auto_describe=False)
        registry.register(self)
        self.path.touch(exist_ok=False)
        self.server, self.thread = start_http_server(0, addr='127.0.0.1', registry=registry)
        self.started = time.time()
        try:
            self.url = f'http://127.0.0.1:{self.server.server_port}/metrics'
            self.session.add_probe(self, self.url)
        except BaseException as exc:
            self.error = self.error or exc
            self.closed = True
            self._close_server()
            raise
        return self

    def _close_server(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise TimeoutError('probe HTTP server did not stop')
            self.server = None

    def stop(self, timeout=5):
        if self.closed:
            if self.error is not None:
                raise RuntimeError('probe collection failed: ' + str(self.error)) from self.error
            return
        if not self.lock.acquire(timeout=max(0, timeout)):
            raise TimeoutError('probe adapter did not finish')
        try:
            self.closed, self.ended = True, time.time()
        finally:
            self.lock.release()
        try:
            if self.started is not None:
                self._export(time.monotonic() + timeout)
        except Exception as exc:
            self.error = self.error or exc
            raise
        finally:
            self._close_server()
        if self.error is not None:
            raise RuntimeError('probe collection failed: ' + str(self.error)) from self.error

    def _export(self, deadline):
        """Preserve real scrape timestamps and missing fields in a frozen artifact."""
        from monitoring.metric_store import atomic_json
        from importlib.metadata import version
        samples, queries = {}, []
        cursor = self.started
        while cursor < self.ended:
            right = min(cursor + 60, self.ended)
            expression = '{job=' + json.dumps(self.job) + '}[' + str(math.ceil((right-cursor)*1000)) + 'ms]'
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("probe archive exceeded shutdown budget")
            result = self.session.query(expression, right, timeout=min(5, remaining))
            queries.append(dict(promql=expression, end=right, result=result))
            for series in result:
                metric = series['metric']['__name__']
                if metric == 'up':
                    if any(float(value) != 1 for _, value in series['values']):
                        self.error = self.error or ValueError('probe exporter scrape failed')
                    continue
                field = next((key for key, name in self.names.items() if name == metric), None)
                if field is None:
                    continue
                identity = series['metric'][self.label]
                for timestamp, value in series['values']:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError('non-finite probe archive')
                    row = samples.setdefault((timestamp, identity), dict(epoch_s=timestamp,
                                                                       **{self.label: identity}))
                    if field in row and row[field] != number:
                        raise ValueError('conflicting probe samples')
                    row[field] = number
            cursor = right
        if not samples and self.error is None:
            self.error = ValueError('probe archive contains no samples')
        frozen = dict(sdk=dict(name='prometheus-client', version=version('prometheus-client')), transport='prometheus', source=self.source, fields=self.fields,
                      job=self.job, start=self.started, end=self.ended,
                      queries=queries, source_endpoints={name: getattr(read, 'url', None)
                          for name, read in self.adapters.items()}, error=str(self.error) if self.error else None)
        atomic_json(self.path.with_suffix('.prometheus.json'), frozen)
        with self.path.open('w') as stream:
            for key in sorted(samples):
                stream.write(json.dumps(samples[key], allow_nan=False) + '\n')
