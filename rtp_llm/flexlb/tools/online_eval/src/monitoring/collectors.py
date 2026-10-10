"""Owned bounded evidence acquisition for exceptional non-Prometheus sources.

Adapters perform one bounded read. They never schedule, spawn threads or write
artifacts. Ordinary time series remain owned by PrometheusSession.
"""

import math
import json
import threading
import time
import urllib.request
from pathlib import Path

from runtime.observation import ObservationClock, SampleBudget


class HttpJsonAdapter:
    """One bounded read; transport outages and malformed payloads are distinct.

    Only protocols where unavailability is an observation provide unavailable().
    Otherwise a failed request aborts collection. Parsing errors always fail.
    """

    def __init__(self, url, project, unavailable=None, *, max_response_bytes=32 * 1024 * 1024):
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("HTTP response byte budget must be positive")
        self.url, self.project, self.unavailable = url, project, unavailable
        self.max_response_bytes = max_response_bytes

    def __call__(self, *, timeout):
        try:
            with urllib.request.urlopen(self.url, timeout=timeout) as response:
                body = response.read(self.max_response_bytes + 1)
                if len(body) > self.max_response_bytes:
                    raise ValueError("HTTP evidence response exceeds byte budget")
                data = json.loads(body)
        except OSError:
            if self.unavailable is None:
                raise
            return self.unavailable()
        return self.project(data)


class EvidenceCollector:
    def __init__(self, path, adapters, *, limits, interval_s=1, timeout_s=.4,
                 clock=time.monotonic, wall_clock=time.time):
        if (not adapters or any(not callable(read) for read in adapters.values())
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in (interval_s, timeout_s))):
            raise ValueError("invalid evidence adapters or sampling budget")
        self.path = Path(path)
        self.adapters = dict(adapters)
        self.interval_s, self.timeout_s = interval_s, min(timeout_s, interval_s)
        self.clock, self.wall_clock = clock, wall_clock
        self.budget = SampleBudget(limits)
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None or self.path.exists():
            raise ValueError("evidence collector requires a fresh resource and artifact")
        # Create the artifact before starting so path/permission errors fail startup.
        self.path.touch(exist_ok=False)
        self._thread = threading.Thread(target=self._run, name="evidence-collector", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout=5):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0, timeout))
            if self._thread.is_alive():
                raise TimeoutError("evidence collector did not stop")
        if self.error is not None:
            raise RuntimeError("evidence collection failed: " + str(self.error)) from self.error

    def _run(self):
        anchor = ObservationClock(self.wall_clock(), self.clock())
        try:
            with self.path.open("w", encoding="utf-8") as stream:
                while not self._stop.is_set():
                    started = self.clock()
                    for read in self.adapters.values():
                        if self._stop.is_set():
                            break
                        values = read(timeout=self.timeout_s)
                        if (not isinstance(values, dict)
                                or set(values) & {"epoch_s", "elapsed_s", "monotonic_s"}):
                            raise ValueError("evidence adapter returned invalid/reserved fields")
                        now = self.clock()
                        row = dict(epoch_s=anchor.origin_epoch_s + now - anchor.origin_monotonic_s,
                                   elapsed_s=now - anchor.origin_monotonic_s,
                                   monotonic_s=now, **values)
                        stream.write(self.budget.append(row) + "\n")
                    stream.flush()
                    self._stop.wait(max(0, self.interval_s - (self.clock() - started)))
        except Exception as exc:
            self.error = exc
            self._stop.set()
