"""HA Master state sampling and ledger projection."""

import math
import threading
import time
from pathlib import Path

from runtime.network import http_get_json, master_url
from runtime.observation import ObservationClock, SampleBudget


def master_state_fields(data):
    """Project a successful response; absent ledger fields are contract errors."""
    def number(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("HA state requires finite nonnegative ledger values")
        return value

    def total(endpoints, field):
        if not isinstance(endpoints, list):
            raise ValueError("HA state endpoint ledger must be a list")
        if any(not isinstance(endpoint, dict) or field not in endpoint for endpoint in endpoints):
            raise ValueError("HA state endpoint lacks " + field)
        return sum(number(endpoint[field]) for endpoint in endpoints)

    required = {"scheduler_inflight", "prefill_endpoints", "decode_endpoints"}
    if not isinstance(data, dict) or not required <= set(data):
        raise ValueError("HA state response lacks required ledger fields")
    return dict(scheduler_inflight=number(data["scheduler_inflight"]),
                prefill_inflight_requests=total(data["prefill_endpoints"], "inflight_requests"),
                decode_master_queued=total(data["decode_endpoints"], "master_queued"),
                decode_confirmed_running=total(data["decode_endpoints"], "confirmed_running"))


class HaMasterStateSampler:
    """Record both Masters' HTTP inflight state during the traffic window."""

    def __init__(self, env, path: Path, interval_s: float = 1.0, *,
                 limits, clock=time.monotonic, wall_clock=time.time):
        self.path = path
        self.urls = {
            name: master_url(spec.bind_ip, spec.http_port, "inflight")
            for name, spec in env.master_specs.items()
        }
        self.clock, self.wall_clock = clock, wall_clock
        self.budget = SampleBudget(limits)
        self.error = None
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="ha-master-state", daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise TimeoutError("HA Master state sampler did not stop")
            if self.error is not None:
                raise RuntimeError("HA Master state sampling failed: " + self.error)

    def _run(self):
        anchor = ObservationClock(self.wall_clock(), self.clock())
        try:
            with self.path.open("w", encoding="utf-8") as stream:
                while not self._stop.is_set():
                    started = self.clock()
                    for name, url in self.urls.items():
                        if self._stop.is_set():
                            break
                        data = http_get_json(url, timeout=min(0.4, self.interval_s))
                        elapsed = self.clock() - anchor.origin_monotonic_s
                        row = dict(epoch_s=anchor.origin_epoch_s + elapsed, elapsed_s=elapsed,
                                   monotonic_s=self.clock(), master=name, http_up=int(data is not None))
                        if data is not None:
                            try:
                                row.update(master_state_fields(data))
                            except ValueError as exc:
                                row["state_error"] = str(exc)
                        stream.write(self.budget.append(row) + "\n")
                    stream.flush()
                    self._stop.wait(max(0, self.interval_s - (self.clock() - started)))
        except Exception as exc:
            self.error = str(exc)
            self._stop.set()


