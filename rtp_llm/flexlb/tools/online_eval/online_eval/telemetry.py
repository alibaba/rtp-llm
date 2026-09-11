"""One owner for destructive mock metric reads; consumers share its samples.

Registry entries are scoped to a workload environment and removed on cleanup.
A registered but unavailable source never falls back to another HTTP scrape.
"""

import json
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

_REGISTRY = {}
_REGISTRY_LOCK = threading.Lock()


def http_text(url, timeout=5.0):
    with _REGISTRY_LOCK:
        owner = _REGISTRY.get(url)
    if owner is not None:
        return owner.read(timeout)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def shared_samples_since(url, sequence):
    """None means unowned; an empty list means no new source sample."""
    with _REGISTRY_LOCK:
        owner = _REGISTRY.get(url)
    if owner is None:
        return None
    with owner.condition:
        if owner.samples and sequence < owner.samples[0]["sequence"] - 1:
            return owner.replay(sequence, owner.sequence)
        return [
            dict(sample) for sample in owner.samples if sample["sequence"] > sequence
        ]


class SharedMetricSource:
    def __init__(self, url, directory, interval_s, fetch=None, history_limit=128):
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError("history_limit must be a positive integer")
        self.url, self.interval = url, interval_s
        self.directory = Path(directory)
        self.fetch = fetch or self._fetch
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.stopping = False
        self.body = None
        self.error = None
        self.fatal_error = None
        self.sequence = 0
        self.samples = deque(maxlen=history_limit)
        self.thread = threading.Thread(
            target=self._run, name="shared-mock-telemetry", daemon=True
        )

    def _fetch(self):
        # Deliberately bypass the registry: only this owner touches the endpoint.
        with urllib.request.urlopen(self.url, timeout=2) as response:
            return response.read().decode("utf-8", "replace")

    def replay(self, sequence, through):
        """Stream evicted samples from the journal without another HTTP read."""
        with (self.directory / "mock.prom").open("rb") as raw, (
            self.directory / "mock-samples.jsonl"
        ).open() as journal:
            for line in journal:
                sample = json.loads(line)
                if sample["sequence"] <= sequence:
                    continue
                if sample["sequence"] > through:
                    break
                sample["body"] = None
                if sample["error"] is None:
                    raw.seek(sample["raw_offset"])
                    body = raw.read(sample["raw_size"])
                    if len(body) != sample["raw_size"]:
                        raise RuntimeError("shared telemetry raw sample truncated")
                    sample["body"] = body.decode("utf-8")
                yield sample

    def start(self):
        with _REGISTRY_LOCK:
            if self.url in _REGISTRY:
                raise RuntimeError("metric source already owned: " + self.url)
            _REGISTRY[self.url] = self
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.thread.start()
        except BaseException:
            with _REGISTRY_LOCK:
                _REGISTRY.pop(self.url, None)
            raise

    def read(self, timeout):
        with self.condition:
            if self.sequence == 0:
                self.condition.wait_for(
                    lambda: self.sequence or self.stop_event.is_set(), timeout
                )
            if self.error is not None:
                raise RuntimeError("shared telemetry unavailable: " + self.error)
            if self.body is None:
                raise TimeoutError("shared telemetry has no sample")
            return self.body

    def _run(self):
        try:
            with (self.directory / "mock.prom").open("a") as out, (
                self.directory / "mock-samples.jsonl"
            ).open("a") as journal:
                while not self.stop_event.is_set():
                    started, epoch = time.monotonic(), time.time()
                    body, error = None, None
                    raw_offset, raw_size = None, None
                    try:
                        body = self.fetch()
                        out.write(f"# ts={int(epoch * 1000)}\n")
                        raw_offset = out.tell()
                        raw_size = len(body.encode("utf-8"))
                        out.write(body + "\n")
                        out.flush()
                    except Exception as exc:
                        error = repr(exc)
                    with self.condition:
                        self.sequence += 1
                        self.body, self.error = body, error
                        self.samples.append(
                            dict(
                                sequence=self.sequence,
                                body=body,
                                error=error,
                                monotonic_s=started,
                                epoch_s=epoch,
                            )
                        )
                        journal.write(
                            json.dumps(
                                dict(
                                    sequence=self.sequence,
                                    epoch_s=epoch,
                                    monotonic_s=started,
                                    error=error,
                                    raw_offset=raw_offset,
                                    raw_size=raw_size,
                                )
                            )
                            + "\n"
                        )
                        journal.flush()
                        self.condition.notify_all()
                    self.stop_event.wait(
                        max(0, self.interval - (time.monotonic() - started))
                    )
        except BaseException as exc:
            with self.condition:
                self.error = repr(exc)
                self.fatal_error = self.error
                self.sequence += 1
                self.condition.notify_all()
        finally:
            # Retain ownership while an outstanding destructive read is alive.
            # A timed-out stop must still release it when that read exits.
            if self.stopping:
                with _REGISTRY_LOCK:
                    if _REGISTRY.get(self.url) is self:
                        del _REGISTRY[self.url]

    def stop(self, timeout):
        self.stopping = True
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError("shared telemetry did not stop")
        with _REGISTRY_LOCK:
            if _REGISTRY.get(self.url) is self:
                del _REGISTRY[self.url]
        if self.fatal_error is not None:
            raise RuntimeError(
                "shared telemetry collector terminated: " + self.fatal_error
            )
