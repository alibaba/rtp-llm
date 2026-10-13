"""Experiment-owned Prometheus. No Python scrape loop or private time-series DB.

Prometheus is required when a plan selects time-series queries (PROMETHEUS_BIN
or PATH); evidence-only plans do not launch it. Each environment has an isolated TSDB and a single owner of /metrics.
Live adapters read the TSDB, never the exporter. Query results carry their
PromQL, evaluation bounds and source; absent/stale data is never replaced by 0.
"""

import json
import math
import os
import shutil
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from artifacts.json_io import write_json
from monitoring.query_plan import DEFAULT_PLAN, load_plan, plan_hash
from monitoring.client import PrometheusClient
from monitoring.archive import export_metrics
from monitoring.recording import ArchiveWindow, record_queries
from runtime.deadline import Deadline
from runtime.cleanup import cleanup_all
from runtime.startup import startup_deadline, wait_ready
from monitoring.collection_plan import scrape_job, collection_plan


class PrometheusSession:
    def __init__(self, directory, targets, interval_s=1, max_gap_s=5, binary=None,
                 query_plan=DEFAULT_PLAN, target_kinds=None, metric_plan=None):

        self.directory = Path(directory).resolve()
        self.query_plan_name = query_plan
        self.query_plan = metric_plan if metric_plan is not None else load_plan(query_plan)
        kinds = dict(target_kinds) if target_kinds is not None else {}
        if set(kinds) != set(targets) or set(kinds.values()) - set(self.query_plan["sources"]):
            raise ValueError("target kinds must identify every Prometheus target")
        self.targets = {name: url for name, url in targets.items() if self.collects(kinds[name])}
        self.target_kinds = {name: kinds[name] for name in self.targets}
        self.collection_plan = collection_plan(self.query_plan)
        self.query_plan_sha256 = plan_hash(self.query_plan)
        self.interval, self.max_gap = float(interval_s), float(max_gap_s)
        if (
            not math.isfinite(self.max_gap)
            or not 0.001 <= self.interval <= self.max_gap
        ):
            raise ValueError("invalid Prometheus targets or sampling budget")
        self.binary = (
            binary or os.environ.get("PROMETHEUS_BIN") or shutil.which("prometheus")
        )
        self.process = None
        self.started = self.monotonic = None
        self.client = PrometheusClient(None, query_plan=self.query_plan,
                                       target_kinds=self.target_kinds, max_gap_s=self.max_gap)
        self.log = None
        self.target_bounds = {}
        self.closed = False
        self.probes = {}
        self.registration_error = None

    def collects(self, kind):
        return bool(self.query_plan["sources"][kind])

    def add_client_target(self, name, url, *, deadline=None):
        self.add_client_targets({name: url}, deadline=deadline)

    def add_client_targets(self, targets, *, deadline=None):
        deadline = startup_deadline(deadline, 20)
        if self.closed or self.registration_error is not None:
            raise RuntimeError("monitor session cannot register targets after closure or rollback failure")
        if set(targets) & set(self.targets):
            raise ValueError("duplicate monitor target")
        selected = {name: url for name, url in targets.items() if self.collects("client")}
        if not selected:
            return
        path = self.directory / "prometheus.json"
        previous_config = json.loads(path.read_text())
        config = dict(previous_config, scrape_configs=list(previous_config["scrape_configs"]))
        jobs = [scrape_job(self.query_plan, name, url, "client") for name, url in selected.items()]
        config["scrape_configs"].extend(jobs)
        previous_targets, previous_kinds = dict(self.targets), dict(self.target_kinds)
        self.targets.update(selected)
        self.target_kinds.update({name: "client" for name in selected})
        reload_attempted = False
        try:
            write_json(path, config)
            reload_attempted = True
            self._reload(deadline)
            self._wait_targets(selected, deadline, 'client metrics')
        except BaseException:
            # Preserve shared inventory objects used by the query client.
            self.targets.clear()
            self.targets.update(previous_targets)
            self.target_kinds.clear()
            self.target_kinds.update(previous_kinds)
            try:
                write_json(path, previous_config)
                if reload_attempted:
                    self._reload(deadline)
            except BaseException as rollback_error:
                self.registration_error = rollback_error
                raise RuntimeError("monitor target registration rollback failed; stop the session") from rollback_error
            raise
        for name in selected:
            self.target_bounds[name] = [time.time(), None]

    def _reload(self, deadline):
        request = urllib.request.Request(self.client.url + "/-/reload", data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=min(5, deadline.remaining())):
            pass
        deadline.check()

    def add_probe(self, probe, url, *, deadline=None):
        """A registered on-demand adapter shares this run's scheduler and TSDB."""
        deadline = startup_deadline(deadline, 20)
        if not self.client.url or self.closed or self.registration_error is not None or probe.job in self.probes:
            raise ValueError("probe requires a live Prometheus and a unique source")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("probe requires an HTTP(S) URL without credentials")
        self.probes[probe.job] = probe
        path = self.directory / "prometheus.json"
        config = json.loads(path.read_text())
        config["scrape_configs"].append(dict(job_name=probe.job, scheme=parsed.scheme, metrics_path=parsed.path or "/metrics", sample_limit=100000,
            body_size_limit="32MB", static_configs=[dict(targets=[parsed.netloc])],
            metric_relabel_configs=[dict(source_labels=["__name__"], action="keep",
                regex="(" + "|".join(probe.names.values()) + ")")]))
        path.write_text(json.dumps(config))
        with urllib.request.urlopen(urllib.request.Request(self.client.url + "/-/reload", data=b"", method="POST"), timeout=min(5, deadline.remaining())):
            pass
        self._write_session()
        def ready():
            if probe.error is not None:
                raise RuntimeError('probe startup failed: ' + str(probe.error)) from probe.error
            health = self.client.query('up{job=' + json.dumps(probe.job) + '}', timeout=min(5, deadline.remaining()))
            return len(health) == 1 and float(health[0]['value'][1]) == 1
        wait_ready(deadline, 'probe/first scrape', ready, interval_s=0.05)

    def end_target(self, name, ended=None):
        self.target_bounds[name][1] = ended or time.time()

    def _wait_targets(self, targets, deadline, phase):
        deadline = startup_deadline(deadline, 15)
        def ready():
            try:
                for name in targets:
                    self.client.instant(name, timeout=min(5, deadline.remaining()))
                return True
            except TimeoutError:
                raise
            except (OSError, RuntimeError):
                return False
        wait_ready(deadline, phase, ready, interval_s=0.1)

    def start(self, *, deadline=None):
        deadline = startup_deadline(deadline, 30)
        deadline.check()
        if self.started is not None:
            raise ValueError("monitor session already started")
        if not any(self.query_plan["sources"].values()) and not self.collection_plan["evidence"]:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.started, self.monotonic = time.time(), time.monotonic()
            self._write_session()
            return self
        if not self.binary:
            raise RuntimeError(
                "Prometheus is required; install it or set PROMETHEUS_BIN (no collector fallback)"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "data").exists():
            raise ValueError("Prometheus directory must be fresh")
        jobs = [scrape_job(self.query_plan, name, url, self.client.target_kind(name))
                for name, url in self.targets.items()]
        config = {
            "global": {
                "scrape_interval": f"{round(self.interval * 1000)}ms",
                "scrape_timeout": f"{round(min(self.interval, 2) * 1000)}ms",
            },
            "scrape_configs": jobs,
        }
        (self.directory / "prometheus.json").write_text(json.dumps(config, indent=2))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.client.url = f"http://127.0.0.1:{port}"
        self.started, self.monotonic = time.time(), time.monotonic()
        self.log = (self.directory / "prometheus.log").open("w")
        self.process = subprocess.Popen(
            [
                self.binary,
                "--config.file=" + str(self.directory / "prometheus.json"),
                "--storage.tsdb.path=" + str(self.directory / "data"),
                "--storage.tsdb.retention.time=24h",
                "--storage.tsdb.retention.size=1GB",
                "--web.listen-address=127.0.0.1:" + str(port),
                "--web.enable-lifecycle",
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = startup_deadline(deadline, 30)
            def ready():
                if self.process.poll() is not None:
                    raise RuntimeError('Prometheus exited; inspect ' + str(self.directory / 'prometheus.log'))
                try:
                    with urllib.request.urlopen(self.client.url + '/-/ready', timeout=min(1, deadline.remaining())):
                        return True
                except TimeoutError:
                    raise
                except OSError:
                    return False
            wait_ready(deadline, 'prometheus/http', ready, interval_s=0.05)
            self._wait_targets(self.targets, deadline, 'prometheus/first scrape')
            self.started, self.monotonic = time.time(), time.monotonic()
            self._write_session()
        except BaseException:
            self.stop(export=False)
            raise
        return self

    def _write_session(self):
        (self.directory / "session.json").write_text(json.dumps(dict(
            backend="prometheus", url=self.client.url, targets=self.targets,
            collection_plan=self.collection_plan,
            evidence_targets={name: dict(source=probe.source, fields=probe.fields, url=probe.url, path=str(probe.path))
                              for name, probe in self.probes.items()}, started_epoch_s=self.started,
            interval_s=self.interval, retention_time="24h", retention_size="1GB"), indent=2))

    def archive(self, *, materialize=True, deadline=None):
        window = ArchiveWindow(self.directory, self.query_plan_name, self.collection_plan,
                               self.started, self.interval, dict(self.targets),
                               {name: list(bounds) for name, bounds in self.target_bounds.items()})
        result = record_queries(window, self.client, deadline=deadline)
        if self.registration_error is not None:
            result["errors"].append(dict(error="monitor target registration rollback failed: "
                                               + str(self.registration_error)))
            write_json(self.directory / "queries.json", result)
        if materialize:
            if deadline is not None:
                deadline.check()
            root = self.directory.parent.parent if self.directory.parent.name == "telemetry" else self.directory
            export_metrics(root)
            if deadline is not None:
                deadline.check()
        if result["errors"]:
            if deadline is not None:
                deadline.check()
            raise RuntimeError("incomplete Prometheus archive")

    def stop(self, timeout=10, export=True, *, materialize=True,
             archive_timeout=None, deadline=None):
        # The enclosing cleanup budget bounds the whole operation. Query export
        # cannot spend the time reserved for terminating and reaping Prometheus.
        archive_timeout = timeout if archive_timeout is None else archive_timeout
        clock = deadline.clock if deadline is not None else time.monotonic

        def bounded(seconds, reserve=0):
            expires = clock() + max(0, seconds)
            if deadline is not None:
                expires = min(expires, deadline.expires_at - reserve)
            return Deadline(expires, clock=clock,
                            cancelled=deadline.cancelled if deadline is not None else None)

        probe_deadline = bounded(timeout)
        operations = [(name, lambda probe=probe: probe.stop(deadline=probe_deadline))
                      for name, probe in self.probes.items()]
        if not self.closed and export and (self.process is not None and self.process.poll() is None
                or self.started is not None and not any(self.query_plan["sources"].values())):
            operations.append(("Prometheus archive", lambda: self.archive(
                materialize=materialize, deadline=bounded(archive_timeout, reserve=timeout))))
        # Construct this deadline when shutdown starts, after archive succeeds
        # or fails, so query time never consumes the process shutdown allowance.
        operations.append(("Prometheus process", lambda: self._stop_process(bounded(timeout))))
        if self.log:
            operations.append(("Prometheus log", self.log.close))
        try:
            cleanup_all(operations)
        finally:
            self.closed = True

    def _stop_process(self, deadline):
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=deadline.remaining())
        except (subprocess.TimeoutExpired, TimeoutError):
            self.process.kill()
            # Reap after escalation even if the cooperative budget has expired.
            self.process.wait(timeout=1)
            raise TimeoutError("Prometheus exceeded shutdown budget")
