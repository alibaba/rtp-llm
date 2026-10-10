"""Experiment-owned Prometheus. No Python scrape loop or private time-series DB.

Prometheus is a required executable for monitored workloads (PROMETHEUS_BIN or
PATH). Each environment has an isolated TSDB and a single owner of /metrics.
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

from monitoring import telemetry
from monitoring.query_plan import DEFAULT_PLAN

def _finite(value):
    number = float(value)
    return number if math.isfinite(number) else None


def exposition(rows):
    lines = []
    for row in rows:
        labels = dict(row["metric"])
        name = labels.pop("__name__")
        for key in ("job", "instance"):
            labels.pop(key, None)
        suffix = ",".join(f"{k}={json.dumps(v)}" for k, v in sorted(labels.items()))
        lines.append(f"{name}{{{suffix}}} {row['value'][1]}")
    return "\n".join(lines) + "\n"


class PrometheusSource:
    """Compatibility view over actual scrape timestamps, without copying raw data."""

    def __init__(self, session, name, url):
        self.session, self.name, self.url = session, name, url

    def read(self, timeout=5):
        return exposition(self.session.instant(self.name, timeout=timeout))

    def samples_since(self, sequence):
        # Cursor is the actual scrape timestamp in microseconds, not query time.
        start = max(self.session.started, sequence / 1_000_000)
        end = time.time()
        output = []
        for chunk in self.session.raw(self.name, start, end):
            by_time = {}
            for row in chunk:
                for stamp, value in row["values"]:
                    cursor = round(stamp * 1_000_000)
                    if cursor > sequence:
                        by_time.setdefault(stamp, []).append(
                            dict(metric=row["metric"], value=[stamp, value])
                        )
            for stamp, rows in sorted(by_time.items()):
                up = [r for r in rows if r["metric"]["__name__"] == "up"]
                healthy = len(up) == 1 and up[0]["value"][1] == "1"
                output.append(
                    dict(
                        sequence=round(stamp * 1_000_000),
                        epoch_s=stamp,
                        monotonic_s=self.session.monotonic
                        + stamp
                        - self.session.started,
                        error=None if healthy else "Prometheus scrape failed",
                        body=(
                            exposition(
                                [r for r in rows if r["metric"]["__name__"] != "up"]
                            )
                            if healthy
                            else None
                        ),
                    )
                )
        return output


class PrometheusSession:
    def __init__(self, directory, targets, interval_s=1, max_gap_s=5, binary=None,
                 query_plan=DEFAULT_PLAN, target_kinds=None):
        from monitoring.query_plan import load_plan

        self.directory = Path(directory).resolve()
        self.targets = dict(targets)
        self.target_kinds = dict(target_kinds) if target_kinds is not None else None
        self.query_plan_name = query_plan
        self.query_plan = load_plan(query_plan)
        from monitoring.query_plan import plan_hash
        self.query_plan_sha256 = plan_hash(self.query_plan)
        self.interval, self.max_gap = float(interval_s), float(max_gap_s)
        if (
            not self.targets
            or not math.isfinite(self.max_gap)
            or not 0.001 <= self.interval <= self.max_gap
        ):
            raise ValueError("invalid Prometheus targets or sampling budget")
        self.binary = (
            binary or os.environ.get("PROMETHEUS_BIN") or shutil.which("prometheus")
        )
        self.process = None
        self.views = []
        self.started = self.monotonic = None
        self.url = None
        self.log = None
        self.target_bounds = {}

    def api(self, endpoint, timeout=5, **params):
        url = self.url + "/api/v1/" + endpoint + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.load(response)
        if data.get("status") != "success" or data.get("warnings"):
            raise RuntimeError("Prometheus query failed or incomplete: " + str(data))
        return data["data"]

    def query(self, expression, when=None, timeout=5):
        return self.api(
            "query", timeout=timeout, query=expression, time=when or time.time()
        )["result"]

    def selector(self, name):
        if name not in self.targets:
            raise ValueError("unknown monitored source " + name)
        return "{job=" + json.dumps(name) + "}"

    def instant(self, name, timeout=5):
        selector = self.selector(name)
        now = time.time()
        health = self.query("up" + selector, now, timeout)
        if len(health) != 1 or health[0]["value"][1] != "1":
            raise RuntimeError("Prometheus target unavailable: " + name)
        rows = self.query(f"{selector}[{round(self.max_gap * 1000)}ms]", now, timeout)
        if not any(r["metric"].get("__name__") == "up" for r in rows):
            raise RuntimeError("Prometheus target stale: " + name)
        latest_scrape = max(
            r["values"][-1][0] for r in rows if r["metric"].get("__name__") == "up"
        )
        return [
            dict(metric=r["metric"], value=r["values"][-1])
            for r in rows
            if r["values"][-1][0] == latest_scrape
            and not r["metric"]["__name__"].startswith(("up", "scrape_"))
        ]

    def raw(self, name, start, end):
        # Range-vector queries preserve scrape timestamps; query_range would
        # repeat old gauges via lookback and hide gaps. Bound each response.
        cursor = start
        while cursor < end:
            right = min(end, cursor + 60)
            span = max(0.001, right - cursor)
            yield self.query(
                f"{self.selector(name)}[{math.ceil(span * 1000)}ms]", right
            )
            cursor = right

    def add_target(self, name, url):
        self.add_targets({name: url})

    def add_targets(self, targets):
        if set(targets) & set(self.targets):
            raise ValueError("duplicate monitor target")
        path = self.directory / "prometheus.json"
        config = json.loads(path.read_text())
        for name, url in targets.items():
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme != "http" or not parsed.hostname or parsed.username:
                raise ValueError("invalid client exporter URL")
            config["scrape_configs"].append(
                dict(
                    job_name=name,
                    metrics_path=parsed.path,
                    sample_limit=100000,
                    body_size_limit="32MB",
                    static_configs=[dict(targets=[parsed.netloc])],
                )
            )
        self.targets.update(targets)
        if self.target_kinds is not None:
            self.target_kinds.update({name: "client" for name in targets})
        path.write_text(json.dumps(config))
        request = urllib.request.Request(
            self.url + "/-/reload", data=b"", method="POST"
        )
        with urllib.request.urlopen(request, timeout=5):
            pass
        deadline = time.monotonic() + 15
        while True:
            try:
                for name in targets:
                    self.instant(name)
                break
            except (OSError, RuntimeError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("client monitor targets not ready")
                time.sleep(0.1)
        for name in targets:
            self.target_bounds[name] = [time.time(), None]

    def end_target(self, name, ended=None):
        self.target_bounds[name][1] = ended or time.time()

    def start(self):
        if not self.binary:
            raise RuntimeError(
                "Prometheus is required; install it or set PROMETHEUS_BIN (no collector fallback)"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "data").exists():
            raise ValueError("Prometheus directory must be fresh")
        jobs = []
        for name, url in self.targets.items():
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme != "http" or not parsed.hostname or parsed.username:
                raise ValueError("monitor target must be an explicit http endpoint")
            jobs.append(
                dict(
                    job_name=name,
                    sample_limit=100000,
                    body_size_limit="32MB",
                    metrics_path=parsed.path or "/metrics",
                    params=urllib.parse.parse_qs(parsed.query),
                    static_configs=[dict(targets=[parsed.netloc])],
                )
            )
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
        self.url = f"http://127.0.0.1:{port}"
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
            end = time.monotonic() + 15
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "Prometheus exited; inspect "
                        + str(self.directory / "prometheus.log")
                    )
                try:
                    with urllib.request.urlopen(self.url + "/-/ready", timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= end:
                        raise TimeoutError("Prometheus not ready")
                    time.sleep(0.05)
            with telemetry._REGISTRY_LOCK:
                if any(url in telemetry._REGISTRY for url in self.targets.values()):
                    raise RuntimeError("monitor endpoint already has an owner")
                for name, url in self.targets.items():
                    view = PrometheusSource(self, name, url)
                    telemetry._REGISTRY[url] = view
                    self.views.append(view)
            # Discovery/scrape startup is preflight, before experiment traffic.
            end = time.monotonic() + 15
            while True:
                try:
                    for name in self.targets:
                        self.instant(name)
                    break
                except (OSError, RuntimeError):
                    if time.monotonic() >= end:
                        raise TimeoutError("Prometheus targets not ready")
                    time.sleep(0.1)
            self.started, self.monotonic = time.time(), time.monotonic()
            (self.directory / "session.json").write_text(
                json.dumps(
                    dict(
                        backend="prometheus",
                        url=self.url,
                        targets=self.targets,
                        started_epoch_s=self.started,
                        interval_s=self.interval,
                        retention_time="24h",
                        retention_size="1GB",
                    ),
                    indent=2,
                )
            )
        except BaseException:
            self.stop(export=False)
            raise
        return self

    def metric_definition(self, metric_id):
        from monitoring.query_plan import definitions
        from monitoring.metric_store import MetricContractError
        definition = definitions(self.query_plan).get(metric_id)
        if definition is None or "promql" not in definition:
            raise MetricContractError("undeclared Prometheus metric: " + metric_id)
        return definition

    def metric_rows(self, metric_id, *, source, start, end):
        """Read a declared raw metric with actual scrape times, never lookback filling."""
        definition = self.metric_definition(metric_id)
        if definition.get("mode") != "scrape":
            raise ValueError("raw gate input requires scrape mode: " + metric_id)
        if self.target_kind(source) != definition["source_kind"]:
            raise ValueError("metric source kind mismatch: " + metric_id)
        if not math.isfinite(start) or not math.isfinite(end) or start >= end:
            raise ValueError("invalid metric query window")
        expression = definition["promql"].replace("${selector}", self.selector(source))
        rows, cursor = [], start
        while cursor < end:
            right = min(end, cursor + 60)
            rows.extend(self.query(expression + f"[{math.ceil((right - cursor) * 1000)}ms]", right))
            cursor = right
        for row in rows:
            if set(definition["labels"]) - set(row["metric"]):
                raise ValueError("metric lacks required labels: " + metric_id)
            row["metric_id"] = metric_id
        return rows

    def metric_snapshot(self, metric_ids, *, source, timeout=5):
        rows = self.instant(source, timeout)
        result = []
        for identity in metric_ids:
            definition = self.metric_definition(identity)
            if definition.get("mode") != "scrape" or self.target_kind(source) != definition["source_kind"]:
                raise ValueError("snapshot requires a raw metric with matching source kind")
            name = definition["promql"].split("${selector}")[0]
            for row in rows:
                if row["metric"].get("__name__") == name:
                    if set(definition["labels"]) - set(row["metric"]):
                        raise ValueError("metric lacks required labels: " + identity)
                    result.append(dict(row, metric_id=identity))
        return result

    def target_kind(self, source):
        if source not in self.targets:
            raise ValueError("unknown monitored source: " + source)
        if self.target_kinds is not None:
            return self.target_kinds[source]
        return "mock" if source == "mock" else "client" if source.startswith("client-") else "master"

    def archive(self):
        end = time.time()
        from monitoring.query_plan import queries_for_targets

        queries, required_queries = queries_for_targets(
            self.query_plan, self.targets, self.selector, self.interval,
            self.target_kinds,
        )
        result = dict(
            missing_queries=[],
            query_plan=self.query_plan_name,
            query_plan_sha256=self.query_plan_sha256,
            backend="prometheus",
            metric_plan=self.query_plan,
            target_kinds={source: self.target_kind(source) for source in self.targets},
            start=self.started,
            end=end,
            step=self.interval,
            targets=self.targets,
            target_bounds=self.target_bounds,
            queries={},
            errors=[],
        )
        for key, expression in queries.items():
            rows = []
            # API query_range has a maximum point count; chunk long experiments.
            start, target_end = self.target_bounds.get(
                key.split("/", 1)[0], [self.started, end]
            )
            query_end = target_end or end
            cursor = start
            try:
                while cursor <= query_end:
                    right = min(query_end, cursor + self.interval * 1000)
                    source, metric = key.split("/", 1)
                    definition = self.query_plan["sources"][self.target_kind(source)].get(metric, {})
                    if definition.get("mode") == "scrape":
                        if right <= cursor:
                            break
                        rows.extend(self.metric_rows(self.target_kind(source) + "/" + metric,
                                                     source=source, start=cursor, end=right))
                    else:
                        data = self.api("query_range", query=expression, start=cursor,
                                        end=right, step=self.interval)
                        rows.extend(data["result"])
                    cursor = right if definition.get("mode") == "scrape" and right < query_end else right + self.interval
                result["queries"][key] = dict(promql=expression, result=rows)
                if not rows:
                    result["missing_queries"].append(key)
                    if key in required_queries:
                        result["errors"].append(
                            dict(query=key, error="required monitor series absent")
                        )
            except Exception as exc:
                result["errors"].append(dict(query=key, error=str(exc)))
        (self.directory / "queries.json").write_text(
            json.dumps(result, separators=(",", ":"))
        )
        from monitoring.metric_store import export_metrics
        root = self.directory.parent.parent if self.directory.parent.name == "telemetry" else self.directory
        export_metrics(root)
        if result["errors"]:
            raise RuntimeError("incomplete Prometheus archive")

    def stop(self, timeout=10, export=True):
        try:
            if export and self.process is not None and self.process.poll() is None:
                self.archive()
        finally:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=max(0.1, timeout))
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                    raise TimeoutError("Prometheus exceeded shutdown budget")
            with telemetry._REGISTRY_LOCK:
                for view in self.views:
                    if telemetry._REGISTRY.get(view.url) is view:
                        del telemetry._REGISTRY[view.url]
            if self.log:
                self.log.close()



def archived_series(directory, anchor):
    from monitoring.metric_store import MetricStore
    return MetricStore.read(directory).series(anchor)
