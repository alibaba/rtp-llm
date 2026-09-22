"""Experiment-owned Prometheus. No Python scrape loop or private time-series DB.

Prometheus is a required executable for monitored workloads (PROMETHEUS_BIN or
PATH). Each environment has an isolated TSDB and a single owner of /metrics.
Live adapters read the TSDB, never the exporter. Query results carry their
PromQL, evaluation bounds and source; absent/stale data is never replaced by 0.
"""

import json
import hashlib
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

# Deliberately use the exporter contract, not similarly named debug fields.
ENGINE_FIELDS = {
    "running": "rtp_llm_running_stream_size",
    "waiting": "rtp_llm_wait_stream_size",
    "hit_tokens_total": "mock_hit_tokens_total",
    "context_tokens_total": "mock_context_tokens_total",
    "context_requests_total": "mock_context_requests_total",
    "cache_evictions": "mock_engine_cache_evictions_total",
    "prefill_ms_avg": "mock_engine_prefill_ms_avg",
    "prefill_batches": "mock_prefill_batches_total",
    "prefill_batch_requests": "mock_prefill_batch_requests_total",
    "cache_key_hits": "mock_engine_cache_key_hits_total",
    "cache_keys_requested": "mock_engine_cache_keys_requested_total",
}


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
    def __init__(self, directory, targets, interval_s=1, max_gap_s=5, binary=None):
        self.directory = Path(directory).resolve()
        self.targets = dict(targets)
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

    def archive(self):
        end = time.time()
        queries = {}
        window_ms = round(max(4 * self.interval, 10) * 1000)
        for name in self.targets:
            sel = self.selector(name)
            queries[name + "/up"] = "up" + sel
            if name == "mock":
                for field in ("running", "waiting"):
                    for reduction in ("sum", "avg", "max"):
                        queries[f"mock/{field}_{reduction}"] = (
                            f"{reduction} by (role) ({ENGINE_FIELDS[field]}{sel})"
                        )
                queries["mock/engine_count"] = (
                    "count by (role) (rtp_llm_running_stream_size" + sel + ")"
                )
                queries["mock/context_wall_tps"] = (
                    f"sum by (role) (rate(mock_context_tokens_total{sel}[{window_ms}ms]))"
                )
                queries["mock/context_completed_qps"] = (
                    f"sum by (role) (rate(mock_context_requests_total{sel}[{window_ms}ms]))"
                )
                queries["mock/cache_hit_ratio"] = (
                    f"sum by (role) (rate(mock_hit_tokens_total{sel}[{window_ms}ms])) / sum by (role) (rate(mock_context_tokens_total{sel}[{window_ms}ms]))"
                )
                for label, metric in (
                    ("context_execution_tps_avg", "rtp_llm_context_tps"),
                    (
                        "context_execution_tps_with_cache_avg",
                        "rtp_llm_context_tps_with_cache",
                    ),
                    ("simulated_prefill_ms_avg", "mock_engine_prefill_ms_avg"),
                    ("simulated_decode_ms_avg", "mock_engine_decode_ms_avg"),
                ):
                    queries["mock/" + label] = f"avg by (role) ({metric}{sel})"
                for metric in (
                    "rtp_llm_kv_cache_pool_total_blocks",
                    "rtp_llm_kv_cache_pool_available_blocks",
                    "mock_engine_held_blocks",
                    "mock_engine_referenced_blocks",
                ):
                    queries["mock/" + metric] = f"sum by (role) ({metric}{sel})"
                for label, metric in (
                    ("completed_qps", "mock_engine_completed_total"),
                    ("accepted_qps", "mock_engine_accepted_total"),
                    ("output_wall_tps", "mock_generate_tokens_total"),
                ):
                    queries["mock/" + label] = (
                        f"sum by (role) (rate({metric}{sel}[{window_ms}ms]))"
                    )
            elif name.startswith("client-"):
                for metric in ("actual_send", "success", "error", "completed"):
                    queries[name + "/" + metric + "_qps"] = (
                        f"sum(rate(flexlb_client_{metric}_total{sel}[{window_ms}ms]))"
                    )
                for metric in ("ttft", "total", "schedule"):
                    queries[name + "/" + metric + "_p99_seconds"] = (
                        f"histogram_quantile(0.99, sum by (le) (rate(flexlb_client_{metric}_seconds_bucket{sel}[{window_ms}ms])))"
                    )
            else:
                for label, metric, group in (
                    ("arrivals_qps", "flexlb_auto_tpm_request_count_total", "priority"),
                    (
                        # Current Schedule API always records this timer; the legacy
                        # balancing counter requires a populated BalanceContext response.
                        # This measures scheduling responses, not inference completion.
                        "schedule_responses_qps",
                        "flexlb_auto_tpm_schedule_latency_ms_seconds_count",
                        "result",
                    ),
                    (
                        "dispatch_qps",
                        "flexlb_app_engine_balancing_master_dispatch_reason_total",
                        "reason",
                    ),
                ):
                    queries[name + "/" + label] = (
                        f"sum by ({group}) (rate({metric}{sel}[{window_ms}ms]))"
                    )
                for metric in (
                    "flexlb_app_flexlb_batcher_queue_size",
                    "flexlb_app_flexlb_scheduler_inflight_size",
                    "flexlb_app_flexlb_inflight_request_count",
                    "flexlb_auto_tpm_decode_reserved_count",
                    "flexlb_auto_tpm_decode_running_count",
                ):
                    queries[name + "/" + metric] = f"sum({metric}{sel})"

        result = dict(
            missing_queries=[],
            backend="prometheus",
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
                    data = self.api(
                        "query_range",
                        query=expression,
                        start=cursor,
                        end=right,
                        step=self.interval,
                    )
                    rows.extend(data["result"])
                    cursor = right + self.interval
                result["queries"][key] = dict(promql=expression, result=rows)
                if not rows:
                    result["missing_queries"].append(key)
                    if key in {"mock/running_avg", "mock/waiting_avg"} or key.endswith(
                        "/up"
                    ):
                        result["errors"].append(
                            dict(query=key, error="required monitor series absent")
                        )
            except Exception as exc:
                result["errors"].append(dict(query=key, error=str(exc)))
        (self.directory / "queries.json").write_text(
            json.dumps(result, separators=(",", ":"))
        )
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


def engine_sample(url, timeout=5):
    """Strict named contract for cache gates. Never map debug `running` here."""
    from monitoring.metrics import parse_prometheus_samples

    engines = {}
    reverse = {value: key for key, value in ENGINE_FIELDS.items()}
    for name, labels, value in parse_prometheus_samples(
        telemetry.http_text(url, timeout), ""
    ):
        if labels.get("role") != "prefill" or name not in reverse:
            continue
        identity = labels.get("engine_name")
        if not identity or not labels.get("engine_incarnation"):
            raise ValueError("metric lacks engine identity/incarnation")
        row = engines.setdefault(
            identity,
            dict(
                engine_incarnation=labels["engine_incarnation"],
                grpc_addr=labels["engine_ip"] + ":" + labels["grpc_port"],
            ),
        )
        field = reverse[name]
        if field in row or _finite(value) is None:
            raise ValueError("duplicate or invalid metric " + name)
        row[field] = value
    if not engines or any(set(ENGINE_FIELDS) - set(row) for row in engines.values()):
        raise ValueError("incomplete engine monitoring contract")
    return engines


def archived_series(directory, anchor):
    """Read only monitor query archives; return curves, provenance and health gaps."""
    series, sources, gaps, errors = {}, {}, {}, []
    for path in sorted(Path(directory).glob("telemetry/*/queries.json")):
        data = json.loads(path.read_text())
        epoch = path.parent.name
        errors.extend(dict(source=epoch, **error) for error in data.get("errors", []))
        errors.extend(
            dict(source=epoch, query=query, error="monitor series absent")
            for query in data.get("missing_queries", [])
        )
        for query_id, query in data["queries"].items():
            source, metric = query_id.split("/", 1)
            for row in query["result"]:
                labels = dict(row["metric"])
                name = labels.pop("__name__", metric)
                for label in ("job", "instance"):
                    labels.pop(label, None)
                key = f"{epoch}/{source}/{name}/" + json.dumps(labels, sort_keys=True)
                points = [[float(t) - anchor, _finite(v)] for t, v in row["values"]]
                series.setdefault(key, []).extend(points)
                sources[key] = dict(
                    path=str(path),
                    promql=query["promql"],
                    start=data["start"],
                    end=data["end"],
                    step=data["step"],
                    backend="prometheus",
                )
                if metric == "up":
                    gaps.setdefault(f"{epoch}/{source}/collection", []).extend(
                        t for t, value in points if value != 1
                    )
        for source in data["targets"]:
            query = data["queries"].get(source + "/up", {})
            points = sorted(
                (float(t), _finite(v))
                for row in query.get("result", [])
                for t, v in row["values"]
            )
            if not points:
                errors.append(
                    dict(source=f"{epoch}/{source}", error="no Prometheus up samples")
                )
            elif (
                points[0][0]
                - data.get("target_bounds", {}).get(source, [data["start"], None])[0]
                > 2 * data["step"]
                or (
                    data.get("target_bounds", {}).get(source, [None, None])[1]
                    or data["end"]
                )
                - points[-1][0]
                > 2 * data["step"]
            ):
                errors.append(
                    dict(
                        source=f"{epoch}/{source}",
                        error="incomplete monitoring coverage",
                    )
                )
    # A failed scrape is a gap in every associated curve; never bridge it.
    for key, points in series.items():
        source = "/".join(key.split("/")[:2])
        failed = set(gaps.get(source + "/collection", []))
        merged = {t: value for t, value in points}
        for t in failed:
            merged[t] = None
        series[key] = sorted([t, value] for t, value in merged.items())
    return series, sources, {k: v for k, v in gaps.items() if v}, errors


def write_monitor_report(root, run_id=None, output=None):
    from reporting import write_bundle, run_meta, details

    root = Path(root)
    archives = sorted(root.glob("telemetry/*/queries.json"))
    if not archives:
        raise ValueError(
            "missing Prometheus query archive; refusing log-derived curves"
        )
    anchor = min(json.loads(path.read_text())["start"] for path in archives)
    series, sources, gaps, errors = archived_series(root, anchor)
    valid = not gaps and not errors
    result = dict(
        monitor_backend="prometheus",
        summary=dict(test_valid=valid, performance_verdict="NOT_EVALUATED"),
        series=series,
        statistic_sources=sources,
        collection_gaps=gaps,
        errors=errors,
    )
    spec = dict(
        run_id=run_id or root.name,
        title="Experiment monitoring",
        subtitle="Prometheus",
        panels=[
            dict(
                id="monitor-" + str(i),
                title=key,
                type="line",
                timeX=True,
                x=[str(t) for t, _ in points],
                xNums=[t for t, _ in points],
                series=[dict(name=key, data=[v for _, v in points])],
                caption=sources[key]["promql"],
            )
            for i, (key, points) in enumerate(series.items())
        ],
        sections=[
            details(
                "Monitoring evidence", dict(sources=sources, gaps=gaps, errors=errors)
            )
        ],
    )
    configuration = {}
    for filename in (
        "master_config.json",
        "client_env.json",
        "mode_plan.json",
        "endpoints.json",
    ):
        path = root / filename
        if path.is_file():
            configuration[filename] = json.loads(path.read_text())
    inputs = []
    client = configuration.get("client_env.json", {})
    for name in ("TRACE_FILE",):
        path = Path(client.get(name, ""))
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            inputs.append(dict(path=str(path), sha256=digest.hexdigest()))
    meta = run_meta(
        dict(id=run_id or root.name),
        clock=dict(epoch_s=anchor),
        configuration=configuration,
        workload=dict(inputs=inputs),
        implementation=dict(source_commit=os.environ.get("FLEXLB_SOURCE_COMMIT")),
        evidence=[
            dict(
                path=str(path.relative_to(root)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in archives
        ],
    )
    result["run_meta"] = meta
    destination = Path(output) if output is not None else root
    write_bundle(
        destination,
        "run",
        run_id or root.name,
        result,
        spec,
        meta=meta,
        producer="prometheus",
    )
    (destination / "aggregate.json").write_text(json.dumps(result, allow_nan=False))
    return valid


def main():
    import argparse
    import signal
    import threading

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "report"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--target", action="append", default=[], help="name=http://host:port/metrics"
    )
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--clients", type=int, default=1)
    args = parser.parse_args()
    if args.command == "report":
        return 0 if write_monitor_report(args.run_dir) else 2
    targets = dict(item.split("=", 1) for item in args.target)
    stop = threading.Event()
    # Signal callbacks assign only, avoiding nested Event lock acquisition.
    stopping = False

    def request_stop(*unused):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    session = PrometheusSession(
        args.run_dir / "telemetry" / "0", targets, args.interval
    )
    parent = os.getppid()
    try:
        session.start()
        (args.run_dir / "monitor-ready").touch()
        while not stopping and not (args.run_dir / "monitor-stop").exists():
            if session.process.poll() is not None or os.getppid() != parent:
                raise RuntimeError("monitor or owner exited unexpectedly")
            # Files describe endpoint discovery only, never metric values.
            paths = list(args.run_dir.glob("load_client/**/metrics-target.json"))
            if len(paths) == args.clients:
                new = {
                    "client-" + p.parent.name: json.loads(p.read_text())["url"]
                    for p in paths
                    if "client-" + p.parent.name not in session.targets
                }
                if new:
                    session.add_targets(new)
                    # All shards share one post-scrape start barrier.
                    start_at = str(int(time.time() * 1000) + 500)
                    for path in paths:
                        (path.parent / "metrics-ready").write_text(start_at)
            for path in paths:
                name = "client-" + path.parent.name
                complete = path.parent / "metrics-complete.json"
                if name in session.target_bounds and complete.exists():
                    session.end_target(
                        name, json.loads(complete.read_text())["epoch_s"]
                    )
            stop.wait(0.2)
        for path in args.run_dir.glob("load_client/**/metrics-complete.json"):
            name = "client-" + path.parent.name
            if name in session.target_bounds:
                session.end_target(name, json.loads(path.read_text())["epoch_s"])
        clients = [name for name in session.targets if name.startswith("client-")]
        incomplete = [
            name for name in clients if session.target_bounds[name][1] is None
        ]
        session.stop()
        if len(clients) != args.clients or incomplete:
            raise RuntimeError(
                "incomplete client monitoring lifecycle: " + str(incomplete)
            )
        (args.run_dir / "monitor-complete").touch()
    finally:
        session.stop(export=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
