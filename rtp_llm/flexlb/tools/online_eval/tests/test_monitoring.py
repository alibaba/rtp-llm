"""Real TSDB checks: no snapshot fallback or private scrape loop."""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from monitoring.session import (
    PrometheusSession,
    archived_series,
)
from monitoring.telemetry import http_text, shared_samples_since
from monitoring.query_plan import load_plan, queries_for_targets
from scenario.loader import ScenarioError


class ContractTest(unittest.TestCase):
    def test_yaml_query_plan_controls_archive_and_required_series(self):
        plan = load_plan("default.yaml")
        queries, required = queries_for_targets(
            plan, {"mock": "", "client-sample": "", "master-a": ""},
            lambda source: '{job="' + source + '"}', 1,
        )
        self.assertIn("master-a/dispatch_qps", queries)
        self.assertIn("mock/running_avg", required)
        self.assertIn("mock/waiting_avg", required)
        self.assertIn("client-sample/up", required)
        self.assertNotIn("master-a/dispatch_qps", required)
        self.assertEqual(
            queries["master-a/schedule_responses_qps"],
            'sum by (result) (rate(flexlb_auto_tpm_schedule_latency_ms_seconds_count{job="master-a"}[10000ms]))',
        )
        with self.assertRaises(ScenarioError):
            load_plan("../default.yaml")

    def test_explicit_target_kinds_cover_dynamic_clients(self):
        plan = load_plan("default.yaml")
        with self.assertRaisesRegex(ValueError, "target kinds"):
            queries_for_targets(plan, {"mock": "", "client-flow": ""},
                                lambda source: '{job="' + source + '"}', 1,
                                {"mock": "mock"})
        queries, _ = queries_for_targets(
            plan, {"worker": ""}, lambda source: '{job="' + source + '"}', 1,
            {"worker": "client"},
        )
        self.assertIn("worker/schedule_p99_seconds", queries)

    def test_prefill_batch_size_uses_histogram(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = PrometheusSession(tmp, {"mock": "http://unused/metrics"})
            session.started = time.time() - 1
            with patch.object(session, "api", return_value={"result": []}):
                with self.assertRaisesRegex(RuntimeError, "incomplete Prometheus archive"):
                    session.archive()
            queries = json.loads((Path(tmp) / "queries.json").read_text())["queries"]
            expression = queries["mock/prefill_batch_size_mean"]["promql"]
            self.assertIn("rate(mock_prefill_batch_size_sum{job=\"mock\"}[10000ms])", expression)
            self.assertIn("rate(mock_prefill_batch_size_count{job=\"mock\"}[10000ms])", expression)
            self.assertIn("histogram_quantile(0.9", queries["mock/prefill_batch_size_p90"]["promql"])
            self.assertIn("rate(mock_prefill_batch_size_bucket{job=\"mock\"}[10000ms])",
                          queries["mock/prefill_batch_size_p90"]["promql"])

    def test_schedule_response_query_uses_current_api_timer(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = PrometheusSession(tmp, {"master": "http://unused/prometheus"})
            session.started = time.time() - 1
            with patch.object(session, "api", return_value={"result": [{"metric": {}, "values": []}]}):
                session.archive()
            queries = json.loads((Path(tmp) / "queries.json").read_text())["queries"]
            self.assertNotIn("master/completions_qps", queries)
            self.assertEqual(queries["master/schedule_responses_qps"]["promql"],
                'sum by (result) (rate(flexlb_auto_tpm_schedule_latency_ms_seconds_count{job="master"}[10000ms]))')

    def test_gate_snapshot_uses_declared_tsdb_metrics(self):
        from cases.cache_scale_in.inputs import engine_snapshot
        from unittest.mock import Mock
        monitor = PrometheusSession("unused", {"mock": "http://unused/metrics"},
                                    query_plan="cache_scale_in.yaml")
        labels = dict(role="prefill", engine_name="P0", engine_incarnation="one",
                      engine_ip="127.0.0.1", grpc_port="7000")
        monitor.instant = Mock(return_value=[
            dict(metric=dict(labels, __name__="rtp_llm_running_stream_size"), value=[1, "2"]),
            dict(metric=dict(labels, __name__="rtp_llm_wait_stream_size"), value=[1, "128"]),
            dict(metric=dict(labels, __name__="mock_engine_running"), value=[1, "130"]),
        ])
        fields = {"running": "mock/running", "waiting": "mock/waiting"}
        row = engine_snapshot(monitor, fields)["P0"]
        self.assertEqual((row["running"], row["waiting"]), (2, 128))
        monitor.instant.assert_called_once_with("mock", 5)
        del monitor.instant.return_value[0]["metric"]["engine_incarnation"]
        with self.assertRaisesRegex(ValueError, "labels"):
            engine_snapshot(monitor, fields)

    def test_gate_rejects_undeclared_id_and_missing_samples(self):
        from cases.cache_scale_in.inputs import engine_snapshot
        from unittest.mock import Mock
        monitor = PrometheusSession("unused", {"mock": "http://unused/metrics"},
                                    query_plan="cache_scale_in.yaml")
        monitor.instant = Mock(return_value=[])
        with self.assertRaisesRegex(ValueError, "undeclared"):
            engine_snapshot(monitor, {"running": "mock/unknown"})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            engine_snapshot(monitor, {"running": "mock/running"})

    def test_snapshot_files_never_supply_curves(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "cache-gate-evidence.json").write_text(
                json.dumps({"samples": [{"waiting": 999, "running": 999}]})
            )
            with self.assertRaises(FileNotFoundError):
                archived_series(tmp, 0)

    def test_missing_query_is_reported_as_monitor_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "telemetry" / "1"
            directory.mkdir(parents=True)
            (directory / "queries.json").write_text(json.dumps({
                "missing_queries": ["master/completions_qps"],
                "start": 1, "end": 2, "step": 1,
                "targets": {}, "queries": {}, "errors": [],
            }))
            from monitoring.metric_store import export_metrics
            export_metrics(tmp)
            _, _, _, errors = archived_series(tmp, 0)
            self.assertEqual(errors, [{
                "source": "1", "query": "master/completions_qps",
                "error": "monitor series absent", "severity": "diagnostic",
            }])

    def test_missing_binary_has_no_private_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"PROMETHEUS_BIN": ""}
        ), patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Prometheus is required"):
                PrometheusSession(tmp, {"mock": "http://127.0.0.1:1/metrics"}).start()


@unittest.skipUnless(
    os.environ.get("PROMETHEUS_BIN") or shutil.which("prometheus"),
    "requires real Prometheus",
)
class RealPrometheusTest(unittest.TestCase):
    def test_owned_scrape_query_archive_failure_and_cleanup(self):
        state = {"bad": False}

        class Exporter(BaseHTTPRequestHandler):
            def log_message(self, *unused):
                pass

            def do_GET(self):
                self.send_response(500 if state["bad"] else 200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write(
                    b'rtp_llm_running_stream_size{engine_name="P0",role="prefill"} 2\nrtp_llm_wait_stream_size{engine_name="P0",role="prefill"} 128\nrtp_llm_context_tps{engine_name="P0",role="prefill",engine_incarnation="one"} 9\n'
                )

        server = ThreadingHTTPServer(("127.0.0.1", 0), Exporter)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/metrics"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                session = PrometheusSession(
                    Path(tmp) / "telemetry" / "1", {"mock": url}, 0.2, 1
                )
                try:
                    session.start()
                    time.sleep(0.5)
                    self.assertIn("rtp_llm_running_stream_size", http_text(url))
                    samples = shared_samples_since(url, 0)
                    self.assertTrue(samples)
                    session.query_plan["sources"]["mock"]["raw_context_tps"] = dict(
                        promql="rtp_llm_context_tps${selector}", mode="scrape", unit="tokens/s",
                        value_kind="gauge", labels=["role", "engine_name", "engine_incarnation"])
                    from monitoring.query_plan import plan_hash
                    session.query_plan_sha256 = plan_hash(session.query_plan)
                    raw = session.metric_rows("mock/raw_context_tps", source="mock",
                                              start=session.started, end=time.time())
                    self.assertTrue(raw)
                    self.assertTrue(all(float(value) == 9 for row in raw for _, value in row["values"]))
                    self.assertTrue(session.metric_snapshot(["mock/raw_context_tps"], source="mock"))
                    self.assertEqual(
                        len({x["sequence"] for x in samples}), len(samples)
                    )
                    session.add_targets({"client-0": url, "client-1": url})
                    self.assertEqual(
                        set(session.target_bounds), {"client-0", "client-1"}
                    )
                    self.assertTrue(session.instant("client-0"))
                    self.assertTrue(session.instant("client-1"))
                    with self.assertRaisesRegex(ValueError, "duplicate"):
                        session.add_target("client-0", url)
                    state["bad"] = True
                    time.sleep(0.5)
                    with self.assertRaisesRegex(RuntimeError, "unavailable"):
                        http_text(url)
                    session.archive()
                    series, sources, gaps, errors = archived_series(
                        tmp, session.started
                    )
                    self.assertTrue(gaps)
                    self.assertTrue(all(error["severity"] == "diagnostic" for error in errors))
                    self.assertTrue(
                        any(v is None for points in series.values() for t, v in points)
                    )
                    self.assertTrue(
                        all(s["backend"] == "prometheus" for s in sources.values())
                    )
                    self.assertFalse(list(Path(tmp).rglob("*.prom")))
                    self.assertFalse(list(Path(tmp).rglob("*.jsonl")))
                finally:
                    session.stop(export=False)
                self.assertIsNotNone(session.process.poll())
                self.assertIsNone(shared_samples_since(url, 0))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
