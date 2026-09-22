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
from flexlb_eval.monitoring.session import (
    PrometheusSession,
    archived_series,
    engine_sample,
    ENGINE_FIELDS,
)
from flexlb_eval.monitoring.telemetry import http_text, shared_samples_since


class ContractTest(unittest.TestCase):
    def test_running_is_execution_not_unfinished_tasks(self):
        values = dict.fromkeys(ENGINE_FIELDS, 10)
        values.update(running=2, waiting=128)
        labels = 'role="prefill",engine_name="P0",engine_ip="127.0.0.1",grpc_port="7000",engine_incarnation="one"'
        body = "\n".join(
            f"{metric}{{{labels}}} {values[field]}"
            for field, metric in ENGINE_FIELDS.items()
        )
        body += "\nmock_engine_running{" + labels + "} 130\n"
        with patch("flexlb_eval.monitoring.telemetry.http_text", return_value=body):
            sample = engine_sample("http://test/metrics")["P0"]
        self.assertEqual((sample["running"], sample["waiting"]), (2, 128))
        with patch(
            "flexlb_eval.monitoring.telemetry.http_text",
            return_value=body.replace(',engine_incarnation="one"', ""),
        ):
            with self.assertRaisesRegex(ValueError, "incarnation"):
                engine_sample("http://test/metrics")

    def test_snapshot_files_never_supply_curves(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "cache-gate-evidence.json").write_text(
                json.dumps({"samples": [{"waiting": 999, "running": 999}]})
            )
            self.assertEqual(archived_series(tmp, 0), ({}, {}, {}, []))

    def test_missing_query_is_reported_as_monitor_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "telemetry" / "1"
            directory.mkdir(parents=True)
            (directory / "queries.json").write_text(json.dumps({
                "missing_queries": ["master/completions_qps"],
                "start": 1, "end": 2, "step": 1,
                "targets": {}, "queries": {}, "errors": [],
            }))
            _, _, _, errors = archived_series(tmp, 0)
            self.assertEqual(errors, [{
                "source": "1", "query": "master/completions_qps",
                "error": "monitor series absent",
            }])

    def test_legacy_gate_rejects_monitor_contract(self):
        from flexlb_eval.analysis.compare_ab import PrecheckError, resolve_run

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "aggregate.json").write_text(
                json.dumps({"monitor_backend": "prometheus"})
            )
            with self.assertRaisesRegex(PrecheckError, "legacy stress gate"):
                resolve_run(tmp)

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
                    b'rtp_llm_running_stream_size{engine_name="P0",role="prefill"} 2\nrtp_llm_wait_stream_size{engine_name="P0",role="prefill"} 128\n'
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
                    self.assertFalse(errors)
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
