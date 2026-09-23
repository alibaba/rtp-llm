import json
import os
import pathlib
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from rtp_llm.test.perf_test import reference_benchmark_test as entry
from rtp_llm.test.perf_test import reference_ttft_client as client


class ReferenceProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = entry._manifest_path()

    def test_real_reference_manifest_and_order(self):
        rows, info = client.load_reference_manifest(self.manifest)
        plan = client.plan_reference_requests(rows)
        self.assertEqual(info["sha256"], client.REFERENCE_MANIFEST_SHA256)
        self.assertEqual(
            [tag for _, tag in plan],
            ["warmup-0", "warmup-1"] + ["measure-%d" % i for i in range(8)],
        )
        self.assertEqual(
            [p["id"] for p, _ in plan[2:]], ["screening-%03d" % i for i in range(8)]
        )
        self.assertEqual(len({p["prompt_sha256"] for p in rows}), 8)
        self.assertEqual(len({p["input_token_ids_sha256"] for p in rows}), 8)

    def test_manifest_mutants_rejected(self):
        doc = json.loads(self.manifest.read_text())
        cases = []
        a = json.loads(json.dumps(doc))
        a["prompts"][1]["prompt"] = a["prompts"][0]["prompt"]
        cases.append(a)
        b = json.loads(json.dumps(doc))
        b["prompts"][0]["input_token_ids"] = b["prompts"][0]["input_token_ids"][:-1]
        cases.append(b)
        c = json.loads(json.dumps(doc))
        c["prompts"][0]["id"] = "wrong"
        cases.append(c)
        with tempfile.TemporaryDirectory() as tmp:
            for i, mutant in enumerate(cases):
                p = pathlib.Path(tmp) / ("m%d.json" % i)
                p.write_text(json.dumps(mutant))
                digest = client.sha256_bytes(p.read_bytes())
                with self.assertRaises(ValueError):
                    client.load_reference_manifest(p, expected_sha256=digest)

    def test_protocol_env_fail_closed(self):
        good = {
            "PERF_GRID_WARMUP_RUNS": "2",
            "PERF_MEASURE_RUNS": "8",
            "PERF_FORMAL_WARMUP_RUNS": "0",
            "PERF_PROFILE_RUNS": "0",
            "DSV4_FWD_PROFILE": "0",
        }
        self.assertEqual(entry._required_protocol_environment(good), good)
        bad = dict(good)
        bad["PERF_MEASURE_RUNS"] = "1"
        with self.assertRaises(RuntimeError):
            entry._required_protocol_environment(bad)

    def test_complete_loopback_roster_and_metrics(self):
        rows, _ = client.load_reference_manifest(self.manifest)
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b": heartbeat\r\n\r\n")
                self.wfile.flush()
                event = {
                    "response": "x",
                    "output_ids": [[42]],
                    "finished": True,
                    "aux_info": {
                        "input_len": 32768,
                        "output_len": 1,
                        "pd_sep": False,
                        "iter_count": 8,
                        "reuse_len": 0,
                        "prefix_reuse_len": 0,
                        "first_token_cost_time": 100.0 + len(seen),
                        "wait_time": 1.0,
                        "cost_time": 102.0 + len(seen),
                    },
                }
                self.wfile.write(("data:" + json.dumps(event) + "\r\n\r\n").encode())
                self.wfile.flush()
                time.sleep(0.001)
                self.wfile.write(b"data:[done]\r\n\r\n")
                self.wfile.flush()

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = pathlib.Path(tmp) / "result.json"
                doc = client.run_reference_benchmark(
                    port=server.server_address[1],
                    manifest_path=self.manifest,
                    output_path=out,
                    timeout=5,
                )
                self.assertTrue(doc["valid_run"])
                self.assertEqual(len(doc["results"]), 10)
                self.assertEqual(
                    [x["prompt"] for x in seen[2:]], [x["prompt"] for x in rows]
                )
                self.assertTrue(all(x["yield_generator"] is True for x in seen))
                self.assertTrue(
                    all(x["generate_config"]["force_sp_accept"] is True for x in seen)
                )
                q = doc["statistics_successes_only"]["server_queue_excluded_ms"]
                self.assertEqual(q["count"], 8)
                self.assertIsNotNone(q["six_middle_trimmed_mean"])
                self.assertEqual(json.loads(out.read_text())["valid_run"], True)
                with self.assertRaises(FileExistsError):
                    client.run_reference_benchmark(
                        port=server.server_address[1],
                        manifest_path=self.manifest,
                        output_path=out,
                        timeout=5,
                    )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_validation_rejects_reuse_iteration_and_protocol(self):
        prompt = {"input_len": 32768}
        base = {
            "finished": True,
            "input_len": 32768,
            "output_len": 1,
            "pd_sep": False,
            "iter_count": 8,
            "reuse_fields": {"reuse_len": 0},
            "output_ids": [1],
            "client_ttft_ms": 1.0,
            "client_response_ms": 2.0,
            "server_first_token_cost_ms": 3.0,
            "server_wait_ms": 0.5,
            "server_queue_excluded_ms": 2.5,
            "protocol_errors": [],
        }
        self.assertTrue(client.validate_result(dict(base), prompt)["ok"])
        for patch in (
            {"iter_count": 7},
            {"reuse_fields": {"reuse_len": 1}},
            {"protocol_errors": ["missing done"]},
        ):
            row = dict(base)
            row.update(patch)
            self.assertFalse(client.validate_result(row, prompt)["ok"])

    def test_summary_is_measurement_only(self):
        rows = []
        for i in range(10):
            rows.append(
                {
                    "ok": True,
                    "tag": "warmup-%d" % i if i < 2 else "measure-%d" % (i - 2),
                    "client_ttft_ms": 1000.0 + i,
                    "server_first_token_cost_ms": 900.0 + i,
                    "server_wait_ms": 1.0,
                    "server_queue_excluded_ms": 899.0 + i,
                    "client_response_ms": 1010.0 + i,
                }
            )
        d = client.summarize(rows)
        self.assertTrue(d["valid_run"])
        self.assertEqual(
            d["statistics_successes_only"]["client_ttft_ms"][
                "samples_in_request_order"
            ],
            [1002.0 + i for i in range(8)],
        )
        self.assertEqual(
            d["statistics_successes_only"]["client_ttft_ms"]["six_middle_trimmed_mean"],
            1005.5,
        )


if __name__ == "__main__":
    unittest.main()
