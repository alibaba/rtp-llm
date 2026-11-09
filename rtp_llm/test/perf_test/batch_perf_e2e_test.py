import argparse
import hashlib
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.batch_perf_impl import (
    BatchPerfImpl,
    _curl_server_single_worker,
)
from rtp_llm.test.perf_test.dataclass import (
    MetricState,
    ResponseInfo,
    TableType,
    create_metrics_table,
)
from rtp_llm.test.perf_test.server import EngineServer


class PerfE2ETest(unittest.TestCase):
    def setUp(self):
        self.payloads = []
        payloads = self.payloads

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                payloads.append((self.path, payload))
                response = {
                    "response": ["generated text"],
                    "aux_info": {
                        "input_len": 7,
                        "output_len": 3,
                        "cost_time": 40,
                        "first_token_cost_time": 20,
                        "wait_time": 5,
                        "speculative_verify_rounds": 2,
                        "speculative_accepted_token_num": 3,
                        "speculative_proposed_draft_tokens": 4,
                    },
                }
                if payload.get("prompt") == "missing_count":
                    del response["aux_info"]["output_len"]
                body = json.dumps(response).encode()
                self.send_response(500 if payload.get("prompt") == "fail" else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever)
        self.thread.start()
        self.addCleanup(self.stop_http)

    def stop_http(self):
        self.http.shutdown()
        self.thread.join()
        self.http.server_close()

    def test_natural_cohorts_use_actual_tokens_and_preserve_engine_tpot(self):
        # Given a natural scheduler and two measurement cohorts.
        with patch.dict(
            os.environ,
            {"PERF_USE_BATCH_DECODE_SCHEDULER": "0", "PERF_MEASURE_E2E": "1"},
        ):
            runner = BatchPerfImpl(
                self.http.server_port,
                1,
                2,
                "prompt",
                profile=False,
                warmup_runs=0,
                measure_runs=2,
                generate_config={"force_sp_accept": False},
            )
            self.addCleanup(runner.executor.shutdown)
            # When requests complete over real local HTTP.
            metrics = runner.run()
        # Then TPS uses observed tokens and elapsed cohort time, not requested length.
        self.assertEqual(metrics.success_requests, 4)
        self.assertEqual(metrics.avg_decode_time, 10)
        self.assertEqual(len(metrics.e2e_cohorts), 2)
        for cohort in metrics.e2e_cohorts:
            self.assertEqual(cohort.output_tokens, 6)
            self.assertGreater(cohort.wall_seconds, 0)
            self.assertEqual(cohort.output_tokens_per_second, 6 / cohort.wall_seconds)
            for request in cohort.requests:
                self.assertEqual((request.input_tokens, request.output_tokens), (7, 3))
                self.assertGreater(request.wall_seconds, 0)
                self.assertEqual(
                    request.input_query_sha256,
                    hashlib.sha256(b"prompt").hexdigest(),
                )
                self.assertEqual(
                    request.response_sha256,
                    hashlib.sha256(b'["generated text"]').hexdigest(),
                )
                self.assertEqual(request.speculative_verify_rounds, 2)
                self.assertEqual(request.speculative_accepted_token_num, 3)
                self.assertEqual(request.speculative_proposed_draft_tokens, 4)
        self.assertEqual(len(self.payloads), 4)
        self.assertTrue(all(path == "/" for path, _ in self.payloads))
        self.assertTrue(
            all(not p["generate_config"]["force_sp_accept"] for _, p in self.payloads)
        )
        with tempfile.TemporaryDirectory() as output:
            create_metrics_table(
                TableType.Decode, [MetricState(7, 2, metrics)], output, {}, "E2E"
            )
            result = json.loads((Path(output) / "E2E.json").read_text())["metrics"][0]
        self.assertEqual(result["e2e_output_tokens"], 12)
        self.assertEqual(result["avg_decode_time"], 10)
        self.assertEqual(len(result["e2e_cohorts"]), 2)

    def test_natural_scheduler_preserves_explicit_forced_acceptance(self):
        with patch.dict(
            os.environ,
            {"PERF_USE_BATCH_DECODE_SCHEDULER": "0", "PERF_MEASURE_E2E": "1"},
        ):
            result = _curl_server_single_worker(
                0,
                self.http.server_port,
                "prompt",
                True,
                10,
                5,
                generate_config={"force_sp_accept": True},
            )
        self.assertTrue(self.payloads[0][1]["generate_config"]["force_sp_accept"])
        self.assertTrue(result.success)
        self.assertGreater(result.wall_seconds, 0)

    def test_prearm_profile_delays_capture_on_all_ranks(self):
        with patch.dict(
            os.environ,
            {
                "PERF_PREARM_PROFILE": "1",
                "PERF_PROFILE_START_STEP": "20",
                "PERF_PROFILE_NUM_STEPS": "12",
                "PERF_PROFILE_ARM_SLEEP": "0",
            },
        ):
            runner = BatchPerfImpl(
                self.http.server_port,
                1,
                1,
                "prompt",
                profile=True,
                decode_test_length=220,
                warmup_runs=0,
                measure_runs=1,
            )
            self.addCleanup(runner.executor.shutdown)
            runner._prearm_profile()
        path, payload = self.payloads[0]
        self.assertEqual(path, "/start_profile")
        self.assertEqual(payload["start_step"], 20)
        self.assertEqual(payload["num_steps"], 12)
        self.assertTrue(payload["enable_all_rank"])

    def test_missing_speculative_counters_remain_absent(self):
        # Given a response without speculative counters.
        response = {"aux_info": {"input_len": 7, "output_len": 3}}
        # When response metrics are parsed.
        result = ResponseInfo(response)
        # Then missing counters are distinguishable from measured zero.
        self.assertIsNone(result.input_query_sha256)
        self.assertIsNone(result.response_sha256)
        self.assertIsNone(result.speculative_verify_rounds)
        self.assertIsNone(result.speculative_accepted_token_num)
        self.assertIsNone(result.speculative_proposed_draft_tokens)

    def test_natural_mode_rejects_failed_requests(self):
        # Given an HTTP failure with natural scheduling.
        with patch.dict(os.environ, {"PERF_USE_BATCH_DECODE_SCHEDULER": "0"}):
            runner = BatchPerfImpl(
                self.http.server_port,
                1,
                1,
                "fail",
                profile=False,
                warmup_runs=0,
                measure_runs=1,
            )
            self.addCleanup(runner.executor.shutdown)
            # When a measurement fails, then no partial throughput is returned.
            with self.assertRaisesRegex(Exception, "perf requests failed"):
                runner.run()

    def test_default_requests_keep_forced_acceptance_and_no_wall_measurement(self):
        # Given the legacy defaults.
        with patch.dict(os.environ, {}, clear=True):
            # When one request is sent.
            result = _curl_server_single_worker(
                0, self.http.server_port, "prompt", True, 10, 5
            )
        # Then existing synthetic acceptance and timing behavior remain.
        self.assertTrue(self.payloads[0][1]["generate_config"]["force_sp_accept"])
        self.assertIsNone(result.input_query_sha256)
        self.assertIsNone(result.response_sha256)
        self.assertEqual(result.wall_seconds, 0)
        self.assertEqual(result.decode_time_per_token, 10)

    def test_natural_mode_rejects_missing_output_counts(self):
        with patch.dict(
            os.environ,
            {"PERF_USE_BATCH_DECODE_SCHEDULER": "0", "PERF_MEASURE_E2E": "1"},
        ):
            runner = BatchPerfImpl(
                self.http.server_port,
                1,
                1,
                "missing_count",
                profile=False,
                warmup_runs=0,
                measure_runs=1,
            )
            self.addCleanup(runner.executor.shutdown)
            with self.assertRaisesRegex(Exception, "perf requests failed"):
                runner.run()

    def test_server_preserves_explicit_routing_and_scheduler_env(self):
        # Given explicit natural-routing settings.
        args = argparse.Namespace(partial=1, result_dir="/tmp", dp_size=1)
        with patch.dict(
            os.environ,
            {"FAKE_BALANCE_EXPERT": "0", "PERF_USE_BATCH_DECODE_SCHEDULER": "0"},
        ), patch("rtp_llm.test.perf_test.server.MagaServerManager") as manager:
            # When the server starts.
            EngineServer(args, []).start(128, 2)
        # Then both settings reach the server process.
        env = manager.call_args.kwargs["env_args"]
        self.assertEqual(env["FAKE_BALANCE_EXPERT"], "0")
        self.assertEqual(env["USE_BATCH_DECODE_SCHEDULER"], "0")


if __name__ == "__main__":
    unittest.main()
