import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import prefill_bench


def response(input_len=32768, first=105.0, wait=5.0, total=105.01):
    return {
        "aux_info": {
            "input_len": input_len,
            "output_len": 1,
            "reuse_len": 0,
            "first_token_cost_time": first,
            "wait_time": wait,
            "cost_time": total,
        },
        "response": "test",
    }


class PrefillBenchTest(unittest.TestCase):
    def setUp(self):
        self.case = {"name": "32k", "input_len": 32768, "prompt": "hello"}

    def test_server_metric_excludes_only_queue_time(self):
        metrics = prefill_bench.response_metrics(response(), 32768)
        self.assertEqual(metrics["prefill_ms"], 100.0)
        self.assertAlmostEqual(metrics["server_total_ms"], 100.01)
        self.assertEqual(metrics["wait_ms"], 5.0)

    def test_rejects_missing_or_invalid_timings(self):
        for key in ("first_token_cost_time", "wait_time", "cost_time"):
            for value in (None, "105", True, float("nan"), float("inf"), -1):
                with self.subTest(key=key, value=value):
                    payload = response()
                    payload["aux_info"][key] = value
                    with self.assertRaises(ValueError):
                        prefill_bench.response_metrics(payload, 32768)

    def test_rejects_wrong_shape_reuse_and_invalid_timing_order(self):
        for key, value in (
            ("input_len", 8192),
            ("output_len", 2),
            ("output_len", True),
            ("reuse_len", 256),
            ("first_token_cost_time", 4),
            ("cost_time", 104),
        ):
            with self.subTest(key=key):
                payload = response()
                payload["aux_info"][key] = value
                with self.assertRaises(ValueError):
                    prefill_bench.response_metrics(payload, 32768)
        for payload in ({}, [], {"aux_info": []}, {**response(), "finished": False}):
            with self.assertRaises(ValueError):
                prefill_bench.response_metrics(payload, 32768)

    def test_connection_reuse_and_request_contract(self):
        with patch("prefill_bench.http.client.HTTPConnection") as connection_type:
            connection = connection_type.return_value
            reply = connection.getresponse.return_value
            reply.status = 200
            reply.read.return_value = json.dumps(response()).encode()
            client = prefill_bench.PrefillClient("http://127.0.0.1:12345/")
            for request_number in range(2):
                metrics, payload = client.request(self.case)
                self.assertEqual(metrics["prefill_ms"], 100)
                self.assertEqual(payload, response())
            connection_type.assert_called_once_with("127.0.0.1", 12345, timeout=120)
            self.assertEqual(connection.request.call_count, 2)
            arguments = connection.request.call_args.args
            self.assertEqual(arguments[:2], ("POST", "/"))
            body = json.loads(arguments[2])
            self.assertEqual(body["prompt"], self.case["prompt"])
            self.assertEqual(body["generate_config"]["max_new_tokens"], 1)
            self.assertEqual(body["generate_config"]["min_new_tokens"], 1)
            self.assertEqual(body["top_k"], 1)
            self.assertNotIn("gen_timeline", body)
            client.close()
            connection.close.assert_called_once()

    def test_http_error_is_not_retried(self):
        with patch("prefill_bench.http.client.HTTPConnection") as connection_type:
            reply = connection_type.return_value.getresponse.return_value
            reply.status = 500
            reply.read.return_value = b"failed"
            client = prefill_bench.PrefillClient("http://localhost:12345")
            with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
                client.request(self.case)
            self.assertEqual(connection_type.return_value.request.call_count, 1)

    def test_endpoint_validation(self):
        for endpoint in (
            "localhost:12345",
            "file:///tmp/server",
            "http://u:p@host",
            "http://host/#fragment",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                prefill_bench.PrefillClient(endpoint)

    def test_statistics_preserve_outliers_and_expose_trimmed_mean(self):
        metrics = prefill_bench.describe([1, 2, 3, 4, 100])
        self.assertEqual(metrics["mean"], 22)
        self.assertEqual(metrics["trimmed_mean"], 3)
        self.assertEqual(metrics["p95"], 100)
        self.assertEqual(metrics["count"], 5)
        self.assertEqual(prefill_bench.describe([2])["sample_std"], 0)
        with self.assertRaises(ValueError):
            prefill_bench.describe([])

    def test_fixture_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "inputs.json"
            fixture.write_text(json.dumps({"cases": [self.case]}))
            self.assertEqual(prefill_bench.load_cases(fixture), [self.case])
            for cases in (
                [],
                [self.case, self.case],
                [{**self.case, "input_len": True}],
                [{**self.case, "prompt": ""}],
            ):
                fixture.write_text(json.dumps({"cases": cases}))
                with self.assertRaises(ValueError):
                    prefill_bench.load_cases(fixture)

    def test_warmup_exclusion_and_group_counts(self):
        client = MagicMock()
        metrics = {"prefill_ms": 100.0, "client_total_ms": 110.0, "wait_ms": 0.1}
        client.request.return_value = (metrics, response())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            result = prefill_bench.run_benchmark(
                client, [self.case], output, warmup=2, repeats=3, requests=4
            )
            self.assertEqual(client.request.call_count, 14)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["points"]["32k"]["prefill_ms"]["count"], 12)
            self.assertEqual(len(result["points"]["32k"]["groups"]), 3)
            rows = (output / "requests.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 14)
            with self.assertRaises(FileExistsError):
                prefill_bench.run_benchmark(client, [self.case], output)

    def test_failure_is_preserved_and_stops_measurement(self):
        client = MagicMock()
        client.request.side_effect = RuntimeError("request failed")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                prefill_bench.run_benchmark(client, [self.case], output, warmup=0)
            self.assertEqual(client.request.call_count, 1)
            result = json.loads((output / "summary.json").read_text())
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["points"], {})
            record = json.loads((output / "requests.jsonl").read_text())
            self.assertFalse(record["success"])

    def test_invalid_run_counts_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            for settings in ({"warmup": -1}, {"repeats": 0}, {"requests": 0}):
                with self.assertRaises(ValueError):
                    prefill_bench.run_benchmark(
                        MagicMock(), [self.case], output, **settings
                    )
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
