"""Portable baseline and golden result contracts."""

import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.regression.baseline import (
    compare_result_to_baseline,
    write_baseline_candidate,
)
from rtp_llm.test.perf_test.regression.golden import (
    GoldenMismatchError,
    validate_golden,
    write_golden,
)
from rtp_llm.test.perf_test.regression.test_support import _baseline, _entry
from rtp_llm.test.perf_test.regression.test_support import _result as _baseline_result
from rtp_llm.test.perf_test.results import RESULT_SCHEMA_VERSION


def _set_measured_metrics(result, *, throughput, total_latency):
    metrics = result["cases"][0]["metrics"]
    metrics["throughput_tokens_per_second"] = throughput
    metrics["latency_ms"]["total"] = total_latency
    measurement = metrics["measurements"][0]
    measurement["throughput_tokens_per_second"] = throughput
    for request in measurement["requests"]:
        request["latency_ms"]["total"] = total_latency


class BaselineRegressionTest(unittest.TestCase):

    def test_higher_and_lower_is_better_include_threshold_boundary(self):
        result = _baseline_result()
        key = result["cases"][0]["case_id"]
        _set_measured_metrics(result, throughput=7600.0, total_latency=11.55)

        comparison = compare_result_to_baseline(
            result,
            _baseline(
                _entry(
                    key,
                    "throughput_tokens_per_second",
                    8000.0,
                    "higher_is_better",
                ),
                _entry(key, "latency_ms.total", 11.0, "lower_is_better"),
            ),
        )

        self.assertTrue(comparison["passed"])
        self.assertEqual("passed", comparison["status"])
        self.assertEqual(
            [7600.0, 11.55], [item["limit"] for item in comparison["comparisons"]]
        )

    def test_partial_result_is_ineligible_even_if_available_metrics_pass(self):
        result = _baseline_result()
        key = result["cases"][0]["case_id"]
        result["status"] = "partial"

        comparison = compare_result_to_baseline(
            result,
            _baseline(_entry(key, "latency_ms.total", 11.0, "lower_is_better")),
        )

        self.assertFalse(comparison["passed"])
        self.assertEqual("ineligible", comparison["status"])
        self.assertEqual([], comparison["comparisons"])
        self.assertIn("partial", comparison["diagnostics"][0])


def _prefill_case(input_length, cache_length, latency, throughput):
    identity = {
        "batch_size": 1,
        "cache_len": cache_length,
        "compute_len": input_length - cache_length,
        "input_len": input_length,
        "measure_runs": 1,
    }
    return {
        "case_id": case_id("prefill", identity),
        "identity": identity,
        "status": "ok",
        "runs": 1,
        "errors": [],
        "metrics": {
            "input_length": input_length,
            "expected_reuse_length": cache_length,
            "observed_reuse_length": cache_length,
            "compute_length": input_length - cache_length,
            "latency_ms": latency,
            "throughput_tokens_per_second": throughput,
            "validation": {"shape": True, "reuse": True, "timing": True},
            "seed": 7,
            "prefix_policy": {"kind": "shared_seed"},
            "measurements": [{"run": 0}],
        },
    }


def _golden_result():
    cases = [
        _prefill_case(2048, 1024, 20.0, 51200.0),
        _prefill_case(1024, 512, 10.0, 51200.0),
    ]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": "run-001",
        "mode": "prefill",
        "backend": "local",
        "status": "completed",
        "config_fingerprint": "a" * 64,
        "workload_fingerprint": "b" * 64,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
            "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
        },
        "summary": {"planned_cases": 2, "successful_cases": 2, "failed_cases": 0},
        "cases": cases,
    }


class BaselineCandidateTest(unittest.TestCase):

    def test_candidate_write_does_not_overwrite_accepted_baseline_or_golden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reports").mkdir()
            accepted = root / "accepted_baseline.json"
            golden = root / "committed_golden.json"
            accepted.write_bytes(b"accepted-baseline\n")
            golden.write_bytes(b"committed-golden\n")

            path = write_baseline_candidate(root, _golden_result(), threshold=0.05)

            self.assertEqual(root / "reports" / "baseline_candidate.json", path)
            self.assertEqual(b"accepted-baseline\n", accepted.read_bytes())
            self.assertEqual(b"committed-golden\n", golden.read_bytes())
            self.assertEqual(1, json.loads(path.read_text())["schema_version"])


class GoldenTest(unittest.TestCase):

    def test_validation_reports_mismatch_without_rewriting_golden(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "golden.json"
            write_golden(_golden_result(), output)
            before = output.read_bytes()
            current = _golden_result()
            current["cases"][0]["metrics"]["latency_ms"] = 21.0

            with self.assertRaisesRegex(GoldenMismatchError, "does not match"):
                validate_golden(current, output)

            self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
