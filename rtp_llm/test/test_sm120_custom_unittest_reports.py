"""Ordinary unittest report contracts; these fixtures execute no Native code."""

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.smoke.sm120_cuda13_test import _run_original_class


class SM120OriginalUnittestReportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sm120-report-contract-")
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.properties = {}

    def run_class(self, case, target="ordinary-report-contract"):
        expected, result = _run_original_class(
            case, target, self.output, self.properties.__setitem__
        )
        report = self.report()
        return expected, result, report

    def report(self):
        payload = self.properties["original_unittest_report"]
        self.assertEqual(
            hashlib.sha256(payload.encode()).hexdigest(),
            self.properties["original_unittest_report_sha256"],
        )
        path = self.properties.get("original_unittest_report_path")
        if path:
            self.assertEqual(payload, Path(path).read_text())
        return json.loads(payload)

    def test_success_records_original_method_and_each_successful_subtest(self):
        class Cases(unittest.TestCase):
            def test_real_method(self):
                for row in ("first", "second"):
                    with self.subTest(row=row):
                        self.assertTrue(row)

        expected, result, report = self.run_class(Cases)
        identity = Cases("test_real_method").id()
        self.assertEqual(1, expected)
        self.assertEqual(1, result.testsRun)
        self.assertEqual({identity: "passed"}, report["method_verdicts"])
        self.assertEqual({"passed": 2}, report["subtest_counts"])
        events = [e for e in report["events"] if e["kind"] == "subtest"]
        self.assertEqual(2, len(events))
        self.assertTrue(all(e["method_id"] == identity for e in events))
        self.assertIn("row='first'", events[0]["subtest_id"])
        self.assertIn("row='second'", events[1]["subtest_id"])
        self.assertTrue(result.wasSuccessful())
        self.assertEqual("ordinary-report-contract", report["outer_target"])
        self.assertEqual(1, self.properties["original_unittest_methods_run"])

    def test_failed_error_and_skipped_subtests_remain_distinct(self):
        class Cases(unittest.TestCase):
            def test_events(self):
                with self.subTest(row="failure"):
                    self.fail("original failure text")
                with self.subTest(row="error"):
                    raise ValueError("original exception text")
                with self.subTest(row="skip"):
                    self.skipTest("original skip reason")

        _, result, report = self.run_class(Cases)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(1, len(result.failures))
        self.assertEqual(1, len(result.errors))
        self.assertEqual(1, len(result.skipped))
        self.assertEqual(
            {"failed": 1, "error": 1, "skipped": 1}, report["subtest_counts"]
        )
        self.assertEqual({"error": 1}, report["method_counts"])
        by_status = {event["status"]: event for event in report["events"]}
        self.assertIn("original failure text", by_status["failed"]["traceback"])
        self.assertIn("original exception text", by_status["error"]["traceback"])
        self.assertEqual("original skip reason", by_status["skipped"]["reason"])
        self.assertTrue(all(event["subtest_id"] for event in report["events"]))

    def test_ordinary_failure_retains_method_id_and_traceback(self):
        class Cases(unittest.TestCase):
            def test_failure(self):
                self.assertEqual("expected", "actual", "ordinary failure")

        _, result, report = self.run_class(Cases)
        self.assertEqual(1, len(result.failures))
        self.assertEqual(
            {Cases("test_failure").id(): "failed"}, report["method_verdicts"]
        )
        self.assertEqual("method", report["events"][0]["kind"])
        self.assertIn("ordinary failure", report["events"][0]["traceback"])

    def test_teardown_and_owned_cleanup_errors_both_survive(self):
        calls = []

        class Cases(unittest.TestCase):
            def setUp(self):
                self.addCleanup(self.cleanup)

            def cleanup(self):
                calls.append("cleanup")
                raise RuntimeError("cleanup error")

            def tearDown(self):
                calls.append("teardown")
                raise ValueError("teardown error")

            def test_body(self):
                calls.append("body")

        _, result, report = self.run_class(Cases)
        self.assertEqual(["body", "teardown", "cleanup"], calls)
        self.assertEqual(2, len(result.errors))
        self.assertEqual({"error": 1}, report["method_counts"])
        self.assertEqual(2, len(report["events"]))
        self.assertIn("teardown error", report["events"][0]["traceback"])
        self.assertIn("cleanup error", report["events"][1]["traceback"])

    def test_class_setup_error_does_not_invent_executed_method(self):
        class Cases(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise RuntimeError("class setup failed")

            def test_not_run(self):
                raise AssertionError("must not execute")

        expected, result, report = self.run_class(Cases)
        self.assertEqual(1, expected)
        self.assertEqual(0, result.testsRun)
        self.assertEqual({}, report["method_counts"])
        self.assertEqual(0, report["actual_methods_run"])
        self.assertEqual("error", report["events"][0]["status"])
        self.assertIn("setUpClass", report["events"][0]["method_id"])
        self.assertIn("class setup failed", report["events"][0]["traceback"])

    def test_expected_failure_and_unexpected_success_are_not_plain_passes(self):
        class Cases(unittest.TestCase):
            @unittest.expectedFailure
            def test_expected(self):
                self.fail("expected failure retained")

            @unittest.expectedFailure
            def test_unexpected(self):
                pass

        _, result, report = self.run_class(Cases)
        self.assertEqual(1, len(result.expectedFailures))
        self.assertEqual(1, len(result.unexpectedSuccesses))
        self.assertEqual(
            {"expected_failure": 1, "unexpected_success": 1}, report["method_counts"]
        )
        self.assertNotIn("passed", report["method_counts"])

    def test_interrupt_is_reported_and_reraised(self):
        class Cases(unittest.TestCase):
            def test_interrupted(self):
                raise KeyboardInterrupt("original interruption")

        with self.assertRaisesRegex(KeyboardInterrupt, "original interruption"):
            self.run_class(Cases)
        report = self.report()
        self.assertIn("original interruption", report["interrupted"])
        self.assertEqual(1, report["actual_methods_run"])
        self.assertEqual({"incomplete": 1}, report["method_counts"])

    def test_empty_class_is_failure_with_zero_execution_report(self):
        class Cases(unittest.TestCase):
            pass

        with self.assertRaisesRegex(AssertionError, "must not be empty"):
            self.run_class(Cases)
        report = self.report()
        self.assertEqual(0, report["expected_methods"])
        self.assertEqual(0, report["actual_methods_run"])
        self.assertEqual({}, report["method_counts"])
        self.assertIn("must not be empty", report["interrupted"])

    def test_report_write_failure_still_binds_full_original_payload_and_fails(self):
        class Cases(unittest.TestCase):
            def test_success(self):
                pass

        self.output = self.output / "not-a-directory"
        self.output.write_text("existing ordinary file")
        with self.assertRaises(OSError):
            self.run_class(Cases)
        report = self.report()
        self.assertEqual(1, report["actual_methods_run"])
        self.assertEqual({"passed": 1}, report["method_counts"])
        self.assertNotIn("original_unittest_report_path", self.properties)

    def test_original_custom_class_methods_and_outer_failure_gates_preserved(self):
        root = Path(__file__).resolve().parents[2] / "rtp_llm/test/smoke"
        expected = {
            "sm120_qwen3_test.py": ("SM120Qwen3Test", "test_requests_share_one_server"),
            "sm120_reuse_cache_test.py": (
                "SM120ReuseCacheTest",
                "test_long_prefix_reuse_matches_uncached_tokens",
            ),
            "sm120_moe_test.py": ("SM120MoeTest", "test_golden_and_decode_graph"),
        }
        for filename, (class_name, method_name) in expected.items():
            tree = ast.parse((root / filename).read_text())
            original = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
            self.assertEqual(
                [method_name],
                [
                    node.name
                    for node in original.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name.startswith("test_")
                ],
            )
        source = (root / "sm120_cuda13_test.py").read_text()
        for condition in (
            "result.testsRun == expected",
            "not result.skipped",
            "not result.expectedFailures",
            "not result.unexpectedSuccesses",
            "not result.errors",
            "not result.failures",
        ):
            self.assertIn(condition, source)
        self.assertIn("suite.run(result)", source)
        self.assertIn("loadTestsFromTestCase(test_class)", source)


if __name__ == "__main__":
    unittest.main()
