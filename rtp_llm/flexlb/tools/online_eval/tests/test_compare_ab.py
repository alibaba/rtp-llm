import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import compare_ab as cab  # noqa: E402


def classify(name, a, b):
    metric = {"name": name, "group": "test", "a": a, "b": b, "ctx": None}
    cab.compute_diff(metric, cab.DEFAULT_NOISE_FLOOR * 100.0)
    return metric


def write_run(root, name, test_valid):
    run = Path(root) / name
    run.mkdir()
    aggregate = {
        "meta": {
            "run_dir": name,
            "duration_s": 100,
            "trace_file_sha256": "deadbeef" * 8,
        },
        "summary": {"test_valid": test_valid, "validity_checks": {}},
    }
    (run / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
    return run


class DirectionClassificationTest(unittest.TestCase):
    def test_significant_improvements_do_not_block(self):
        for metric in (
            classify("schedule_latency_ms_p50", 100.0, 80.0),
            classify("input_token_tps", 1000.0, 1100.0),
            classify("test_valid", False, True),
        ):
            self.assertTrue(metric["significant"])
            self.assertFalse(metric["significant_regression"])
            self.assertEqual(metric["tier"], 3)
            self.assertEqual(metric["note"], "[显著改善]")

    def test_directional_regressions_block(self):
        for metric in (
            classify("schedule_latency_ms_p50", 100.0, 120.0),
            classify("input_token_tps", 1000.0, 900.0),
            classify("test_valid", True, False),
        ):
            self.assertTrue(metric["significant_regression"])
            self.assertEqual(metric["tier"], 1)

    def test_significant_neutral_mismatch_blocks(self):
        metric = classify("actual_send_qps", 100.0, 110.0)
        self.assertTrue(metric["significant"])
        self.assertTrue(metric["significant_regression"])
        self.assertEqual(metric["tier"], 1)

    def test_missing_boolean_value_is_not_comparable(self):
        for metric in (
            classify("test_valid", None, True),
            classify("test_valid", False, None),
        ):
            self.assertFalse(metric["comparable"])
            self.assertFalse(metric["significant"])
            self.assertFalse(metric["significant_regression"])
            self.assertEqual(metric["tier"], 3)
            self.assertIsNone(metric["abs_diff"])
            self.assertEqual(metric["note"], "[不可比较]")
            self.assertEqual(cab.fmt_rel(metric), "N/A")

    def test_significant_improvement_is_not_rendered_as_noise(self):
        metric = classify("input_token_tps", 1000.0, 1100.0)
        payload = cab.build_payload(
            {"label": "a", "path": "a"},
            {"label": "b", "path": "b"},
            {},
            [],
            {"lo_s": 25.0, "hi_s": 92.0, "source": "test", "rows_used": 0},
            [metric],
            cab.DEFAULT_NOISE_FLOOR,
        )
        output = cab.render_stdout(payload)
        line = next(line for line in output.splitlines() if "input_token_tps" in line)
        self.assertIn("显著改善", line)
        self.assertIn("改善", line)
        self.assertNotIn("噪声内", line)
        self.assertTrue(payload["gate"]["passed"])

    def test_classification_summary_preserves_legacy_semantics(self):
        regression = classify("input_token_tps", 1000.0, 900.0)
        improvement = classify("input_token_tps", 1000.0, 1100.0)
        unchanged = classify("test_valid", True, True)
        payload = cab.build_payload(
            {"label": "a", "path": "a"},
            {"label": "b", "path": "b"},
            {},
            [],
            {"lo_s": 25.0, "hi_s": 92.0, "source": "test", "rows_used": 0},
            [regression, improvement, unchanged],
            cab.DEFAULT_NOISE_FLOOR,
        )
        summary = payload["classification_summary"]
        self.assertEqual(summary["significant_critical_regressions"], 1)
        self.assertEqual(summary["significant_critical"], 2)
        self.assertEqual(summary["critical_not_regressed"], 2)
        self.assertEqual(summary["critical_unchanged"], 1)


class CliGateTest(unittest.TestCase):
    def test_improvement_only_exits_zero_and_regression_exits_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            invalid = write_run(tmp, "invalid", False)
            valid = write_run(tmp, "valid", True)
            with redirect_stdout(io.StringIO()):
                improvement_exit = cab.main(
                    ["--run-a", str(invalid), "--run-b", str(valid), "--out", "-"]
                )
                regression_exit = cab.main(
                    ["--run-a", str(valid), "--run-b", str(invalid), "--out", "-"]
                )
            self.assertEqual(improvement_exit, 0)
            self.assertEqual(regression_exit, 1)


if __name__ == "__main__":
    unittest.main()
