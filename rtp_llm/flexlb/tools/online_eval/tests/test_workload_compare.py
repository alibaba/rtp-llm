import copy
import json
import tempfile
import unittest
from pathlib import Path

from flexlb_test_framework.workload.compare import compare
from flexlb_test_framework.workload.report import mature_series


class WorkloadCompareTest(unittest.TestCase):
    def report(self, offset, points):
        return dict(
            id="same",
            configuration_sha256="abc",
            workload={"runtime_validity": "VALID"},
            clock_anchor={"monotonic_s": offset},
            stages=[
                dict(
                    id="load",
                    status="PASS",
                    started_s=offset + 2,
                    finished_s=offset + 5,
                )
            ],
            series={"metric": points},
        )

    def test_mature_percentile_curve_uses_same_comparison_and_source(self):
        a = self.report(100, [])
        b = self.report(200, [])
        key = "statistics/1/per_second/e2e_p99"
        a["series"] = {key: [[2, 10], [3, 10]]}
        b["series"] = {key: [[2, 30], [3, 30]]}
        a["statistic_sources"] = {key: {"field": "per_second.e2e_p99"}}
        result = compare(a, b)["changes"][0]
        self.assertEqual(result["metric_kind"], "derived_statistic")
        self.assertEqual(result["relative_delta"], 2)
        self.assertEqual(result["statistic_sources"][0]["field"], "per_second.e2e_p99")

    def test_clock_origins_are_not_compared_directly(self):
        a = self.report(100, [[2, 10], [3, 10]])
        b = self.report(999, [[2, 20], [3, 20]])
        r = compare(a, b)["changes"][0]
        self.assertEqual(r["relative_delta"], 1)
        self.assertEqual(r["baseline"], [[0, 10], [1, 10]])
        self.assertEqual(r["verdict"], "DESCRIPTIVE_ONLY")

    def test_missing_window_is_not_unchanged(self):
        a = self.report(1, [[2, 10]])
        b = self.report(2, [[6, 10]])
        r = compare(a, b)["changes"][0]
        self.assertEqual(r["status"], "MISSING_DATA")
        self.assertIsNone(r["rank_score"])

    def test_zero_baseline_and_changed_configuration_are_explicit(self):
        a = self.report(1, [[2, 0]])
        b = self.report(2, [[2, 10]])
        self.assertIsNone(compare(a, b)["changes"][0]["relative_delta"])
        b["configuration_sha256"] = "other"
        with self.assertRaises(ValueError):
            compare(a, b)

    def test_mature_series_preserves_missing_values_and_time_origin(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "canvas.json"
            p.write_text(
                json.dumps(
                    {"per_second": [{"t": 0, "e2e_p99": 8}, {"t": 1, "e2e_p99": None}]}
                )
            )
            (p.parent / "client_events.jsonl").write_text(
                json.dumps({"send_start_epoch_ms": 103000}) + "\n"
            )
            series, sources, errors = mature_series(
                [dict(status="GENERATED", path=str(p), env_epoch=1)], 100
            )
            key = "statistics/1/per_second/e2e_p99"
            self.assertEqual(series[key], [[3, 8], [4, None]])
            self.assertEqual(sources[key]["path"], str(p))
            self.assertFalse(errors)
            (p.parent / "client_events.jsonl").unlink()
            self.assertTrue(
                mature_series(
                    [dict(status="GENERATED", path=str(p), env_epoch=1)], 100
                )[2]
            )
