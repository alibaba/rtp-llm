import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from online_eval.reporting import (
    write_bundle,
    read_bundle,
    load_analysis,
    render,
    run_meta,
    compare_controls,
    details,
    table,
)
from online_eval.reporting.statistics import select_window, counter_delta
from flexlb_test_framework.workload.report import write_report


class ReportBundleTest(unittest.TestCase):
    def test_relocated_bundle_renders_without_evidence_or_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = {"verdict": "FAIL", "window": [1, 3], "value": None}
            spec = dict(
                run_id="x",
                title="x < y",
                panels=[],
                sections=[
                    details("Evidence", data),
                    table("Checks", ["name", "state"], [["<script>", "FAIL"]]),
                ],
            )
            original = copy.deepcopy(spec)
            bundle = write_bundle(
                root / "original",
                "run",
                "x",
                data,
                spec,
                meta=run_meta({"id": "x"}),
                producer="test",
            )
            shutil.copytree(bundle, root / "relocated")
            shutil.rmtree(root / "original")
            moved = read_bundle(root / "relocated")
            self.assertEqual(load_analysis(moved), data)
            saved = json.loads((moved / "report-spec.json").read_text())
            self.assertEqual(render(saved), (moved / "report.html").read_text())
            self.assertEqual(spec, original)
            self.assertIn("&lt;script&gt;", render(saved))
            self.assertEqual(
                set(json.loads((moved / "manifest.json").read_text())["files"]),
                {"analysis.json", "report-spec.json", "report.html"},
            )
            (moved / "analysis.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_bundle(moved)

    def test_policy_compares_new_controls_and_marks_missing_required(self):
        a = {"config": {"qps": 10, "new_option": True}, "master": "old"}
        b = copy.deepcopy(a)
        b["master"] = "new"
        self.assertTrue(
            compare_controls(a, b, allowed=("/master",), required=("/config/qps",))[
                "aligned"
            ]
        )
        b["config"]["new_option"] = False
        self.assertEqual(
            compare_controls(a, b, allowed=("/master",))["differences"][0]["path"],
            "/config/new_option",
        )
        self.assertEqual(
            compare_controls({}, {}, required=("/config/qps",))["status"], "UNKNOWN"
        )
        self.assertFalse(
            compare_controls({"config": None}, {"config": None}, required=("/config",))[
                "aligned"
            ]
        )

    def test_window_edges_and_counter_reset_are_explicit(self):
        points = [(0, 1), (1, 2), (2, 3)]
        self.assertEqual(select_window(points, 0, 2, time=lambda p: p[0]), points[:2])
        self.assertEqual(
            select_window(points, 0, 2, time=lambda p: p[0], include_end=True), points
        )
        self.assertEqual(counter_delta([10, 12, 15]), (5, "AVAILABLE"))
        self.assertEqual(counter_delta([10, 2, 15]), (None, "COUNTER_RESET"))
        self.assertEqual(counter_delta([10, None]), (None, "MISSING_COUNTER"))

    def test_workload_presentation_cannot_change_status_or_read_telemetry(self):
        payload = dict(
            id="case",
            status="ERROR",
            workload={"runtime_validity": "INVALID"},
            implementation={},
            configuration={},
            configuration_sha256=None,
            clock_anchor={"epoch_s": 10},
            request_sources=[],
            series={},
            checks=[],
            iterations=[],
            traffic_manifests=[],
            gate_report=None,
        )
        original = copy.deepcopy(payload)
        with tempfile.TemporaryDirectory() as d, patch(
            "flexlb_test_framework.workload.evidence_analysis.read_series",
            side_effect=AssertionError("renderer read telemetry"),
        ):
            result = write_report(d, payload)
            self.assertEqual(load_analysis(result)["status"], "ERROR")
            self.assertEqual(payload, original)

    def test_remote_cli_writes_one_comparison_bundle(self):
        from test_remote_compare import fixture
        from remote_compare import main

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("a", "b"):
                (root / (name + ".json")).write_text(
                    json.dumps(fixture([(1000, 10), (2000, 20), (3000, 30)]))
                )
            self.assertEqual(
                main(
                    [
                        "--real",
                        str(root / "a.json"),
                        "--mock",
                        str(root / "b.json"),
                        "--steady-lo-ms",
                        "1000",
                        "--steady-hi-ms",
                        "3000",
                        "--out-dir",
                        str(root / "out"),
                    ]
                ),
                0,
            )
            bundle = read_bundle(root / "out/reports/comparison/remote")
            self.assertEqual(
                load_analysis(bundle)["classification"], "descriptive_only"
            )
            self.assertFalse((root / "out/remote_comparison.html").exists())
            self.assertEqual(len(list((root / "out").rglob("*.html"))), 1)

    def test_case_cli_combines_table_and_duration_chart(self):
        from test_compare_case_runs import case
        from compare_case_runs import main

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("a", "b"):
                (root / (name + ".json")).write_text(
                    json.dumps({"schema_version": 1, "instances": [case("PASS")]})
                )
            self.assertEqual(
                main(
                    [
                        "--run-a",
                        str(root / "a.json"),
                        "--run-b",
                        str(root / "b.json"),
                        "--out-dir",
                        str(root / "out"),
                    ]
                ),
                0,
            )
            bundle = read_bundle(root / "out/reports/comparison/case-ab")
            spec = json.loads((bundle / "report-spec.json").read_text())
            self.assertTrue(spec["panels"])
            self.assertEqual(spec["sections"][0]["type"], "table")
            self.assertEqual(len(list((root / "out").rglob("*.html"))), 1)

    def test_unsafe_report_identity_stays_inside_artifact_root(self):
        with tempfile.TemporaryDirectory() as d:
            result = write_bundle(d, "run", "../../other", {}, dict(panels=[]))
            self.assertEqual(result.parent, Path(d) / "reports/run")


if __name__ == "__main__":
    unittest.main()
