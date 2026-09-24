import copy
import json
import tempfile
import unittest
from pathlib import Path

from cases.config import configure_program
from reporting import discover_reports, read_bundle, render, write_bundle
from reporting.view_config import declaration
from scenario.loader import ScenarioError, load_document
from workload.report import write_report
from workload.report_panels import build_panels


ROOT = Path(__file__).resolve().parents[1]


def payload(series=None, sources=None, gates=None):
    return dict(
        id="example", status="PASS", workload={"runtime_validity": "VALID"},
        implementation={}, configuration={}, configuration_sha256=None,
        clock_anchor={"epoch_s": 10}, request_sources=[],
        series=series or {}, statistic_sources=sources or {},
        checks=[], iterations=[], traffic_manifests=[], gate_reports=gates or [],
    )


class WorkloadReportViewsTest(unittest.TestCase):
    def test_high_cardinality_is_one_panel_with_bounded_html_and_auditable_details(self):
        series = {}
        sources = {}
        for engine in range(128):
            key = f'1/mock/running_streams/{{"engine_name":"p-{engine}"}}'
            series[key] = [[t, None if t == 201 else engine + t % 11]
                           for t in range(1800)]
            sources[key] = dict(path="telemetry/1/queries.json", promql="running_streams")
        original = copy.deepcopy(series)
        with tempfile.TemporaryDirectory() as d:
            bundle = write_report(d, payload(series, sources))
            read_bundle(bundle)
            spec = json.loads((bundle / "report-spec.json").read_text())
            self.assertEqual(len(spec["panels"]), 1)
            panel = spec["panels"][0]
            self.assertEqual(len(panel["series"]), 130)
            self.assertEqual(sum(not row["hidden"] for row in panel["series"]), 2)
            self.assertEqual(len(panel["presets"]["明细"]), 128)
            self.assertTrue(all(len(row["points"]) <= 256 for row in panel["series"]))
            self.assertTrue(all(row["provenance"]["source_series_key"] in series
                                for row in panel["series"][2:]))
            self.assertEqual(panel["series"][0]["provenance"]["kind"], "derived")
            self.assertIsNone(next(point["y"] for point in panel["series"][0]["points"]
                                   if point["x"] == 201))
            html = (bundle / "report.html").read_text()
            self.assertIn("FlexMultiCurve.mount", html)
            self.assertIn("p-127", html)
            self.assertLess(len(html.encode()), 6_000_000)
            self.assertEqual(series, original)
            analysis = json.loads((bundle / "analysis.json").read_text())["result"]
            self.assertEqual(len(analysis["series"]), 128)
            self.assertEqual(len(next(iter(analysis["series"].values()))), 1800)
            self.assertEqual(render(spec), html)

    def test_template_reuse_and_local_difference_do_not_mutate_other_case(self):
        reports = {}
        for case in ("cache_scale_in", "balance_distribution"):
            path = ROOT / "config/scenarios" / f"{case}.yaml"
            doc = configure_program(load_document(path), str(path))
            reports[case] = doc["variants"][0]["test"]["reports"]
        self.assertEqual(reports["cache_scale_in"]["default"]["template"], "workload")
        self.assertEqual(reports["balance_distribution"]["default"]["template"], "workload")
        self.assertEqual(reports["cache_scale_in"]["custom"], ["gate"])
        self.assertEqual(reports["balance_distribution"]["custom"], [])
        series = {
            f'1/mock/qps/{{"engine_name":"p-{i}"}}': [[0, i], [1, i + 1]]
            for i in range(2)
        }
        left = build_panels(series, {}, reports["cache_scale_in"]["default"])[0]
        right = build_panels(series, {}, reports["balance_distribution"]["default"])[0]
        self.assertEqual(sum(not s["hidden"] for s in left["series"]), 2)
        self.assertEqual(sum(not s["hidden"] for s in right["series"]), 1)

    def test_other_labels_remain_distinct_metric_families(self):
        series = {
            f'1/mock/waiting/{{"priority":"{priority}","engine_name":"p-{engine}"}}':
                [[0, engine + 1]]
            for priority in ("high", "low") for engine in range(3)
        }
        panels = build_panels(series, {})
        self.assertEqual(len(panels), 2)
        self.assertEqual([len(panel["series"]) for panel in panels], [5, 5])

    def test_declared_gate_and_default_are_independent_verified_bundles(self):
        with tempfile.TemporaryDirectory() as d:
            gate = write_bundle(d, "run", "gate", {"verdict": "PASS"},
                                dict(title="Gate", timeAxis=dict(min=0, max=2), panels=[]),
                                role="gate")
            declared = declaration({"default": {"template": "workload"}, "custom": ["gate"]},
                                   kind="workload")
            run = write_report(d, payload(gates=discover_reports(d, role="gate")),
                               reports=declared)
            read_bundle(run)
            read_bundle(gate)
            spec = json.loads((run / "report-spec.json").read_text())
            self.assertIn("../gate/report.html", json.dumps(spec["sections"]))
            self.assertEqual(json.loads((gate / "report-spec.json").read_text())["timeAxis"],
                             dict(min=0, max=2))
            self.assertEqual(spec["timeAxis"], dict(min=0, max=1))

    def test_invalid_declarations_reject_unsafe_or_unknown_views(self):
        for value in (
            {"custom": ["unknown"]},
            {"default": {"template": "unknown"}},
            {"default": {"default_visible": ["detail"]}},
            {"default": False},
            {"custom": ["gate", "gate"]},
            {"custom": [{"action": "write"}]},
        ):
            with self.subTest(value=value), self.assertRaises(ScenarioError):
                declaration(value, kind="workload")
        with self.assertRaisesRegex(ScenarioError, "test.kind=workload"):
            declaration({"custom": ["gate"]}, kind="functional")


if __name__ == "__main__":
    unittest.main()
