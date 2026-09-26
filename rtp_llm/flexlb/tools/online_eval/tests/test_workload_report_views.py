import json
import tempfile
import unittest
from pathlib import Path

from cases.config import configure_program
from reporting import discover_reports, load_analysis, read_bundle, write_bundle
from reporting.view_config import DEFAULT_VIEW, declaration, view
from scenario.loader import ScenarioError, load_document
from workload.report import write_report, write_views
from workload.report_panels import build_panels

ROOT = Path(__file__).resolve().parents[1]


def payload(series=None, gates=None):
    return dict(
        id="example", status="PASS", workload={"runtime_validity": "VALID"},
        implementation={}, configuration={}, configuration_sha256=None,
        clock_anchor={"epoch_s": 10}, request_sources=[], series=series or {},
        statistic_sources={}, checks=[dict(stage="observe", id="cache", status="PASS")],
        iterations=[], traffic_manifests=[], gate_reports=gates or [],
    )


class WorkloadReportViewsTest(unittest.TestCase):
    def test_per_engine_metrics_share_one_chart_with_switchable_detail(self):
        series = {
            f'1/mock/running_streams/{{"engine_name":"p-{i}"}}':
                [[t, None if t == 201 else i + t % 11] for t in range(1800)]
            for i in range(128)
        }
        with tempfile.TemporaryDirectory() as d:
            bundle = write_report(d, payload(series))
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
            self.assertIsNone(next(point["y"] for point in panel["series"][0]["points"]
                                   if point["x"] == 201))
            self.assertIn("FlexMultiCurve.mount", (bundle / "report.html").read_text())
            self.assertLess((bundle / "report.html").stat().st_size, 6_000_000)
            self.assertEqual(load_analysis(bundle)["series"], series)

    def test_cases_declare_real_view_files(self):
        for path in (ROOT / "config/scenarios").glob("*.yaml"):
            doc = load_document(path)
            if doc.get("test", {}).get("kind") != "workload":
                continue
            configured = configure_program(doc, str(path))
            for variant in configured["variants"]:
                if variant["test"]["kind"] != "workload":
                    self.assertNotIn("reports", variant["test"])
                    continue
                names = variant["test"]["reports"]
                self.assertIn(DEFAULT_VIEW, names)
                self.assertTrue(all((ROOT / "config/report_views" / name).is_file()
                                    for name in names))
        cache = load_document(ROOT / "config/scenarios/cache_scale_in.yaml")
        self.assertEqual(cache["reports"], ["workload.yaml", "cache_scale_in.yaml"])
        series = {
            f'1/mock/qps/{{"engine_name":"p-{i}"}}': [[0, i], [1, i + 1]]
            for i in range(2)
        }
        self.assertEqual(len(build_panels(series, {}, view(DEFAULT_VIEW))), 1)

    def test_unlabelled_legacy_statistic_is_still_visible(self):
        key = "statistics/1/per_second/e2e_p99"
        panels = build_panels({key: [[0, 123]]}, {})
        self.assertEqual(len(panels), 1)
        self.assertEqual(panels[0]["series"][0]["provenance"]["source_series_key"], key)

    def test_selected_views_include_common_gate_detail_and_link_each_other(self):
        curves = [
            dict(name="P engine count", axis="count", points=[dict(x=0, y=2)]),
            dict(name="P cache hit ratio", axis="ratio", points=[dict(x=0, y=0.8)]),
            dict(name="P Waiting / engine", axis="queue", points=[dict(x=0, y=4)]),
        ]
        with tempfile.TemporaryDirectory() as d:
            write_bundle(d, "run", "cache-scale-in", {"verdict": "PASS", "threshold": 0.5},
                         dict(title="Gate evidence", timeAxis=dict(min=0, max=2),
                              timeOriginLabel="observation", panels=[dict(
                                  id="gate", title="All", overlay=True,
                                  axes={"count": {}, "ratio": {}, "queue": {}}, series=curves,
                              )]), role="gate")
            data = payload({'1/mock/running/{"engine_name":"p0"}': [[0, 1]]},
                           discover_reports(d, role="gate"))
            links = write_views(d, data, ["workload.yaml", "cache_scale_in.yaml"])
            custom = links["cache_scale_in.yaml"].parent
            default = links["workload.yaml"].parent
            selected = json.loads((custom / "report-spec.json").read_text())
            self.assertEqual([row["name"] for row in selected["panels"][0]["series"]],
                             ["P engine count", "P cache hit ratio"])
            self.assertEqual(selected["timeOriginLabel"], "observation")
            self.assertIn("../example/report.html", json.dumps(selected["sections"]))
            for bundle in (default, custom):
                read_bundle(bundle)
                sections = json.dumps(json.loads((bundle / "report-spec.json").read_text())["sections"], ensure_ascii=False)
                self.assertIn("门禁详细结果", sections)
                self.assertIn('"threshold": 0.5', sections)

    def test_missing_view_source_and_invalid_declarations_fail_loud(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(ValueError, "cache-scale-in"):
            write_report(d, payload(), view_name="cache_scale_in.yaml")
        for names in ([], ["missing.yaml", "workload.yaml"], ["../workload.yaml"],
                      ["cache_scale_in.yaml"], ["workload.yaml", "workload.yaml"],
                      ["workload.yaml", {"action": "render"}]):
            with self.subTest(names=names), self.assertRaises(ScenarioError):
                declaration(names, kind="workload")
        with self.assertRaisesRegex(ScenarioError, "test.kind=workload"):
            declaration([DEFAULT_VIEW], kind="functional")


if __name__ == "__main__":
    unittest.main()
