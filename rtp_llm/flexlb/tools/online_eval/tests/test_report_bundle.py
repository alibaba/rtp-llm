import copy
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reporting import (
    write_bundle,
    discover_reports,
    read_bundle,
    load_analysis,
    render,
    run_meta,
    compare_controls,
    details,
    table,
)
from reporting.statistics import select_window, counter_delta
from reporting.pairing import event_anchor, shifted_panel, pair_samples, paired_overlay
from reporting.renderer import render_context, render_sections
from workload.report import write_report


class ReportBundleTest(unittest.TestCase):
    def test_report_producers_do_not_declare_new_color_literals(self):
        source = Path(__file__).resolve().parents[1] / "src"
        offenders = [str(path.relative_to(source)) for path in source.rglob("*.py")
                     if path.name != "catalog.py"
                     and re.search(r"#[0-9a-fA-F]{6}\b", path.read_text())]
        self.assertEqual(offenders, [])

    def test_common_context_uses_actual_runs_and_omits_missing_fields(self):
        spec = {"run_meta": run_meta(
            {"id": "comparison"},
            runs={
                "old": run_meta({"id": "old"}, implementation={"master": {"source_commit": "abc"}}, workload=None),
                "new": run_meta({"id": "new"}, implementation={"master": {"source_commit": "def"}}, workload={"sha256": "trace"}),
            },
        )}
        rendered = render_context(spec)
        self.assertIn('>old</h3>', rendered)
        self.assertIn('>new</h3>', rendered)
        self.assertIn('&quot;source_commit&quot;: &quot;abc&quot;', rendered)
        self.assertIn('&quot;source_commit&quot;: &quot;def&quot;', rendered)
        self.assertNotIn('null', rendered)
        self.assertNotIn('未提供', rendered)
        self.assertNotIn('schema_version', rendered)
        self.assertEqual(render_context({"meta": {"version": {"branch": None}}}), "")

    def test_freeform_subtitle_and_scrollable_evidence(self):
        page = render({"title": "实验 A", "subtitle": {"流量": "240 QPS", "结论": "PASS"}})
        self.assertIn('"流量": "240 QPS"', page)
        self.assertIn('"结论": "PASS"', page)
        self.assertNotIn('id="hint"', page)
        self.assertNotIn('id="meta"', page)
        self.assertIn('class="attachment"', render_sections([details("证据", {"a": 1})]))
        self.assertIn('max-height:280px;overflow:auto', page)

    def test_overlay_preserves_old_new_line_styles(self):
        html = render({
            "title": "A/B",
            "panels": [{
                "id": "ab", "title": "A/B", "overlay": True,
                "axes": {"ratio": {"title": "命中率"}},
                "series": [
                    {"name": "old · hit", "axis": "ratio", "color": "#123456", "dash": [6, 4], "points": [{"x": 0, "y": 0.9}]},
                    {"name": "new · hit", "axis": "ratio", "color": "#654321", "dash": [], "points": [{"x": 0, "y": 0.9}]},
                ],
            }],
        })
        self.assertIn('"dash": [6, 4]', html)
        self.assertIn('"dash": []', html)

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

    def test_shared_pairing_supports_stage_and_event_anchors_without_zero_fill(self):
        self.assertEqual(event_anchor([dict(name="start", t=12)], "start"), 12)
        self.assertIsNone(event_anchor([dict(name="start", t=12), dict(name="start", t=13)], "start"))
        panel = dict(series=[dict(name="hit", color="#123456",
                                  points=[dict(x=12, y=None), dict(x=13, y=2)])])
        aligned = shifted_panel(panel, 12)
        self.assertEqual(panel["series"][0]["points"][0]["x"], 12)
        self.assertEqual(aligned["series"][0]["points"],
                         [dict(x=0, y=None), dict(x=1, y=2)])
        axis, left, right, delta = pair_samples([(0, None), (1, 2)], [(0, 3), (2, 4)])
        self.assertEqual((axis, left, right, delta),
                         ([0, 1, 2], [None, 2, None], [3, None, 4], [None, None, None]))
        curves, _ = paired_overlay([aligned, aligned])
        self.assertEqual([curve["name"] for curve in curves], ["A · hit", "B · hit"])
        self.assertEqual([curve["dash"] for curve in curves], [[6, 4], []])

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
            "workload.evidence_analysis.read_series",
            side_effect=AssertionError("renderer read telemetry"),
        ):
            result = write_report(d, payload)
            self.assertEqual(load_analysis(result)["status"], "ERROR")
            self.assertEqual(payload, original)

    def test_unsafe_report_identity_stays_inside_artifact_root(self):
        with tempfile.TemporaryDirectory() as d:
            result = write_bundle(d, "run", "../../other", {}, dict(panels=[]))
            self.assertEqual(result.parent, Path(d) / "reports/run")

    def test_all_bundle_kinds_are_discoverable_and_verified(self):
        with tempfile.TemporaryDirectory() as d:
            for kind in ("run", "comparison", "sweep"):
                write_bundle(d, kind, "example", {}, dict(panels=[]), role="audit")
                self.assertEqual(len(discover_reports(d, kind=kind, role="audit")), 1)
                self.assertEqual(len(discover_reports(d, kind=kind)), 1)
            self.assertEqual(discover_reports(d, role="missing"), [])
            with self.assertRaisesRegex(ValueError, "unsupported report kind"):
                discover_reports(d, kind="timeline")

    def test_time_panel_adapter_preserves_gaps_and_legacy_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            source = dict(title="mixed", panels=[
                dict(id="legacy", type="line", timeX=True, x=["0", "1"],
                     xNums=[0, 1], series=[dict(name="x", data=[3, None])]),
                dict(id="multi", overlay=True, axes={"y": dict(title="count")},
                     series=[dict(name="y", points=[dict(x=0, y=None)])]),
            ])
            bundle = write_bundle(d, "run", "mixed", {}, source)
            saved = json.loads((bundle / "report-spec.json").read_text())
            self.assertEqual(source["panels"][0]["series"][0]["data"], [3, None])
            self.assertEqual(saved["panels"][0]["series"][0]["points"],
                             [dict(x=0, y=3), dict(x=1, y=None)])
            self.assertEqual(saved["panels"][1]["series"][0]["points"],
                             [dict(x=0, y=None)])
            self.assertEqual([p["representation"] for p in saved["panels"]],
                             ["standard", "multi"])
            self.assertIn("FlexMultiCurve.mount", (bundle / "report.html").read_text())


if __name__ == "__main__":
    unittest.main()
