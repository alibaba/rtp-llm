import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cases.config import configure_program
from reporting import discover_reports, load_analysis, read_bundle, write_bundle
from reporting.view_config import DEFAULT_VIEW, declaration, view
from scenario.loader import ScenarioError, load_document
from workload.report import write_report, write_views
from workload.report_panels import build_panels
from cases.master_ha_failover.metrics import _prefill_balance_series

ROOT = Path(__file__).resolve().parents[1]


def payload(series=None, gates=None):
    return dict(
        id="example", status="PASS", workload={"runtime_validity": "VALID"},
        implementation={}, configuration={}, configuration_sha256=None,
        clock_anchor={"epoch_s": 10}, request_sources=[], series=series or {},
        statistic_sources={k: dict(unit="count") for k in series or {}}, checks=[dict(stage="observe", id="cache", status="PASS")],
        iterations=[], traffic_manifests=[], gate_reports=gates or [],
    )


class WorkloadReportViewsTest(unittest.TestCase):
    def test_case_metric_ids_and_axes_are_validated(self):
        presentation = view("cache_scale_in.yaml")
        self.assertEqual(
            presentation["charts"]["panels"][0]["curve_ids"][1], "mock/cache_hit_ratio"
        )
        broken = copy.deepcopy(presentation)
        broken["charts"]["panels"] = [dict(panel) for panel in presentation["charts"]["panels"]]
        broken["charts"]["panels"][0]["axes"] = {"count": {"title": "数量", "position": "left"}}
        with mock.patch("reporting.view_config.load_document", return_value=broken):
            with self.assertRaisesRegex(ScenarioError, "axis is not declared"):
                view("cache_scale_in.yaml")

    def test_unused_catalog_query_does_not_force_collection_or_presentation(self):
        from monitoring.query_plan import load_plan

        plan = load_plan("cache_scale_in.yaml")
        plan["sources"]["master"]["new_metric"] = {"promql": "new_metric${selector}"}
        with mock.patch("monitoring.query_plan.load_plan", return_value=plan):
            from monitoring.collection_plan import select_plan
            selected = select_plan(plan, {}, [view("cache_scale_in.yaml")])
            self.assertNotIn("new_metric", selected["sources"]["master"])

    def test_ha_core_view_aligns_events_requests_and_master_state(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            requests = root / "client_events.jsonl"
            requests.write_text("\n".join(json.dumps(row) for row in [
                {"send_start_epoch_ms": 11_100, "status": "ok", "prefill": "P0"},
                {"send_start_epoch_ms": 11_800, "status": "exception", "prefill": "P0"},
                {"send_start_epoch_ms": 12_100, "status": "ok", "prefill": "P1"},
            ]) + "\n")
            state = root / "master_states.jsonl"
            state.write_text("\n".join(json.dumps(row) for row in [
                {"epoch_s": 11, "master": "A", "http_up": 1,
                 "scheduler_inflight": 3, "prefill_inflight_requests": 2,
                 "decode_master_queued": 1, "decode_confirmed_running": 4},
                {"epoch_s": 11, "master": "B", "http_up": 0},
                {"epoch_s": 15, "master": "A", "http_up": 0},
            ]) + "\n")
            analysis = payload()
            analysis["id"] = "master_ha_failover::default::batch-window"
            analysis["status"] = "FAIL"
            analysis["configuration"] = {"environment": {"n_prefill": 2}}
            analysis["stages"] = [dict(id="finish", output=dict(rows=dict(kind="ha_rows", env_epoch=1)),
                                       artifacts=[str(requests), str(state)])]
            analysis["phases"] = [dict(stage="kill_a", event="end", epoch_s=11)]
            from monitoring.metric_store import export_metrics
            from monitoring.query_plan import load_plan
            from monitoring.producers import produce
            from metric_fixtures import freeze_metrics
            freeze_metrics(root, "master_ha_failover")
            produce(root, analysis)
            requests.unlink()
            state.unlink()
            paths = write_views(root, analysis, ["default.yaml", "master_ha_failover.yaml"])
            path = paths["master_ha_failover.yaml"]
            spec = json.loads((path.parent / "report-spec.json").read_text())
            panels = {panel["id"]: panel for panel in spec["panels"]}
            self.assertEqual({"request_qps", "inflight", "prefill_balance"}, set(panels))
            self.assertEqual([1, 0], [point["y"] for point in panels["request_qps"]["series"][2]["points"]])
            self.assertEqual([1, 0], [point["y"] for point in panels["request_qps"]["series"][3]["points"]])
            self.assertEqual([3, None], [point["y"] for point in panels["inflight"]["series"][0]["points"]])
            self.assertEqual([1, 0], [point["y"] for point in panels["inflight"]["series"][-2]["points"]])
            balance = panels["prefill_balance"]["series"]
            self.assertTrue(all(not curve["points"] for curve in balance[:3]))
            self.assertEqual(["A · HTTP 可回读", "B · HTTP 可回读"],
                             [curve["name"] for curve in balance[3:]])
            self.assertEqual([1, 0], [point["y"] for point in balance[3]["points"]])
            self.assertEqual([0], [point["y"] for point in balance[4]["points"]])
            self.assertTrue(all(curve["axis"] == "up" for curve in balance[3:]))
            self.assertTrue(all(panel["axes"]["up"]["position"] == "right"
                                and panel["events"] for panel in panels.values()))
            self.assertEqual(1, panels["request_qps"]["events"][0]["t"])
            self.assertEqual(5, spec["timeAxis"]["max"])
            self.assertEqual(spec["title"], "master_ha_failover : default : batch-window")
            self.assertEqual(spec["subtitle"], view("master_ha_failover.yaml")["report"]["subtitle"])
            default = json.loads((paths["default.yaml"].parent / "report-spec.json").read_text())
            self.assertNotIn("../ha-ha-core/report.html", json.dumps(default["sections"]))

    def test_ha_balance_chart_matches_five_second_gate_window(self):
        rows = [
            {"send_start_epoch_ms": 11_000 + (i // 60) * 1000,
             "status": "exception" if i < 30 else "ok",
             "prefill": f"P{i % 125}"}
            for i in range(300)
        ]
        series = _prefill_balance_series(rows, 10, 125)
        self.assertEqual([dict(x=5, y=0.6)], series["prefill_peak_qps"])
        self.assertEqual([dict(x=5, y=0.48)], series["prefill_mean_qps"])
        self.assertEqual([dict(x=5, y=1.25)], series["prefill_skew"])

    def test_per_engine_metrics_share_one_chart_with_switchable_detail(self):
        series = {
            f'1/mock/running_streams/{{"engine_name":"p-{i}"}}':
                [[t, None if t == 201 else i + t % 11] for t in range(1800)]
            for i in range(128)
        }
        with tempfile.TemporaryDirectory() as d:
            bundle = write_report(d, payload(series), name=DEFAULT_VIEW)
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
            if doc.get("metadata", {}).get("kind") != "workload":
                continue
            configured = configure_program(doc, str(path))
            for variant in configured["variants"]:
                if variant["metadata"]["kind"] != "workload":
                    self.assertNotIn("reports", variant)
                    continue
                names = variant["reports"]
                self.assertEqual(len(names), 1)
                self.assertNotIn(DEFAULT_VIEW, names)
                self.assertTrue(all((ROOT / "config/report_views" / name).is_file()
                                    for name in names))
        cache = load_document(ROOT / "config/scenarios/cache_scale_in.yaml")
        self.assertEqual(cache["reports"], ["cache_scale_in.yaml"])
        series = {
            f'1/mock/qps/{{"engine_name":"p-{i}"}}': [[0, i], [1, i + 1]]
            for i in range(2)
        }
        self.assertEqual(len(build_panels(series, {k: dict(unit="req/s") for k in series}, view(DEFAULT_VIEW))), 1)

    def test_unlabelled_legacy_statistic_is_still_visible(self):
        key = "statistics/1/per_second/e2e_p99"
        panels = build_panels({key: [[0, 123]]}, {key: dict(unit="ms")})
        self.assertEqual(len(panels), 1)
        self.assertEqual(panels[0]["series"][0]["provenance"]["source_series_key"], key)

    def test_full_view_uses_frozen_units_without_metric_name_inference(self):
        key = '1/mock/cache_hit_ratio/{"role":"prefill"}'
        values = [[0, 123], [1, None]]
        panels = build_panels({key: values}, {key: dict(unit="ms")})
        self.assertEqual(panels[0]["axes"], {"y": dict(title="ms", position="left")})
        self.assertEqual(panels[0]["series"][0]["unit"], "ms")
        self.assertEqual(panels[0]["series"][0]["points"],
                         [dict(x=0, y=123), dict(x=1, y=None)])
        self.assertIn("cache_hit_ratio", panels[0]["title"])
        with self.assertRaisesRegex(ValueError, "missing frozen metric unit"):
            build_panels({key: values}, {})

    def test_selected_view_renders_verified_frozen_gate_once(self):
        from cases.cache_scale_in.analysis import analyze
        from workload.gate_result import freeze_gate
        from test_cache_scale_gate import CacheGateTest
        from metric_fixtures import freeze_metrics
        with tempfile.TemporaryDirectory() as d:
            e = CacheGateTest().evidence(.8)
            decision = analyze(e)
            freeze_gate(d, "cache", e, decision)
            freeze_metrics(d, "cache_scale_in")
            before = (Path(d) / "cache-gate-result.json").read_bytes()
            from cases.cache_scale_in import report
            with mock.patch.object(report, "write_report", wraps=report.write_report) as render:
                links = write_views(d, payload(), ["default.yaml", "cache_scale_in.yaml"])
            self.assertEqual(render.call_count, 1)
            frozen = load_analysis(links["cache_scale_in.yaml"].parent)
            self.assertEqual({k: v for k, v in frozen.items() if k != "run"}, decision)
            self.assertEqual(frozen["run"]["status"], "PASS")
            self.assertEqual(before, (Path(d) / "cache-gate-result.json").read_bytes())
            read_bundle(links["default.yaml"].parent)

    def test_missing_view_source_and_invalid_declarations_fail_loud(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(OSError):
            write_views(d, payload(), ["default.yaml", "cache_scale_in.yaml"])
        for names in ([], ["missing.yaml", "default.yaml"], ["../default.yaml"],
                      ["default.yaml", "default.yaml"],
                      ["default.yaml", {"action": "render"}]):
            with self.subTest(names=names), self.assertRaises(ScenarioError):
                declaration(names, kind="workload")
        with self.assertRaisesRegex(ScenarioError, "metadata.kind=workload"):
            declaration([DEFAULT_VIEW], kind="functional")

    def test_timeout_keeps_monitoring_report_when_gate_was_not_produced(self):
        with tempfile.TemporaryDirectory() as d:
            data = payload({'1/mock/running/{"engine_name":"p0"}': [[0, 1]]})
            data.update(status="TIMEOUT", checks=[])
            data["workload"]["runtime_validity"] = "INVALID"
            paths = write_views(d, data, ["cache_scale_in.yaml"])
            self.assertEqual({DEFAULT_VIEW}, set(paths))
            frozen = load_analysis(paths[DEFAULT_VIEW].parent)
            self.assertEqual("TIMEOUT", frozen["status"])
            self.assertEqual("INVALID", frozen["workload"]["runtime_validity"])
            self.assertEqual("NOT_PRODUCED", frozen["unavailable_report_views"][0]["status"])
            self.assertEqual(data["series"], frozen["series"])
            self.assertNotIn("verdict", frozen)
            self.assertFalse((Path(d) / "reports/run/cache-scale-in").exists())
            self.assertIn("未生成的报告视角", paths[DEFAULT_VIEW].read_text())

    def test_failed_run_does_not_hide_corrupt_frozen_gate(self):
        from cases.cache_scale_in.analysis import analyze
        from workload.gate_result import freeze_gate
        from test_cache_scale_gate import CacheGateTest
        with tempfile.TemporaryDirectory() as d:
            e = CacheGateTest().evidence(.8)
            freeze_gate(d, "cache", e, analyze(e))
            (Path(d) / "cache-gate-result.json").write_text("{}")
            data = payload()
            data["status"] = "ERROR"
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                write_views(d, data, ["default.yaml", "cache_scale_in.yaml"])

    def test_performance_report_title_follows_frozen_run_identity(self):
        from cases.master_performance.report import write_report as report

        presentation = view("master_performance.yaml").copy()
        with tempfile.TemporaryDirectory() as d, mock.patch(
            "reporting.view_config.view", return_value=presentation
        ), mock.patch(
            "cases.master_performance.panels.prepare_curves",
            return_value=([
                dict(curve_id="mock/rtp_llm_context_tps_engine_mean/P", metric_id="mock/rtp_llm_context_tps_engine_mean/P", name="P throughput", group="Prefill TPS", axis="forward", hidden=False, points=[]),
                dict(curve_id="mock/mock_decode_wall_tps_engine_mean/D", metric_id="mock/mock_decode_wall_tps_engine_mean/D", name="D detail", group="Decode 逐引擎 TPS", axis="forward", hidden=True, points=[]),
            ], {}),
        ):
            bundle = report(d, {"criteria": {"measure_s": 1}, "window": {"start_epoch_ms": 0},
                                "provenance": {"instance": "master_performance::default::single-nonbatch"}},
                            {"verdict": "PASS", "checks": [], "errors": [], "metrics": {}, "windows": []})
            spec = json.loads((bundle / "report-spec.json").read_text())
            self.assertEqual(spec["title"], "master_performance : default : single-nonbatch")
            self.assertEqual([panel["title"] for panel in spec["panels"]],
                             [panel["title"] for panel in presentation["charts"]["panels"]])
            self.assertEqual([panel["id"] for panel in spec["panels"]],
                             ["engine-tps", "client-qps", "latency", "cache-hit",
                              "prefill-batch", "prefill-state"])


if __name__ == "__main__":
    unittest.main()
