import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reporting import discover_reports, load_analysis, read_bundle, render, table, write_bundle
from reporting.run_context import canonical_spec, provenance_from
from workload.report import write_views
from workload.run_provenance import collect


def analysis():
    return dict(id="case::variant::profile", status="FAIL", series={'1/mock/qps/{"role":"prefill"}': [[0, None], [1, 3]]},
                statistic_sources={'1/mock/qps/{"role":"prefill"}': dict(unit="requests/s")},
                clock_anchor=dict(epoch_s=10),
                workload={"runtime_validity": "VALID"},
                checks=[dict(stage="end", id="inflight", status="PASS",
                             actual={"endpoint_loads": {"PREFILL": [0] * 125, "DECODE": [0] * 536}},
                             expected=0)], configuration={"n_prefill": 125},
                runtime_provenance={"topology": {"n_prefill": 125}}, implementation={"sha": "abc"})


class ReportPresentationTest(unittest.TestCase):
    def test_default_is_single_report_with_all_archived_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            source = analysis()
            paths = write_views(directory, source)
            self.assertEqual(["default.yaml"], list(paths))
            self.assertEqual([str(next(iter(paths.values())).resolve())], discover_reports(directory))
            bundle = read_bundle(next(iter(paths.values())))
            spec = json.loads((bundle / "report-spec.json").read_text())
            self.assertEqual(1, len(spec["panels"]))
            self.assertEqual("case : variant : profile", spec["title"])
            self.assertEqual(source["series"], load_analysis(bundle)["series"])
            self.assertEqual("FAIL", load_analysis(bundle)["status"])
            html = (bundle / "report.html").read_text()
            self.assertEqual(1, html.count("运行信息（制品与配置）"))
            self.assertNotIn("Run provenance", html)
            self.assertIn('class="cell-detail"', html)
            self.assertIn("536 项；min=0，max=0", html)

    def test_dedicated_is_primary_even_when_full_view_is_first_in_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            original = {"verdict": "FAIL", "checks": [], "threshold": 0.3}
            bundle = write_bundle(directory, "run", "cache-scale-in", original,
                                  dict(title="old", panels=[], kpis=[], timeOriginEpochS=10),
                                  producer="cache-gate", role="gate")
            source = analysis()
            with mock.patch("workload.report.build_panels", return_value=[]):
                paths = write_views(directory, source, ["default.yaml", "cache_scale_in.yaml"])
            self.assertEqual(["cache_scale_in.yaml", "default.yaml"], list(paths))
            frozen = load_analysis(bundle)
            self.assertEqual(original, {key: frozen[key] for key in original})
            self.assertEqual(source, frozen["run"])
            self.assertEqual(2, len(discover_reports(directory)))
            self.assertEqual(1, len(discover_reports(directory, role="gate")))

    def test_common_sections_are_idempotent_and_verdict_diagnostics_remain_frozen(self):
        source = analysis()
        spec = dict(panels=[], sections=[dict(type="details", title="策略诊断", value={"verdict": "FAIL"})])
        first = canonical_spec(spec, source)
        self.assertEqual(first, canonical_spec(first, source))
        self.assertEqual(["门禁检查", "有效性与证据完整性", "策略诊断"],
                         [item["title"] for item in first["sections"]])
        self.assertNotIn("title", spec)

    def test_context_preserves_gate_sources_and_raw_evidence_is_escaped(self):
        source = analysis()
        meta = provenance_from(source, {"identity": {"verdict": "FAIL"},
                                       "evidence": {"sha256": "gate-hash"}})
        self.assertEqual("gate-hash", meta["evidence"]["gate"]["sha256"])
        self.assertEqual("FAIL", meta["identity"]["verdict"])
        page = render(dict(title="case", run_meta=meta, panels=[], sections=[
            table("诊断", ["原始值"], [[{"raw": "<script>" * 100}]], opened=False)]))
        self.assertIn('&lt;script&gt;', page)
        self.assertIn('<details class="report-block attachment"><summary>诊断', page)

    def test_provenance_is_observed_not_fabricated_and_malformed_inputs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "environment"
            environment.mkdir()
            (environment / "actual-master-config.json").write_text('{"dispatcher":"batch"}')
            flow = root / "ha-flow"
            flow.mkdir()
            (flow / "flow-input.json").write_text('{"trace":{"sha256":"trace"},"environment":{"QPS":10}}')
            with mock.patch("runtime.paths.MOCK_JAR", root / "absent.jar"):
                value = collect(root, {"epoch": environment}, {"epoch": {"n_prefill": 125}})
            self.assertEqual({"dispatcher": "batch"}, value["environments"]["epoch"]["actual_master_config"])
            self.assertEqual("trace", value["flows"][0]["trace"]["sha256"])
            self.assertIsNone(value["mock_artifact"])
            self.assertNotIn("performance", value["environments"]["epoch"])
            (environment / "actual-master-config.json").write_text("broken")
            with self.assertRaises(json.JSONDecodeError):
                collect(root, {"epoch": environment})
