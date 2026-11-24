"""Performance framework behavior and failure regressions."""

import copy
import csv
import json
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from rtp_llm.test.perf_test.decode.report import generate_decode_report
from rtp_llm.test.perf_test.entry import plan_from_source
from rtp_llm.test.perf_test.execution.artifacts import copy_member_traces
from rtp_llm.test.perf_test.execution.engine_session import EngineSession
from rtp_llm.test.perf_test.multi_node.analyzer import batch_analyze as batch
from rtp_llm.test.perf_test.multi_node.analyzer.trace_analyze import analyze_trace
from rtp_llm.test.perf_test.prefill.report import generate_prefill_report
from rtp_llm.test.perf_test.tests.support.fixtures import (
    PERF_TEST_ROOT,
    fixture_config,
    fixture_execution,
    fixture_model,
    fixture_workload,
)
from rtp_llm.test.perf_test.tests.support.reports import (
    _setup_prefill_report_run,
    decode_report_exact_cases,
    prefill_report_case_bytes,
)


def kernel_trace(*events):
    return {
        "traceEvents": [
            {"cat": "kernel", "ph": "X", "name": name, "ts": start, "dur": duration}
            for name, start, duration in events
        ]
    }


class PrefillReportTest(unittest.TestCase):

    def test_complete_input_generates_formal_observations_formula_and_chart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            plan = _setup_prefill_report_run(root)
            report = generate_prefill_report(root)
            self.assertEqual("formal", report["report_kind"])
            self.assertEqual(2, report["counts"]["valid"])
            self.assertEqual([], report["excluded"])
            self.assertEqual(
                [case.case_id for case in plan.cases],
                [row["case_id"] for row in report["observations"]],
            )
            self.assertTrue((root / "reports" / "prefill_report.json").is_file())
            self.assertTrue((root / "reports" / "prefill_observations.json").is_file())
            html = (root / "reports" / "prefill_report.html").read_text()
            self.assertIn("formal", html)

    def test_invalid_case_is_excluded_and_report_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _setup_prefill_report_run(root, statuses=("ok", "invalid_reuse"))
            report = generate_prefill_report(root)
            self.assertEqual("diagnostic", report["report_kind"])
            self.assertEqual(1, report["counts"]["valid"])
            self.assertEqual(1, report["counts"]["excluded"])
            self.assertEqual("invalid_reuse", report["excluded"][0]["status"])

    def test_rerun_never_launches_engine_or_changes_case_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _setup_prefill_report_run(root)
            before = prefill_report_case_bytes(root)
            with patch.object(
                EngineSession, "start", side_effect=AssertionError("no engine")
            ) as start:
                first = generate_prefill_report(root)
                second = generate_prefill_report(root)
            self.assertEqual(first, second)
            start.assert_not_called()
            self.assertEqual(before, prefill_report_case_bytes(root))


class PrefillChartTest(unittest.TestCase):

    def test_plotly_available_tracks_the_offline_import(self):
        from rtp_llm.test.perf_test.prefill.chart import plotly_available

        with patch.dict(
            "sys.modules",
            {"plotly": None, "plotly.graph_objects": None, "plotly.offline": None},
        ):
            self.assertFalse(plotly_available())
        offline = types.ModuleType("plotly.offline")
        offline.get_plotlyjs = lambda: ""
        package = types.ModuleType("plotly")
        package.offline = offline
        with patch.dict("sys.modules", {"plotly": package, "plotly.offline": offline}):
            self.assertTrue(plotly_available())

    def test_standalone_charts_and_static_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            _setup_prefill_report_run(root, concurrent=True)
            report = generate_prefill_report(root)
            self.assertTrue(report["observations"])
            for row in report["observations"]:
                self.assertGreater(row["latency_ms"], 0)
                self.assertGreater(row["throughput_tokens_per_second"], 0)
            for metric in ("latency", "tpm"):
                document = (root / "reports" / f"{metric}.interactive.html").read_text()
                self.assertIn("<html", document.lower())
                self.assertEqual(
                    "{http://www.w3.org/2000/svg}svg",
                    ElementTree.parse(root / "reports" / f"{metric}.svg").getroot().tag,
                )
            with patch.dict(
                "sys.modules",
                {"plotly": None, "plotly.graph_objects": None, "plotly.offline": None},
            ):
                fallback_report = generate_prefill_report(root)
            self.assertEqual(report["observations"], fallback_report["observations"])
            for metric in ("latency", "tpm"):
                document = (root / "reports" / f"{metric}.interactive.html").read_text()
                self.assertIn("<svg", document)
                self.assertIn("request_index", document)


class DecodeReportTest(unittest.TestCase):

    def test_exact_report_groups_input_output_matrix_by_batch(self):
        cases = decode_report_exact_cases((128, 512), (16, 64), (1, 4))
        before = copy.deepcopy(cases)
        report = generate_decode_report(cases)
        self.assertEqual(2, report["schema_version"])
        self.assertEqual("formal", report["report_kind"])
        self.assertEqual({"planned": 8, "valid": 8, "excluded": 0}, report["counts"])
        self.assertEqual(
            [1, 4], [matrix["batch_size"] for matrix in report["matrices"]]
        )
        first = report["matrices"][0]
        self.assertEqual([128, 512], first["input_lengths"])
        self.assertEqual([16, 64], first["output_lengths"])
        self.assertEqual(4, len(first["cells"]))
        self.assertEqual(1.0, first["cells"][0]["tpot_ms"])
        self.assertEqual(100.0, first["cells"][0]["throughput_tokens_per_second"])
        self.assertEqual(17.0, first["cells"][0]["total_latency_ms"])
        self.assertEqual(1.0, first["cells"][0]["success_rate"])
        self.assertEqual(
            [item["case_id"] for item in cases],
            [item["case_id"] for item in report["observations"]],
        )
        for row in report["observations"]:
            self.assertNotIn("search", row)
        for matrix in report["matrices"]:
            for cell in matrix["cells"]:
                self.assertNotIn("search", cell)
        self.assertEqual(before, cases)
        first["cells"][0]["errors"].append({"message": "changed"})
        self.assertEqual([], cases[0]["errors"])


class TraceReportTest(unittest.TestCase):

    def make_run(
        self,
        root,
        mode,
        session="capture",
        *,
        status="completed",
        format="chrome_trace",
        tp=1,
    ):
        source = fixture_config(
            model=fixture_model(topology={"tp": tp, "dp": 1, "pp": 1, "ep": tp}),
            execution=fixture_execution(devices=list(range(tp))),
            output={"result_root": str(root.parent)},
            workload=(
                fixture_workload(
                    cache_lengths=[0], min_compute_len=128, max_input_len=128
                )
                if mode == "prefill"
                else {
                    "kind": "grid",
                    "input_lengths": [128],
                    "output_lengths": [16],
                    "batch": {"strategy": "fixed", "sizes": [1]},
                    "profiling": {"enabled": True, "steps": 2},
                }
            ),
        )
        plan, _ = plan_from_source(
            source, source_dir=root.parent, result_directory=str(root)
        )
        root.mkdir(exist_ok=True)
        (root / "execution_plan.json").write_text(json.dumps(plan.to_dict()))
        case_id = plan.cases[0].case_id
        trace = root / "traces" / session / case_id / "trace_wr0_1.json"
        trace.parent.mkdir(parents=True)
        trace.write_text(
            json.dumps(kernel_trace(("gemm", 0, 10), ("gemm", 5, 20), ("rare", 30, 2)))
        )
        manifest = {
            "schema_version": 2,
            "session_id": session,
            "mode": mode,
            "backend": "kineto" if format == "chrome_trace" else "nsys",
            "status": status,
            "diagnostic_only": True,
            "capture_scope": "one forward step",
            "cases": [
                {
                    "case_id": case_id,
                    "trace_name": "trace",
                    "status": status,
                    "error": None,
                    "artifacts": [
                        {
                            "path": trace.relative_to(root).as_posix(),
                            "format": format,
                            "rank": 0,
                            "phase": mode,
                        }
                    ],
                }
            ],
        }
        path = root / "profiles" / session / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(manifest))
        return path, trace

    def test_both_modes_and_repeated_sessions_keep_independent_raw_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("prefill", "decode"):
                self.make_run(root / mode, mode)
            self.make_run(root / "prefill", "prefill", "another")
            report = batch.analyze_runs(root)
            self.assertEqual([], report["skipped"])
            self.assertEqual(3, len(report["traces"]))
            self.assertEqual(
                {"prefill", "decode"}, {row["mode"] for row in report["traces"]}
            )
            for row in report["traces"]:
                self.assertEqual(0, row["rank"])
                self.assertEqual(1, row["topology"]["tp"])
                self.assertTrue(row["case_identity"])
                self.assertEqual(32, row["statistics"]["kernel_time_us"])
                self.assertEqual(32, row["statistics"]["kernel_span_us"])
                kernels = row["statistics"]["kernels"]
                self.assertEqual(["gemm", "rare"], [k["kernel_name"] for k in kernels])
                self.assertEqual(30, kernels[0]["total_duration_us"])
                self.assertEqual(15, kernels[0]["avg_duration_us"])

    def test_invalid_missing_and_unsupported_artifacts_are_explicitly_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root / "valid", "decode")
            for kind in (
                "failed",
                "running",
                "missing",
                "invalid",
                "nsys",
                "escape",
                "symlink",
            ):
                path, trace = self.make_run(
                    root / kind,
                    "prefill",
                    status=kind if kind in ("failed", "running") else "completed",
                )
                manifest = json.loads(path.read_text())
                if kind == "missing":
                    trace.unlink()
                if kind == "invalid":
                    trace.write_text('{"traceEvents": "bad"}')
                if kind == "nsys":
                    manifest["cases"][0]["artifacts"][0]["format"] = "nsys"
                if kind == "escape":
                    manifest["cases"][0]["artifacts"][0][
                        "path"
                    ] = "../valid/execution_plan.json"
                if kind == "symlink":
                    trace.unlink()
                    trace.symlink_to(root / "valid/execution_plan.json")
                path.write_text(json.dumps(manifest))
            report = batch.analyze_runs(root)
            self.assertEqual(1, len(report["traces"]))
            self.assertEqual(7, len(report["skipped"]))
            self.assertTrue(all(row["reason"] for row in report["skipped"]))

    def test_copied_run_cli_exports_csv_without_mutating_captured_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root / "original", "decode")
            copied = root / "copied"
            shutil.copytree(root / "original", copied)
            before = {
                str(p.relative_to(copied)): p.read_bytes()
                for p in copied.rglob("*")
                if p.is_file()
            }
            self.assertEqual(0, batch.main([str(copied)]))
            output = copied / "reports/trace"
            with (output / "kernels.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(["gemm", "rare"], [row["kernel_name"] for row in rows])
            self.assertEqual("0", rows[0]["rank"])
            for name, content in before.items():
                self.assertEqual(content, (copied / name).read_bytes())
            self.assertFalse((root / "original/reports").exists())
            self.assertTrue((output / "analysis.json").is_file())

    def test_member_trace_collection_updates_manifest_without_rank_counter_assumptions(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            path, trace = self.make_run(root, "decode", tp=2)
            logs = Path(directory) / "worker/logs"
            logs.mkdir(parents=True)
            (logs / "trace_wr1_77.json").write_bytes(trace.read_bytes())
            copy_member_traces(logs, root / "traces", "1")
            copy_member_traces(logs, root / "traces", "1")
            report = batch.analyze_runs(root)
            self.assertEqual([], report["skipped"])
            self.assertEqual([0, 1], sorted(row["rank"] for row in report["traces"]))
            stored = json.loads(path.read_text())
            paths = [item["path"] for item in stored["cases"][0]["artifacts"]]
            self.assertEqual(2, len(set(paths)))
            self.assertTrue(
                all(
                    p.startswith(
                        "traces/capture/" + stored["cases"][0]["case_id"] + "/"
                    )
                    for p in paths
                )
            )

    def test_parent_scan_stops_at_run_roots_and_keeps_sibling_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            self.make_run(parent / "controller", "decode")
            self.make_run(parent / "sibling", "prefill")
            nested = parent / "controller/execution/ssh/members/0/collected/run"
            shutil.copytree(parent / "controller", parent / "temporary-copy")
            nested.parent.mkdir(parents=True)
            shutil.move(str(parent / "temporary-copy"), nested)
            report = batch.analyze_runs(parent)
            self.assertEqual([], report["skipped"])
            self.assertEqual(2, len(report["traces"]))
            self.assertEqual(
                {"controller", "sibling"},
                {Path(row["run_directory"]).name for row in report["traces"]},
            )

    def test_missing_manifest_fields_are_diagnostic_instead_of_crashing_batch(self):
        for field in ("rank", "phase"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "run"
                path, _ = self.make_run(root, "decode")
                manifest = json.loads(path.read_text())
                del manifest["cases"][0]["artifacts"][0][field]
                path.write_text(json.dumps(manifest))
                report = batch.analyze_runs(root)
                self.assertEqual([], report["traces"])
                self.assertEqual(1, len(report["skipped"]))
                self.assertIn(field, report["skipped"][0]["reason"])

    def test_module_totals_include_rare_kernels_and_use_actual_event_counts(self):
        trace = kernel_trace(
            ("gemm_a", 0, 10), ("gemm_a", 10, 10), ("gemm_b", 20, 7), ("other", 27, 3)
        )
        result = analyze_trace(
            trace, min_occurrences=2, module_config={"gemm": "gemm_"}
        )
        self.assertEqual(["gemm_a"], [row["kernel_name"] for row in result["kernels"]])
        module = result["modules"][0]
        self.assertEqual(["gemm_a", "gemm_b"], module["kernel_names"])
        self.assertEqual(3, module["count"])
        self.assertEqual(27, module["total_duration_us"])
        self.assertEqual(90, module["kernel_time_percentage"])

        from rtp_llm.test.perf_test.multi_node.analyzer.configs.module_base import (
            ModuleConf,
            TestSetUp,
        )

        spec = ModuleConf(name_pattern="gemm", num_flop_calc_func=lambda setup: 1000)
        spec.formula_per_kernel = True
        rates = []
        for repeats in (1, 4):
            events = [
                {
                    "cat": "kernel",
                    "ph": "X",
                    "name": "gemm",
                    "ts": index * 10,
                    "dur": 10,
                }
                for index in range(repeats)
            ]
            row = analyze_trace(
                {"traceEvents": events},
                module_config={"gemm": spec},
                setup=TestSetUp(mode="prefill", steps=1),
            )["modules"][0]
            self.assertIsNotNone(row["estimated_tflops"])
            rates.append(row["estimated_tflops"])
        self.assertAlmostEqual(rates[0], rates[1])
        spec.formula_per_kernel = False
        module = analyze_trace(
            {"traceEvents": events},
            module_config={"gemm": spec},
            setup=TestSetUp(mode="prefill", steps=1),
        )["modules"][0]
        self.assertIsNone(module["estimated_tflops"])
        self.assertIn("invocation", module["estimation_error"])

    def test_restored_model_configs_export_modules_from_new_run_artifacts(self):
        configs = PERF_TEST_ROOT / "multi_node/analyzer/configs"
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "run"
                _, trace = self.make_run(root, mode)
                payload = json.loads(trace.read_text())
                for event in payload["traceEvents"][:2]:
                    event["name"] = "qkv<14336u, 6144u>"
                trace.write_text(json.dumps(payload))
                config = configs / (
                    "qwen_480b_prefill_fp8.py"
                    if mode == "prefill"
                    else "qwen_480b_decode-fp8.py"
                )
                self.assertEqual(
                    0, batch.main([str(root), "--module-conf", str(config)])
                )
                report = json.loads((root / "reports/trace/analysis.json").read_text())
                self.assertEqual([], report["skipped"])
                module = report["traces"][0]["statistics"]["modules"][0]
                tokens = 128 if mode == "prefill" else 1
                expected = 2 * 14336 * 6144 * tokens
                self.assertEqual("qkv_gemm", module["module_name"])
                self.assertEqual(30, module["total_duration_us"])
                self.assertEqual(expected, module["estimated_flops_per_invocation"])
                self.assertAlmostEqual(
                    expected * module["count"] / 30 / 1e6,
                    module["estimated_tflops"],
                )
                with (root / "reports/trace/modules.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual("qkv_gemm", rows[0]["module_name"])
                self.assertEqual("0", rows[0]["rank"])
                self.assertEqual("30", rows[0]["total_duration_us"])
                self.assertEqual(0, batch.main([str(root)]))
                self.assertFalse((root / "reports/trace/modules.csv").exists())
        self.assertEqual(7, len(batch.load_module_config(configs / "module_conf.py")))

    def test_module_estimates_explain_unknown_shapes_phases_and_formula_failures(self):
        from rtp_llm.test.perf_test.multi_node.analyzer.configs.module_base import (
            ModuleConf,
            TestSetUp,
            calc_gemm_mem_io,
        )

        self.assertEqual(41, calc_gemm_mem_io(2, 3, 5, 1, 2))
        trace = kernel_trace(("gemm", 0, 10))
        config = {
            "gemm": ModuleConf(name_pattern="gemm", num_flop_calc_func=lambda setup: -1)
        }
        config["gemm"].formula_per_kernel = True
        result = analyze_trace(
            trace, module_config=config, setup=TestSetUp(steps=1, mode="prefill")
        )
        module = result["modules"][0]
        self.assertIsNone(module["estimated_flops_per_invocation"])
        self.assertIn("nonnegative", module["estimation_error"])
        config["gemm"].num_flop_calc_func = lambda setup: 2000
        module = analyze_trace(
            trace, module_config=config, setup=TestSetUp(mode="decode")
        )["modules"][0]
        self.assertEqual(2000, module["estimated_flops_per_invocation"])
        self.assertIsNone(module["estimated_tflops"])
        self.assertIn("capture", module["estimation_error"])
        row = {
            "mode": "prefill",
            "phase": None,
            "topology": {"tp": 1, "dp": 1},
            "case_identity": {
                "batch_size": 2,
                "request_groups": [
                    {"count": 1, "input_len": 128, "cache_len": 0},
                    {"count": 1, "input_len": 256, "cache_len": 0},
                ],
            },
        }
        setup = batch.module_setup(row)
        self.assertIn("heterogeneous", setup.estimation_error)

    def test_kernel_filter_preserves_raw_span_and_rejects_nonfinite_timing(self):
        trace = kernel_trace(("long", 0, 100), ("short", 1, 2), ("short", 5, 2))
        stats = analyze_trace(trace, min_occurrences=2)
        self.assertEqual(104, stats["kernel_time_us"])
        self.assertEqual(100, stats["kernel_span_us"])
        self.assertEqual(["short"], [row["kernel_name"] for row in stats["kernels"]])
        trace["traceEvents"][0]["dur"] = float("nan")
        with self.assertRaises(ValueError):
            analyze_trace(trace)


if __name__ == "__main__":
    unittest.main()
