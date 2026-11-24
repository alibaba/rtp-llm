"""Performance framework behavior and failure regressions."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.ci_delivery import deliver_config
from rtp_llm.test.perf_test.tests.support.fixtures import (
    PERF_TEST_ROOT,
    fixture_config,
    fixture_envelope,
)


class CiDeliveryTest(unittest.TestCase):

    def setUp(self):
        self.checkout = PERF_TEST_ROOT.parents[2]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.destination = Path(self.temp.name) / "delivered.json"

    def deliver(self, *, MODE="decode", CONFIG="auto"):
        return deliver_config(MODE, CONFIG, self.checkout, self.destination)

    def test_missing_ep_is_resolved_by_planner(self):
        base = self.deliver(MODE="prefill")
        base["model"]["topology"].pop("ep")
        delivered = self.deliver(MODE="prefill", CONFIG=json.dumps(base))
        self.assertNotIn("ep", delivered["model"]["topology"])
        self.assertEqual(
            2, self.plan_delivered("prefill").resolved_config["model"]["topology"]["ep"]
        )

    def test_explicit_inline_config_is_authoritative(self):
        base = self.deliver(MODE="prefill")
        base["model"]["label"] = "custom"
        base["model"]["type"] = "configured"
        base["model"]["checkpoint"] = "/configured/model"
        base["model"]["tokenizer"] = "/configured/tokenizer"
        base["execution"]["backend"] = "local"
        base["execution"]["test_timeout_seconds"] = 19500
        config = self.deliver(MODE="prefill", CONFIG=json.dumps(base))
        self.assertEqual(config["model"], base["model"])
        self.assertEqual(
            config["execution"],
            dict(
                backend="aone",
                resource="daily",
                gpu_type="H20_CU13",
                max_parallel=2,
                gpus_per_machine=1,
                test_timeout_seconds=19500,
            ),
        )
        self.assertEqual(config["workload"], base["workload"])

    def test_both_modes_output_root_matches_ci_gate_and_upload(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                source = self.deliver(MODE=mode)
                source["output"]["result_root"] = "custom-results"
                delivered = self.deliver(MODE=mode, CONFIG=json.dumps(source))
                expected = "perf_test_results"
                self.assertEqual(expected, delivered["output"]["result_root"])
                self.assertEqual(
                    expected,
                    json.loads(self.destination.read_text())["output"]["result_root"],
                )

    def plan_delivered(self, mode):
        from rtp_llm.test.perf_test.cli import parse_cli
        from rtp_llm.test.perf_test.entry import build_request_plan

        return build_request_plan(
            parse_cli(
                ["--mode=" + mode, "--action=run", "--config=" + str(self.destination)]
            )
        )[0]

    def test_single_host_config_preserves_case_shards(self):
        base = self.deliver(MODE="prefill")
        base["model"]["topology"] = dict(tp=2, dp=1, pp=1, ep=1)
        base["execution"].update(shard_count=2, max_parallel=2)
        base["execution"].pop("gpus_per_machine", None)
        source_path = Path(self.temp.name) / "single_host.json"
        source_path.write_text(json.dumps(base))
        for spec in (json.dumps(base), str(source_path)):
            with self.subTest(spec=spec):
                config = self.deliver(MODE="prefill", CONFIG=spec)
                self.assertNotIn("gpus_per_machine", config["execution"])
                plan = self.plan_delivered("prefill")
                self.assertEqual(1, plan.resource_plan.machine_count)
                self.assertEqual(2, plan.resource_plan.shard_count)

    def test_explicit_gang_config_selects_machine_count(self):
        base = self.deliver(MODE="prefill")
        base["model"]["topology"] = dict(tp=2, dp=1, pp=1, ep=1)
        base["execution"] = dict(
            backend="aone",
            resource="daily",
            gpu_type="H20_CU13",
            max_parallel=2,
            gpus_per_machine=1,
        )
        config = self.deliver(MODE="prefill", CONFIG=json.dumps(base))
        self.assertNotIn("shard_count", config["execution"])
        self.assertEqual(2, self.plan_delivered("prefill").resource_plan.machine_count)

    def test_relocated_configs_keep_distribution_content(self):
        from rtp_llm.test.perf_test.aone.config_schema import parse_execution
        from rtp_llm.test.perf_test.config_resolver import load_resolved_config

        for mode, named in (("prefill", False), ("prefill", True), ("decode", False)):
            with self.subTest(mode=mode, named=named):
                execution = dict(
                    backend="aone",
                    resource="daily",
                    gpu_type="H20_CU13",
                    shard_count=1,
                    max_parallel=1,
                )
                if mode == "prefill":
                    workload = {
                        "kind": "cache_grid",
                        "cache_alignment": 64,
                        "commit_tail_tokens": 64,
                        "probe_block_size": 64,
                        "workspace_tokens": 4096,
                        "seed": 7,
                        "traffic_request_batches": {
                            "distribution_file": "distribution.json",
                            "num_cases": 3,
                            "batch_sizes": [1],
                            "min_input_tokens": 256,
                            "max_input_tokens": 1024,
                        },
                    }
                    distribution = {
                        "schema_version": 1,
                        "kind": "conditional_request_lengths",
                        "input_bucket_width": 256,
                        "reuse_ratio_bins": 4,
                        "source_rows": 2,
                        "incomplete_rows": 0,
                        "cells": [
                            dict(
                                input_bucket=1,
                                reuse_bin=-1,
                                count=1,
                                input_min=256,
                                input_max=511,
                            ),
                            dict(
                                input_bucket=2,
                                reuse_bin=2,
                                count=1,
                                input_min=512,
                                input_max=767,
                            ),
                        ],
                    }
                    base = fixture_config(
                        workload=workload,
                        execution={
                            "backend": "aone",
                            "platform": "cuda13",
                            "local": {"devices": [0]},
                            "aone": {
                                key: value
                                for key, value in execution.items()
                                if key != "backend"
                            },
                        },
                    )
                    base["schema_version"] = 2
                else:
                    base = fixture_config(
                        execution=execution,
                        workload={
                            "kind": "distribution",
                            "distribution_file": "distribution.json",
                            "output_length": 16,
                            "batch": {"strategy": "fixed", "sizes": [1]},
                        },
                    )
                    distribution = [128, 256]
                source_copy = (
                    Path(self.temp.name) / "source" / f"{mode}-{named}" / "config.json"
                )
                source_copy.parent.mkdir(parents=True)
                source_copy.write_text(json.dumps(base))
                (source_copy.parent / "distribution.json").write_text(
                    json.dumps(distribution)
                )
                expected = load_resolved_config(
                    source_copy, execution_parser=parse_execution
                ).resolved_config["workload"]
                if named:
                    traffic = base["workload"].pop("traffic_request_batches")
                    base["workload"]["grids"] = [
                        dict(name="traffic", traffic_request_batches=traffic)
                    ]
                    source_copy.write_text(json.dumps(base))
                self.deliver(MODE=mode, CONFIG=str(source_copy))
                actual = load_resolved_config(
                    self.destination, execution_parser=parse_execution
                ).resolved_config["workload"]
                if named:
                    self.assertEqual(
                        expected["request_batches"],
                        actual["grids"][0]["workload"]["request_batches"],
                    )
                else:
                    self.assertEqual(expected, actual)

    def test_inline_distribution_paths_are_relative_to_checkout(self):
        source = self.deliver(MODE="prefill")
        source["workload"] = {
            "kind": "cache_grid",
            "traffic_request_batches": {"distribution_file": "requests.json"},
        }
        delivered = self.deliver(MODE="prefill", CONFIG=json.dumps(source))
        path = delivered["workload"]["traffic_request_batches"]["distribution_file"]
        self.assertEqual(str(self.checkout / "requests.json"), path)


class PrefillVerificationTest(unittest.TestCase):

    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("rtp_llm.test.perf_test.ci_prefill_verify"),
            "collected-artifact verifier is absent",
        )
        from rtp_llm.test.perf_test.ci_prefill_verify import verify_prefill_run

        self.verify = verify_prefill_run
        from rtp_llm.test.perf_test.prefill.chart import plotly_available

        self.require_interactive_charts = plotly_available()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build_artifacts(batch_size=4)

    def build_artifacts(self, batch_size):
        self.batch_size = batch_size
        self.cases = []
        for i in range(30):
            length = 128 + i * 64
            identity = dict(
                input_len=length + 64,
                cache_len=64,
                compute_len=length + 64,
                batch_size=self.batch_size,
                measure_runs=1,
                prefix_policy="independent",
                measurement_contract="concurrent_grouped_batch_v1",
                request_groups=[
                    dict(count=self.batch_size - 1, input_len=length, cache_len=64),
                    dict(count=1, input_len=length + 64, cache_len=0),
                ],
            )
            members = [
                dict(
                    run=0,
                    request_index=j,
                    group_index=0 if j < self.batch_size - 1 else 1,
                    input_length=n,
                    expected_reuse_length=c,
                    observed_reuse_length=c,
                    compute_length=n - c,
                    latency_ms=10.0,
                    round_elapsed_ms=20.0,
                    measurement_contract=identity["measurement_contract"],
                )
                for (j, (n, c)) in enumerate(
                    [(length, 64)] * (self.batch_size - 1) + [(length + 64, 0)]
                )
            ]
            compute = (self.batch_size - 1) * (length - 64) + length + 64
            metrics = dict(
                input_length=compute + 64 * (self.batch_size - 1),
                expected_reuse_length=64 * (self.batch_size - 1),
                observed_reuse_length=64 * (self.batch_size - 1),
                compute_length=compute,
                latency_ms=10.0,
                throughput_tokens_per_second=compute * 50.0,
                validation=dict(shape=True, reuse=True, timing=True),
                seed=1,
                prefix_policy=dict(kind="independent"),
                measurements=members,
            )
            self.cases.append(
                dict(
                    case_id=f"case_{i}",
                    identity=identity,
                    status="ok",
                    runs=1,
                    errors=[],
                    metrics=metrics,
                )
            )
        self.result = fixture_envelope(
            "prefill",
            self.cases,
            run_id="prefill-test",
            backend="aone",
            model=dict(
                label="test",
                type="test",
                checkpoint="/m",
                topology=dict(tp=2, dp=1, pp=1, ep=1),
            ),
        )
        self.write(
            "resolved_config.json", dict(workload=dict(transport="dashsc_input_ids"))
        )
        self.write("results/result.json", self.result)
        from types import SimpleNamespace

        from rtp_llm.test.perf_test.prefill.report import _observation

        rows = [
            _observation(
                SimpleNamespace(case_id=c["case_id"], identity=c["identity"]), c
            )[0]
            for c in self.cases
        ]
        self.report = dict(
            mode="prefill",
            report_kind="formal",
            counts=dict(planned=30, valid=30, excluded=0),
            observations=rows,
            excluded=[],
            config_fingerprint="a" * 64,
            workload_fingerprint="b" * 64,
        )
        self.write("reports/prefill_report.json", self.report)
        self.write(
            "checkpoints/prefill_cache_probe.json",
            dict(
                status="ok",
                block_size=64,
                responses=dict(seed=dict(reuse_len=0), measure=dict(reuse_len=64)),
            ),
        )
        self.write(
            "reports/formula/fit_report.json",
            dict(
                candidate_generated=True,
                complete=True,
                unique_geometry_count=30,
                production_acceptance=False,
            ),
        )
        self.write(
            "reports/formula/model.json",
            dict(formula="1+I", production_acceptance=False),
        )
        for name, content in [
            ("PREFILL_TIME_FORMULA.txt", "PREFILL_TIME_FORMULA=1+I"),
            ("predictions.csv", "source,predicted_ms\ncase_0,10"),
            ("fit_gap.svg", "<svg></svg>"),
        ]:
            self.write_text("reports/formula/" + name, content)
        from rtp_llm.test.perf_test.prefill.chart import render_charts

        if not hasattr(type(self), "chart_artifacts"):
            type(self).chart_artifacts = render_charts(self.report)
        for name, content in self.chart_artifacts.items():
            self.write_text("reports/" + name, content)
        self.write_text("reports/prefill_report.html", "<html>formal report</html>")

    def write_text(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def write(self, name, value):
        self.write_text(name, json.dumps(value))

    def test_candidate_without_production_acceptance_passes(self):
        summary = self.verify(
            self.root, require_interactive_charts=self.require_interactive_charts
        )
        self.assertEqual(summary["valid_geometries"], 30)

    def test_failed_cache_probe_fails(self):
        for probe in (
            dict(status="failed"),
            dict(
                status="ok",
                block_size=64,
                responses=dict(seed=dict(reuse_len=0), measure=dict(reuse_len=0)),
            ),
        ):
            self.write("checkpoints/prefill_cache_probe.json", probe)
            with self.assertRaises(ValueError):
                self.verify(
                    self.root,
                    require_interactive_charts=self.require_interactive_charts,
                )

    SVG_FALLBACK_CHART = "<!doctype html><html><body><h1>Prefill 3D chart</h1><p>Plotly unavailable; showing static chart.</p><svg></svg></body></html>"
    EMBEDDED_PLOTLY_CHART = "<!doctype html><html><body><div id=\"chart\"></div><script>/* plotly.js v1.2.3 */ window.Plotly = {}; window.Plotly.newPlot = function () {}; window.Plotly.react = function () {}; window.Plotly.relayout = function () {};</script><script>const layout = {updatemenus: []}; Plotly.newPlot('chart', [], layout);</script></body></html>"

    def write_interactive_charts(self, content):
        for chart in ("latency", "tpm"):
            self.write_text(f"reports/{chart}.interactive.html", content)

    def test_relaxed_chart_check_accepts_static_fallback(self):
        self.write_interactive_charts(self.SVG_FALLBACK_CHART)
        summary = self.verify(self.root, require_interactive_charts=False)
        self.assertEqual(summary["valid_geometries"], 30)

    def test_strict_chart_check_rejects_static_fallback(self):
        self.write_interactive_charts(self.SVG_FALLBACK_CHART)
        with self.assertRaises(ValueError) as raised:
            self.verify(self.root, require_interactive_charts=True)
        self.assertIn("missing interactive chart", str(raised.exception))

    def test_strict_chart_check_accepts_embedded_plotly_bundle(self):
        self.write_interactive_charts(self.EMBEDDED_PLOTLY_CHART)
        summary = self.verify(self.root)
        self.assertEqual(summary["valid_geometries"], 30)


if __name__ == "__main__":
    unittest.main()
