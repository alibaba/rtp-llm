import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cases.master_performance.analysis import analyze, validate
from cases.master_performance.publication import publish_performance as report
from workload.gate_evidence import trace_workload_sha
from scenario import compile_scenarios, load_scenarios
from scenario.catalog import handlers

ROOT = Path(__file__).resolve().parents[1]



def metric_panel(directory, evidence, result, presentation=None):
    from cases.master_performance.metrics import produce
    from cases.master_performance.panels import panel
    produce(directory, evidence, result)
    return panel(directory, evidence, result, presentation)


def evidence():
    # Small arithmetic fixture only; never a runnable benchmark.
    c = {
        "benchmark_id": "synthetic-100ms-10qps-v1",
        "warmup_s": 5,
        "measure_s": 30,
        "sample_s": 1,
        "max_gap_s": 3,
        "qps": 10,
        "qps_tolerance": 0.1,
        "min_requests": 250,
        "max_pacing_lag_ms": 100,
        "min_input_tps": 9000,
        "min_output_tps": 70,
        "min_goodput_rps": 9,
        "min_slo_fraction": 0.95,
        "max_error_rate": 0,
        "max_ttft_p99_ms": 1000,
        "max_e2e_p99_ms": 2000,
        "max_tpot_p99_ms": 150,
        "slo_ttft_ms": 1000,
        "slo_e2e_ms": 2000,
        "slo_tpot_ms": 150,
        "max_inflight_growth_rps": 0.2,
    }
    c.update(measure_s=10, min_requests=80, warmup_s=0)
    issued = []
    records = []
    for i in range(100):
        r = dict(
            rid=str(i),
            send_start_epoch_ms=100000 + i * 100,
            input_len=1024,
            output_len=8,
            pacing_lag_ms=0,
        )
        issued.append(r)
        records.append(
            dict(r, status="ok", total_ms=100, ttft_ms=50, observed_output_tokens=8)
        )
    return dict(
        performance_evidence_schema_version=1,
        criteria=c,
        errors=[],
        window=dict(start_epoch_ms=100000, end_epoch_ms=110000),
        samples=[dict(epoch_ms=100000 + i * 1000) for i in range(11)],
        flow=dict(complete=True, errors=[], issued=issued, records=records),
        provenance=dict(
            instance="master_performance::default::batch-window",
            benchmark_id=c["benchmark_id"],
            master_artifact=dict(jar_sha256="a" * 64),
            mock_jar_sha256="b" * 64,
            actual_master_config=dict(dispatcher=dict(type="BATCH")),
            performance=dict(prefill=dict(fixed_ms=100)),
            topology=dict(prefill=2, decode=4),
            capacity=dict(prefill_cache_blocks=1000, decode_cache_blocks=1000),
            trace=dict(sha256="c" * 64, workload_sha256="d" * 64),
            client_environment=dict(FETCH_OUTPUT_STREAM="true"),
            analyzer_sha256="e" * 64,
        ),
    )


class PerformanceGateTest(unittest.TestCase):
    def test_yaml_metric_binding_controls_raw_query_and_gate_validation(self):
        from cases.master_performance.inputs import engine_tps
        e = self.engine_evidence()
        renamed = copy.deepcopy(e["gate_input"])
        roles = renamed["metric_roles"]
        roles["mock/replacement_context_tps"] = roles.pop("mock/rtp_llm_context_tps")
        with self.assertRaisesRegex(ValueError, "match YAML metric_roles"):
            validate(e["criteria"], renamed)
        e["criteria"]["engine_tps"]["mock/replacement_context_tps"] = e["criteria"]["engine_tps"].pop(
            "mock/rtp_llm_context_tps")
        e["gate_input"] = renamed
        for row in e["engine_tps_samples"]:
            if row["metric_id"] == "mock/rtp_llm_context_tps":
                row["metric_id"] = "mock/replacement_context_tps"
        engine_tps(renamed, e["criteria"]["engine_tps"])
        self.assertEqual(analyze(e)["verdict"], "PASS")

    def test_completion_buckets_do_not_depend_on_journal_order(self):
        from workload.gate_evidence import compact_flow
        original = evidence()
        expected = analyze(original)
        shuffled = copy.deepcopy(original)
        shuffled["flow"]["records"].reverse()
        shuffled["flow"]["issued"].reverse()
        actual = analyze(shuffled)
        self.assertEqual(actual["metrics"], expected["metrics"])
        self.assertEqual(actual["windows"], expected["windows"])
        shuffled["flow"] = compact_flow(shuffled["flow"])
        self.assertEqual(analyze(shuffled), actual)

    def test_observe_freezes_run_identity_even_when_provenance_collection_fails(self):
        from types import SimpleNamespace
        from cases.master_performance.actions import observe

        identity = "master_performance::default::single-nonbatch"
        with tempfile.TemporaryDirectory() as d:
            ctx = SimpleNamespace(
                artifact_dir=Path(d), instance=dict(id=identity, profile="single-nonbatch"),
                resource=lambda name, kind: mock.Mock(),
                register_resource=lambda kind, value, **kwargs: value,
            )
            with mock.patch("cases.master_performance.actions.provenance",
                            side_effect=ValueError("artifact missing")):
                observe(ctx, dict(flow="flow", criteria=evidence()["criteria"], gate_input=None), mock.Mock())
            frozen = json.loads((Path(d) / "performance-gate-evidence.json").read_text())
            self.assertEqual(frozen["provenance"]["instance"], identity)
            self.assertEqual(frozen["errors"], ["artifact missing"])
            self.assertEqual(analyze(frozen)["verdict"], "INVALID")

    def test_finish_archives_scoped_raw_evidence_before_analysis(self):
        from types import SimpleNamespace
        from cases.master_performance.actions import finish
        e = self.engine_evidence()
        snapshot = e.pop("flow")
        raw = e.pop("engine_tps_samples")
        flow = mock.Mock()
        flow.evidence_snapshot.return_value = snapshot
        monitor = mock.Mock()
        monitor.metric_rows.side_effect = lambda identity, **kw: [row for row in raw if row["metric_id"] == identity]
        with tempfile.TemporaryDirectory() as d:
            flow.directory = Path(d)
            (flow.directory/"client_lifecycle.jsonl").write_text("journal retained")
            ctx = SimpleNamespace(artifact_dir=Path(d), monitor=monitor,
                resource=lambda name, kind: flow if kind == "java_flow" else e)
            with mock.patch("cases.master_performance.actions.analyze", side_effect=RuntimeError("analysis interrupted")):
                with self.assertRaisesRegex(RuntimeError, "analysis interrupted"):
                    finish(ctx, dict(flow="flow", evidence="evidence"), mock.Mock())
            archived = json.loads((Path(d)/"performance-gate-evidence.json").read_text())
            self.assertEqual(analyze(archived)["metrics"], analyze(dict(e, flow=snapshot))["metrics"])
            self.assertEqual(len(archived["flow"]["journal"]["sha256"]), 64)
            self.assertEqual(archived["engine_tps_samples"], raw)
            self.assertEqual([call.args[0] for call in monitor.metric_rows.call_args_list], list(e["gate_input"]["metric_roles"]))
            monitor.query.assert_not_called()
            monitor.raw.assert_not_called()

    def engine_evidence(self):
        import yaml
        gate_input = yaml.safe_load((ROOT / "config/scenarios/master_performance.yaml").read_text())[
            "parameters"]["gate_inputs"]["engine_tps"]
        roles = gate_input["metric_roles"]
        e = evidence()
        e["gate_input"] = gate_input
        e["criteria"]["engine_tps"] = {name: 100 for name in roles}
        e["engine_tps_samples"] = [
            dict(metric_id=name, metric=dict(__name__=name.split("/")[1], role=role, engine_name=f"{role}-{i}", priority=str(priority)),
                 values=[[t, "60"] for t in range(100, 111)])
            for name, role in roles.items()
            for i in range(e["provenance"]["topology"][role])
            for priority in (0, 50)
        ]
        return e

    def test_html_engine_curves_use_monitor_archive(self):
        panel = metric_panel
        e = self.engine_evidence()
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / "telemetry/1/queries.json"
            archive.parent.mkdir(parents=True)
            archive.write_text(json.dumps(dict(
                start=100, end=110, step=1, targets={}, queries={
                    "mock/rtp_llm_generate_tps_engine_mean": dict(
                        promql="avg by (role) (rtp_llm_generate_tps{job=\"mock\"})",
                        result=[dict(metric=dict(role="decode"),
                                     values=[[100, "120"], [110, "130"]])],
                    ),
                },
            )))
            chart, audit = panel(d, e, analyze(e))
            self.assertTrue(audit["available"])
            self.assertEqual(len(chart["presets"]["Decode TPS"]), 1)
            self.assertTrue(set(chart["presets"]["Decode TPS"]) <= set(chart["presets"]["核心"]))
            self.assertEqual(next(curve for curve in chart["series"]
                                  if curve["name"] == "D generate TPS")["points"],
                             [dict(x=0, y=120), dict(x=10, y=130)])
            self.assertIn("成功 QPS", chart["presets"]["流量"])
            archive.unlink()
            # A new artifact with neither query evidence nor frozen metrics.
            (Path(d) / "metrics.json").unlink()
            chart, _ = panel(d, evidence(), analyze(evidence()))
            self.assertIn("完成输入 TPS", chart["presets"]["核心"])
            # HTML must still exist for INVALID runs, with embedded plotting code.
            bundle = report(d, e, analyze(e))
            self.assertTrue((bundle / "report.html").is_file())

    def test_archived_actual_hit_ratio_is_shown_as_percent(self):
        from cases.master_performance.panels import report_panels
        panel = metric_panel
        from reporting.view_config import view

        e = evidence()
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / "telemetry/1/queries.json"
            archive.parent.mkdir(parents=True)
            archive.write_text(json.dumps(dict(
                start=100, end=110, step=1, targets={},
                queries={"mock/cache_hit_ratio": dict(
                    promql="sum(rate(hit)) / sum(rate(context))",
                    result=[dict(metric=dict(role="prefill"),
                                 values=[[100, "0.42"], [101, "NaN"]])],
                )},
            )))
            chart, _ = panel(d, e, analyze(e))
            panels = report_panels(chart["series"], e["criteria"],
                                   view("master_performance.yaml"))
            hit = panels[3]
            self.assertEqual(hit["id"], "cache-hit")
            self.assertEqual([series["name"] for series in hit["series"]],
                             ["P 实际 token 命中率"])
            self.assertEqual(hit["series"][0]["points"][:2],
                             [dict(x=0, y=42), dict(x=1, y=None)])

    def test_archived_prefill_batch_and_state_panels(self):
        from cases.master_performance.panels import report_panels
        panel = metric_panel
        from reporting.view_config import view

        e = evidence()
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / "telemetry/1/queries.json"
            archive.parent.mkdir(parents=True)
            metrics = {
                "prefill_batch_size_mean": 12.5,
                "prefill_batch_size_p50": 10,
                "prefill_batch_size_p90": 18,
                "prefill_batch_size_p99": 24,
                "waiting_avg": 3,
                "waiting_max": 7,
                "running_avg": 11,
                "running_max": 18,
            }
            archive.write_text(json.dumps(dict(
                start=100, end=110, step=1, targets={},
                queries={"mock/" + name: dict(
                    promql=name,
                    result=[dict(metric=dict(role="prefill"),
                                 values=[[100, str(value)]])],
                ) for name, value in metrics.items()},
            )))
            chart, _ = panel(d, e, analyze(e))
            panels = report_panels(chart["series"], e["criteria"],
                                   view("master_performance.yaml"))
            batch, state = panels[4:]
            self.assertEqual([series["name"] for series in batch["series"]],
                             ["P batch size 均值", "P batch size P50",
                              "P batch size P90", "P batch size P99"])
            self.assertEqual(batch["series"][0]["points"], [dict(x=0, y=12.5)])
            self.assertEqual(batch["series"][0]["axis"], "batch")
            self.assertEqual([series["name"] for series in state["series"]],
                             ["P Waiting / engine", "P Waiting max",
                              "P Running / engine", "P Running max"])
            self.assertEqual(state["series"][0]["points"], [dict(x=0, y=3)])

    def test_observer_gap_retains_request_metrics_without_promoting_verdict(self):
        e = self.engine_evidence()
        e["samples"] = [dict(epoch_ms=100000), dict(epoch_ms=110000)]
        result = analyze(e)
        self.assertEqual(result["verdict"], "INVALID")
        self.assertIn("observer coverage gap", result["errors"])
        self.assertEqual(result["metrics"]["error_rate"], 0)
        self.assertEqual(result["metrics"]["mock/rtp_llm_generate_tps"], 120)
        self.assertEqual(len(result["windows"]), 10)

    def test_engine_tps_floors_are_independent_of_client_tps(self):
        e = self.engine_evidence()
        self.assertEqual(analyze(e)["verdict"], "PASS")
        self.assertEqual(analyze(e)["metrics"]["mock/rtp_llm_context_tps"], 120)
        for row in e["engine_tps_samples"]:
            row["values"] = [[t, "0"] for t in range(100, 111)]
        self.assertEqual(analyze(e)["verdict"], "FAIL")

    def test_engine_tps_missing_engine_gap_and_restart_are_invalid(self):
        e = self.engine_evidence()
        e["engine_tps_samples"] = e["engine_tps_samples"][2:]
        self.assertEqual(analyze(e)["verdict"], "INVALID")
        e = self.engine_evidence()
        for row in e["engine_tps_samples"]:
            row["values"] = [[100, "60"], [110, "60"]]
        self.assertEqual(analyze(e)["verdict"], "INVALID")
        e = self.engine_evidence()
        extra = copy.deepcopy(e["engine_tps_samples"][0])
        extra["metric"]["engine_incarnation"] = "restarted"
        e["engine_tps_samples"].append(extra)
        self.assertEqual(analyze(e)["verdict"], "INVALID")

    def test_request_curve_label_comes_from_yaml_presentation(self):
        from reporting.view_config import view
        from cases.master_performance.panels import report_panels
        panel = metric_panel

        presentation = copy.deepcopy(view("master_performance.yaml"))
        presentation["charts"]["curves"]["request/sent_qps"]["name"] = "YAML sent rate"
        e = evidence()
        with tempfile.TemporaryDirectory() as d:
            chart, _ = panel(d, e, analyze(e), presentation)
        selected = report_panels(chart["series"], e["criteria"], presentation)
        self.assertEqual(selected[1]["series"][0]["name"], "YAML sent rate")

    def test_absolute_success_and_renderer(self):
        e = evidence()
        r = analyze(e)
        self.assertEqual(r["verdict"], "PASS", r)
        self.assertEqual(r["metrics"]["goodput_rps"], 10)
        with tempfile.TemporaryDirectory() as d:
            path = report(d, e, analyze(e))
            self.assertTrue((path / "report.html").is_file())
            spec = json.loads((path / "report-spec.json").read_text())
            self.assertEqual([p["id"] for p in spec["panels"]],
                ["engine-tps", "client-qps", "latency", "cache-hit",
                 "prefill-batch", "prefill-state"])
            for panel in spec["panels"]:
                self.assertTrue(panel["timeX"])
                for series in panel["series"]:
                    self.assertTrue(series["points"])
                    self.assertIn(series["axis"], panel["axes"])
            self.assertEqual(
                analyze(
                    json.loads((Path(d) / "performance-gate-evidence.json").read_text())
                ),
                r,
            )

    def test_multiview_ab_preserves_decode_monitoring_and_request_buckets(self):
        panel = metric_panel

        e = evidence()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = root / "telemetry/1/queries.json"
            archive.parent.mkdir(parents=True)
            archive.write_text(
                json.dumps(
                    dict(
                        start=100,
                        end=110,
                        step=1,
                        targets={},
                        queries={
                            "mock/rtp_llm_context_tps_engine_mean": dict(
                                promql="avg by (role) (sum without (priority) (rtp_llm_context_tps))",
                                result=[dict(metric=dict(role="prefill"), values=[[100, 60000], [101, 61000]])],
                            ),
                            "mock/rtp_llm_context_tps_per_engine": dict(
                                promql="sum without (priority) (rtp_llm_context_tps)",
                                result=[dict(metric=dict(role="prefill", engine_name="p0"), values=[[100, 60000], [101, 61000]])],
                            ),
                            "mock/engine_count": dict(
                                promql="sum by(role)(engines)",
                                result=[
                                    dict(
                                        metric=dict(role=role),
                                        values=[[100, count], [101, count]],
                                    )
                                    for role, count in [("prefill", 2), ("decode", 4)]
                                ],
                            )
                        },
                    )
                )
            )
            chart, audit = panel(root, e, analyze(e))
            self.assertTrue(audit["available"])
            self.assertNotIn("规模", chart["presets"])
            self.assertEqual(len(chart["presets"]["Prefill TPS"]), 1)
            self.assertNotIn("Prefill 逐引擎 TPS", chart["presets"])
            self.assertTrue(set(chart["presets"]["Prefill TPS"]) <= set(chart["presets"]["核心"]))
            self.assertNotIn("完成输入 TPS", chart["presets"]["核心"])
            self.assertIn("完成输入 TPS", chart["presets"]["客户端吞吐"])
            by_name = {c["name"]: c for c in chart["series"]}
            self.assertNotIn("mock · D engine count", by_name)
            from monitoring.metric_store import MetricStore
            store = MetricStore.read(root)
            self.assertEqual(store.select("mock/engine_count", labels={"role": "decode"})[0]["points"][0], [100, 4])
            self.assertEqual(store.select("mock/rtp_llm_context_tps_per_engine", labels={"role": "prefill"})[0]["points"][0], [100, 60000])
            self.assertEqual(by_name["TTFT p99"]["points"][0]["y"], 50)
            self.assertEqual(by_name["到达 cohort 成功率"]["points"][0]["y"], 1)
            self.assertEqual(
                sum(p["y"] for p in by_name["完成输出 TPS"]["points"]), 792
            )

    def test_tail_cohort_and_actual_tokens_not_requested_budget(self):
        e = evidence()
        for r in e["flow"]["records"]:
            r["observed_output_tokens"] = 2
        r = analyze(e)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertLess(r["metrics"]["output_tps"], 20)
        # Last request finishes beyond measurement, remains in cohort SLO denominator.
        e = evidence()
        e["flow"]["records"][-1]["total_ms"] = 5000
        r = analyze(e)
        self.assertEqual(r["metrics"]["cohort_requests"], 100)
        self.assertEqual(r["metrics"]["goodput_rps"], 9.9)
        self.assertEqual(r["metrics"]["slo_fraction"], 0.99)

    def test_any_failure_including_warmup_blocks_gate(self):
        for index in (0, 99):
            e = evidence()
            e["flow"]["records"][index]["status"] = "timeout"
            self.assertEqual(analyze(e)["verdict"], "FAIL")
        e = evidence()
        for r in (e["flow"]["issued"][0], e["flow"]["records"][0]):
            r["send_start_epoch_ms"] = 99900
        e["flow"]["records"][0]["status"] = "timeout"
        self.assertEqual(analyze(e)["verdict"], "FAIL")

    def test_zero_success_is_failure_not_missing_percentile_invalid(self):
        e = evidence()
        for r in e["flow"]["records"]:
            r["status"] = "engine_error"
        self.assertEqual(analyze(e)["verdict"], "FAIL")

    def test_missing_or_corrupt_evidence_never_passes(self):
        mutations = [
            lambda e: e["flow"]["records"][0].pop("observed_output_tokens"),
            lambda e: e["provenance"].pop("mock_jar_sha256"),
            lambda e: e["provenance"].pop("analyzer_sha256"),
            lambda e: e.update(provenance=[]),
            lambda e: e.update(errors=None),
            lambda e: e["flow"].update(complete=False),
            lambda e: e["flow"]["records"].pop(),
            lambda e: e["flow"]["records"].append(e["flow"]["records"][0]),
            lambda e: e["flow"]["records"][0].update(ttft_ms=float("nan")),
            lambda e: e["flow"]["records"][0].update(status="scheduled"),
            lambda e: e.update(samples=e["samples"][:2] + e["samples"][8:]),
            lambda e: e["provenance"]["client_environment"].update(
                FETCH_OUTPUT_STREAM="false"
            ),
        ]
        for mutate in mutations:
            e = evidence()
            mutate(e)
            self.assertEqual(analyze(e)["verdict"], "INVALID")

    def test_sender_invalid_and_delayed_completion_backlog_fails(self):
        e = evidence()
        e["flow"]["issued"][1]["pacing_lag_ms"] = 1000
        self.assertEqual(analyze(e)["verdict"], "INVALID")
        e = evidence()
        for r in e["flow"]["records"][50:]:
            r["total_ms"] = 20000
        r = analyze(e)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertGreater(r["metrics"]["inflight_growth_rps"], 4)

    def test_one_token_tpot_not_applicable(self):
        e = evidence()
        for r in e["flow"]["records"]:
            r["observed_output_tokens"] = 1
        e["criteria"]["min_output_tps"] = 1
        r = analyze(e)
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(
            next(x for x in r["checks"] if x["metric"] == "tpot_p99_ms")["status"],
            "NOT_APPLICABLE",
        )

    def test_workload_checksum_preserves_everything_except_rid(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "a"
            b = Path(d) / "b"
            a.write_text('{"rid":"left:1","il":1,"ol":2,"input_ids":[3]}\n')
            b.write_text('{"rid":"right:1","il":1,"ol":2,"input_ids":[3]}\n')
            self.assertEqual(trace_workload_sha(a), trace_workload_sha(b))
            b.write_text('{"rid":"right:1","il":1,"ol":3,"input_ids":[3]}\n')
            self.assertNotEqual(trace_workload_sha(a), trace_workload_sha(b))

    def test_contract_rejects_relaxed_success_and_compiles_both_profiles(self):
        c = evidence()["criteria"]
        c["max_error_rate"] = 0.01
        with self.assertRaises(ValueError):
            validate(c)
        with mock.patch("scenario.compiler.VICTIM_OFFSETS", (300, 301, 302)):
            plans = compile_scenarios(
                load_scenarios(ROOT / "config/scenarios/master_performance.yaml"),
                handlers=handlers(),
            )
        self.assertEqual(len(plans), 2)
        for p in plans:
            self.assertIn("performance_finish", [s["action"] for s in p["stages"]])

    def test_default_runner_selects_core_without_external_performance_input(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/commands/run_cases.py"),
             "--dry-run", "--parallel", "1"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout[result.stdout.index("\n{") + 1:])
        self.assertEqual(len(plan["instances"]), 1)
        self.assertNotIn("master_performance::", result.stdout)
